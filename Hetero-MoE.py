import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CompressedInteractionNet, LogisticRegression
from fuxictr.pytorch.layers import CrossNetMix
import torch.nn.functional as F

class DHEN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DHEN", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                #  model_structure="cin_only",
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU",
                 cin_hidden_units=[16, 16, 16],
                 num_cross_layers=3, 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DHEN, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)  
        self.num_models = kwargs.get("num_models", None)
        print('Num expert:', self.num_models)

        #Multi Embedding Table
        self.multi_embedding_layers = nn.ModuleList([FeatureEmbedding(feature_map, embedding_dim) for _ in range(self.num_models)] )
        # self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.expert_name = kwargs.get("expert_name", [])

        self.use_corrloss = kwargs.get("use_corrloss", None)
        self.corr_weight = kwargs.get("corr_weight", 0.0)

        input_dim = feature_map.sum_emb_out_dim()
        
        ### Add Simple Gating Mechanism
        self.gating_embeddinglayer = FeatureEmbedding(feature_map, embedding_dim)
        # self.gating_network = nn.Linear(input_dim, self.num_models)
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_models),
        )

        ### CIN
        self.CIN = CompressedInteractionNet(feature_map.num_fields, cin_hidden_units, output_dim=1)
        cin_final_dim = cin_hidden_units[-1] * len(cin_hidden_units)
        self.DCN = CrossNetV2(input_dim, num_cross_layers, embedding_dim,kwargs)
        dcn_final_dim = input_dim

        self.DNN = MLP_Block(input_dim=input_dim,
                            output_dim=None,
                            hidden_units=dnn_hidden_units,
                            hidden_activations=dnn_activations,
                            output_activation=None,
                            dropout_rates=net_dropout,
                            batch_norm=batch_norm)

        self.CIN_align_layers = nn.Sequential(
            nn.Linear(cin_final_dim, 500),
            nn.ReLU(),
            # nn.Linear(500, 500)
        )
        self.DCN_align_layers = nn.Sequential(
            nn.Linear(dcn_final_dim, 500),
            nn.ReLU(),
            # nn.Linear(500, 500)
        )
        self.FM_align_layers = nn.Sequential(
            nn.Linear(embedding_dim, 500),
            nn.ReLU(),
            # nn.Linear(500, 500)
        )
        self.fc = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(),
            # nn.Linear(500, 500),
            # nn.ReLU(),
            nn.Linear(500, 1)
        )

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()


    def init_record(self):
        self.record_feature_emb = []
        self.record_expert_out = []

    def compute_loss(self, return_dict, y_true):
        loss = super().compute_loss(return_dict, y_true)
        # print(loss,self.corr_loss,self.corr_loss/loss)
        loss += self.corr_loss
        return loss
    
    def get_expert_output(self, feature_emb, expert_name):
        if expert_name == 'CIN':
            expert_out = torch.cat(self.CIN(feature_emb), dim=-1)
        elif expert_name == 'DCN':
            feature_emb = feature_emb.reshape(feature_emb.shape[0], feature_emb.shape[1] * feature_emb.shape[2])
            expert_out = self.DCN(feature_emb)
        elif expert_name == 'DNN':
            feature_emb = feature_emb.reshape(feature_emb.shape[0], feature_emb.shape[1] * feature_emb.shape[2])
            expert_out = self.DNN(feature_emb)
        elif expert_name == 'FM':
            row, col = torch.triu_indices(feature_emb.shape[1], feature_emb.shape[1], offset=1)
            rst = feature_emb[:, row] * feature_emb[:, col]
            expert_out = rst.sum(-2)
        # print(expert_out.shape)
        return expert_out

    
    def compute_corr_loss(self, A, B):
        #std 
        A = (A - A.mean(0)) / (A.std(0) + 1e-8)
        B = (B - B.mean(0)) / (B.std(0) + 1e-8)
        #cov = (A.T @ B / (A.shape[0] - 1))
        cov = (A.T @ B / (A.shape[0] - 1)) ** 2
        loss = self.corr_weight * cov.sum()

        return loss / (cov.shape[0] * cov.shape[1])
    
    def forward(self, inputs):
        X = self.get_inputs(inputs)

        feature_emb = [None] * self.num_models
        # # flatten_feat_emb = [None] * self.num_models
        for i in range(self.num_models):
            feature_emb[i] = self.multi_embedding_layers[i](X)
        #     flatten_feat_emb[i] = feature_emb[i].reshape(feature_emb[i].shape[0],feature_emb[i].shape[1]*feature_emb[i].shape[2])
        
        # feature_emb = self.embedding_layer(X)


        # feature_emb_CIN = self.multi_embedding_layers[0](X)
        # feature_emb_DCN = self.multi_embedding_layers[1](X, flatten_emb=True)
        # feature_emb_DNN = self.multi_embedding_layers[2](X, flatten_emb=True)
        # row, col = torch.triu_indices(feature_emb[i].shape[1], feature_emb[i].shape[1], offset=1)
        # rst = feature_emb[i][:, row] * feature_emb[i][:, col]
        # bi_pooling_vec = rst.sum(-2)

        #Gating
        gating_emb = self.gating_embeddinglayer(X, flatten_emb=True)
        gating_weights = self.gating_network(gating_emb)
        gating_weights = torch.softmax(gating_weights, dim=1)


        final_outs = [None] * self.num_models
        for i in range(self.num_models):
            final_outs[i] = self.get_expert_output(feature_emb[i], self.expert_name[i])
            if self.expert_name[i] == 'FM':
                final_outs[i] = self.FM_align_layers(final_outs[i])
            elif self.expert_name[i] == 'CIN':
                final_outs[i] = self.CIN_align_layers(final_outs[i])
            elif self.expert_name[i] == 'DCN':
                final_outs[i] = self.DCN_align_layers(final_outs[i])

        if self.analyzing:
            for i in range(self.num_models):
                # self.record_feature_emb.append(feature_emb.detach().clone().cpu())
                self.record_feature_emb.append(feature_emb[i].detach().clone().cpu())
            for i in range(self.num_models):
                self.record_expert_out.append(final_outs[i].detach().clone().cpu())
                print(self.record_expert_out[0].shape, final_outs[i].shape)
        # print(final_outs[i].shape)
        # self.record_feature_emb = [None] * self.num_models
        # if self.analyzing:
        #     self.record_feature_emb = [None] * self.num_models
        #     self.record_expert_out = [None] * self.num_models
        #     for i in range(self.num_models):
        #         self.record_feature_emb[i] = feature_emb[i].detach().clone().cpu()
        #         self.record_expert_out[i] = final_outs[i].detach().clone().cpu()
                
            # print(len(self.record_expert_out))
                
            # for i in range(self.num_models):
            #     self.record_expert_out.append(final_outs[i].detach().clone().cpu().unsqueeze(0))  # 每个 final_outs[i] 是 [100, 20]
            # self.record_expert_out = torch.cat(self.record_expert_out,dim=0)  # 堆叠为 [2, 100, 20]

        # final_outs[0] = torch.cat(self.CIN(feature_emb_CIN), dim=-1)
        # final_outs[1] = self.DCN(feature_emb_DCN)
        # final_outs[2] = self.DNN(feature_emb_DNN)

        ### align   
        # final_outs[0] = self.CIN_align_layers(final_outs[0])
        # final_outs[1] = self.DCN_align_layers(final_outs[1])
        
        self.corr_loss = 0.0
        if self.use_corrloss == True and self.num_models > 1:
            corrloss = 0.0
            for i in range(self.num_models):
                for j in range(i+1, self.num_models):
                    corrloss += self.compute_corr_loss(final_outs[i],final_outs[j])
                    # print(corrloss)
            self.corr_loss = corrloss / (self.num_models * (self.num_models - 1) / 2.0)
        gating_weights = gating_weights.permute(1, 0).unsqueeze(-1)
        final_outs = torch.stack(final_outs)
        gating_weights = gating_weights.expand_as(final_outs)
        final_out = torch.sum(gating_weights * final_outs, dim=0)

        y_pred = self.fc(final_out)

        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
    
#### DCN Block
class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers, embedding_dim=None, kwargs=None):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
                                          for _ in range(self.num_layers))
        self.transform_layers = nn.ModuleList(nn.Linear(input_dim, embedding_dim, bias=False)
                                              for _ in range(self.num_layers - 1))
        self.batch_norm_layers = nn.ModuleList(nn.BatchNorm1d(1)
                                          for _ in range(self.num_layers))
        self.embedding_dim = embedding_dim
        self.num_field = input_dim // embedding_dim


    def init_record(self):
        self.record_cross_emb = []
        self.record_cross_emb_residual = []
        self.record_crossnet_representation = []

    def forward(self, feature_embedding, gating=None):
        # self.grad_var_list = []
        X_0 = feature_embedding
        if gating is not None:
            X_0 = gating
        X_i = feature_embedding # b x dim

        for i in range(self.num_layers):
            tmp = self.cross_layers[i](X_i)
            X_i = X_i + X_0 * tmp
            #X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i