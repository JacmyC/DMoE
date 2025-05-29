import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetMix
import torch.nn.functional as F
# from ptflops import get_model_complexity_info

class DCNv2(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCNv2", 
                 gpu=-1,
                 model_structure="stacked_parallel",
                 use_low_rank_mixture=False,
                 low_rank=32,
                 num_experts=4,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 stacked_dnn_hidden_units=[], 
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None, 
                 **kwargs):
        super(DCNv2, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        
        self.embedding_type = kwargs.get("embedding_type", None)
        self.num_models = kwargs.get("num_models", None)
        print('Num expert:', self.num_models)
        
        # self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
    

        self.multi_embedding_layers = nn.ModuleList([FeatureEmbedding(feature_map, embedding_dim) for _ in range(self.num_models)] )


        ### use corrloss
        self.use_corrloss = kwargs.get("use_corrloss", None)
        self.use_gating = kwargs.get("use_gating", None)
        input_dim = feature_map.sum_emb_out_dim()

        # self.feature_gating = FeatureSelection(feature_map, feature_map.sum_emb_out_dim(), embedding_dim, fs_hidden_units=[1000])
        self.feature_gating = nn.Sequential(
            nn.Linear(feature_map.sum_emb_out_dim(), feature_map.sum_emb_out_dim()),
        )
        activation_dict = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'prelu': nn.PReLU(),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),
            'linear': nn.Identity(),
        }


        self.corr_weight = kwargs.get("corr_weight", None)




        ### Add Simple Gating Mechanism

        self.gating_embeddinglayer = FeatureEmbedding(feature_map, embedding_dim)
        # self.gating_network = nn.Linear(input_dim, self.num_models)
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_models),
        )



        
        self.crossnets = nn.ModuleList([CrossNetV2(input_dim, num_cross_layers, embedding_dim, self.nonlinear, self.concat_emb, self.gamma, self.symmetric, kwargs) for _ in range(self.num_models)])


        self.model_structure = model_structure
        assert self.model_structure in ["crossnet_only", "stacked", "parallel", "stacked_parallel", "MEMoE_gating","single_parallel","single_crossnet"], \
               "model_structure={} not supported!".format(self.model_structure)
        if self.model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnns = nn.ModuleList([
                                                MLP_Block(input_dim=input_dim,
                                                        output_dim=None,
                                                        hidden_units=stacked_dnn_hidden_units,
                                                        hidden_activations=dnn_activations,
                                                        output_activation=None,
                                                        dropout_rates=net_dropout,
                                                        batch_norm=batch_norm)
                                                for _ in range(self.num_models)
                                                ])
            final_dim = stacked_dnn_hidden_units[-1]
            
        if self.model_structure in ["parallel", "stacked_parallel","MEMoE_gating","single_parallel"]:
            self.parallel_dnns = nn.ModuleList([
                                                MLP_Block(input_dim=input_dim,
                                                        output_dim=None,
                                                        hidden_units=parallel_dnn_hidden_units,
                                                        hidden_activations=dnn_activations,
                                                        output_activation=None,
                                                        dropout_rates=net_dropout,
                                                        batch_norm=batch_norm)
                                                for _ in range(self.num_models)
                                                ])

            final_dim = input_dim + parallel_dnn_hidden_units[-1]
        if self.model_structure == "stacked_parallel":
            final_dim = stacked_dnn_hidden_units[-1] + parallel_dnn_hidden_units[-1]
        if self.model_structure in ["crossnet_only","single_crossnet"]: # only CrossNet
            final_dim = input_dim

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 500),
            nn.ReLU(),
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
        loss += self.corr_loss
        return loss
    
    def compute_corr_loss(self, A, B):
        #std 
        A = (A - A.mean(0)) / (A.std(0) + 1e-8)
        B = (B - B.mean(0)) / (B.std(0) + 1e-8)
        cov = (A.T @ B / (A.shape[0] - 1)) ** 2
        loss = self.corr_weight * cov.sum()

        return loss / (cov.shape[0] ** 2)
    


    def forward(self, inputs):
        self.grad_var_list = []
        X = self.get_inputs(inputs)
        feature_emb = [None] * self.num_models
        #feature_emb = self.embedding_layer(X, flatten_emb=True)

        for i in range(self.num_models):
            feature_emb[i] = self.multi_embedding_layers[i](X, flatten_emb=True)
            #print(feature_emb[i].shape)
        #feature_emb = self.single_embedding_layer(X, flatten_emb=True)

        cross_out = [None] * self.num_models
        for i in range(self.num_models):
            # cross_out[i] = self.crossnets[i](feature_emb[i],gating=None)
            cross_out[i] = self.crossnets[i](feature_emb[i])

        gating_emb = self.gating_embeddinglayer(X, flatten_emb=True)
        gating_weights = self.gating_network(gating_emb)
        
        gating_weights = torch.softmax(gating_weights, dim=1)


        if self.model_structure == "crossnet_only":
            self.corr_loss = 0.0
            final_outs = [None] * self.num_models
            for i in range(self.num_models):
                final_outs[i] = self.model_fc_layers[i](cross_out[i])

            if self.use_corrloss == True and self.num_models > 1:
                corrloss = 0.0
                for i in range(self.num_models):
                    for j in range(i+1, self.num_models):
                        corrloss += self.compute_corr_loss(final_outs[i],final_outs[j])
        
                self.corr_loss = corrloss / (self.num_models * (self.num_models - 1) / 2.0)
                
            gating_weights = gating_weights.permute(1, 0).unsqueeze(-1)
            final_outs = torch.stack(final_outs)
            gating_weights = gating_weights.expand_as(final_outs)
            final_out = torch.sum(gating_weights * final_outs, dim=0)
    

    
        elif self.model_structure == "stacked":
            self.corr_loss = 0.0
            final_outs = [None] * self.num_models
            # for i in range(self.num_models):
            #     final_outs[i] = self.stacked_dnns[i](cross_out[i])

            if self.use_corrloss and self.num_models >= 2:
                corrloss = 0.0
                for i in range(self.num_models):
                    for j in range(i+1, self.num_models):
                        corrloss += self.compute_corr_loss(final_outs[i],final_outs[j])
                self.corr_loss = corrloss / (self.num_models * (self.num_models - 1) / 2.0)

            final_out = sum(final_outs)/self.num_models

            gating_weights = gating_weights.permute(1, 0).unsqueeze(-1)
            final_outs = torch.stack(final_outs)
            gating_weights = gating_weights.expand_as(final_outs)
            final_out = torch.sum(gating_weights * final_outs, dim=0)

        elif self.model_structure == "parallel":
            self.corr_loss = 0.0
            final_outs = [None] * self.num_models
            dnn_outs = [None] * self.num_models
            for i in range(self.num_models):
                dnn_outs[i] = self.parallel_dnns[i](feature_emb[i])
                #dnn_outs[i] = self.parallel_dnns[i](feature_emb)
                final_outs[i] = torch.cat([cross_out[i], dnn_outs[i]], dim=-1)
                # final_outs[i] = self.model_fc_layers[i](final_outs[i])

            if self.use_corrloss and self.num_models >= 2:
                corrloss = 0.0
                for i in range(self.num_models):
                    for j in range(i+1, self.num_models):
                        corrloss += self.compute_corr_loss(final_outs[i],final_outs[j])
                self.corr_loss = corrloss / (self.num_models * (self.num_models - 1) / 2.0)
            # print(self.corr_loss)

            gating_weights = gating_weights.permute(1, 0).unsqueeze(-1)
            final_outs = torch.stack(final_outs)
            gating_weights = gating_weights.expand_as(final_outs)

            final_out = torch.sum(gating_weights * final_outs, dim=0)


        y_pred = self.fc(final_out)

        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

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
        self.transform_input = nn.Linear(input_dim, input_dim, bias=False)


    def init_record(self):
        self.record_cross_emb = []
        self.record_cross_emb_residual = []
        self.record_crossnet_representation = []
    

    def forward(self, feature_embedding, gating=None):
        self.grad_var_list = []
        X_0 = feature_embedding
        if gating is not None:
            X_0 = gating
        X_i = feature_embedding # b x dim
        self.inner_corr_loss = 0
        #print(self.num_layers)
        for i in range(self.num_layers):
            tmp = self.cross_layers[i](X_i)
            X_i = X_i + X_0 * tmp

        return X_i
