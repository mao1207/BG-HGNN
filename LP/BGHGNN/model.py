import time

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, GraphConv

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


class BGHGNN(nn.Module):
    def __init__(self, G, input, in_dim, hidden_dim, num_layers, heads, feat_drop = 0.1, attn_drop = 0.1, negative_slope=0.2,
                 residual=False, beta=0.0):
        super(BGHGNN, self).__init__()
        current_time = int(time.time())
        th.manual_seed(current_time)
        print(current_time)
    
        # create layers
        self.input = input
        self.ntypes = G.ntypes
        print(self.ntypes)
        self.hgn_layers = nn.ModuleList()
        self.activation = F.elu
        self.edge_dim = 512
        self.num_etypes = len(G.etypes)
        self.num_layers = num_layers
        self.rank = 4
        self.embedding_dim = 512
        self.combine_dim = 64

        print(G.canonical_etypes)
        self.node_embedding = (th.rand(size=(len(G.ntypes), self.embedding_dim)).to(device))
        self.edge_type_embeddings = (th.rand(size=(len(G.canonical_etypes), self.edge_dim)).to(device))

        self.ntype_to_start = {}
        begin = 0
        for ntype in G.ntypes:
            self.ntype_to_start[ntype] = begin
            begin += G.number_of_nodes(ntype)

        self.common_features_factor = nn.Parameter(th.Tensor(self.rank, in_dim + 1, self.combine_dim))
        self.type_embedding_factor = nn.Parameter(th.Tensor(self.rank, self.embedding_dim + 1, self.combine_dim))
        self.relation_embedding_factor = nn.Parameter(th.Tensor(self.rank, self.edge_dim + 1, self.combine_dim))

        self.fusion_weights = nn.Parameter(th.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(th.Tensor(1, self.combine_dim))

        self.h_type_features = th.empty(0).to(device)
        self.h_relation_features = th.zeros((G.number_of_nodes(), self.edge_dim)).to(device)
        
        degrees = th.zeros(G.number_of_nodes(), dtype=th.long).to(device)

        begin = 0
        for i, ntype in enumerate(G.ntypes):
            end = begin + G.number_of_nodes(ntype)
            h_temp = self.node_embedding[i].repeat(G.number_of_nodes(ntype), 1)
            self.h_type_features = th.cat((self.h_type_features, h_temp))
            begin = end

        for index, etype in enumerate(G.canonical_etypes):
            G_src, G_dst = G.edges(etype=etype)
            src = G_src + self.ntype_to_start[etype[0]]
            dst = G_dst + self.ntype_to_start[etype[2]]
            edge_embedding = self.edge_type_embeddings[index].repeat(len(src), 1)
            # edge_embedding_reverse = self.edge_type_embeddings[index + len(G.canonical_etypes)].repeat(len(src), 1)
            self.h_relation_features.index_add_(0, dst, edge_embedding)
            self.h_relation_features.index_add_(0, src, -edge_embedding)
            deg_for_etype_src = th.bincount(dst, minlength=G.number_of_nodes())
            deg_for_etype_dst = th.bincount(src, minlength=G.number_of_nodes())
            degrees += deg_for_etype_src
            degrees += deg_for_etype_dst

        degree = degrees.float().clamp(min=1).to(device)
        self.h_relation_features /= degree.view(-1,1)

        nn.init.xavier_uniform_(self.common_features_factor)
        nn.init.xavier_uniform_(self.type_embedding_factor)
        nn.init.xavier_uniform_(self.relation_embedding_factor)
        nn.init.xavier_uniform_(self.fusion_weights)
        nn.init.xavier_uniform_(self.fusion_bias)
        nn.init.xavier_uniform_(self.node_embedding)
        nn.init.xavier_uniform_(self.edge_type_embeddings)
        print(self.common_features_factor)

        
        ones_tensor = th.ones(self.input.to(device).size(0), 1, device=device)

        self.h_common_features = th.cat([self.input.to(device), ones_tensor], dim=-1)
        self.h_type_features = th.cat([self.h_type_features, ones_tensor], dim=-1)
        self.h_relation_features = th.cat([self.h_relation_features, ones_tensor], dim=-1)

        self.hgn_layers.append(
            GATConv(
                self.combine_dim,
                hidden_dim,
                heads[0],
                residual = residual,
                feat_drop = feat_drop,
                attn_drop = attn_drop
            )
        )
        # hidden layers
        for l in range(1, self.num_layers):  # noqa E741
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.hgn_layers.append(
                GATConv(
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    residual = True,
                    feat_drop = feat_drop,
                    attn_drop = attn_drop
                )
            )

        print(self.hgn_layers)

    def forward(self, H):

        fusion_common_features = th.matmul(self.h_common_features, self.common_features_factor)
        fusion_type_features = th.matmul(self.h_type_features, self.type_embedding_factor)
        # print(self.h_type_features.shape)
        # print(self.type_embedding_factor.shape)
        # print(fusion_type_features.shape)
        # print(fusion_type_features)
        fusion_relation_features = th.matmul(self.h_relation_features, self.relation_embedding_factor)
        fusion_zy = fusion_common_features * fusion_type_features * fusion_relation_features

        h = th.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        h = self.activation(h)

        # h=self.input.to(device)

        for l in range(self.num_layers):
            h, att = self.hgn_layers[l](H, h, get_attention = True)
            h = h.flatten(1)
        
        att = att.squeeze()

        return h, att