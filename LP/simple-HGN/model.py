import math
import random
import time
from abc import ABCMeta

import dgl
import dgl.function as fn
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import TypedLinear
from gensim.models import Word2Vec
from transformers import BertModel, BertTokenizer

# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


device = torch.device("cuda")
# device = torch.device("cpu")


class simpleHGN(nn.Module):
    def __init__(self, G, input, in_dim, hidden_dim, num_layers, heads, feat_drop=0.0, negative_slope=0.2,
                 residual=True, beta=0.0):
        super(simpleHGN, self).__init__()
        current_time = int(time.time())
        torch.manual_seed(current_time)
        # create layers
        self.h_dict = input
        self.ntypes = G.ntypes
        self.hgn_layers = nn.ModuleList()
        self.activation = F.elu
        self.edge_dim = 256
        self.num_etypes = len(G.etypes)
        self.num_layers = num_layers
        self.G = G

        # input projection (no residual)
        self.hgn_layers.append(
            SimpleHGNConv(
                self.edge_dim,
                in_dim,
                hidden_dim,
                heads[0],
                self.num_etypes,
                feat_drop,
                negative_slope,
                False,
                self.activation,
                beta=beta,
            )
        )
        # hidden layers
        for l in range(1, self.num_layers - 1):  # noqa E741
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.hgn_layers.append(
                SimpleHGNConv(
                    self.edge_dim,
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    self.num_etypes,
                    feat_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    beta=beta,
                )
            )

    def forward(self, G):
        if hasattr(G, 'ntypes'):
            # full graph training,
            with G.local_scope():
                max_length = max([feat.shape[1] for feat in self.h_dict.values()])  # 假设特征维度在第二个轴上
                padded_h_dict = {ntype: torch.nn.functional.pad(feat.to_dense(), (0, max_length - feat.to_dense().shape[1])) for ntype, feat in self.h_dict.items()}
                
                for ntype in G.ntypes:
                    G.nodes[ntype].data['h'] = padded_h_dict[ntype]
                    padded_h = padded_h_dict[ntype]
                    
                if len(G.nypes) == 1:
                    G.ndata['h'] = padded_h
                else:
                    G.ndata['h'] = padded_h_dict

                g = dgl.to_homogeneous(G, ndata='h')
                h = g.ndata['h']
                for l in range(self.num_layers - 1):  # noqa E741
                    h = self.hgn_layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)
                    h = h.flatten(1)
        else:
            # for minibatch training, input h_dict is a tensor
            for l, block in enumerate(G):
                edge_type_tensor = torch.full((block.number_of_edges(),), 0).int().to(device)

                begin = 0
                for index, etype in enumerate(block.canonical_etypes):
                    end = begin + block.edges(etype=etype)[0].shape[0]
                    edge_type_tensor[begin:end] = index
                    begin = end

                if l == 0:
                    h = block.srcdata['inp']
                # for ntype in block.ntypes:
                #     sample_nodes = block.dstdata[dgl.NID][ntype]
                #     h = torch.cat((h, self.h_dict[ntype][sample_nodes]), dim=0)

                h = self.hgn_layers[l](block, h, edge_type_tensor, edge_type_tensor, presorted=False)
            # torch.set_printoptions(threshold=torch.inf)
            # print('x1,', h.shape)
            # h_dict = to_hetero_feat(h, block.ndata['_TYPE'][block.num_src_nodes() + 1:] - 7, self.ntypes)
            # print('x2,', h.shape)
            h_dict = h

        return h
        # H = dgl.to_homogeneous(G).to(device)
        # h = self.input
        # for l in range(0, self.num_layers):  # noqa E741
        #     h = self.hgn_layers[l](H, h, H.ndata['_TYPE'], H.edata['_TYPE'], True)
        #     h = h.flatten(1)
        # h_dict = to_hetero_feat(h, H.ndata['_TYPE'], G.ntypes)
        #
        # return h_dict[out_key]


class SimpleHGNConv(nn.Module):
    r"""
    The SimpleHGN convolution layer.

    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    num_heads: int
        the number of heads
    num_etypes: int
        the number of edge type
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    activation: str
        the activation function
    beta: float
        the hyperparameter used in edge residual
    """

    def __init__(self, edge_dim, in_dim, out_dim, num_heads, num_etypes, feat_drop=0.0,
                 negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super(SimpleHGNConv, self).__init__()
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_etypes = num_etypes

        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim)))

        self.W = nn.Parameter(torch.FloatTensor(
            in_dim, out_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)

        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))

        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)
        else:
            self.register_buffer("residual", None)

        self.beta = beta

    def forward(self, g, h, project, etype, presorted=False):
        """
        The forward part of the SimpleHGNConv.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        h: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``

        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        if g.is_block:
            h_t = torch.empty(0).to(device)
            for ntype in g.srctypes:
                h_t = torch.cat((h_t, h[ntype]), dim=0)

            emb = self.feat_drop(h_t)
            emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
            emb[torch.isnan(emb)] = 0.0
            edge_emb = self.W_r(self.edge_emb[etype], etype, presorted).view(-1, self.num_heads, self.edge_dim)

            emb_t = {}
            begin = 0
            for (key, value) in h.items():
                end = begin + value.shape[0]
                emb_t[key] = emb[begin:end].squeeze()
                begin = end

            row = torch.empty(0).int().to(device)
            col = torch.empty(0).int().to(device)
            for etype in g.canonical_etypes:
                if g.edges(etype=etype)[0].shape[0] != 0:
                    row = torch.cat((row, g.edges(etype=etype)[0]), dim=0)
                    col = torch.cat((col, g.edges(etype=etype)[1]), dim=0)

            h_l = (self.a_l * emb).sum(dim=-1)[row]
            h_r = (self.a_r * emb).sum(dim=-1)[col]
            h_e = (self.a_e * edge_emb).sum(dim=-1)
            att = self.leakyrelu(h_l + h_e + h_r)

            edge_attention_t = {}
            begin = 0
            for etype in g.canonical_etypes:
                end = begin + g.edges(etype=etype)[0].shape[0]
                edge_attention_t[etype] = att[begin:end]
                begin = end

            edge_attention = {etype: edge_softmax(g[etype], attn) for etype, attn in edge_attention_t.items()}

            if 'alpha' in g.edata.keys():
                res_attn = g.edata['alpha']
                edge_attention = edge_attention * \
                                 (1 - self.beta) + res_attn * self.beta

            with g.local_scope():
                g.edata['alpha'] = edge_attention
                g.srcdata['emb'] = emb_t
                g.multi_update_all({etype: (fn.u_mul_e('emb', 'alpha', 'm'), fn.sum('m', 'emb')) \
                                    for etype in g.canonical_etypes}, cross_reducer='mean')
                for ntype in g.dsttypes:
                    g.dstdata['emb'][ntype] = g.dstdata['emb'][ntype].view(-1, self.out_dim * self.num_heads)
                h_output = g.dstdata['emb']

        else:
            emb = self.feat_drop(h)
            emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
            emb[torch.isnan(emb)] = 0.0

            edge_emb = self.W_r(self.edge_emb[etype], etype, presorted).view(-1, self.num_heads, self.edge_dim)

            row = g.edges()[0]
            col = g.edges()[1]

            h_l = (self.a_l * emb).sum(dim=-1)[row]
            h_r = (self.a_r * emb).sum(dim=-1)[col]
            h_e = (self.a_e * edge_emb).sum(dim=-1)

            edge_attention = self.leakyrelu(h_l + h_e + h_r)
            edge_attention = edge_softmax(g, edge_attention)

            with g.local_scope():
                emb = emb.permute(0, 2, 1).contiguous()
                g.edata['alpha'] = edge_attention
                g.srcdata['emb'] = emb

                g.update_all(fn.u_mul_e('emb', 'alpha', 'm'), fn.sum('m', 'emb'))
                # g.update_all(fn.copy_u('emb', 'm'), fn.sum('m', 'emb'))
                h_output = g.dstdata['emb'].view(-1, self.out_dim * self.num_heads)
            if self.residual:
                res = self.residual(h)
                h_output += res
            if self.activation is not None:
                h_output = self.activation(h_output)

        return h_output


# class myLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, negative_slope=0.2,
#                  residual=True, activation=F.elu):
#         super(myLayer, self).__init__()

#         self.in_dim        = in_dim
#         self.out_dim       = out_dim
#         self.n_heads       = num_heads
#         self.d_k           = out_dim // num_heads
#         self.sqrt_dk       = math.sqrt(self.d_k)
#         self.att           = None

#         self.W = nn.Parameter(torch.FloatTensor(
#             in_dim, out_dim * num_heads))

#         self.k_linears = nn.Linear(in_dim, out_dim * num_heads)
#         self.q_linears = nn.Linear(in_dim, out_dim * num_heads)
#         self.v_linears = nn.Linear(in_dim, out_dim * num_heads)

#         self.drop           = nn.Dropout(dropout)
#         self.activation = activation
#         self.leakyrelu = nn.LeakyReLU(negative_slope)

#         if residual:
#             self.residual = nn.Linear(in_dim, out_dim * num_heads)

#     def forward(self, G, h_dict, h):
#             v = self.v_linears(h).view(-1, self.n_heads, self.out_dim)
#             q = self.q_linears(h).view(-1, self.n_heads, self.out_dim)
#             k = self.k_linears(h).view(-1, self.n_heads, self.out_dim)

#             G.srcdata['k'] = k
#             G.dstdata['q'] = q
#             G.srcdata['v'] = v

#             with G.local_scope():
#                 G.apply_edges(fn.v_dot_u('q', 'k', 't'))
#                 attn_score = self.leakyrelu(G.edata.pop('t').sum(-1)/self.sqrt_dk)
#                 attn_score = edge_softmax(G, attn_score, norm_by='dst')

#                 G.edata['t'] = attn_score.unsqueeze(-1)
#                 G.update_all(fn.u_mul_e('v', 't', 'm'), fn.sum('m', 't'))
#                 # G.update_all(fn.copy_u('v', 'm'), fn.sum('m', 't'))
#                 h_output = G.dstdata['t'].view(-1, self.out_dim * self.n_heads)

#             if self.residual:
#                 res = self.residual(h)
#                 h_output += res
#             if self.activation is not None:
#                 h_output = self.activation(h_output)
#             return h_output
