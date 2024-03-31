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

def find_2_hop_attention(G, src, dst, attention):

    results = []
    
    edge_to_attention = {(s.item(), d.item()): att.item() for s, d, att in zip(src, dst, attention)}
    
    for s in src.unique():
        first_hop_dst = dst[src == s]
        for fhd in first_hop_dst:
            second_hop_dst = dst[src == fhd]
            att_value1 = edge_to_attention[(s.item(), fhd.item())]
            for shd in second_hop_dst:
                if (fhd.item(), shd.item()) in edge_to_attention:
                    att_value2 = att_value1 + edge_to_attention[(fhd.item(), shd.item())]
                    results.append((s, shd, att_value2))
                    
    return results


def to_hetero_feat(h, type, name):
    h_dict = {}
    for index, ntype in enumerate(name):
        h_dict[ntype] = h[torch.where(type == index)]

    return h_dict


class NodeTypePredictor(nn.Module):
    def __init__(self, g, embed_dim):
        super(NodeTypePredictor, self).__init__()
        self.embeddings = nn.Parameter(torch.empty((len(g.ntypes), embed_dim))).to(device)
        self.classifier = nn.Linear(embed_dim, len(g.ntypes)).to(device)
        self.ntype_dict = torch.empty((g.number_of_nodes())).int().to(device)

        nn.init.xavier_uniform_(self.embeddings)

        begin = 0
        for i, ntype in enumerate(g.ntypes):
            end = begin + g.number_of_nodes(ntype)
            self.ntype_dict[begin:end] = i
            begin = end

    # def message_func(self, edges):
    #     print(edges.src)
    #     return {'m': self.embeddings[edges.src['_TYPE']]}

    # def reduce_func(self, nodes):
    #     # print(nodes.mailbox['m'][0].shape)
    #     # print(nodes.mailbox['m'][0])
    #     return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

    # def forward(self, hg):
    #     for etype in hg.canonical_etypes:
    #         src, dst = hg.in_edges(256, etype=etype)
    #         print(self.ntype_dict[src])

    #     h = torch.empty(0).to(device)
    #     begin = 0
    #     for i, ntype in enumerate(hg.ntypes):
    #         end = begin + hg.number_of_nodes(ntype)
    #         h = torch.cat((h, self.embeddings[i].repeat(hg.number_of_nodes(ntype), 1)))
    #         begin = end

    #     g = dgl.to_homogeneous(hg)
    #     g.update_all(self.message_func, self.reduce_func)
    #     h_output = g.ndata.pop('h')
    #     torch.set_printoptions(threshold=float('inf'))
    #     # print(self.embeddings[0])
    #     # print(self.embeddings[2])
    #     print((h_output)[256])
    #     return self.classifier(h)

    def congregate(self, g):
        all_neighbors = torch.zeros((g.number_of_nodes(), self.embeddings.shape[1])).to(device)

        in_degrees = torch.zeros(g.number_of_nodes(), dtype=torch.long).to(device)

        for etype in g.canonical_etypes:
            src, dst = g.edges(etype=etype)
            src_embeddings = self.embeddings[self.ntype_dict[src]]
            all_neighbors.index_add_(0, dst, src_embeddings)
            dst_embeddings = self.embeddings[self.ntype_dict[dst]]
            all_neighbors.index_add_(0, src, dst_embeddings)
            deg_for_etype_src = torch.bincount(dst, minlength=g.number_of_nodes())
            deg_for_etype_dst = torch.bincount(src, minlength=g.number_of_nodes())
            in_degrees += deg_for_etype_src
            in_degrees += deg_for_etype_dst

        degree = in_degrees.float().clamp(min=1).to(device)  # Avoid division by 0
        all_neighbors /= degree.view(-1, 1)

        return all_neighbors

    def forward(self, g):
        h = self.congregate(g)
        return self.classifier(h)


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout=0.2,
                 use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h_hdict, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[(srctype, etype, dsttype)]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype: (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer='mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT(nn.Module):
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm=True):
        super(HGT, self).__init__()
        current_time = int(time.time())
        torch.manual_seed(current_time)
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm=use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, h_dict, out_key):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h_dict, h)
        return self.out(h[out_key])


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
            str(name): nn.Linear(in_size, out_size) for name in etypes
        })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[str((srctype, etype, dsttype))](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % str((srctype, etype, dsttype))] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[(srctype, etype, dsttype)] = (
            fn.copy_u('Wh_%s' % str((srctype, etype, dsttype)), 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        current_time = int(time.time())
        torch.manual_seed(current_time)
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.canonical_etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, hidden_size, G.canonical_etypes)
        self.layer3 = HeteroRGCNLayer(hidden_size, out_size, G.canonical_etypes)

    def forward(self, G, h_dict, out_key):
        input_dict = {ntype: G.nodes[ntype].data['inp'] for ntype in G.ntypes}
        h_dict = self.layer1(G, input_dict)
        h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer3(G, h_dict)
        # get paper logits
        return h_dict[out_key]


class BGHGNN(nn.Module):
    def __init__(self, G, input, in_dim, hidden_dim, out_dim, num_layers, heads, feat_drop = 0.2, attn_drop = 0, negative_slope=0.2,
                 residual=False, beta=0.0):
        super(BGHGNN, self).__init__()
        current_time = int(time.time())
        torch.manual_seed(current_time)
        print(current_time)
    
        # create layers
        self.input = input
        self.ntypes = G.ntypes
        self.hgn_layers = nn.ModuleList()
        self.activation = F.relu
        self.edge_dim = 512
        self.num_etypes = len(G.etypes)
        self.num_layers = num_layers
        self.rank = 4
        self.embedding_dim = 512
        self.combine_dim = 64
        # self.node_embedding = nn.Parameter(torch.empty(
        #     size=(len(G.ntypes), self.embedding_dim)
        # ))
        # self.edge_type_embeddings = nn.Parameter(torch.empty(
        #     size=(len(G.canonical_etypes), self.edge_dim)
        # ))

        # self.node_embedding = torch.eye(len(G.ntypes), self.embedding_dim).to(device)
        # self.edge_type_embeddings = torch.eye(len(G.canonical_etypes), self.edge_dim).to(device)
        # torch.manual_seed(23)
        # self.node_embedding = torch.rand(size=(len(G.ntypes), self.embedding_dim)).to(device)

        # self.node_embedding = torch.empty(0).to(device)
        # sentense = []
        # for ntype in G.ntypes:
        #     sentense.append([ntype])
        # model = Word2Vec(sentense, vector_size=512, window=5, min_count=1, workers=4)
        # for word in model.wv.key_to_index:
        #     self.node_embedding = torch.cat((self.node_embedding, torch.tensor(model.wv[word]).unsqueeze(0).to(device)), dim = 0)

        # self.edge_type_embeddings = torch.empty(0).to(device)
        # sentense = []
        # for etype in G.canonical_etypes:
        #     sentense.append([etype[0] + "->" + etype[1] + "->" + etype[2]])
        # model = Word2Vec(sentense, vector_size=512, window=5, min_count=1, workers=4)
        # for word in model.wv.key_to_index:
        #     self.edge_type_embeddings = torch.cat((self.edge_type_embeddings, torch.tensor(model.wv[word]).unsqueeze(0).to(device)), dim = 0)

        self.node_embedding = (torch.rand(size=(len(G.ntypes), self.embedding_dim)).to(device))
        self.edge_type_embeddings = (torch.rand(size=(len(G.canonical_etypes), self.edge_dim)).to(device))

        # node_emb_dict = {}
        # for i, ntype in enumerate(G.ntypes):
        #     node_emb_dict[ntype] = self.node_embedding[i]
        # relation_offset = torch.rand(size=(len(G.canonical_etypes), self.embedding_dim)).to(device) * 0.2
        # self.edge_type_embeddings = torch.empty(0).to(device)
        # for i, edge in enumerate(G.canonical_etypes):
        #     edge_embedding = ((node_emb_dict[edge[2]] - node_emb_dict[edge[0]] + relation_offset[i])).unsqueeze(0)
        #     self.edge_type_embeddings = torch.cat((self.edge_type_embeddings, edge_embedding), dim = 0)

        # self.edge_type_embeddings = torch.rand(size=(len(G.canonical_etypes), self.edge_dim)).to(device)

        ##########################
        self.node_type_dict = {}
        ##########################

        # self.node_embedding = torch.tensor(np.eye(len(list(set(G.ntypes))))).float().to(device)

        print(in_dim + 1)

        self.ntype_to_start = {}
        begin = 0
        for i, ntype in enumerate(G.ntypes):
            self.ntype_to_start[ntype] = begin
            ##########
            for node_id in range(begin, begin + G.number_of_nodes(ntype)):
                self.node_type_dict[node_id] = i

            ##########
            begin += G.number_of_nodes(ntype)

        self.common_features_factor = nn.Parameter(torch.Tensor(self.rank, in_dim + 1, self.combine_dim))
        self.type_embedding_factor = nn.Parameter(torch.Tensor(self.rank, self.embedding_dim + 1, self.combine_dim))
        self.relation_embedding_factor = nn.Parameter(torch.Tensor(self.rank, self.edge_dim + 1, self.combine_dim))

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.combine_dim))

        self.h_type_features = torch.empty(0).to(device)
        self.h_relation_features = torch.zeros((G.number_of_nodes(), self.edge_dim)).to(device)
        
        degrees = torch.zeros(G.number_of_nodes(), dtype=torch.long).to(device)

        begin = 0
        for i, ntype in enumerate(G.ntypes):
            end = begin + G.number_of_nodes(ntype)
            h_temp = self.node_embedding[i].repeat(G.number_of_nodes(ntype), 1)
            self.h_type_features = torch.cat((self.h_type_features, h_temp))
            begin = end

        for index, etype in enumerate(G.canonical_etypes):
            G_src, G_dst = G.edges(etype=etype)
            src = G_src + self.ntype_to_start[etype[0]]
            dst = G_dst + self.ntype_to_start[etype[2]]

            edge_embedding = self.edge_type_embeddings[index].repeat(len(src), 1)
            self.h_relation_features.index_add_(0, dst, edge_embedding)
            self.h_relation_features.index_add_(0, src, -edge_embedding)
            deg_for_etype_src = torch.bincount(dst, minlength=G.number_of_nodes())
            deg_for_etype_dst = torch.bincount(src, minlength=G.number_of_nodes())
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

        
        ones_tensor = torch.ones(self.input.to(device).size(0), 1, device=device)

        self.h_common_features = torch.cat([self.input.to(device), ones_tensor], dim=-1)
        self.h_type_features = torch.cat([self.h_type_features, ones_tensor], dim=-1)
        self.h_relation_features = torch.cat([self.h_relation_features, ones_tensor], dim=-1)

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
        for l in range(1, self.num_layers - 1):  # noqa E741
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.hgn_layers.append(
                GATConv(
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    residual = residual,
                    feat_drop = feat_drop,
                    attn_drop = attn_drop
                )
            )
        # output projection
        self.hgn_layers.append(
            GATConv(
                hidden_dim * heads[-2],
                out_dim,
                heads[-1],
                residual = True,
                feat_drop = feat_drop,
                attn_drop = attn_drop
            )
        )  

        self.H = dgl.to_homogeneous(G).to(device)
        self.H = dgl.add_self_loop(self.H)  


    def forward(self, G, h_dict, out_key, att_matrix = False): 


        fusion_common_features = torch.matmul(self.h_common_features, self.common_features_factor)
        fusion_type_features = torch.matmul(self.h_type_features, self.type_embedding_factor)
        # print(self.h_type_features.shape)
        # print(self.type_embedding_factor.shape)
        # print(fusion_type_features.shape)
        # print(fusion_type_features)
        fusion_relation_features = torch.matmul(self.h_relation_features, self.relation_embedding_factor)
        fusion_zy = fusion_common_features * fusion_type_features * fusion_relation_features

        h = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        h = self.activation(h)

        # h=self.input.to(device)

        for l in range(self.num_layers):
            h, att = self.hgn_layers[l](self.H, h, get_attention = True)
            h = h.flatten(1)
        h_dict = to_hetero_feat(h, self.H.ndata['_TYPE'], G.ntypes)
        
        # att = att.squeeze()
        
        # if att_matrix == True:
        #     num_node_type = len(G.ntypes)
        #     sum_for_each_type = torch.ones((num_node_type, num_node_type))
        #     attention_matrix = torch.zeros((num_node_type, num_node_type))

        #     # Get the source and destination node IDs for each edge
        #     src_ids, dst_ids = self.H.edges()
        #     print(self.node_type_dict)

        #     two_hop_att = find_2_hop_attention(G, src_ids, dst_ids, att)

        #     # Populate the attention matrix
        #     for (src, dst, attention_weight) in two_hop_att:
        #         attention_matrix[self.node_type_dict[src.item()], self.node_type_dict[dst.item()]] += attention_weight
        #         sum_for_each_type[self.node_type_dict[src.item()], self.node_type_dict[dst.item()]]+= 1

        #     attention_matrix /= sum_for_each_type
        #     print(attention_matrix)

        #     return h_dict[out_key], att, attention_matrix

        return h_dict[out_key], att
    
    # def forward(self, H, G, h_dict, out_key):
    #     h_common_features = self.input.to(device)[H[0].ndata[dgl.NID]['_N']]
    #     h_type_features = self.h_type_features[H[0].ndata[dgl.NID]['_N']]
    #     h_relation_features = self.h_relation_features[H[0].ndata[dgl.NID]['_N']]

    #     ones_tensor = torch.ones(h_common_features.size(0), 1, device=device)

    #     h_common_features = torch.cat([h_common_features, ones_tensor], dim=-1)
    #     h_type_features = torch.cat([h_type_features, ones_tensor], dim=-1)
    #     h_relation_features = torch.cat([h_relation_features, ones_tensor], dim=-1)

    #     fusion_common_features = torch.matmul(h_common_features, self.common_features_factor)
    #     fusion_type_features = torch.matmul(h_type_features, self.type_embedding_factor)
    #     fusion_relation_features = torch.matmul(h_relation_features, self.relation_embedding_factor)
    #     fusion_zy = fusion_common_features * fusion_type_features * fusion_relation_features

    #     h = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
    #     h = self.activation(h)

    #     # h=self.input.to(device)

    #     for l in range(self.num_layers):
    #         h = self.hgn_layers[l](H[l], h)
    #         h = h.flatten(1)
    #     h_dict = to_hetero_feat(h, H[3].ndata['_TYPE']['_N'], G.ntypes)

    #     return h_dict[out_key]
    


class simpleHGN(nn.Module):
    def __init__(self, G, input, in_dim, hidden_dim, out_dim, num_layers, heads, feat_drop=0.0, negative_slope=0.2,
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
        # output projection
        self.hgn_layers.append(
            SimpleHGNConv(
                self.edge_dim,
                hidden_dim * heads[-2],
                out_dim,
                heads[-1],
                self.num_etypes,
                feat_drop,
                negative_slope,
                residual,
                None,
                beta=beta,
            )
        )

    def forward(self, G, h_dict, out_key):
        if hasattr(G, 'ntypes'):
            # full graph training,
            with G.local_scope():
                G.ndata['h'] = self.h_dict
                g = dgl.to_homogeneous(G, ndata='h')
                h = g.ndata['h']
                for l in range(self.num_layers):  # noqa E741
                    h = self.hgn_layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)
                    h = h.flatten(1)
            h_dict = to_hetero_feat(h, g.ndata['_TYPE'], G.ntypes)
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

        return h_dict[out_key]
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

class myLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, negative_slope=0.2,
                 residual=True, activation=F.elu):
        super(myLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = num_heads
        self.d_k = out_dim // num_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.conv = nn.Conv1d(in_dim, out_dim * num_heads, kernel_size=3, padding=1)

        self.feat_drop = nn.Dropout(dropout)
        self.activation = activation
        self.leakyrelu = nn.LeakyReLU(negative_slope)

        self.W = nn.Parameter(torch.FloatTensor(
            in_dim, out_dim * num_heads))

        nn.init.xavier_uniform_(self.W)

        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)

    def forward(self, G, h_dict, h):
        h = self.feat_drop(h)
        # h_output = self.conv(h.transpose(0, 1))
        # h_output = h_output.transpose(0, 1)

        h_output = torch.matmul(h, self.W)
        h_output[torch.isnan(h_output)] = 0.0

        G.srcdata['h'] = h_output

        src, dst = G.edges()
        edge_attention= self.leakyrelu((h_output[src] * h_output[dst]).sum(-1))
        edge_attention = edge_softmax(G, edge_attention)
        G.edata['att'] = edge_attention


        with G.local_scope():
            # G.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
            G.update_all(fn.u_mul_e('h', 'att', 'm'), fn.mean('m', 'h'))
            h_output = G.dstdata['h']

        if self.residual:
            res = self.residual(h)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)
        return h_output