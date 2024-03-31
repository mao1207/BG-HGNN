import argparse
import gc
import os
import sys
import time
from collections import defaultdict

import dgl
import numpy as np
import psutil
import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from dgl.nn.pytorch import RelGraphConv
from model import H2H
from utils.data import load_data
from utils.pytorchtools import EarlyStopping

pynvml.nvmlInit()
sys.path.append('../../')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        print("=============")
        print(p.size())
        nn=1
        for s in list(p.size()):
            nn = nn*s
        print(nn)
        pp += nn
    return pp


class LinkPredict(nn.Module):
    def __init__(self, g, features, in_dims, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, reg_param=0):
        super(LinkPredict, self).__init__()
        # input = h_t
        self.h2h = BGHGNN(g, features, features.shape[-1], args.n_hid, 3, [1, 1, 1]).to(device)
        print(self.h2h)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        triplets = triplets.long()

        s = embedding[0][triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[0][triplets[:, 2]]
        print(s.shape)
        print(r.shape)
        print(o.shape)
        exit(-1)
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h):
        return self.h2h.forward(g)

    def regularization_loss(self, embedding):
        return torch.mean(embedding[0].pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        logp = torch.sigmoid(score)
        predict_loss = F.binary_cross_entropy_with_logits(logp, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def run_model_DBLP(args):
    feats_type = args.feats_type
    features_list, adjM, dl = load_data(args.dataset)
    use_gpu = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    else:
        torch.manual_seed(42)
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            features_list[i] = torch.rand((dim, 512)).to(device)
            
    # 初始化一个空的特征列表
    total_width = sum(features.shape[1] for features in features_list)
    cumulative_widths = [0] + list(torch.cumsum(torch.tensor([features.shape[1] for features in features_list]), dim=0))
    padded_features = []

    # 对每种类型的节点特征进行填充
    for i, features in enumerate(features_list):
        print("======================")
        print(features)
        # 计算每个特征的前后填充
        features = features.to_dense()
        padding_before = torch.zeros(features.shape[0], cumulative_widths[i]).to(device)
        padding_after = torch.zeros(features.shape[0], total_width - cumulative_widths[i+1]).to(device)
        # 创建填充后的特征
        padded_feature = torch.cat([padding_before, features, padding_after], dim=1)
        padded_features.append(padded_feature)

    # 将特征在行上拼接
    concatenated_features = torch.cat(padded_features, dim=0)
    print(concatenated_features)

    in_dims = []
    handle = pynvml.nvmlDeviceGetHandleByIndex(use_gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_begin = meminfo.used
    print("begin:", gpu_begin / 1024 / 1024, 'MB')
    process = psutil.Process(os.getpid())
    memory_begin = process.memory_info().rss
    print('Used Memory begin:', memory_begin / 1024 / 1024, 'MB')
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)

    edge2type = {}
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u, v)] = k
    for i in range(dl.nodes['total']):
        if (i, i) not in edge2type:
            edge2type[(i, i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            if (v, u) not in edge2type:
                edge2type[(v, u)] = k+1+len(dl.links['count'])

    edge_dict = {}
    # edge_type2meta = {}
    for i, meta_path in dl.links['meta'].items():
        edge_dict[(str(meta_path[0]), str(i), str(meta_path[1]))] = (torch.tensor(dl.links['data'][i].tocoo().row - dl.nodes['shift'][meta_path[0]]), torch.tensor(dl.links['data'][i].tocoo().col - dl.nodes['shift'][meta_path[1]]))
        # edge_type2meta[i] = str(i) #str(meta_path[0]) + '_' + str(meta_path[1])

    print(edge_dict)


    node_count = {}
    for i, count in dl.nodes['count'].items():
        print(i, node_count)
        node_count[str(i)] = count

    G = dgl.heterograph(edge_dict, num_nodes_dict = node_count, device=device)
    print(G)
    """
    for ntype in G.ntypes:
        G.nodes[ntype].data['inp'] = dataset.nodes['attr'][ntype]
        # print(G.nodes['attr'][ntype].shape)
    """

    G.node_dict = {}
    G.edge_dict = {}
    for ntype in G.ntypes:
        print(ntype, dl.nodes['shift'][int(ntype)])
    for ntype in G.ntypes:
        G.node_dict[ntype] = len(G.node_dict)
    for etype in G.etypes:
        G.edge_dict[etype] = len(G.edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long).to(device) * G.edge_dict[etype] 


    g = dgl.to_homogeneous(G).to(device)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)   
    test_norm = utils.data.comp_deg_norm(g.cpu())
    edge_norm = utils.data.node_norm_to_edge_norm(
        g.cpu(), torch.from_numpy(test_norm).view(-1, 1)).to(device)

    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u, v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    res_2hop = defaultdict(float)
    res_random = defaultdict(float)
    total = len(list(dl.links_test['data'].keys()))
    print(g)
    print(e_feat.shape)

    if True:
        # edge_types=[test_edge_type])
        train_pos, valid_pos = dl.get_train_valid_pos()
        num_classes = args.hidden_dim
        heads = [args.num_heads] * args.num_layers + [args.num_heads]
        print(concatenated_features.shape)
        net = LinkPredict(
            G,
            concatenated_features,
            in_dims,
            args.hidden_dim,
            len(dl.links['count'])+1,
            num_bases=args.n_bases,
            num_hidden_layers=args.num_layers,
            dropout=args.dropout,
            reg_param=args.regularization)
        net.to(device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                       save_path='/local/scratch/hcui25/Project/xin/H2H/LP/H2H/checkpoint/rgcn_checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
        loss_func = nn.BCELoss()
    for epoch in range(args.epoch):
        train_pos_head_full = np.array([])
        train_pos_tail_full = np.array([])
        train_neg_head_full = np.array([])
        train_neg_tail_full = np.array([])
        r_id_full = np.array([])
        for test_edge_type in dl.links_test['data'].keys():
            train_neg = dl.get_train_neg(edge_types=[test_edge_type])[
                test_edge_type]
            train_pos_head_full = np.concatenate(
                [train_pos_head_full, np.array(train_pos[test_edge_type][0])])
            train_pos_tail_full = np.concatenate(
                [train_pos_tail_full, np.array(train_pos[test_edge_type][1])])
            train_neg_head_full = np.concatenate(
                [train_neg_head_full, np.array(train_neg[0])])
            train_neg_tail_full = np.concatenate(
                [train_neg_tail_full, np.array(train_neg[1])])
            r_id_full = np.concatenate([r_id_full, np.array(
                [test_edge_type]*len(train_pos[test_edge_type][0]))])
        train_idx = np.arange(len(train_pos_head_full))
        np.random.shuffle(train_idx)
        batch_size = args.batch_size
        print('Training with #param: %d' % (get_n_params(net)))
        for step, start in enumerate(range(0, len(train_pos_head_full), args.batch_size)):
            t_start = time.time()
            # training
            net.train()
            train_pos_head = train_pos_head_full[train_idx[start:start+batch_size]]
            train_neg_head = train_neg_head_full[train_idx[start:start+batch_size]]
            train_pos_tail = train_pos_tail_full[train_idx[start:start+batch_size]]
            train_neg_tail = train_neg_tail_full[train_idx[start:start+batch_size]]
            r_id = r_id_full[train_idx[start:start+batch_size]]
            left = np.concatenate([train_pos_head, train_neg_head])
            right = np.concatenate([train_pos_tail, train_neg_tail])
            mid = np.concatenate([r_id, r_id])
            labels = torch.FloatTensor(np.concatenate(
                [np.ones(train_pos_head.shape[0]), np.zeros(train_neg_head.shape[0])])).to(device)

            embed = net(g, features_list)
            triplets = [[i[0], i[1], i[2]]for i in zip(left, mid, right)]
            train_loss = net.get_loss(
                embed, torch.LongTensor(triplets), labels)

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            # print('Epoch {:05d}, Step{:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
            #     epoch, step, train_loss.item(), t_end-t_start))

            t_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                valid_pos_head = np.array([])
                valid_pos_tail = np.array([])
                valid_neg_head = np.array([])
                valid_neg_tail = np.array([])
                valid_r_id = np.array([])
                for test_edge_type in dl.links_test['data'].keys():
                    valid_neg = dl.get_valid_neg(edge_types=[test_edge_type])[
                        test_edge_type]
                    valid_pos_head = np.concatenate(
                        [valid_pos_head, np.array(valid_pos[test_edge_type][0])])
                    valid_pos_tail = np.concatenate(
                        [valid_pos_tail, np.array(valid_pos[test_edge_type][1])])
                    valid_neg_head = np.concatenate(
                        [valid_neg_head, np.array(valid_neg[0])])
                    valid_neg_tail = np.concatenate(
                        [valid_neg_tail, np.array(valid_neg[1])])
                    valid_r_id = np.concatenate([valid_r_id, np.array(
                        [test_edge_type]*len(valid_pos[test_edge_type][0]))])
                left = np.concatenate([valid_pos_head, valid_neg_head])
                right = np.concatenate([valid_pos_tail, valid_neg_tail])
                mid = np.concatenate([valid_r_id, valid_r_id])
                labels = torch.FloatTensor(np.concatenate(
                    [np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)
                embed = net(g, features_list)
                triplets = [[i[0], i[1], i[2]]for i in zip(left, mid, right)]
                val_loss = net.get_loss(
                    embed, torch.LongTensor(triplets), labels)
            t_end = time.time()
            # print validation info
            # print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
            #     epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                # print('Early stopping!')
                break
        if early_stopping.early_stop:
            # print('Early stopping!')
            break

    torch.cuda.empty_cache()
    for test_edge_type in dl.links_test['data'].keys():
        # testing with evaluate_results_nc
        net.load_state_dict(torch.load(
            '/local/scratch/hcui25/Project/xin/H2H/LP/H2H/checkpoint/rgcn_checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            test_neigh, test_label = dl.get_test_neigh()
            test_neigh = test_neigh[test_edge_type]
            test_label = test_label[test_edge_type]
            left = np.array(test_neigh[0])
            right = np.array(test_neigh[1])
            mid = np.zeros(left.shape[0], dtype=np.int32)
            mid[:] = test_edge_type
            labels = torch.FloatTensor(test_label).to(device)
            embed = net(g, features_list)
            triplets = [[i[0], i[1], i[2]]for i in zip(left, mid, right)]
            logits = net.calc_score(embed, torch.LongTensor(triplets))
            pred = torch.sigmoid(logits).cpu().numpy()
            edge_list = np.concatenate(
                [left.reshape((1, -1)), right.reshape((1, -1))], axis=0)
            labels = labels.cpu().numpy()
            res = dl.evaluate(edge_list, pred, labels)
            # print(res)
            for k in res:
                res_2hop[k] += res[k]
        with torch.no_grad():
            test_neigh, test_label = dl.get_test_neigh_w_random()
            test_neigh = test_neigh[test_edge_type]
            test_label = test_label[test_edge_type]
            left = np.array(test_neigh[0])
            right = np.array(test_neigh[1])
            mid = np.zeros(left.shape[0], dtype=np.int32)
            mid[:] = test_edge_type
            labels = torch.FloatTensor(test_label).to(device)
            embed = net(g, features_list)
            triplets = [[i[0], i[1], i[2]]for i in zip(left, mid, right)]
            logits = net.calc_score(embed, torch.LongTensor(triplets))
            pred = torch.sigmoid(logits).cpu().numpy()
            edge_list = np.concatenate(
                [left.reshape((1, -1)), right.reshape((1, -1))], axis=0)
            labels = labels.cpu().numpy()
            res = dl.evaluate(edge_list, pred, labels)
            # print(res)
            for k in res:
                res_random[k] += res[k]
    for k in res_2hop:
        res_2hop[k] /= total
    for k in res_random:
        res_random[k] /= total
    print(res_2hop)
    print(res_random)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_end = meminfo.used
    print("GPU usage test end:", gpu_end / 1024 / 1024, 'MB')
    memory_end = process.memory_info().rss
    print('Memory usage test end:',
          memory_end / 1024 / 1024, 'MB')

    print("Net GPU usage:", (gpu_end-gpu_begin) / 1024 / 1024, 'MB')
    print('Net Memory usage:', (memory_end-memory_begin) / 1024 / 1024, 'MB')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                    '4 - only term features (id vec for others);' +
                    '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64,
                    help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=2,
                    help='Number of the attention heads. Default is 2.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=50, help='Patience.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.01)
    ap.add_argument('--dataset', type=str, default='amazon')
    ap.add_argument('--edge-feats', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--gpu', type=int, default=-1)
    ap.add_argument('--n-bases', type=int, default=16)
    ap.add_argument("--regularization", type=float, default=0.01,
                    help="regularization weight")
    ap.add_argument('--n_hid',   type=int, default=64)

    args = ap.parse_args()
    run_model_DBLP(args)