import argparse
import heapq
import itertools
import math
import random
import time
import urllib.request
import warnings
from collections import defaultdict

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
from model import *
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.metrics import f1_score
from utils import load_data

warnings.filterwarnings("ignore")

torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Training GNN on ogbn-products benchmark')


parser.add_argument('--mode', type=str, default='GCN')
parser.add_argument('--sample', type=bool, default=False)
parser.add_argument('--n_epoch', type=int, default=10000)
parser.add_argument('--n_hid',   type=int, default=256)
parser.add_argument('--n_inp',   type=int, default=256)
parser.add_argument('--clip',    type=int, default=1.0)
parser.add_argument('--max_lr',  type=float, default=5e-5)
parser.add_argument('--dataset',  type=str, default='IMDB')
parser.add_argument('--distributed',  type=bool, default=False)

args = parser.parse_args()

def find_high_attention_paths(graph, node_types, edge_attention, expected_length, starting_type, top_k=10):
    """
    Find paths in a graph with the highest cumulative attention scores.
    
    :param graph: DGL Graph object
    :param node_types: Dictionary mapping node IDs to their types
    :param edge_attention: Dictionary mapping edge IDs to their attention scores
    :param expected_length: Expected length of the paths
    :param top_k: Number of top paths to return
    :return: List of top-k paths with the highest attention scores
    """
    # Function to get the attention score for an edge between two nodes
    def get_attention_score(path):
        attention_score = 0
        for i in range(len(path) - 1):
            edge_id = graph.edge_ids(path[i], path[i + 1])
            attention_score += edge_attention[edge_id]
        return attention_score
    
    def dfs(current_path, current_path_type, all_paths, all_paths_types, length):
        if len(current_path) == length:
            all_paths.append(current_path)
            all_paths_types.append(current_path_type)
            return
        last_node = current_path[-1]

        for next_node in graph.successors(last_node):
            if next_node not in current_path:
                # Pruning to avoid too much computation
                if random.randint(0,100) < 4:
                    return
                dfs(current_path + [next_node], current_path_type + "-" + node_type_dict[next_node], all_paths, all_paths_types, length)

    def find_path(node_id, length):
        all_paths = []
        all_paths_types = []
        dfs([node_id], node_type_dict[node_id], all_paths, all_paths_types, length)
        return all_paths, all_paths_types
        
    
    # Generate all possible paths of the expected length
    attention_score_path = {}
    nums_path = {}
    for node_id in graph.nodes():
        if node_id >= 50:
            break
        # Only consider paths that start with the specified node type
        if node_types[node_id] == starting_type:
            print(node_id)
            paths, paths_types = find_path(node_id, expected_length)
        else:
            break
            
        sample_size_per_type = 2  # For example, we want 2 samples per type

        # Organize paths by type
        paths_by_type = {}
        for path, path_type in zip(paths, paths_types):
            if path_type not in paths_by_type:
                paths_by_type[path_type] = []
            paths_by_type[path_type].append(path)

        # Sample uniformly from each type
        paths_sample = []
        paths_types_sample = []
        for path_type, paths_list in paths_by_type.items():
            sample_size = min(len(paths_list), sample_size_per_type)  # Adjust sample size if not enough paths
            sampled_paths = random.sample(paths_list, sample_size)
            paths_sample.extend(sampled_paths)
            paths_types_sample.extend([path_type] * sample_size)  # Extend with repeated path type
                    
        # number_of_samples = 20  # The number of samples you want to take
        # number_of_samples = min(number_of_samples, len(paths))
        # sampled_indices = random.sample(range(len(paths)), number_of_samples)
        # sampled_pairs = [(paths[i], paths_types[i]) for i in sampled_indices]
        
        for path, paths_type in zip(paths_sample, paths_types_sample):
            print(path, paths_type)
            # # Skip paths that don't match the expected node type sequence
            # if not all(node_types[node] == expected_node_type for node, expected_node_type in zip(path, expected_types)):
            #     continue
            # Calculate the cumulative attention score for the path
            attention_score = get_attention_score(path)
            if paths_type not in attention_score_path:
                attention_score_path[paths_type] = 0
                nums_path[paths_type] = 0
                
            attention_score_path[paths_type] += attention_score
            nums_path[paths_type] += 1
            
            print(attention_score_path)
                        
            
    for key in attention_score_path:
        attention_score_path[key] /= nums_path[key]
        
    return attention_score_path



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def train(model, G):
    total_time = 0
    if args.distributed:
        dgl.distributed.initialize('ip_config.txt')
        th.distributed.init_process_group(backend='gloo')
        G = dgl.distributed.DistGraph('graph_name', 'part_config.json')
        model = th.nn.parallel.DistributedDataParallel(model)

    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)
    best_macro = torch.tensor(0)
    best_micro = torch.tensor(0)
    train_step = torch.tensor(0)
    for epoch in np.arange(args.n_epoch) + 1:
        start_time = time.time()
        model.train()
        if args.mode == 'BG-HGNN':
            logits, att = model(G, h_dict, category)
        else:
            logits = model(G, h_dict, category)

        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))
        # print("output",torch.argmax(logits[train_idx], dim = 1))
        #print("label",labels[train_idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        # print((model.projection[0].grad).cpu().numpy())
        #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time
        if epoch % 5 == 0:
            model.eval()
            if args.mode == 'GCN':
                logits, att = model(G, h_dict, category)
            else:
                logits = model(G, h_dict, category)
            pred   = logits.argmax(1).cpu()
            train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
            val_acc   = (pred[val_idx]   == labels[val_idx]).float().mean()
            test_acc  = (pred[test_idx]  == labels[test_idx]).float().mean()
            micro_f1 = f1_score(labels[test_idx].cpu(), pred[test_idx], average='micro')
            macro_f1 = f1_score(labels[test_idx].cpu(), pred[test_idx], average='macro')
            if best_val_acc <= val_acc:
                best_val_acc = val_acc
                if best_test_acc < test_acc:
                    best_test_acc = test_acc
                    best_micro = micro_f1
                    best_macro = macro_f1

            # if epoch % 300 == 0:
            #     logits, att, att_matrix = model(G, h_dict, category, True)
            #     plt.figure(figsize=(8, 6))
            #     plt.imshow(att_matrix, cmap='viridis', interpolation='nearest', vmin=0, vmax=att_matrix.max())
            #     plt.colorbar()
            #     plt.xlabel('Destination Node Type', fontsize = 18)
            #     plt.ylabel('Source Node Type', fontsize = 18)
            #     plt.savefig(f"att{epoch}.png")

            print('Epoch: %d LR: %.5f Loss %.10f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f), Micro-F1: %.4f, Macro-F1: %.4f' % (
                epoch,
                optimizer.param_groups[0]['lr'],
                loss.item(),
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),
                best_micro,
                best_macro
            ), flush=True)
            print(total_time, flush=True)
            
            # if epoch == 620:
            #     H = dgl.to_homogeneous(G).to(device)
            #     H = dgl.add_self_loop(H)
            #     H = dgl.reverse(H)
            #     att_path = find_high_attention_paths(H, node_type_dict, att, 5, category_name, 10)
            #     top_20_paths = sorted(att_path.items(), key=lambda x: x[1], reverse=True)[:20]
            #     print(top_20_paths)

# Defining Conversion Functions
def encode_and_extend_features(G, node_types, node_type_features, all_features):
    for ntype in node_types:
        print(len(all_features))

        new_feat = torch.full((G.number_of_nodes(ntype), len(all_features)), 0.0)
        for i, f in enumerate(all_features):
            if f in node_type_features[ntype]:
                old_idx = node_type_features[ntype].index(f)
                new_feat[:, i] = G.nodes[ntype].data['inp'][:, old_idx]

        print(new_feat.shape)
        G.nodes[ntype].data['inp'] = new_feat.to(device)

device = torch.device("cuda")
# device = torch.device("cpu")

if args.dataset == 'aifb':
    dataset = AIFBDataset()
elif args.dataset == 'mutag':
    dataset = MUTAGDataset()
elif args.dataset == 'bgs':
    dataset = BGSDataset()
elif args.dataset == 'am':
    dataset = AMDataset()
elif args.dataset == 'ogbn' or args.dataset == 'DBLP' or args.dataset == 'ACM' or args.dataset == 'Freebase' or args.dataset == 'IMDB':
    pass
else:
    raise ValueError()

if args.dataset == 'ogbn':
    dataset = DglNodePropPredDataset(name = 'ogbn-mag')
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"]["paper"], split_idx["valid"]["paper"], split_idx["test"]["paper"]
    hg, labels = dataset[0]
    labels = labels['paper']
    labels = torch.tensor(labels).squeeze()
    hg = hg.to(device)
    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.canonical_etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = (torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]).to(device)

    # Random initialize input feature
    torch.manual_seed(42)
    for ntype in hg.ntypes:
        emb = torch.rand(hg.number_of_nodes(ntype), 64)
        hg.nodes[ntype].data['inp'] = emb.to(device)

    h_dict = {}
    for ntype in hg.ntypes:
        print(ntype)
        h_dict[ntype] = hg.nodes[ntype].data['inp'].to(device)
    category = 'paper'

elif args.dataset == 'DBLP' or args.dataset == 'ACM' or args.dataset == 'Freebase' or args.dataset == 'IMDB':
    hg, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask, meta_paths = load_data(args.dataset, feat_type=0)
    
    hg = hg.to(device)

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
        

    features = features.to_dense ().to(device)
    labels = labels
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    
    torch.manual_seed(42)
    for ntype in hg.ntypes:
        print(features.shape[-1])
        emb = torch.rand(hg.number_of_nodes(ntype), 3489)
        print(emb)
        hg.nodes[ntype].data['inp'] = emb.to(device)
    # for ntype in hg.ntypes:
    #     emb = torch.eye(hg.number_of_nodes(ntype))
    #     hg.nodes[ntype].data['inp'] = emb.to(device)
    
    category = '0'
    
    if args.dataset == 'ACM':
        ntype_name = ["paper", "author", "subject", "term"]
    elif args.dataset == 'DBLP':
        ntype_name = ["author", "paper", "term", "venue"]
    elif args.dataset == 'Freebase':
        ntype_name = ["BOOK", "FILM", "MUSIC", "SPORTS", "PEOPLE", "LOCATION", "ORGANIZATION", "BUSINESS"]
    elif args.dataset == 'IMDB':
        ntype_name = ["movie", "director", "actor", "keyword"]
        
    category_name = ntype_name[0]
    
    node_type_dict = [''] * hg.number_of_nodes()
    begin = 0
    for index, ntype in enumerate(hg.ntypes):
        end = begin + hg.number_of_nodes(ntype)
        for i in range(begin, end):
            node_type_dict[i] = ntype_name[index]
        begin = end
        
    
    hg.nodes[category].data['inp'] = features
    
    h_dict = {}
    for ntype in hg.ntypes:
        print(ntype)
        h_dict[ntype] = hg.nodes[ntype].data['inp'].to(device)
        
    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.canonical_etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long).to(device) * edge_dict[etype]


else:
    # Load from hetero-graph
    hg = dataset[0]

    num_rels = len(hg.canonical_etypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = torch.nonzero(train_mask).squeeze()
    test_idx = torch.nonzero(test_mask).squeeze()
    labels = hg.nodes[category].data.pop('labels')

    # split dataset into train, validate, test
    val_idx = train_idx[:len(train_idx) // 5]
    train_idx = train_idx[len(train_idx) // 5:]

    # calculate norm for each edge type and store in edge
    for canonical_etype in hg.canonical_etypes:
        u, v, eid = hg.all_edges(form='all', etype=canonical_etype)
        _, inverse_index, count = torch.unique(
            v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = torch.ones(eid.shape[0]).float() / degrees.float()
        norm = norm.unsqueeze(1)
        hg.edges[canonical_etype].data['norm'] = norm

    # get target category id
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
    
    category_begin = 0
    category_end = 0
    number = 0
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_begin = number
            category_end = category_begin + hg.number_of_nodes(ntype)
        else:
            number += hg.number_of_nodes(ntype)

    g = dgl.to_homogeneous(hg, edata=['norm'])
    num_nodes = g.number_of_nodes()
    node_ids = torch.arange(num_nodes)
    edge_norm = g.edata['norm']
    edge_type = g.edata[dgl.ETYPE].long()

    # find out the target node ids in g
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]

    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.canonical_etypes:
        print(etype)
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    # Random initialize input feature
    torch.manual_seed(42)
    super_node_begin = hg.number_of_nodes()
    for ntype in hg.ntypes:
        # emb = nn.Parameter(torch.Tensor(hg.number_of_nodes(ntype), 256), requires_grad = False)
        # nn.init.xavier_uniform_(emb)
        emb = torch.rand((hg.number_of_nodes(ntype), 256))
        hg.nodes[ntype].data['inp'] = emb
        # print(ntype,emb)

    h_dict = {}
    if args.sample == False:
        for ntype in hg.ntypes:
            h_dict[ntype] = hg.nodes[ntype].data['inp'].to(device)

    hg = hg.to(device)
    
if args.mode == 'BG-HGNN':
    # h_t = torch.empty(0).to(device)
    # for ntype in hg.srctypes:
    #     h_t = torch.cat((h_t, hg.nodes[ntype].data['inp']), dim = 0)

    node_type_features = {}
    for ntype in hg.ntypes:
        node_type_features[ntype] = [str(ntype) + str(i) for i in range(hg.nodes[ntype].data['inp'].shape[1])]

    # Find all possible features
    all_features = sorted(list(set([f for features in node_type_features.values() for f in features])))

    encode_and_extend_features(hg, hg.ntypes, node_type_features, all_features)
    G_feat = torch.cat([hg.nodes[ntype].data['inp'] for ntype in hg.ntypes])
    input = G_feat.to(device).float()
    model = BGHGNN(hg, input, input.shape[-1], args.n_hid, labels.max().item() + 1, 3,[1, 1, 1]).to(device)

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
    print('Training BGHGNN with #param: %d' % (get_n_params(model)))
    train(model, hg)

if args.mode == 'simple-HGN':

    model = simpleHGN(hg, h_dict, args.n_inp, args.n_hid, labels.max().item() + 1, 3,[1, 1, 1]).to(device)
    # for name, param in model.named_parameters():
    #     print(name)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
    print('Training simple-HGN with #param: %d' % (get_n_params(model)))
    train(model, hg)

if args.mode == 'HGT':
    model = HGT(hg,
                node_dict, edge_dict,
                n_inp=args.n_inp,
                n_hid=args.n_hid,
                n_out=labels.max().item()+1,
                n_layers=3,
                n_heads=4,
                use_norm = True).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
    print('Training HGT with #param: %d' % (get_n_params(model)))
    train(model, hg)

if args.mode == 'RGCN':
    model = HeteroRGCN(hg,
                       in_size=args.n_inp,
                       hidden_size=args.n_hid,
                       out_size=labels.max().item()+1).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
    print('Training RGCN with #param: %d' % (get_n_params(model)))
    train(model, hg)


if args.mode == 'MLP':
    model = HGT(hg,
                node_dict, edge_dict,
                n_inp=args.n_inp,
                n_hid=args.n_hid,
                n_out=labels.max().item()+1,
                n_layers=0,
                n_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
    print('Training MLP with #param: %d' % (get_n_params(model)))
    train(model, hg)