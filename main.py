import scipy.io
import urllib.request
import dgl
import math
import numpy as np
from model import *
import argparse

torch.manual_seed(0)
data_url = 'https://data.dgl.ai/dataset/ACM.mat'
data_file_path = '/tmp/ACM.mat'

urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)


parser = argparse.ArgumentParser(description='Training GNN on ogbn-products benchmark')


parser.add_argument('--mode', type=str, default='GCN')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--n_hid',   type=int, default=256)
parser.add_argument('--n_inp',   type=int, default=256)
parser.add_argument('--clip',    type=int, default=1.0)
parser.add_argument('--max_lr',  type=float, default=1e-3)

args = parser.parse_args()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def train(model, G):
    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)
    train_step = torch.tensor(0)
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        logits = model(G, 'paper')
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        if epoch % 5 == 0:
            model.eval()
            logits = model(G, 'paper')
            pred   = logits.argmax(1).cpu()
            train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
            val_acc   = (pred[val_idx]   == labels[val_idx]).float().mean()
            test_acc  = (pred[test_idx]  == labels[test_idx]).float().mean()
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            print('Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                epoch,
                optimizer.param_groups[0]['lr'],
                loss.item(),
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),
            ))

# Defining Conversion Functions
def encode_and_extend_features(G, node_types, node_type_features, all_features, typeonehot):
    for ntype in node_types:
        type_vec = torch.tensor(typeonehot[ntype]).unsqueeze(0)
        type_vec = type_vec.repeat(G.number_of_nodes(ntype), 1)
        G.nodes[ntype].data['type'] = type_vec.to(device)

        new_feat = torch.full((G.number_of_nodes(ntype), len(all_features)), -1.0)
        for i, f in enumerate(all_features):
            if f in node_type_features[ntype]:
                old_idx = node_type_features[ntype].index(f)
                new_feat[:, i] = G.nodes[ntype].data['inp'][:, old_idx]

        G.nodes[ntype].data['inp'] = new_feat.to(device)

device = torch.device("cuda:0")

G = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
    })

pvc = data['PvsC'].tocsr()
p_selected = pvc.tocoo()

# generate labels
labels = pvc.indices
labels = torch.tensor(labels).long()

# generate train/val/test split
pid = p_selected.row
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:800]).long()
val_idx = torch.tensor(shuffle[800:900]).long()
test_idx = torch.tensor(shuffle[900:]).long()

node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict)
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

# Random initialize input feature
for ntype in G.ntypes:
    emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 256), requires_grad = False)
    nn.init.xavier_uniform_(emb)
    G.nodes[ntype].data['inp'] = emb

G = G.to(device)

if args.mode == 'GCN':
    # Creating a one-hot encoded dictionary
    typeonehot = {n: np.eye(len(list(set(G.ntypes))))[i] for i, n in enumerate(list(set(G.ntypes)))}
    node_type_features = {}
    for ntype in G.ntypes:
        node_type_features[ntype] = [str(ntype) + str(i) for i in range(G.nodes[ntype].data['inp'].shape[1])]

    # Find all possible features
    all_features = sorted(list(set([f for features in node_type_features.values() for f in features])))

    encode_and_extend_features(G, G.ntypes, node_type_features, all_features, typeonehot)
    H = dgl.to_homogeneous(G).to(device)
    G_feat = torch.cat([G.nodes[ntype].data['inp'] for ntype in G.ntypes])
    G_type = torch.cat([G.nodes[ntype].data['type'] for ntype in G.ntypes])
    H.ndata['feat'] = torch.cat((G_feat, G_type), dim = 1).to(device)

    model = GCN(H, len(all_features) + len(node_type_features), args.n_hid, labels.max().item()+1).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
    print('Training GCN with #param: %d' % (get_n_params(model)))
    train(model, H)

if args.mode == 'HGT':
    model = HGT(G,
                node_dict, edge_dict,
                n_inp=args.n_inp,
                n_hid=args.n_hid,
                n_out=labels.max().item()+1,
                n_layers=2,
                n_heads=4,
                use_norm = True).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
    print('Training HGT with #param: %d' % (get_n_params(model)))
    train(model, G)

if args.mode == 'RGCN':
    model = HeteroRGCN(G,
                       in_size=args.n_inp,
                       hidden_size=args.n_hid,
                       out_size=labels.max().item()+1).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
    print('Training RGCN with #param: %d' % (get_n_params(model)))
    train(model, G)


if args.mode == 'MLP':
    model = HGT(G,
                node_dict, edge_dict,
                n_inp=args.n_inp,
                n_hid=args.n_hid,
                n_out=labels.max().item()+1,
                n_layers=0,
                n_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
    print('Training MLP with #param: %d' % (get_n_params(model)))
    train(model, G)
