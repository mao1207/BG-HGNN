import scipy.io
import urllib.request
import dgl
import math
import numpy as np
import torch
import torch.nn.functional as F
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset

from model import *
import argparse
from dgl.sampling import sample_neighbors

torch.manual_seed(0)
data_url = 'https://data.dgl.ai/dataset/ACM.mat'
data_file_path = '/tmp/ACM.mat'

# urllib.request.urlretrieve(data_url, data_file_path)
data = AIFBDataset()


parser = argparse.ArgumentParser(description='Training GNN on ogbn-products benchmark')


parser.add_argument('--mode', type=str, default='GCN')
parser.add_argument('--sample', type=bool, default=False)
parser.add_argument('--n_epoch', type=int, default=2000)
parser.add_argument('--n_hid',   type=int, default=256)
parser.add_argument('--n_inp',   type=int, default=256)
parser.add_argument('--clip',    type=int, default=1.0)
parser.add_argument('--max_lr',  type=float, default=1e-3)
parser.add_argument('--dataset',  type=str, default='bgs')
parser.add_argument('--trian_embedding',  type=bool, default=False)

args = parser.parse_args()

def get_node_embedding(model, G, epoch):
    G = G.to(device)
    for epoch in np.arange(epoch) + 1:
        model.train()
        optimizer_pre.zero_grad()
        logits = model(G)
        node_type = torch.argmax(logits, dim=1).float()
        loss = F.cross_entropy(logits, model.ntype_dict.long())
        loss.backward()
        optimizer_pre.step()
        train_acc = (node_type == model.ntype_dict).float().mean()
        if epoch % 5 == 0:
            model.eval()
            print(node_type)
            print('Epoch: %d, Loss %.10f, train_acc %.5f'% (
                epoch,
                loss.item(),
                train_acc.item()
            ))



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

if args.sample == True:
    def train(model, G):
        G = G.to(device)
        train_nid = {}
        for ntype in G.ntypes:
            nodes = G.nodes(ntype)
            train_nid[ntype] = nodes.to(device)
        sample_dict = {}
        for etype in G.canonical_etypes:
            sample_dict[etype] = 10
        sampler = dgl.dataloading.NeighborSampler([sample_dict]*2)
        dataloader = dgl.dataloading.DataLoader(
            G, train_nid, sampler,
            batch_size=30000, shuffle=True, drop_last=False)
        for epoch in np.arange(args.n_epoch) + 1:
            best_acc = torch.tensor(0)
            train_step = torch.tensor(0)
            for input_nodes, output_nodes, blocks in dataloader:
                model.train()
                logits = model(blocks, h_dict, category)
                # The loss is computed only for labeled nodes.
                label_block = blocks[-1].dstdata['label'][category][blocks[-1].dstdata['label'][category] != -1]
                indices_block = torch.nonzero(blocks[-1].dstdata['label'][category] != -1).squeeze()

                loss = F.cross_entropy(logits[indices_block], label_block)
                print(epoch, loss)
                pred   = logits.argmax(1).cpu()
                print(pred.shape)
                train_acc = (pred[indices_block.cpu()] == label_block.cpu()).float().mean()
                print(pred[indices_block.cpu()])
                print(label_block)
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
            train_step += 1
            scheduler.step(train_step)
            if epoch % 1 == 0:
                model.eval()
                logits = model(blocks, h_dict, category)
                pred   = logits.argmax(1).cpu()
                train_acc = (pred[indices_block.cpu()] == labels[indices_block.cpu()]).float().mean()
                print(pred[indices_block.cpu()])
                print(labels[indices_block.cpu()])
                if train_acc <= best_acc:
                    best_acc = train_acc
                print('Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Best Acc %.4f' % (
                    epoch,
                    optimizer.param_groups[0]['lr'],
                    loss.item(),
                    train_acc.item(),
                    best_acc.item(),
                ))
else:
    def train(model, G):
        best_val_acc = torch.tensor(0)
        best_test_acc = torch.tensor(0)
        train_step = torch.tensor(0)
        for epoch in np.arange(args.n_epoch) + 1:
            model.train()
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
            if epoch % 5 == 0:
                model.eval()
                logits = model(G, h_dict, category)
                pred   = logits.argmax(1).cpu()
                train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
                val_acc   = (pred[val_idx]   == labels[val_idx]).float().mean()
                test_acc  = (pred[test_idx]  == labels[test_idx]).float().mean()
                if best_val_acc <= val_acc:
                    best_val_acc = val_acc
                    if best_test_acc < test_acc:
                        best_test_acc = test_acc
                print('Epoch: %d LR: %.5f Loss %.10f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
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
        type_vec = torch.tensor(typeonehot[ntype]).unsqueeze(0).detach()
        type_vec = type_vec.repeat(G.number_of_nodes(ntype), 1)
        G.nodes[ntype].data['type'] = type_vec.to(device)

        new_feat = torch.full((G.number_of_nodes(ntype), len(all_features)), 0.0)
        for i, f in enumerate(all_features):
            if f in node_type_features[ntype]:
                old_idx = node_type_features[ntype].index(f)
                new_feat[:, i] = G.nodes[ntype].data['inp'][:, old_idx]

        G.nodes[ntype].data['inp'] = new_feat.to(device)

device = torch.device("cuda:0")

if args.dataset == 'aifb':
    dataset = AIFBDataset()
elif args.dataset == 'mutag':
    dataset = MUTAGDataset()
elif args.dataset == 'bgs':
    dataset = BGSDataset()
elif args.dataset == 'am':
    dataset = AMDataset()
elif args.dataset == 'ACM':
    data_url = 'https://data.dgl.ai/dataset/ACM.mat'
    data_file_path = '/tmp/ACM.mat'
    # urllib.request.urlretrieve(data_url, data_file_path)
    data = scipy.io.loadmat(data_file_path)
else:
    raise ValueError()

if args.dataset == 'ACM':
    hg = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
    }).to(device)

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
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.canonical_etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = (torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]).to(device)

    # Random initialize input feature
    for ntype in hg.ntypes:
        emb = torch.rand(hg.number_of_nodes(ntype), 256)
        hg.nodes[ntype].data['inp'] = emb.to(device)

    h_dict = {}
    for ntype in hg.ntypes:
        h_dict[ntype] = hg.nodes[ntype].data['inp'].to(device)
    category = 'paper'

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
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    # Random initialize input feature
    super_node_begin = hg.number_of_nodes()
    for ntype in hg.ntypes:
        # emb = nn.Parameter(torch.Tensor(hg.number_of_nodes(ntype), 256), requires_grad = False)
        # nn.init.xavier_uniform_(emb)
        emb = torch.rand((hg.number_of_nodes(ntype), 256))
        hg.nodes[ntype].data['inp'] = emb
        print(ntype,emb)

    h_dict = {}
    if args.sample == False:
        for ntype in hg.ntypes:
            h_dict[ntype] = hg.nodes[ntype].data['inp'].to(device)

    hg = hg.to(device)
if args.mode == 'GCN':
    # Creating a one-hot encoded dictionary
    if args.trian_embedding == True:
        model_pre = NodeTypePredictor(hg, 64)
        optimizer_pre = torch.optim.AdamW(model_pre.parameters(), lr = 0.01)
        print('Training node type embeddings with #param: %d' % (get_n_params(model_pre)))
        get_node_embedding(model_pre, hg, 40)
        ntype_embedding = model_pre.embeddings
    else:
        ntype_embedding = torch.rand((len(hg.ntypes), 64))

    h_t = torch.empty(0).to(device)
    for ntype in hg.srctypes:
        h_t = torch.cat((h_t, hg.nodes[ntype].data['inp']), dim = 0)

    typeonehot = {n: 1 * np.eye(len(list(set(hg.ntypes))))[i] for i, n in enumerate(list(set(hg.ntypes)))}
    # typeonehot = {n: ntype_embedding[i] for i, n in enumerate(list(set(hg.ntypes)))}
    node_type_features = {}
    for ntype in hg.ntypes:
        node_type_features[ntype] = [str(ntype) + str(i) for i in range(hg.nodes[ntype].data['inp'].shape[1])]

    # Find all possible features
    all_features = sorted(list(set([f for features in node_type_features.values() for f in features])))

    encode_and_extend_features(hg, hg.ntypes, node_type_features, all_features, typeonehot)
    G_feat = torch.cat([hg.nodes[ntype].data['inp'] for ntype in hg.ntypes])
    G_type = torch.cat([hg.nodes[ntype].data['type'] for ntype in hg.ntypes])
    # input = torch.cat((G_feat, G_type), dim=1).to(device).float()
    input = G_feat.to(device).float()
    # input = torch.rand((hg.number_of_nodes(), 256))
    model = GAT(hg, input, input.shape[-1], args.n_hid, labels.max().item() + 1, 3,[1, 1, 1]).to(device)
    # for name, param in model.named_parameters():
    #     print(name)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
    print('Training GCN with #param: %d' % (get_n_params(model)))
    train(model, hg)

if args.mode == 'simple-HGN':
    # Creating a one-hot encoded dictionary
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
