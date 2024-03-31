import scipy.io
import time
import dgl
import numpy as np
import torch
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from ogb.nodeproppred import DglNodePropPredDataset
from model import *
import argparse

data_url = 'https://data.dgl.ai/dataset/ACM.mat'
data_file_path = '/tmp/ACM.mat'

parser = argparse.ArgumentParser(description='Training GNN on ogbn-products benchmark')

parser.add_argument('--dataset',  type=str, default='bgs')
args = parser.parse_args()

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
elif args.dataset == 'ogbn':
    pass
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
    torch.manual_seed(42)
    for ntype in hg.ntypes:
        emb = torch.rand(hg.number_of_nodes(ntype), 256)
        hg.nodes[ntype].data['inp'] = emb.to(device)
    torch.manual_seed(int(time.time()))

    h_dict = {}
    for ntype in hg.ntypes:
        h_dict[ntype] = hg.nodes[ntype].data['inp'].to(device)
    category = 'paper'

elif args.dataset == 'ogbn':
    dataset = DglNodePropPredDataset(name = 'ogbn-mag')
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    hg, labels = dataset[0]
    labels = labels['paper']
    labels = torch.tensor(labels).to(device)
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
        emb = torch.rand(hg.number_of_nodes(ntype), 32)
        hg.nodes[ntype].data['inp'] = emb.to(device)
    torch.manual_seed(int(time.time()))

    h_dict = {}
    for ntype in hg.ntypes:
        print(ntype)
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
    torch.manual_seed(int(time.time()))

    h_dict = {}
    if args.sample == False:
        for ntype in hg.ntypes:
            h_dict[ntype] = hg.nodes[ntype].data['inp'].to(device)

    hg = hg.to(device)