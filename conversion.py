import torch
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

# Define node types and corresponding features
node_types = ['paper', 'author']
node_type_features = {
    'paper': ['f1', 'f4'],
    'author': ['f1', 'f2', 'f3'],
}

# Find all possible features
all_features = sorted(list(set([f for features in node_type_features.values() for f in features])))

# Creating a one-hot encoded dictionary
type2onehot = {n: np.eye(len(node_types))[i] for i, n in enumerate(node_types)}

# Defining Conversion Functions
def encode_and_extend_features(G, node_types, node_type_features, all_features, type2onehot):
    for ntype in node_types:
        type_vec = torch.tensor(type2onehot[ntype]).unsqueeze(0)
        type_vec = type_vec.repeat(G.number_of_nodes(ntype), 1)
        G.nodes[ntype].data['type'] = type_vec

        new_feat = torch.full((G.number_of_nodes(ntype), len(all_features)), -1.0)
        for i, f in enumerate(all_features):
            if f in node_type_features[ntype]:
                old_idx = node_type_features[ntype].index(f)
                new_feat[:, i] = G.nodes[ntype].data['feat'][:, old_idx]

        G.nodes[ntype].data['feat'] = new_feat

# Data preparation
data_dict = {
    ('paper', 'write', 'author'): (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
    ('author', 'write', 'paper'): (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
    ('paper', 'cite', 'paper'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
}
G = dgl.heterograph(data_dict)
G.nodes['paper'].data['feat'] = torch.randn(3, 2)
G.nodes['author'].data['feat'] = torch.randn(3, 3)
encode_and_extend_features(G, node_types, node_type_features, all_features, type2onehot)

# GNN model
class GNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_features, hidden_features)
        self.conv2 = dglnn.GraphConv(hidden_features, out_features)

    def forward(self, H, inputs):
        h = self.conv1(H, inputs.float())
        h = F.relu(h)
        h = self.conv2(H, h)
        return h

    def train(self, H, inputs, labels):
        optimizer = Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(100):
            outputs = model(H, inputs)

            # Calculation of losses
            loss = 0
            loss += loss_fn(outputs, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Output Log
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss {loss.item()}")

# Converting Isomorphic Images to Isomorphic Images
H = dgl.to_homogeneous(G)
G_feat = torch.cat([G.nodes[ntype].data['feat'] for ntype in G.ntypes])
G_type = torch.cat([G.nodes[ntype].data['type'] for ntype in G.ntypes])
H.ndata['feat'] = torch.cat((G_feat, G_type), dim = 1)

# Train model
model = GNN(len(all_features) + len(node_type_features), 10, 2)
inputs = H.ndata['feat']

# Define labels
labels = {ntype: torch.randint(0, 2, (G.number_of_nodes(ntype),)) for ntype in node_types}
labels = torch.cat([labels[ntype] for ntype in G.ntypes])

model.train(H, inputs, labels)
