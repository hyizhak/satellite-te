import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl.function as fn

from .utils import weight_initialization

class TopoGNN(nn.Module): 

    def __init__(self, teal_env, gnn_type):
        super(TopoGNN, self).__init__()
        self.edge_gat_1 = dglnn.EdgeGATConv(
            in_feats=1,
            edge_feats=1,
            out_feats=16,
            num_heads=2,)
        self.edge_gat_2 = dglnn.EdgeGATConv(
            in_feats=16,
            edge_feats=1,
            out_feats=1,
            num_heads=2,)

    def forward(self, G, efeatures):
        
        G = dgl.add_self_loop(G)

        # Initialize a temporary edge feature for message passing
        G.edata['temp'] = G.edata['capacity']

        # Use message passing to sum the capacities of incoming edges for each node
        G.update_all(fn.copy_e('temp', 'm'), fn.sum('m', 'sum_cap'))

        x = self.edge_gat_1(G, G.ndata['sum_cap'], efeatures)
        # (N, H, O) -> (N, O)
        x = torch.sum(x, 1)

        x = F.relu(x)

        x = self.edge_gat_2(G, x, efeatures)

        x = torch.sum(x, 1)

        # Set the new node features in the graph
        G.ndata['feat'] = x

        # Define a function to concatenate the features of source and destination nodes
        def generate_edge_features(edges):
            # edges.src['feat'] and edges.dst['feat'] are the features of source and destination nodes
            return {'edge_feat': (edges.src['feat'] * edges.dst['feat']) ** 0.5}

        # Apply the function to all edges in the graph
        G.apply_edges(generate_edge_features)

        # Now G.edata['edge_feat'] contains the generated edge features
        edge_features = G.edata['edge_feat']

        return edge_features


