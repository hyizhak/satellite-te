import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl.function as fn

from .utils import weight_initialization

class TopoGNN(nn.Module): 

    def __init__(self, dytop_env, gnn_type):
        super(TopoGNN, self).__init__()
        self.edge_gat_1 = dglnn.EdgeGATConv(
            in_feats=1,
            edge_feats=1,
            out_feats=16,
            num_heads=16,)
        self.edge_gat_2 = dglnn.EdgeGATConv(
            in_feats=16,
            edge_feats=1,
            out_feats=8,
            num_heads=8,)
        self.edge_feature = nn.Linear(16, 8)

    def forward(self, G, efeatures):

        efeatures = efeatures.reshape(-1, 1)

        num_nodes = G.number_of_nodes()
        
        G = dgl.add_self_loop(G)

        # may cause bugs to directly set the device
        self_loop_features = torch.zeros((num_nodes, 1)).to('cuda')

        efeatures = torch.cat((efeatures, self_loop_features), dim=0)

        # Initialize a temporary edge feature for message passing
        G.edata['temp'] = G.edata['capacity']

        # Use message passing to sum the capacities of incoming edges for each node
        G.update_all(fn.copy_e('temp', 'm'), fn.sum('m', 'sum_cap'))

        nfeatures = torch.unsqueeze(G.ndata['sum_cap'], 1)

        x = self.edge_gat_1(G, nfeatures, efeatures)
        # (N, H, O) -> (N, O)
        x = torch.sum(x, 1)

        x = F.relu(x)

        x = self.edge_gat_2(G, x, efeatures)

        x = torch.sum(x, 1)

        # Set the new node features in the graph
        G.ndata['feat'] = x

        G = dgl.remove_self_loop(G)

        # Define a function to concatenate the features of source and destination nodes
        def generate_edge_features(edges):
            # edges.src['feat'] and edges.dst['feat'] are the features of source and destination nodes
            # return {'edge_feat': (edges.src['feat'] * edges.dst['feat']) ** 0.5}
            concatenated_features = torch.cat([edges.src['feat'], edges.dst['feat']], dim=1) 
            edge_features = self.edge_feature(concatenated_features)
            return {'edge_feat': edge_features}

        # Apply the function to all edges in the graph
        G.apply_edges(generate_edge_features)

        # Now G.edata['edge_feat'] contains the generated edge features
        edge_features = G.edata['edge_feat']

        return edge_features


