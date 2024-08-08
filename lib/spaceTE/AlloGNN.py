import torch
import torch.nn.functional as F
import torch.nn as nn

from .res_EdgeGAT import ResidualEdgeGATConv

class AlloGNN(nn.Module):
    def __init__(
        self,
        env,
        in_sizes,
        hidden_size,
        out_sizes,
        num_heads,
        decoder,
        canonical_etypes,
        dropout=0.05,
    ):
        """Graph convolution model inspired by `SCENE <https://arxiv.org/pdf/2301.03512.pdf>`.
            The model cascades multiple layers of graph convolution to aggregate information
            into the path nodes, then allocate flows.

        Args:
            in_sizes (dict): Dictionary containing node types and size of ndata per node type
                of the knowledge graph to be trained on.
            hidden_size (int): Hidden size used during graph convolution.
            out_sizes (dict): Dictionary containing the node type to be classified and the number
                of possible classes of this node type.
            num_heads (int): Number of attention heads of the EdgeGAT operator.
            canonical_etypes (list[(str, str, str)]): List of the canonical edge types of the knowledge
                graph to be trained on.
            learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 0.001.
            weight_decay (float, optional): Weight decay for Adam optimizer. Defaults to 0.0.
            dropout (float, optional): Dropout applied during decoding. Defaults to 0.0.
        """
        super().__init__()
        self.env = env
        self.in_sizes = in_sizes
        self.hidden_size = hidden_size
        self.out_sizes = out_sizes
        self.num_heads = num_heads
        self.canonical_etypes = canonical_etypes
        self.dropout_p = dropout
        self.device = self.env.device
        
        # Extract training objective
        self.category = list(out_sizes.keys())[0]
        self.out_size = out_sizes[self.category]

        # self.node_embeddings = torch.nn.ParameterDict()

        self.projector = torch.nn.ModuleDict()
        for type, in_size in in_sizes.items():
            self.projector[type] = torch.nn.Linear(in_size, hidden_size)
            
        # self.path_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)

        # Initialize graph convolutions (cascaded style)
        self.full_graph_conv = torch.nn.ModuleDict()
        self.full_graph_conv["conv_1"] = torch.nn.ModuleDict()
        self.full_graph_conv["conv_2"] = torch.nn.ModuleDict()
        # self.full_graph_conv["conv_3"] = torch.nn.ModuleDict()
        # self.full_graph_conv["conv_4"] = torch.nn.ModuleDict()
        for edge in canonical_etypes:
            # [('flow', 'uses', 'path'), ('link', 'constitutes', 'path')]
            which_graph_conv = None
            # if (edge[0] == edge[2]) and (edge[0] != self.category):
            #     # Self-conv excluding the target node
            #     which_graph_conv = "conv_1"
            # elif (edge[0] != edge[2]) and (edge[2] != self.category):
            #     # Conv from all nodes to others but the target node
            #     which_graph_conv = "conv_2"
            # elif (edge[2] == self.category) and (edge[0] != self.category):
            #     # Conv to the target node
            #     which_graph_conv = "conv_3"
            # elif (edge[2] == self.category) and (edge[0] == self.category):
            #     # Self update
            #     which_graph_conv = "conv_4"
            # else:
            #     NotImplementedError(
            #         f"Undefined graph convolution for edge {edge}")

            if (edge[0] == 'link'):
                which_graph_conv = "conv_1"
            elif (edge[0] == 'flow'):
                which_graph_conv = "conv_2"

            if which_graph_conv == "conv_1":
                self.full_graph_conv[which_graph_conv][str(edge)] = ResidualEdgeGATConv(in_feats=hidden_size, out_feats=hidden_size, num_heads=num_heads, edge_feats=1)
            elif which_graph_conv is not None:
                self.full_graph_conv[which_graph_conv][str(edge)] = ResidualEdgeGATConv(
                    in_feats=hidden_size, out_feats=hidden_size, num_heads=num_heads)
                
        # Initialize decoder
        if decoder == "linear":
            self.decoder_1 = torch.nn.Linear(
                in_features=hidden_size*3, out_features=hidden_size)
            self.dropout_1 = torch.nn.Dropout(p=dropout)
            self.decoder_2 = torch.nn.Linear(
                in_features=hidden_size, out_features=self.out_size)
            self.dropout_2 = torch.nn.Dropout(p=dropout)
        elif decoder == 'transformer':
            self.decoder_1 = nn.TransformerEncoderLayer(d_model=hidden_size*3, nhead=num_heads, dropout=dropout)
            self.transformer_encoder = nn.TransformerEncoder(self.decoder_1, num_layers=1)
            self.decoder_2 = torch.nn.Linear(
                in_features=hidden_size*3, out_features=self.out_size)
            self.dropout_2 = torch.nn.Dropout(p=dropout)
        else:
            raise ValueError("Unsupported decoder type: {}".format(decoder))

    def forward(self, graph, e2p_feature):
        """Forward pass of the model.

        Args:
            graph (dgl.heterograph.DGLHeteroGraph): Input graph.

        Returns:
            torch.Tensor: Allocation.
        """        
        # if not self.node_embeddings.keys():
        #     in_nodes = {}
        #     for ntype in graph:
        #         in_nodes[ntype] = graph.num_nodes(ntype)
        #     # Create node embeddings
        #     for key in self.in_nodes:
        #         if key == 'link':
        #             embed = torch.nn.Parameter(
        #                 self.edge_index_values.unsqueeze(1).repeat(1, self.hidden_size)
        #             )
        #             self.node_embeddings[key] = embed
        #         else: 
        #             embed = torch.nn.Parameter(
        #                 torch.Tensor(self.in_nodes[key], self.hidden_size))
        #             torch.nn.init.xavier_uniform_(
        #                 embed, gain=torch.nn.init.calculate_gain('relu'))
        #             self.node_embeddings[key] = embed

        # Node embedding update
        for ntype in graph.ntypes:
            graph.nodes[ntype].data["x"] = F.relu(self.projector[ntype](graph.nodes[ntype].data["x"].reshape(-1, self.in_sizes[ntype])))

        # Iterate over the cascaded layers
        for conv_key, conv_dict in self.full_graph_conv.items():
            # Collect the node types that are possible targets during this layer of graph convolution
            conv_dict_key_tuples = [
                tuple(map(str, string[2:-2].split("', '"))) for string in conv_dict.keys()]

            targets = [x[2] for x in conv_dict_key_tuples]
            targets = list(set(targets))

            # embeddings = {x: 0.0 for x in targets}
            embeddings = {node: torch.zeros_like(graph.nodes[node].data['x']) for node in targets}
            # Do graph convolution
            for curr_conv_key, curr_conv in conv_dict.items():
                # Get key of target node
                curr_tuple = tuple(map(str, curr_conv_key[2:-2].split("', '")))
                src_ntype = curr_tuple[0]
                target_ntype = curr_tuple[2]

                # Extract subgraph
                curr_subgraph = graph.edge_type_subgraph([curr_tuple])
                src_feats = graph.nodes[curr_tuple[0]].data["x"]
                dst_feats = graph.nodes[curr_tuple[2]].data["x"]

                if src_ntype == 'link' :
                    embeddings[target_ntype] += curr_conv(
                        curr_subgraph, (src_feats, dst_feats), e2p_feature)
                else:
                    embeddings[target_ntype] += curr_conv(
                            curr_subgraph, (src_feats, dst_feats))

            # Update each node simultaneously
            for node_key, embedding in embeddings.items():
                # embedding = nn.LayerNorm(embedding.size()).to(embedding.device)(embedding)
                graph.nodes[node_key].data["x"] = F.relu(embedding)

            # x = graph.nodes[self.category].data["x"]
            # x = self.path_encoder(x)
            # graph.nodes[self.category].data["x"] = x
            
            # Residual values
            if conv_key == "conv_1":
                x_res1 = graph.nodes[self.category].data["x"]
            if conv_key == "conv_2":
                x_res2 = graph.nodes[self.category].data["x"]

        # Decoder
        x = graph.nodes[self.category].data["x"]
        x = torch.cat([x, x_res1, x_res2], dim=-1)
        if hasattr(self, 'transformer_encoder'):
            x = self.transformer_encoder(x)
        else:
            x = self.decoder_1(x)
            x = self.dropout_1(x)
            x = F.relu(x)
        
        x = self.decoder_2(x)
        x = self.dropout_2(x)
        
        return x
