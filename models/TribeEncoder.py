import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import Set2Set
import numpy as np
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling


class SELayer(nn.Module):
    """Squeeze-and-excitation networks"""

    def __init__(self, in_channels, se_channels):
        super(SELayer, self).__init__()

        self.in_channels = in_channels
        self.se_channels = se_channels

        self.encoder_decoder = nn.Sequential(
            nn.Linear(in_channels, se_channels),
            nn.ELU(),
            nn.Linear(se_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """"""
        # Aggregate input representation
        x_global = torch.mean(x, dim=0)
        # Compute reweighting vector s
        s = self.encoder_decoder(x_global)

        return x * s


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp, use_selayer):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = (
            SELayer(self.mlp.output_dim, int(np.sqrt(self.mlp.output_dim)))
            if use_selayer
            else nn.BatchNorm1d(self.mlp.output_dim)
        )

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_selayer):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(
                    SELayer(hidden_dim, int(np.sqrt(hidden_dim)))
                    if use_selayer
                    else nn.BatchNorm1d(hidden_dim)
                )

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)

class UnsupervisedGIN(nn.Module):

    def __init__(
        self,
        num_layers,
        num_mlp_layers,
        input_dim,
        hidden_dim,
        output_dim,
        final_dropout,
        learn_eps,
        graph_pooling_type,
        neighbor_pooling_type,
        use_selayer,
    ):
        super(UnsupervisedGIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(
                    num_mlp_layers, input_dim, hidden_dim, hidden_dim, use_selayer
                )
            else:
                mlp = MLP(
                    num_mlp_layers, hidden_dim, hidden_dim, hidden_dim, use_selayer
                )

            self.ginlayers.append(
                GINConv(
                    ApplyNodeFunc(mlp, use_selayer),
                    neighbor_pooling_type,
                    0,
                    self.learn_eps,
                )
            )
            self.batch_norms.append(
                SELayer(hidden_dim, int(np.sqrt(hidden_dim)))
                if use_selayer
                else nn.BatchNorm1d(hidden_dim)
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == "sum":
            self.pool = SumPooling()
        elif graph_pooling_type == "mean":
            self.pool = AvgPooling()
        elif graph_pooling_type == "max":
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h, efeat):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        all_outputs = []
        for i, h in list(enumerate(hidden_rep)):
            pooled_h = self.pool(g, h)
            all_outputs.append(pooled_h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer, all_outputs[1:]


class GraphEncoder(nn.Module):
    """
    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 15.
    edge_input_dim : int
        Dimension of input edge feature, default to be 15.
    output_dim : int
        Dimension of prediction, default to be 12.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 64.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    num_step_set2set : int
        Number of set2set steps
    num_layer_set2set : int
        Number of set2set layers
    """

    def __init__(
        self,
        positional_embedding_size=32,
        max_node_freq=8,
        max_edge_freq=8,
        max_degree=128,
        freq_embedding_size=32,
        degree_embedding_size=16,
        output_dim=32,
        node_hidden_dim=32,
        edge_hidden_dim=32,
        num_layers=6,
        num_heads=4,
        num_step_set2set=6,
        num_layer_set2set=3,
        norm=False,
        gnn_model="mpnn",
        degree_input=False,
        lstm_as_gate=False,
        shortest_path_input=True,
        max_path_length=6,
        sp_embedding_size=16,
        directional_subgraph=False,
        ntype_embedding_size=32,
        ntype_num = 2,
    ):
        super(GraphEncoder, self).__init__()

        if degree_input:
            node_input_dim = positional_embedding_size + degree_embedding_size + 1
        else:
            node_input_dim = positional_embedding_size + 1
        self.directional_subgraph = directional_subgraph
        if shortest_path_input:
            if directional_subgraph:
                node_input_dim += sp_embedding_size * 2
            else:
                node_input_dim += sp_embedding_size
        node_input_dim += ntype_embedding_size
        print('node_input_dim:', node_input_dim)
        if gnn_model == "gin":
            self.gnn = UnsupervisedGIN(
                num_layers=num_layers,
                num_mlp_layers=2,
                input_dim=node_input_dim,
                hidden_dim=node_hidden_dim,
                output_dim=output_dim,
                final_dropout=0.5,
                learn_eps=False,
                graph_pooling_type="sum",
                neighbor_pooling_type="sum",
                use_selayer=False,
            )
        self.gnn_model = gnn_model

        self.max_node_freq = max_node_freq
        self.max_edge_freq = max_edge_freq
        self.max_degree = max_degree
        self.degree_input = degree_input
        self.shortest_path_input = shortest_path_input
        self.max_path_length = max_path_length
        if shortest_path_input:
            if directional_subgraph:
                self.sp_in_embedding = nn.Embedding(num_embeddings=max_path_length + 2, embedding_dim=sp_embedding_size)
                self.sp_out_embedding = nn.Embedding(num_embeddings=max_path_length + 2, embedding_dim=sp_embedding_size)
            else:
                self.sp_embedding = nn.Embedding(num_embeddings=max_path_length + 2, embedding_dim=sp_embedding_size)
        self.ntype_embedding = nn.Embedding(num_embeddings=ntype_num, embedding_dim=ntype_embedding_size)
        if degree_input:
            self.degree_embedding = nn.Embedding(
                num_embeddings=max_degree + 1, embedding_dim=degree_embedding_size
            )

        self.set2set = Set2Set(node_hidden_dim, num_step_set2set, num_layer_set2set)
        self.lin_readout = nn.Sequential(
            nn.Linear(2 * node_hidden_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, output_dim),
        )
        self.norm = norm

    def forward(self, g, return_all_outputs=False):
        # nfreq = g.ndata["nfreq"]
        if self.shortest_path_input and self.degree_input:
            device = g.ndata["seed"].device
            degrees = g.in_degrees()
            # print(device, g.device)
            if device != torch.device("cpu"):
                degrees = degrees.cuda(device)
            if self.directional_subgraph:
                n_feat = torch.cat(
                    (
                        g.ndata["pos_undirected"], # 32
                        self.degree_embedding(degrees.clamp(0, self.max_degree)), # 16
                        self.sp_in_embedding(g.ndata['sp_in_emb'].clamp(-1, self.max_path_length).add(1)), # 16
                        self.sp_out_embedding(g.ndata['sp_out_emb'].clamp(-1, self.max_path_length).add(1)), # 16
                        g.ndata["seed"].unsqueeze(1).float(), # 1
                        self.ntype_embedding(g.ndata['ntype'].long()),
                    ),dim=-1)
            else:
                n_feat = torch.cat(
                    (
                        g.ndata["pos_undirected"], # 32
                        self.degree_embedding(degrees.clamp(0, self.max_degree)), # 16
                        self.sp_embedding(g.ndata['sp_in_emb'].clamp(-1, self.max_path_length).add(1)), # 16
                        g.ndata["seed"].unsqueeze(1).float(), # 1
                        self.ntype_embedding(g.ndata['ntype'].long()),
                    ),dim=-1)
        elif self.degree_input:
            device = g.ndata["seed"].device
            degrees = g.in_degrees()
            if device != torch.device("cpu"):
                degrees = degrees.cuda(device)

            n_feat = torch.cat(
                (
                    g.ndata["pos_undirected"], # 32
                    self.degree_embedding(degrees.clamp(0, self.max_degree)), # 16
                    g.ndata["seed"].unsqueeze(1).float(), # 1
                    g.ndata['ntype'].float(), # 2
                ),
                dim=-1,
            )
        else:
            n_feat = torch.cat(
                (
                    g.ndata["pos_undirected"],
                    g.ndata["seed"].unsqueeze(1).float(),

                ),
                dim=-1,
            )
        e_feat = None
        if self.gnn_model == "gin":
            x, all_outputs = self.gnn(g, n_feat, e_feat)
        else:
            x, all_outputs = self.gnn(g, n_feat, e_feat), None
            x = self.set2set(g, x)
            x = self.lin_readout(x)
        if self.norm:
            x = F.normalize(x, p=2, dim=-1, eps=1e-5)
        if return_all_outputs:
            return x, all_outputs
        else:
            return x