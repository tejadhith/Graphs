import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gnnfunctions import GNNEssentials


class GCNLayer(nn.Module):
    """
    Implementation of a single GCN layer
    """

    def __init__(self, input_dim, output_dim, device):
        """
        Initialize the layer

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension
        device : str
            Device to run on (cpu, cuda, mps)
        """

        super(GCNLayer, self).__init__()
        self.device = device
        self.gnn_essential = GNNEssentials(self.device)

        self.layerW = nn.Linear(input_dim, output_dim)
        self.layerB = nn.Linear(input_dim, output_dim)

    def forward(self, features, edge_index):
        """
        Forward pass

        Parameters
        ----------
        features : torch.Tensor
            Input tensor
        edge_index : torch.Tensor
            Edge index tensor

        Returns
        -------
        features : torch.Tensor
            Output tensor
        """

        n = features.shape[0]
        sparse_adj_matrix = self.gnn_essential.getSparseAdjacency(edge_index, (n, n))
        degree_matrix = self.gnn_essential.getInvesreDegreeMatrix(edge_index)

        featureW = torch.spmm(sparse_adj_matrix, features)
        featureW = torch.spmm(degree_matrix, featureW)
        featureW = self.layerW(featureW)

        featureB = self.layerB(features)
        return featureW + featureB


class GraphSageLayer(nn.Module):
    """
    Implementation of a single GraphSage layer
    """

    def __init__(self, input_dim, output_dim, agg_layer, device):
        """
        Initialize the layer

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension
        agg_layer : str
            Aggregation layer to use (mean, sum, mlp)
        device : str
            Device to run on (cpu, cuda, mps)
        """

        super(GraphSageLayer, self).__init__()
        self.device = device
        self.gnn_essential = GNNEssentials(self.device)

        self.agg_layer = agg_layer
        if self.agg_layer == "mlp":
            self.aggregate_layer = nn.Linear(input_dim, input_dim)

        self.linearW = nn.Linear(2 * input_dim, output_dim)

    def forward(self, features, edge_index):
        """
        Forward pass

        Parameters
        ----------
        features : torch.Tensor
            Input tensor
        edge_index : torch.Tensor
            Edge index tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """

        n = features.shape[0]
        if self.agg_layer == "mean":
            neighbour_features = self.gnn_essential.meanAggregation(
                features, edge_index
            )
        elif self.agg_layer == "sum":
            neighbour_features = self.gnn_essential.sumAggregation(features, edge_index)
        elif self.agg_layer == "mlp":
            neighbour_features = self.gnn_essential.mlpAggregation(
                features, edge_index, self.aggregate_layer
            )

        features = torch.cat((features, neighbour_features), dim=1)
        return self.linearW(features)


class GraphAttentionNetwork(nn.Module):
    """
    Implementation of a single Graph Attention Network layer
    """

    def __init__(
        self, input_dim, attention_hidden_dim, number_of_heads, output_dim, device
    ):
        """
        Initialize the layer

        Parameters
        ----------
        input_dim : int
            Input dimension
        attention_hidden_dim : int
            Hidden dimension of the attention layer
        number_of_heads : int
            Number of attention heads
        output_dim : int
            Output dimension
        device : str
            Device to run on (cpu, cuda, mps)
        """

        super(GraphAttentionNetwork, self).__init__()
        self.device = device
        if (output_dim % number_of_heads) != 0:
            raise ValueError("Hidden dimension must be divisible by number of heads")
        self.number_of_heads = number_of_heads
        self.output_dim = output_dim // number_of_heads

        self.gnn_essential = GNNEssentials(self.device)

        self.query, self.key = nn.ModuleList(), nn.ModuleList()
        for i in range(number_of_heads):
            self.query.append(nn.Linear(input_dim, attention_hidden_dim))
            self.key.append(nn.Linear(input_dim, attention_hidden_dim))

        self.linearW = nn.Linear(input_dim, self.output_dim)
        self.linearB = nn.Linear(input_dim, output_dim)

    def forward(self, features, edge_index):
        """
        Forward pass

        Parameters
        ----------
        features : torch.Tensor
            Input tensor
        edge_index : torch.Tensor
            Edge index tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """

        n = features.shape[0]
        all_neighbour_features = []
        node_features = self.linearB(features)

        for query_layer, key_layer in zip(self.query, self.key):
            query = query_layer(features)
            key = key_layer(features)

            attention_weights = torch.mm(query, key.t())

            att_weights = -torch.inf * torch.ones(attention_weights.shape)
            att_weights = att_weights.to(self.device)

            att_weights[edge_index[0], edge_index[1]] = attention_weights[
                edge_index[0], edge_index[1]
            ]

            attention_weights = F.softmax(att_weights, dim=1).to_sparse()

            neighbour_features = self.linearW(features)
            neighbour_features = torch.spmm(attention_weights, neighbour_features)
            all_neighbour_features.append(neighbour_features)

        all_neighbour_features = torch.cat(all_neighbour_features, dim=1)
        return node_features + all_neighbour_features
