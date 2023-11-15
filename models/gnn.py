import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP
from models.gnnlayers import GCNLayer, GraphSageLayer, GraphAttentionNetwork


class GCNModel(nn.Module):
    """
    GCN model with 2 layers and an MLP classifier
    """

    def __init__(
        self, input_dim, hidden_dims, output_dim, device, dropout=0.2, batch_norm=False
    ):
        """
        Initialize the model

        Parameters
        ----------
        input_dim : int
            Input dimension
        hidden_dims : list
            List of hidden dimensions of size 3 (GCN hidden dim, embedding dim, MLP hidden dim)
        output_dim : int
            Output dimension
        device : str
            Device to run on (cpu, cuda, mps)
        dropout : float, optional
            Dropout probability, by default 0.2
        batch_norm : bool, optional
            Whether to use batch normalization, by default False
        """

        super(GCNModel, self).__init__()
        self.device = device
        self.gcnconv1 = GCNLayer(input_dim, hidden_dims[0], self.device)
        self.gcnconv2 = GCNLayer(hidden_dims[0], hidden_dims[1], self.device)

        if batch_norm:
            self.dropout_batch_norm1 = nn.Sequential(
                nn.BatchNorm1d(hidden_dims[0]), nn.Dropout(p=dropout)
            )
            self.dropout_batch_norm2 = nn.Sequential(
                nn.BatchNorm1d(hidden_dims[1]), nn.Dropout(p=dropout)
            )
        else:
            self.dropout_batch_norm1 = nn.Dropout(p=dropout)
            self.dropout_batch_norm2 = nn.Dropout(p=dropout)

        self.mlp = MLP(
            hidden_dims[1],
            hidden_dims[2],
            output_dim,
            dropout=dropout,
            batch_norm=batch_norm,
        )

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

        features = self.gcnconv1(features, edge_index)
        features = F.sigmoid(self.dropout_batch_norm1(features))
        features = self.gcnconv2(features, edge_index)
        features = F.sigmoid(self.dropout_batch_norm2(features))
        return self.mlp(features)


class GraphSageModel(nn.Module):
    """
    GraphSage model with 2 layers and an MLP classifier
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        device,
        agg_layer="mean",
        dropout=0.2,
        batch_norm=False,
    ):
        """
        Parameters
        ----------
        input_dim : int
                Input dimension
        hidden_dims : list
            List of hidden dimensions of size 3 (GCN hidden dim, embedding dim, MLP hidden dim)
        output_dim : int
            Output dimension
        device : str
            Device to run on (cpu, cuda, mps)
        agg_layer : str, optional
            Aggregation layer to use (mean, sum, mlp), by default "mean"
        dropout : float, optional
            Dropout probability, by default 0.2
        batch_norm : bool, optional
            Whether to use batch normalization, by default False
        """

        super(GraphSageModel, self).__init__()
        self.device = device
        self.sage1 = GraphSageLayer(input_dim, hidden_dims[0], agg_layer, self.device)
        self.sage2 = GraphSageLayer(
            hidden_dims[0], hidden_dims[1], agg_layer, self.device
        )

        if batch_norm:
            self.dropout_batch_norm1 = nn.Sequential(
                nn.BatchNorm1d(hidden_dims[0]), nn.Dropout(p=dropout)
            )
            self.dropout_batch_norm2 = nn.Sequential(
                nn.BatchNorm1d(hidden_dims[1]), nn.Dropout(p=dropout)
            )
        else:
            self.dropout_batch_norm1 = nn.Dropout(p=dropout)
            self.dropout_batch_norm2 = nn.Dropout(p=dropout)

        self.mlp = MLP(
            hidden_dims[1],
            hidden_dims[2],
            output_dim,
            dropout=dropout,
            batch_norm=batch_norm,
        )

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

        features = self.sage1(features, edge_index)
        features = F.sigmoid(self.dropout_batch_norm1(features))
        features = self.sage2(features, edge_index)
        features = F.sigmoid(self.dropout_batch_norm2(features))
        return self.mlp(features)


class GATModel(nn.Module):
    """
    Graph Attention Network model with 2 layers and an MLP classifier
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        attention_dim,
        num_heads,
        device,
        dropout=0.2,
        batch_norm=False,
    ):
        """
        Parameters
        ----------
        input_dim : int
                Input dimension
        hidden_dims : list
            List of hidden dimensions of size 3 (GCN hidden dim, embedding dim, MLP hidden dim)
        output_dim : int
            Output dimension
        attention_dim : int
            Dimension of NN used to compute attention coefficients
        num_heads : int
            Number of attention heads
        device : str
            Device to run on (cpu, cuda, mps)
        dropout : float, optional
            Dropout probability, by default 0.2
        batch_norm : bool, optional
            Whether to use batch normalization, by default False
        """

        super(GATModel, self).__init__()
        self.device = device
        self.gat1 = GraphAttentionNetwork(
            input_dim, attention_dim, num_heads, hidden_dims[0], self.device
        )
        self.gat2 = GraphAttentionNetwork(
            hidden_dims[0], attention_dim, num_heads, hidden_dims[1], self.device
        )

        if batch_norm:
            self.dropout_batch_norm1 = nn.Sequential(
                nn.BatchNorm1d(hidden_dims[0]), nn.Dropout(p=dropout)
            )
            self.dropout_batch_norm2 = nn.Sequential(
                nn.BatchNorm1d(hidden_dims[1]), nn.Dropout(p=dropout)
            )
        else:
            self.dropout_batch_norm1 = nn.Dropout(p=dropout)
            self.dropout_batch_norm2 = nn.Dropout(p=dropout)

        self.mlp = MLP(
            hidden_dims[1],
            hidden_dims[2],
            output_dim,
            dropout=dropout,
            batch_norm=batch_norm,
        )

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

        features = self.gat1(features, edge_index)
        features = F.sigmoid(self.dropout_batch_norm1(features))
        features = self.gat2(features, edge_index)
        features = F.sigmoid(self.dropout_batch_norm2(features))
        return self.mlp(features)
