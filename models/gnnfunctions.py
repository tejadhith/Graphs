import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric


class GNNEssentials(nn.Module):
    """
    Implement functions for GNNs in general
    """

    def __init__(self, device):
        """
        Initialize the module

        Parameters
        ----------
        device : str
            Device to run on (cpu, cuda, mps)
        """
        super(GNNEssentials, self).__init__()
        self.device = device

    def getSparseAdjacency(self, edge_index, shape, edge_weight=None):
        """
        Get sparse representation of the adjacency matrix

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor
        shape : tuple
            Shape of the sparse representation
        edge_weight : torch.Tensor
            Edge weight tensor

        Returns
        -------
        sparse_repr : torch.Tensor
            Sparse representation of the adjacency matrix
        """

        if edge_weight is None:
            values = torch.ones(edge_index.shape[1]).to(self.device)
        else:
            values = edge_weight
        return torch.sparse_coo_tensor(edge_index, values, shape)

    def getInvesreDegreeMatrix(self, edge_index):
        """
        Get inverse degree vector of the layer

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor

        Returns
        -------
        degree_vector : torch.Tensor
            Inverse degree vector
        """

        degree_vector = 1 / torch_geometric.utils.degree(edge_index[0])
        indices = torch.arange(degree_vector.shape[0]).to(self.device)
        indices = torch.stack([indices, indices])
        n = degree_vector.shape[0]

        return torch.sparse_coo_tensor(indices, degree_vector, (n, n))

    def meanAggregation(self, features, edge_index):
        """
        Mean aggregation

        Parameters
        ----------
        features : torch.Tensor
            Input tensor
        edge_index : torch.Tensor
            Edge index tensor

        Returns
        -------
        torch.Tensor
            Aggregated tensor
        """

        n = features.shape[0]
        sparse_adj_matrix = self.getSparseAdjacency(edge_index, (n, n))
        degree_inverse_matrix = self.getInvesreDegreeMatrix(edge_index)

        features = torch.spmm(sparse_adj_matrix, features)
        features = torch.spmm(degree_inverse_matrix, features)
        return features

    def sumAggregation(self, features, edge_index):
        """
        Sum aggregation

        Parameters
        ----------
        features : torch.Tensor
            Input tensor
        edge_index : torch.Tensor
            Edge index tensor

        Returns
        -------
        torch.Tensor
            Aggregated tensor
        """

        n = features.shape[0]
        sparse_adj_matrix = self.getSparseAdjacency(edge_index, (n, n))

        features = torch.spmm(sparse_adj_matrix, features)
        return features

    def mlpAggregation(self, features, edge_index, linear_layer):
        """
        Multi-layer perceptron aggregation i.e. we transform the features and then apply mean aggregation

        Parameters
        ----------
        features : torch.Tensor
            Input tensor
        edge_index : torch.Tensor
            Edge index tensor
        linear_layer : torch.nn.Linear
            Linear layer to transform the features

        Returns
        -------
        torch.Tensor
            Aggregated tensor
        """

        n = features.shape[0]
        features = linear_layer(features)

        return self.meanAggregation(features, edge_index)
