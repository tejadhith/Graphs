import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from models.gnnfunctions import GNNEssentials
from models.hyperbolic_models import LorentzModel, PoincareModel


class HGCNLayer(nn.Module):
    """
    Implementation of a single HGCN layer with Lorentz Hyperbolic space

    by Divyanshu
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        device,
        model="lorentz",
    ):
        """
        Initialize the layer

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension
        c : float
            Curvature of the hyperbolic space
        device : str
            Device to run on (cpu, cuda, mps)
        model : str, optional
            Hyperbolic space model, by default "lorentz"
        """

        super(HGCNLayer, self).__init__()
        self.device = device
        self.c = torch.tensor(1.0).to(self.device)

        if model == "lorentz":
            self.model = LorentzModel(self.device)
            self.biasW = nn.Parameter(
                torch.randn(output_dim + 1, 1), requires_grad=True
            )
            self.biasB = nn.Parameter(
                torch.randn(output_dim + 1, 1), requires_grad=True
            )
        elif model == "poincare":
            self.model = PoincareModel(self.device)
            self.biasW = nn.Parameter(torch.randn(output_dim, 1), requires_grad=True)
            self.biasB = nn.Parameter(torch.randn(output_dim, 1), requires_grad=True)
        self.gnn_essential = GNNEssentials(self.device)

        self.layerW = nn.Linear(input_dim, output_dim, bias=True)
        self.layerB = nn.Linear(input_dim, output_dim, bias=True)

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
        feature : torch.Tensor
            Output tensor
        """

        n = features.shape[0]

        sparse_adj_matrix = self.gnn_essential.getSparseAdjacency(edge_index, (n, n))
        degree_matrix = self.gnn_essential.getInvesreDegreeMatrix(edge_index)

        featureW = self.model.matrixMultiplication(features, self.layerW, self.c)
        featureW = self.model.logarithmicMap(featureW, self.c)
        featureW = torch.spmm(sparse_adj_matrix, featureW)
        featureW = torch.spmm(degree_matrix, featureW)

        featureB = self.model.matrixMultiplication(features, self.layerB, self.c)
        featureB = self.model.logarithmicMap(featureB, self.c)

        features = featureW + featureB
        features = self.model.exponentialMap(features, self.c)

        return features
