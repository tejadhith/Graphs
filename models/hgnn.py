import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP
from models.hgnnlayers import HGCNLayer
from models.hyperbolic_models import LorentzModel, PoincareModel


class HGCNModel(nn.Module):
    """
    Hyperbolic GCN model with 2 layers and an MLP classifier
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        device,
        model="lorentz",
        dropout=0.2,
        batch_norm=False,
    ):
        """
        Initialize the model

        Parameters
        ----------
        input_dim : int
            Input dimension
        hidden_dims : list
            List of hidden dimensions of size 3 (HGCN hidden dim, embedding dim, MLP hidden dim)
        output_dim : int
            Output dimension
        device : str
            Device to run on (cpu, cuda, mps)
        model : str, optional
            Hyperbolic space model, by default "lorentz"
        dropout : float, optional
            Dropout probability, by default 0.2
        batch_norm : bool, optional
            Whether to use batch normalization, by default False
        """

        super(HGCNModel, self).__init__()
        self.device = device
        self.hgcnconv1 = HGCNLayer(input_dim, hidden_dims[0], self.device, model=model)
        self.hgcnconv2 = HGCNLayer(
            hidden_dims[0], hidden_dims[1], self.device, model=model
        )

        self.hyperbolic_space = model
        self.c = torch.tensor(1.0).to(self.device)

        if model == "lorentz":
            self.model = LorentzModel(self.device)
        elif model == "poincare":
            self.model = PoincareModel(self.device)

        self.batch_norm = batch_norm
        if self.batch_norm:
            if model == "lorentz":
                self.batch_norm_layer1 = nn.BatchNorm1d(hidden_dims[0] + 1)
                self.batch_norm_layer2 = nn.BatchNorm1d(hidden_dims[1] + 1)
            else:
                self.batch_norm_layer1 = nn.BatchNorm1d(hidden_dims[0])
                self.batch_norm_layer2 = nn.BatchNorm1d(hidden_dims[1])

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

        if self.hyperbolic_space == "lorentz":
            features = torch.cat(
                [torch.zeros(features.shape[0], 1).to(self.device).float(), features],
                dim=1,
            )
            features = self.model.exponentialMap(features, self.c)
        elif self.hyperbolic_space == "poincare":
            features = self.model.exponentialMap(features, self.c)

        features = self.hgcnconv1(features, edge_index)

        if self.batch_norm:
            features = self.model.logarithmicMap(features, self.c)
            features = self.batch_norm_layer1(features)
            features = self.model.exponentialMap(features, self.c)

        features = self.model.sigmoidNonLinearity(features, self.c, self.c)

        features = self.hgcnconv2(features, edge_index)
        if self.batch_norm:
            features = self.model.logarithmicMap(features, self.c)
            features = self.batch_norm_layer2(features)
            features = self.model.exponentialMap(features, self.c)

        features = self.model.sigmoidNonLinearity(features, self.c, self.c)
        features = self.model.logarithmicMap(features, self.c)

        if self.hyperbolic_space == "lorentz":
            features = features[:, 1:]
        features = self.mlp(features)

        return features


class HyperbolicGraphSage(nn.Module):
    """
    Hyperbolic GraphSage model with 2 layers and an MLP classifier
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        device,
        model="lorentz",
        dropout=0.2,
        batch_norm=False,
    ):
        """
        Initialize the model

        Parameters
        ----------
        input_dim : int
            Input dimension
        hidden_dims : list
            List of hidden dimensions of size 3 (HGCN hidden dim, embedding dim, MLP hidden dim)
        output_dim : int
            Output dimension
        device : str
            Device to run on (cpu, cuda, mps)
        model : str, optional
            Hyperbolic space model, by default "lorentz"
        dropout : float, optional
            Dropout probability, by default 0.2
        batch_norm : bool, optional
            Whether to use batch normalization, by default False
        """

        super(HGCNModel, self).__init__()
        self.device = device
        self.hgcnconv1 = HGCNLayer(input_dim, hidden_dims[0], self.device, model=model)
        self.hgcnconv2 = HGCNLayer(
            hidden_dims[0], hidden_dims[1], self.device, model=model
        )

        self.hyperbolic_space = model
        self.c = torch.tensor(1.0).to(self.device)

        if model == "lorentz":
            self.model = LorentzModel(self.device)
        elif model == "poincare":
            self.model = PoincareModel(self.device)

        self.mlp = MLP(
            hidden_dims[1],
            hidden_dims[2],
            output_dim,
            dropout=dropout,
            batch_norm=batch_norm,
        )
