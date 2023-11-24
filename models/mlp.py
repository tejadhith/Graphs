import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-layer perceptron with 1 hidden layer

    by Tejadhith
    """

    def __init__(
        self, input_dim, hidden_dim, output_dim, dropout=0.2, batch_norm=False
    ):
        """
        Initialize the model

        Parameters
        ----------
        input_dim : int
            Input dimension
        hidden_dim : int
            Hidden dimension
        output_dim : int
            Output dimension
        dropout : float, optional
            Dropout probability, by default 0.2
        batch_norm : bool, optional
            Whether to use batch normalization, by default False
        """

        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        if batch_norm:
            self.dropout_batch_norm = nn.Sequential(
                nn.BatchNorm1d(hidden_dim), nn.Dropout(p=dropout)
            )
        else:
            self.dropout_batch_norm = nn.Dropout(p=dropout)

    def forward(self, features):
        """
        Forward pass

        Parameters
        ----------
        features : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """

        features = self.linear1(features)
        features = self.dropout_batch_norm(features)
        features = F.relu(features)
        features = self.linear2(features)
        return features
