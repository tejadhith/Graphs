import torch
import torch.nn as nn
import torch.nn.functional as F


class PoincareModel(nn.Module):
    """
    Implement functions for Poincare Model

    by Divyanshu + Mirza
    """

    def __init__(self, device):
        """
        Initialize the module

        Parameters
        ----------
        device : str
            Device to run on (cpu, cuda, mps)
        """
        super(PoincareModel, self).__init__()
        self.device = device

    def matrixMultiplication(self, features, linear_layer, c):
        """
        Matrix multiplication in Poincare Model

        Parameters
        ----------
        features : torch.Tensor (n, d)
            Input tensor
        linear_layer : torch.nn.Linear
            Linear layer
        c : torch.Tensor (1, 1)
            Constant defining the curvature of the Poincare Model

        Returns
        -------
        z : torch.Tensor (n, d') n_samples x d_dimensions
            Output tensor
        """

        z = self.logarithmicMap(features, c)  # logarithmic map about origin
        z = linear_layer(z)  # linear layer
        z = self.exponentialMap(z, c)  # exponential map about origin

        return z

    def biasAddition(self, features, bias, c):
        """
        Bias addition in Poincare Model

        Parameters
        ----------
        features : torch.Tensor (n, d)
            Input tensor
        bias : torch.Tensor (d, 1)
            Input tensor
        c : torch.Tensor (1, 1)

        Returns
        -------
        z : torch.Tensor (n, d) n_samples x d_dimensions
            Output tensor
        """

        bias_ = bias.repeat(1, features.shape[0]).T
        z = self.parallelTransport(
            bias_, features, c
        )  # parallel transport about features
        z = self.exponentialMap(z, c, features)  # exponential map about features

        return z

    def sigmoidNonLinearity(self, features, c1, c2):
        """
        Sigmoid non-linearity in Poincare Model

        Parameters
        ----------
        features : torch.Tensor (n, d)
            Input tensor
        c1 : torch.Tensor (1, 1)
            Constant defining the curvature of the Poincare Model
        c2 : torch.Tensor (1, 1)
            Constant defining the curvature of the Poincare Model

        Returns
        -------
        z : torch.Tensor (n, d) n_samples x d_dimensions
            Output tensor
        """

        z = self.logarithmicMap(features, c1)
        z = torch.relu(z)
        return self.exponentialMap(z, c2)

    def exponentialMap(self, v, c, x=None):
        """
        Exponential map in Lorentz Model

        Parameters
        ----------
        v : torch.Tensor (n, d)
            Input tensor
        c : torch.Tensor (1, 1)
            Constant defining the curvature of the c Model
        x : torch.Tensor (n, d) (optional)
            Input tensor

        Returns
        -------
        z : torch.Tensor (n, d) n_samples x d_dimensions
            Output tensor as z = mobiusaddition(x , (alpha * v)/beta)
        """

        if x is None:
            v_norm = torch.norm(v, dim=1).view(-1, 1)
            alpha = torch.tanh(torch.sqrt(c) * v)
            beta = torch.sqrt(c) * v_norm

            z = alpha / (beta + 1e-8)
            return z * v

        x_norm_sq = torch.sum(x**2, dim=1).view(-1, 1)
        v_norm_sq = torch.sum(v**2, dim=1).view(-1, 1)

        lambda_x = 2 / (1 - c * x_norm_sq)

        alpha = torch.tanh(torch.sqrt(c) * lambda_x * torch.sqrt(v_norm_sq) / 2)
        beta = torch.sqrt(c) * torch.sqrt(v_norm_sq)

        z = (alpha * v) / (beta + 1e-8)

        return self.mobiusAddition(x, z, c)

    def logarithmicMap(self, v, c, x=None):
        """
        Logarithmic map in Lorentz Model

        Parameters
        ----------
        v : torch.Tensor (n, d)
            Input tensor
        c : torch.Tensor (1, 1)
            Constant defining the curvature of the Poincare Model
        x : torch.Tensor (n, d) (optional)
            Input tensor

        Returns
        -------
        z : torch.Tensor (n, d) n_samples x d_dimensions
            Output tensor as z = (alpha * beta * mobiusaddition(-x, v))/norm(mobiusaddition(-x, v))
        """

        if x is None:
            alpha = torch.arctanh(torch.sqrt(c) * torch.norm(v, dim=1).view(-1, 1))
            beta = torch.sqrt(c) * torch.norm(v, dim=1).view(-1, 1)

            z = alpha / (beta + 1e-8)
            return z * v

        x_norm_sq = torch.sum(x**2, dim=1).view(-1, 1)

        lambda_x = 2 / (1 - c * x_norm_sq)
        alpha = 2 / (torch.sqrt(c) * lambda_x)

        mobius_addition = self.mobiusAddition(-x, v, c)
        mobius_addition_norm = torch.sqrt(torch.sum(mobius_addition**2, dim=1))
        mobius_addition_norm = mobius_addition_norm.view(-1, 1)

        beta = torch.arctanh(torch.sqrt(c) * mobius_addition_norm)

        z = alpha * beta * mobius_addition / (mobius_addition_norm + 1e-8)
        return z

    def parallelTransport(self, v, x, c):
        """
        Parallel transport in Poincare Model

        Parameters
        ----------
        v : torch.Tensor (d, 1)
            Input tensor
        x : torch.Tensor (n, d)
            Input tensor
        c : torch.Tensor (1, 1)
            Constant defining the curvature of the Poincare Model

        Returns
        -------
        z : torch.Tensor (n, d) n_samples x d_dimensions
            Output tensor
        """

        z = self.exponentialMap(v, c)
        z = self.mobiusAddition(x, z, c)
        z = self.logarithmicMap(z, c, x)
        return z

    def mobiusAddition(self, x, y, c):
        """
        Mobius addition in Poincare Model

        Parameters
        ----------
        x : torch.Tensor (n, d) n_samples x d_dimensions
            Input tensor
        y : torch.Tensor (n, d) n_samples x d_dimensions
            Input tensor
        c : torch.Tensor (1, 1)
            Constant defining the curvature of the Poincare Model

        Returns
        -------
        z : torch.Tensor (n, d) n_samples x d_dimensions
            Output tensor as z = (alpha * x + beta * y)/gamma
        """

        x_norm_sq = torch.sum(x**2, dim=1).view(-1, 1)
        y_norm_sq = torch.sum(y**2, dim=1).view(-1, 1)
        xy_dot_product = torch.sum(x * y, dim=1).view(-1, 1)

        alpha = 1 + (2 * c * xy_dot_product) + (c * y_norm_sq)
        beta = 1 - (c * x_norm_sq)
        gamma = 1 + (2 * c * xy_dot_product) + (c**2 * x_norm_sq * y_norm_sq)

        z = (alpha * x + beta * y) / gamma

        return z

    def getZeroVector(self, n, d):
        """
        Get zero vector in Poincare Model of shape (n, d)

        Parameters
        ----------
        n : int
            Number of samples
        d : int
            Dimension of the vector

        Returns
        -------
        zero_vector : torch.Tensor (n, d)
            Zero vector
        """

        return torch.zeros(n, d).to(self.device)


class LorentzModel(nn.Module):
    """
    Implement functions for Lorentz Model

    by Divyanshu + Tejadhith + Hitesh
    """

    def __init__(self, device):
        """
        Initialize the module

        Parameters
        ----------
        device : str
            Device to run on (cpu, cuda, mps)
        """
        super(LorentzModel, self).__init__()
        self.device = device

    def matrixMultiplication(self, features, linear_layer, c):
        """
        Matrix multiplication in Lorentz Model

        Parameters
        ----------
        features : torch.Tensor (n, d)
            Input tensor
        linear_layer : torch.nn.Linear
            Linear layer
        c : torch.Tensor (1, 1)
            Constant defining the curvature of the Lorentz Model

        Returns
        -------
        z : torch.Tensor (n, d') n_samples x d_dimensions
            Output tensor
        """

        z = self.logarithmicMap(features, c)[:, 1:]  # logarithmic map about origin
        z = linear_layer(z)  # linear layer

        zeros = torch.zeros(z.shape[0], 1).to(self.device)
        z = torch.cat((zeros, z), dim=1)  # add 0 to first dimension

        return self.exponentialMap(z, c)  # exponential map about origin

    def biasAddition(self, features, bias, c):
        """
        Bias addition in Lorentz Model

        Parameters
        ----------
        features : torch.Tensor (n, d)
            Input tensor
        bias : torch.Tensor (n, d)
            Input tensor
        c : torch.Tensor (1, 1)

        Returns
        -------
        z : torch.Tensor (n, d) n_samples x d_dimensions
            Output tensor
        """

        z = self.parallelTransport(
            bias, features, c
        )  # parallel transport about features
        z = self.exponentialMap(z, c, features)  # exponential map about features

        return z

    def sigmoidNonLinearity(self, features, c1, c2):
        """
        Sigmoid non-linearity in Lorentz Model

        Parameters
        ----------
        features : torch.Tensor (n, d)
            Input tensor
        c1 : torch.Tensor (1, 1)
            Constant defining the curvature of the Lorentz Model
        c2 : torch.Tensor (1, 1)
            Constant defining the curvature of the Lorentz Model

        Returns
        -------
        z : torch.Tensor (n, d) n_samples x d_dimensions
            Output tensor
        """

        z = self.logarithmicMap(features, c1)[:, 1:]
        z = torch.nn.functional.relu(z)

        zeros = torch.zeros(z.shape[0], 1).to(self.device)
        z = torch.cat((zeros, z), dim=1)

        return self.exponentialMap(z, c2)

    def exponentialMap(self, v, c, x=None):
        """
        Exponential map in Lorentz Model

        Parameters
        ----------
        v : torch.Tensor (n, d)
            Input tensor
        c : torch.Tensor (1, 1)
            Constant defining the curvature of the c Lorentz Model
        x : torch.Tensor (n, d) (optional)
            Input tensor

        Returns
        -------
        z : torch.Tensor (n, d) n_samples x d_dimensions
            Output tensor z = alpha * x + beta * v
        """

        if x is None:
            x = self.getZeroVector(v.shape[0], v.shape[1])

        scaled_norm = torch.sqrt(c) * self.lorentzNorm(v)
        # breakpoint()
        alpha = torch.cosh(scaled_norm)
        beta = torch.sinh(scaled_norm) / (scaled_norm + 1e-8)

        z = alpha * x + beta * v

        return z

    def logarithmicMap(self, v, c, x=None):
        """
        Logarithmic map in Lorentz Model

        Parameters
        ----------
        v : torch.Tensor (n, d)
            Input tensor
        c : torch.Tensor (1, 1)
            Constant defining the curvature of the Lorentz Model
        x : torch.Tensor (n, d) (optional)
            Input tensor

        Returns
        -------
        z : torch.Tensor (n, d) n_samples x d_dimensions
            Output tensor z = alpha/beta * (v + gamma * x)
        """

        if x is None:
            x = self.getZeroVector(v.shape[0], v.shape[1])

        xy_dot_product = self.lorentzDotProduct(x, v)

        alpha = torch.arccosh(-c * xy_dot_product)
        beta = torch.sinh(alpha)
        gamma = c * xy_dot_product

        z = (alpha / (beta + 1e-8)) * (v + gamma * x)
        return z

    def parallelTransport(self, v, x, c):
        """
        Parallel transport in Lorentz Model

        Parameters
        ----------
        v : torch.Tensor (d, 1)
            Input tensor
        x : torch.Tensor (n, d)
            Input tensor
        c : torch.Tensor (1, 1)
            Constant defining the curvature of the Lorentz Model

        Returns
        -------
        z : torch.Tensor (n, d) n_samples x d_dimensions
            Output tensor
        """

        zero_vector = self.getZeroVector(x.shape[0], x.shape[1])

        alpha = c * self.lorentzDotProduct(x, v)
        beta = 1 - c * self.lorentzDotProduct(zero_vector, x)
        z = (alpha / beta) * (x + zero_vector) + v

        return z

    def lorentzDotProduct(self, x, y):
        """
        Lorentz dot product in Lorentz Model

        Parameters
        ----------
        x : torch.Tensor (n, d)
            Input tensor
        y : torch.Tensor (n, d)
            Input tensor

        Returns
        -------
        xy_dot_product : torch.Tensor (n, 1)
            Output tensor
        """

        xy_dot_product = x * y
        xy_dot_product[:, 0] *= -1
        xy_dot_product = torch.sum(xy_dot_product, dim=1, keepdim=True)

        return xy_dot_product

    def lorentzNorm(self, x):
        """
        Lorentz norm in Lorentz Model

        Parameters
        ----------
        x : torch.Tensor (n, d)
            Input tensor

        Returns
        -------
        norm : torch.Tensor (n, 1)
            Output tensor
        """

        return torch.sqrt(self.lorentzDotProduct(x, x))

    def getZeroVector(self, n, d):
        """
        Get zero vector in Poincare Model of shape (n, d)

        Parameters
        ----------
        n : int
            Number of samples
        d : int
            Dimension of the vector

        Returns
        -------
        zero_vector : torch.Tensor (n, d)
            Zero vector
        """

        zero_vector = torch.zeros(n, d).to(self.device)
        zero_vector[:, 0] = torch.ones(n)

        return zero_vector
