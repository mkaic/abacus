import torch
import torch.nn as nn
from typing import Tuple


class FuzzyNAND(nn.Module):
    def __init__(self, input_shape: Tuple[int] = None, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Assuming fuzzy AND is multiplication and fuzzy NOT is (1 - x), fuzzy NAND is therefore defined as (NOT a) AND (NOT b), or: (1 - a) * (1 - b).
        :param activations: Expected shape: (B x N x degree). Each value is assumed to be in [0, 1].
        :return: Aggregated inputs, of shape (B x N).
        """
        #
        assert activations.shape[-1] == 2, "FuzzyNAND is only defined for degree 2."

        activations = activations.clamp(0, 1)
        activations = 1 - activations
        activations = torch.prod(activations, dim=self.dim)
        return activations

    def clamp_params(self):
        pass


class FuzzyNOR(nn.Module):
    def __init__(self, input_shape: Tuple[int] = None, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Assuming fuzzy OR is addition and fuzzy NOT is (1 - x), fuzzy NOR is therefore defined as (NOT a) OR (NOT b), or: 1 - (a + b).
        :param activations: Expected shape: (B x N x degree). Each value is assumed to be in [0, 1].
        :return: Aggregated inputs, of shape (B x N).
        """
        #
        assert activations.shape[-1] == 2, "FuzzyNOR is only defined for degree 2."

        activations = activations.clamp(0, 1)
        activations = 1 - torch.sum(activations, dim=self.dim)
        activations = activations.clamp(0, 1)
        return activations

    def clamp_params(self):
        pass


class FuzzyNANDNOR(nn.Module):
    """
    A weighted linear combination of fuzzy NAND and fuzzy NOR.
    """

    def __init__(self, input_shape: Tuple[int], dim=-1):
        super().__init__()

        self.nand = FuzzyNAND(dim=dim)
        self.nor = FuzzyNOR(dim=dim)

        self.weights = torch.rand(*input_shape[:-1])
        self.weights = nn.Parameter(self.weights)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        nand = self.nand(activations)
        nor = self.nor(activations)

        activations = ((1 - self.weights) * nand) + (self.weights * nor)

    def clamp_params(self):
        self.weights.data.clamp_(0, 1)


class LinearCombination(nn.Module):
    def __init__(self, input_shape: Tuple[int], dim: int = -1):
        super().__init__()
        self.dim = dim

        self.weights = torch.full(input_shape, 1 / input_shape[dim])
        self.weights = nn.Parameter(self.weights)

        biases_shape = list(input_shape)
        biases_shape.pop(dim)

        self.biases = torch.zeros(biases_shape)
        self.biases = nn.Parameter(self.biases)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        :param activations: Some tensor with shape (B, ...), with dimension self.dim being the index of the dimension to reduce along.
        :return: A tensor with shape (B, ...), with the dimension at index self.dim being collapsed.
        """

        activations = activations * self.weights
        activations = torch.sum(activations, dim=self.dim)

        activations = activations + self.biases

        activations = torch.relu(activations)

        return activations

    def clamp_params(self):
        pass
