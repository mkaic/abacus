import torch
import torch.nn as nn
from typing import Tuple


class LinearFuzzyNAND(nn.Module):
    def __init__(self, input_shape: Tuple[int], dim: int = -1):
        super().__init__()
        self.dim = dim

        self.weights = torch.full(input_shape, 1 / input_shape[dim])
        self.weights = nn.Parameter(self.weights)

        biases_shape = list(input_shape)
        biases_shape.pop(dim)

        self.biases = torch.zeros(biases_shape)
        self.biases = nn.Parameter(self.biases)

        # self.activation_func = nn.LeakyReLU(0.5)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        :param activations: Some tensor with shape (B, ...), with dimension self.dim being the index of the dimension to reduce along.
        :return: A tensor with shape (B, ...), with the dimension at index self.dim being collapsed.
        """

        activations = activations * self.weights

        # FuzzyNAND
        activations = 1 - activations
        activations = torch.prod(activations, dim=self.dim)

        activations = activations + self.biases

        # activations = self.activation_func(activations)

        return activations
