import torch
import torch.nn as nn
from typing import Tuple
import numpy as np


class LinearFuzzyNAND(nn.Module):
    def __init__(self, input_shape: Tuple[int]):
        super().__init__()

        self.weights = torch.full(input_shape, 1 / input_shape[-1])
        self.weights = nn.Parameter(self.weights)

        biases_shape = list(input_shape)
        biases_shape.pop(-1)

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


class RecursiveLinearFuzzyNAND(nn.Module):
    def __init__(self, input_shape: Tuple[int]):
        super().__init__()

        self.degree = input_shape[-1]
        self.depth = np.log2(self.degree)

        self.binarized_input_shape = [*input_shape[:-1], +[[2] * self.depth]]

        aggregators = []
        for i in range(self.depth):
            aggregators.append(
                LinearFuzzyNAND(input_shape=self.binarized_input_shape[:i])
            )

        self.aggregators = nn.Sequential(*aggregators)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:

        activations = activations.reshape(*self.binarized_input_shape)

        activations = self.aggregators(activations)

        return activations
