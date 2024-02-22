#!/usr/bin/env python3
# encoding: utf-8
import torch
import torch.nn as nn
from collections.abc import Iterable
from typing import Tuple

from .aggregators import FuzzyNAND, LinearCombination
from .interpolators import LinearInterpolator


EPSILON = 1e-8

# +
# Many thanks to user enrico-stauss on the PyTorch forums for this implementation,
# which I have butchered to fit my specific needs.
# https://discuss.pytorch.org/t/linear-interpolation-in-pytorch/66861/10

# Additional thanks to GitHub user aliutkus, whose implementation I used as a reference
# https://github.com/aliutkus/torchinterp1d/blob/master/torchinterp1d/interp1d.py


class SparseAbacusLayer(nn.Module):
    """
    Each neuron in the layer can sample from the input tensor at different points, and then aggregates those samples in some manner. The sampling points can be simple learnable parameters, or they can be made data-dependent (a la attention) by being predicted on the fly from the input using a provided predictor.

    :param input_shape: The shape of the input tensor, ignoring the batch dimension.
    :param output_shape: The shape of the output tensor, ignoring the batch dimension.
    :param aggregator: A function that takes the `degree` samples for each neuron and aggregates them into a single value. By default, this is the fuzzy NAND function because `degree` defaults to 2.
    :param degree: The number of samples to take for each neuron. Defaults to 2 so that each neuron can act like a fuzzy binary logic gate.
    :param sample_points_predictor: A predictor that takes the input tensor and returns the sampling points for the output neurons.
    :param lookbehind: The number of previous layers of activations the current layer can sample from. Defaults to 1. If set > 1, the network can theoretically learn skip-connections.
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        interpolator: nn.Module = LinearInterpolator,
        aggregator: nn.Module = LinearCombination,
        degree: int = 2,
        sample_points_predictor: nn.Module = None,
        lookbehind: int = 1,
    ) -> None:
        super().__init__()
        self.input_shape = (
            input_shape if isinstance(input_shape, Iterable) else (input_shape,)
        )
        self.ndims_in = len(self.input_shape)

        self.output_shape = (
            output_shape if isinstance(output_shape, Iterable) else (output_shape,)
        )

        self.degree = degree

        self.activations_shape = (*self.output_shape, self.degree)
        self.interpolator = interpolator(
            input_shape=self.input_shape, output_shape=self.activations_shape
        )
        self.aggregator = aggregator(input_shape=self.activations_shape, dim=-1)

        self.sample_points_predictor = sample_points_predictor
        self.lookbehind = lookbehind

        if self.sample_points_predictor is None:
            # linspaces = [torch.linspace(0, 1, n) for n in self.output_shape]
            # sample_points = torch.cartesian_prod(*linspaces)

            # sample_points = sample_points.reshape(*self.output_shape, 1, self.ndims_out)

            # sample_points = sample_points.expand(
            #     *self.output_shape, self.degree, self.ndims_out
            # )
            # sample_points = sample_points + torch.rand_like(sample_points) * 0.01
            # sample_points = torch.clamp(sample_points, 0, 1)

            sample_points = torch.rand(*self.output_shape, self.degree, self.ndims_in)

            self.sample_points = nn.Parameter(sample_points)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        :param activations: Expected shape: (B x N_in). Will be clamped to [0, 1].
        :return: Expected shape: (B x N_out).
        """
        batch_size = activations.shape[0]

        if self.sample_points_predictor is None:
            sample_points = self.sample_points
            sample_points = sample_points.expand(
                batch_size, *self.output_shape, self.degree, self.ndims_in
            )  # B x *self.output_shape x degree x ndims_in
        else:
            sample_points = self.sample_points_predictor(activations)

        sample_points = torch.clamp(sample_points, 0, 1)

        # Make activations continuous and sample from them at variable points
        activations = self.interpolator(activations, sample_points)

        activations = activations.view(
            batch_size, *self.output_shape, self.degree
        )  # B x N_out x degree

        activations = self.aggregator(activations)  # B x N_out

        return activations

    def clamp_params(self):
        if self.sample_points_predictor is None:
            self.sample_points.data.clamp_(0, 1)
        else:
            self.sample_points_predictor.clamp_params()

        self.aggregator.clamp_params()
