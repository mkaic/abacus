#!/usr/bin/env python3
# encoding: utf-8
import torch
import torch.nn as nn
from collections.abc import Iterable
from typing import Tuple
from .samplers import LinearInterpolator, BinaryTreeLinearInterpolator, AnisotropicGaussianSampler
from .aggregators import LinearCombination, LinearFuzzyNAND


EPSILON = 1e-8


class SamplerLayer(nn.Module):
    """
    Each neuron in the layer can sample from the input tensor at different points, and then aggregates those samples in some manner. The sampling points can be simple learnable parameters, or they can be made data-dependent (a la attention) by being predicted on the fly from the input using a provided predictor.

    :param input_shape: The shape of the input tensor, ignoring the batch dimension.
    :param output_shape: The shape of the output tensor, ignoring the batch dimension.
    :param aggregator: A function that takes the `degree` samples for each neuron and aggregates them into a single value. By default, this is the fuzzy NAND function because `degree` defaults to 2.
    :param degree: The number of samples to take for each neuron. Defaults to 2 so that each neuron can act like a fuzzy binary logic gate.
    :param sample_parameters_predictor: A predictor that takes the input tensor and returns the sampling points for the output neurons.
    :param lookbehind: The number of previous layers of activations the current layer can sample from. Defaults to 1. If set > 1, the network can theoretically learn skip-connections.
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        sampler: nn.Module,
        aggregator: nn.Module,
        degree: int = 2,
        sample_parameters_predictor: nn.Module = None,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.input_shape = (
            input_shape if isinstance(input_shape, Iterable) else (input_shape,)
        )
        self.input_shape = torch.tensor(self.input_shape, dtype=torch.float32)
        self.register_buffer("input_shape_tensor", self.input_shape)

        self.ndims_in = len(self.input_shape)

        self.output_shape = (
            output_shape if isinstance(output_shape, Iterable) else (output_shape,)
        )

        self.degree = degree
        self.residual = residual

        self.activations_shape = (*self.output_shape, self.degree)
        self.sampler = sampler(
            input_shape=self.input_shape, output_shape=self.activations_shape
        )
        self.aggregator = aggregator(input_shape=self.activations_shape, dim=-1)

        self.sample_parameters_predictor = sample_parameters_predictor

        self.sample_parameters = self.init_sampling_parameters()

    def init_sampling_parameters(self) -> Tuple[torch.Tensor]:
        raise NotImplementedError

    def forward(self, activations: torch.Tensor) -> torch.Tensor:

        if self.residual:
            og_activations = activations.clone()

        if self.sample_parameters_predictor is not None:
            sample_parameters = self.sample_parameters_predictor(activations)
        else:
            sample_parameters = self.sample_parameters

        # Make activations continuous and sample according to trainable parameters
        activations = self.sampler(activations, sample_parameters)

        # B x N_out
        activations = self.aggregator(activations)

        if self.residual:
            activations = activations + og_activations

        return activations


class BinaryTreeSparseAbacusLayer(SamplerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, sampler=BinaryTreeLinearInterpolator, aggregator=LinearFuzzyNAND
        )

    def init_sampling_parameters(self):
        if self.sample_parameters_predictor is None:
            sample_parameters = torch.rand(
                *self.output_shape, self.degree, self.ndims_in
            )
            return nn.ParameterList([nn.Parameter(sample_parameters)])

    def clamp_params(self):
        if self.sample_parameters_predictor is None:
            self.sample_parameters[0].data.clamp_(0, 1)
        else:
            self.sample_parameters_predictor.clamp_params()

class GaussianLayer(SamplerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, sampler=AnisotropicGaussianSampler, aggregator=LinearFuzzyNAND
        )

    def init_sampling_parameters(self):
        if self.sample_parameters_predictor is None:
            self.mu = nn.Parameter(
                torch.rand(*self.output_shape, self.degree, self.ndims_in)
            )
            self.sigma = nn.Parameter(
                torch.randn(*self.output_shape, self.degree, self.ndims_in) + 0.5
            )
            return nn.ParameterList([self.mu, self.sigma])

    def clamp_params(self):
        if self.sample_parameters_predictor is None:
            self.sample_parameters[0].data.clamp_(0, 1)
            self.sample_parameters[1].data.clamp_(EPSILON, 1)
        else:
            self.sample_parameters_predictor.clamp_params()
