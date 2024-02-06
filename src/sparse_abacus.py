#!/usr/bin/env python3
# encoding: utf-8
import torch
import torch.nn as nn
from typing import Callable
from collections.abc import Iterable
from torch.nn.functional import grid_sample
from functools import partial


EPSILON = 1e-8

# +
# Many thanks to user enrico-stauss on the PyTorch forums for this implementation,
# which I have butchered to fit my specific needs.
# https://discuss.pytorch.org/t/linear-interpolation-in-pytorch/66861/10

# Additional thanks to GitHub user aliutkus, whose implementation I used as a reference
# https://github.com/aliutkus/torchinterp1d/blob/master/torchinterp1d/interp1d.py


def interp1d(x: torch.Tensor, y: torch.Tensor, xnew: torch.Tensor) -> torch.Tensor:
    """
    Given input points x, values y, and desired sampling points xnew, return the linearly interpolated values ynew at xnew.
    :param x: The original coordinates.
    :param y: The original values.
    :param xnew: The xnew points to which y shall be interpolated.
    """

    assert not (
        torch.any(xnew < 0)
        or torch.any(xnew > 1)
        or torch.any(x < 0)
        or torch.any(x > 1)
    ), "All x and xnew values must be in [0,1]"

    assert not (
        torch.any(xnew < x.min(dim=-1).values.unsqueeze(-1))
        or torch.any(xnew > x.max(dim=-1).values.unsqueeze(-1))
    )

    assert x.shape == y.shape, "x and y must have the same shape"
    assert (
        x.dim() == 2 and y.dim() == 2 and xnew.dim() == 2
    ), "x, y, and xnew must be 2D"
    assert x.shape[-1] > 1 and y.shape[-1] > 1, "x and y must have at least 2 points"

    # Evaluate the forward difference
    slope = (y[:, 1:] - y[:, :-1]) / (x[:, 1:] - x[:, :-1] + EPSILON)  # B, N-1

    # Get the first indices of x where each xnew value could be inserted into x without disrupting the sort.
    # That is, for each value v in xnew, an index i is returned which satisfies:
    # x[i-1] < v <= x[i]
    # i-1 therefore gives the index of highest value in x which is still less than v.
    xnew_searchsort_left_indices = torch.searchsorted(x, xnew) - 1

    # Number of intervals is x.shape-1

    xnew_searchsort_left_indices = torch.clamp(
        xnew_searchsort_left_indices,
        0,
        (x.shape[-1] - 1),
    )

    # Get the offset from the point to the left to the xnew point
    xnew_offset = xnew - x.gather(dim=-1, index=xnew_searchsort_left_indices)

    # Calculate the value for the nonzero xnew: value of the point to the left plus slope times offset
    ynew = y.gather(dim=-1, index=xnew_searchsort_left_indices)
    ynew_offset = slope.gather(dim=-1, index=xnew_searchsort_left_indices) * xnew_offset
    ynew = ynew + ynew_offset

    return ynew


def fuzzy_nand(activations: torch.Tensor) -> torch.Tensor:
    """
    Assuming fuzzy AND is multiplication and fuzzy NOT is (1 - x), fuzzy NAND is therefore defined as (NOT a) AND (NOT b), or: (1 - a) * (1 - b).
    :param activations: Expected shape: (B x N x degree). Each value is assumed to be in [0, 1].
    :return: Aggregated inputs, of shape (B x N).
    """
    #
    activations = 1 - activations
    activations = torch.prod(activations, dim=-1)
    return activations


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
        input_shape: int,
        output_shape: int,
        aggregator: Callable[[torch.Tensor], torch.Tensor] = fuzzy_nand,
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
        self.ndims_out = len(self.output_shape)

        self.aggregator = aggregator
        self.degree = degree
        self.sample_points_predictor = sample_points_predictor
        self.lookbehind = lookbehind

        if self.sample_points_predictor is None:
            self.sample_points = nn.Parameter(
                torch.rand(*output_shape, degree, self.ndims_in)
            )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        :param activations: Expected shape: (B x N_in). Will be clamped to [0, 1].
        :return: Expected shape: (B x N_out).
        """
        batch_size = activations.shape[0]

        if self.sample_points_predictor is None:
            sample_points = self.sample_points
            sample_points = sample_points.expand(
                batch_size, [-1] * self.ndims_out, -1
            )  # B x *self.output_shape x degree
        else:
            sample_points = self.sample_points_predictor(activations)

        sample_points = torch.clamp(sample_points, 0, 1)

        activations = torch.clamp(activations, 0, 1)

        # Make activations continuous and sample from them at variable points

        if self.ndims == 1:  # 1D vector input case
            x = torch.linspace(0, 1, self.n_in, device=activations.device).repeat(
                batch_size, 1
            )  # B x N_in

            activations = interp1d(
                x,  # B x N_in
                activations,  # B x N_in
                sample_points,  # B x (N_out * degree)
            )  # B x (N_out * degree)

        elif (
            self.ndims > 1 and self.ndims < 3
        ):  # 2D (HxW) or 3D (CxHxW) image input cases
            # imagining we have a B x C x H x W input and a B x N_out x degree sample_points tensor
            # In order to allow for interpolation along the channel axis, we need to reshape the activations tensor to B x 1 x C x H x W so that grid_sample treats the channel dim as a a spatial dim
            # Sample points tensor needs to be of shape B x 1 x 1 x (N_out * degree) x ndims_in.
            # This will result in a sampled tensor of shape B x 1 x 1 x (N_out * degree)
            # which we want to then reshape into B x N_out x degree

            # First we collapse the degree dim into the last spatial dim
            sample_points = torch.flatten(
                sample_points, start_dim=-2, end_dim=-1
            )  # B x *output_shape[:-1] x (output_shape[-1] * degree)
            # In our example, this moves sample_points from B x N_out x degree to B x (N_out * degree)

            # Next we need to add empty dims to sample_points until it
            for _ in range(self.ndims_in - self.ndims_out):
                sample_points = sample_points.unsqueeze(1)

            activations = grid_sample(
                activations.unsqueeze(
                    1
                ),  # B x 1 x *input_shape (looks like either a 1-channel image or a 1-channel 3D volume, either way we'll get proper interpolation from grid_sample now)
                sample_points.unsqueeze(1),  # B x 1 x *output_shape x ndims
                align_corners=False,
                mode="bilinear",
                padding_mode="reflection",
            )

            for _ in range(self.ndims_in - self.ndims_out):
                activations = activations.squeeze(1)
            # B x *output_shape x degree
        else:
            raise ValueError(
                "Input shape must be 1D, 2D, or 2D plus a channel dimension."
            )

        activations = activations.view(
            batch_size, *self.output_shape, self.degree
        )  # B x N_out x degree

        activations = self.aggregator(activations)  # B x N_out

        return activations
