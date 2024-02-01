#!/usr/bin/env python3
# encoding: utf-8
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

# +
# Many thanks to user enrico-stauss on the PyTorch forums for this implementation,
# which I have butchered to fit my specific needs.
# https://discuss.pytorch.org/t/linear-interpolation-in-pytorch/66861/10


# # Generate random sorted and unique x values in the range from -21 to 19 and corresponding y values
# x = torch.linspace(0, 1, 5)
# y = torch.rand_like(x)

# # Set the new sample points to the range [-25, 25]
# x_new = torch.linspace(0, 1, 24)


def interp1d(x: torch.Tensor, y: torch.Tensor, xnew: torch.Tensor) -> torch.Tensor:
    """
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

    # Evaluate the forward difference
    slope = (y[1:] - y[:-1]) / (x[1:] - x[:-1])

    # Get the indices of the closest point to the left for each xnew point
    xnew_closest_left_indices = torch.searchsorted(x, xnew)

    print(xnew)
    print(xnew_closest_left_indices)

    # Get the offset from the point to the left to the xnew point
    xnew_offset = xnew - x[xnew_closest_left_indices]

    # Calculate the value for the nonzero xnew: value of the point to the left plus slope times offset
    ynew = (
        y[xnew_closest_left_indices]
        + slope[xnew_closest_left_indices - 1] * xnew_offset
    )

    return ynew


# plt.plot(x_new, interp1d(x, y, x_new) - 0.02, "go", label="Custom interpolation")
# plt.plot(x_new, np.interp(x_new, x, y, left=0, right=0) + 0.02, "ro", label="np.interp")
# plt.plot(x, y, "b--", label="original values")
# plt.legend()
# plt.show()

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

    :param n_in: The number of input neurons.
    :param n_out: The number of output neurons.
    :param sample_points_predictor: A predictor that takes the input tensor and returns the sampling points for the output neurons.
    """

    def __init__(
        self, n_in: int, n_out: int, sample_points_predictor: nn.Module = None
    ) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.register_buffer("pos", torch.linspace(0, 1, n_in))

        if sample_points_predictor is not None:
            self.sample_points_predictor = sample_points_predictor
        else:
            self.sample_points = nn.Parameter(torch.rand(n_out, 2))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        :param activations: Expected shape: (B x N_in). Will be clamped to [0, 1].
        :return: Expected shape: (B x N_out).
        """
        if self.sample_points_predictor is not None:
            sample_points = self.sample_points_predictor(activations)
        else:
            sample_points = self.sample_points

        sample_points = torch.clamp(sample_points, 0, 1)

        # Make activations continuous and sample from them at variable points
        activations = torch.clamp(activations, 0, 1)
        activations = interp1d(
            self.pos, activations, sample_points.view(-1)
        )  # B x 2N_out
        activations = activations.view(*self.sample_points.shape)  # B x N_out x 2

        activations = self.aggregator(activations)

        return activations
