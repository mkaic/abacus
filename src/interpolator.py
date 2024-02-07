import torch
import torch.nn as nn
from typing import Tuple

EPSILON = 1e-8


def interp_1d(x: torch.Tensor, y: torch.Tensor, xnew: torch.Tensor) -> torch.Tensor:
    """
    Given input points x, values y, and desired sampling points xnew, return the linearly interpolated values ynew at xnew.
    :param x: The original coordinates.
    :param y: The original values.
    :param xnew: The xnew points to which y shall be interpolated.
    """

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


def interp_nd(values: torch.Tensor, sample_points: torch.Tensor) -> torch.Tensor:
    """Assumes a regular grid of original x values. Assumes sample points has B x -1 x Ndims shape"""

    assert len(sample_points.shape) == 3, "Sample points must be B x -1 x Ndims"

    batchless_input_shape = values.shape[1:]
    n_dims = len(batchless_input_shape)

    assert sample_points.shape[-1] == n_dims, "Sample point coordinates must have Ndims values"

    stepsizes = torch.tensor(
        [1 / (dim - 1) for dim in values.shape[1:]], device=values.device
    )
    stepsizes = stepsizes.view(1, 1, -1)

    print(f"{list(sample_points.shape)=}", f"{list(stepsizes.shape)=}")

    raw_indices = sample_points / stepsizes
    left_indices = raw_indices.floor()

    print(f"{list(left_indices.shape)=}")

    offsets = raw_indices - left_indices
    left_indices = left_indices.long()
    right_indices = left_indices + 1

    input_shape_tensor = torch.tensor(batchless_input_shape, device=values.device).unsqueeze(0)
    right_indices = torch.where(
        right_indices >= input_shape_tensor, left_indices, right_indices
    )



    discrete_points_to_sample = []
    
    for dim in batchless_input_shape:
    torch.stack(
        [torch.cartesian_prod(l, r) for l, r in zip(left_indices, right_indices)],
        dim=-1,
    )


class LinearInterpolator(nn.Module):
    def __init__(self, input_shape: Tuple[int], output_shape: Tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, xnew: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: The original coordinates.
        :param y: The original values.
        :param xnew: The xnew points to which y shall be interpolated.
        :return: The interpolated values ynew at xnew.
        """
