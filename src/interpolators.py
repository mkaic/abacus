import torch
import torch.nn as nn
from typing import Tuple
from scipy.interpolate import interpn
import numpy as np

EPSILON = 1e-8


def unrolling_cartesian_product(cube: torch.Tensor) -> torch.Tensor:
    return torch.cartesian_prod(*cube)


double_batched_unrolling_cartesian_product = torch.vmap(
    torch.vmap(unrolling_cartesian_product)
)


def n_linear_interp(
    original_values: torch.Tensor, sample_points: torch.Tensor
) -> torch.Tensor:
    """Assumes a regular grid of original x values. Assumes sample points has B x output_num_points x Ndims shape and are in the range [0,1]"""

    device = original_values.device
    assert (
        len(sample_points.shape) == 3
    ), "Sample points must be B x output_num_points x Ndims"

    # what the dimension of the data is, ignoring the batch dim
    batch_size = original_values.shape[0]
    batchless_input_shape = original_values.shape[1:]
    n_dims = len(batchless_input_shape)
    output_num_points = sample_points.shape[1]

    assert (
        sample_points.shape[-1] == n_dims
    ), "Sample point coordinates must have Ndims values"

    # Calculate the stepsize in each dimension, assuming the coordinates should
    # range from 0 to 1.
    stepsizes = torch.tensor(
        [1 / (dim - 1) for dim in original_values.shape[1:]], device=device
    )
    stepsizes = stepsizes.view(1, 1, -1)

    # First, we divide to get exactly how many steps you'd need to take in each dimension to reach each
    # sample point. Then, we floor to get the index of the closest grid point to the left of the sample point
    raw_indices = sample_points / stepsizes
    left_indices = raw_indices.floor()
    offsets = raw_indices - left_indices
    left_indices = left_indices.long()  # B x output_num_points x Ndims

    # Cap right-indices so that they don't try to index out of bounds
    # This means any attempts to interpolate out of bounds will just result
    # in repeating the last value in that dimension
    right_indices = left_indices + 1
    input_shape_tensor = torch.tensor(batchless_input_shape, device=device).unsqueeze(
        0
    )  # 1 x Ndims
    right_indices = torch.where(
        right_indices >= input_shape_tensor, left_indices, right_indices
    )  # B x output_num_points x Ndims

    discrete_indices = torch.stack(
        [left_indices, right_indices], dim=-1
    )  # B x output_num_points x Ndims x 2

    ncube_corner_coords = double_batched_unrolling_cartesian_product(
        discrete_indices
    )  # B x output_num_points x 2^Ndims x Ndims

    ncube_corner_coords: torch.Tensor = ncube_corner_coords.view(
        batch_size, output_num_points, 2**n_dims, n_dims
    )  # B x output_num_points x 2^Ndims x Ndims

    # Add the batch dimension to the corner coordinates
    batch_coordinates = (
        torch.arange(batch_size, device=device)
        .view(-1, 1, 1, 1)
        .expand(-1, output_num_points, 2**n_dims, -1)
    )  # B x output_num_points x 2^Ndims x 1

    ncube_corner_coords = torch.cat(
        [batch_coordinates, ncube_corner_coords], dim=-1
    )  # B x output_num_points x 2^Ndims x (Ndims+1)

    ncube_corner_coords = ncube_corner_coords.flatten(
        start_dim=0, end_dim=-2
    )  # (B*output_num_points*2^Ndims) x (Ndims+1)
    ncube_corner_coords = list(
        ncube_corner_coords.T
    )  # list with len=(B*output_num_points*2^Ndims) of tensors with shape (Ndims)

    corner_values = original_values[
        ncube_corner_coords
    ]  # (B*output_num_points*2^Ndims) x 1

    corner_values = corner_values.view(
        batch_size, output_num_points, 2**n_dims
    )  # B x output_num_points x 2^Ndims

    interpolated = corner_values
    for i in range(n_dims):

        # Split the points in half to get pairs of points to interpolate
        # between according to the offsets. The number of points halves
        # each iteration, and since there are 2^Ndims points to start with,
        # after Ndims iterations there will only be 1 value left, the final
        # interpolated value.
        length = interpolated.shape[-1]
        a = interpolated[..., : length // 2]
        b = interpolated[..., length // 2 :]

        slope = b - a
        # offsets is of shape B x output_num_points x Ndims, after indexing and unsqueezing is B x output_num_points x 1.
        interpolated = a + slope * offsets[..., i].unsqueeze(-1)

    return interpolated.squeeze(-1)


class LinearInterpolator(nn.Module):
    def __init__(self, input_shape: Tuple[int], output_shape: Tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_output_el = np.prod(output_shape)

    def forward(self, y: torch.Tensor, xnew: torch.Tensor) -> torch.Tensor:
        """
        :param y: The original values.
        :param xnew: The xnew points to which y shall be interpolated.
        :return: The interpolated values ynew at xnew.
        """
        batch_size = y.shape[0]

        xnew = xnew.reshape(batch_size, self.n_output_el, len(self.input_shape))
        ynew = n_linear_interp(y, xnew)
        ynew = ynew.view(batch_size, *self.output_shape)

        return ynew


class SciPyLinearInterpolator(LinearInterpolator):
    """Exclusively for testing purposes, to sanity check my batched tensor implementation"""

    def forward(self, y: torch.Tensor, xnew: torch.Tensor) -> torch.Tensor:
        """
        :param y: The original values.
        :param xnew: The xnew points to which y shall be interpolated.
        :return: The interpolated values ynew at xnew.
        """
        batch_size = y.shape[0]

        xnew = xnew.reshape(batch_size, self.n_output_el, len(self.input_shape))
        ynew = torch.stack(
            [
                torch.tensor(
                    interpn(
                        [
                            torch.linspace(0, 1, self.input_shape[i]).numpy()
                            for i in range(len(self.input_shape))
                        ],
                        y[i].numpy(),
                        xnew[i].flatten(end_dim=-2).numpy(),
                        method="linear",
                    )
                )
                for i in range(y.shape[0])
            ]
        )
        ynew = ynew.view(batch_size, *self.output_shape)
        ynew = ynew.float()

        return ynew


def n_fourier_interp(
    original_values: torch.Tensor, sample_points: torch.Tensor
) -> torch.Tensor:

    device = original_values.device
    ndims = len(original_values.shape[1:])
    batch_size = original_values.shape[0]
    
    shape_tensor = torch.tensor(original_values.shape[1:], device=device) # Ndims
    num_output_points = sample_points.shape[1]
    sample_points = sample_points * shape_tensor # B x output_num_points x Ndims

    m = torch.meshgrid(*[torch.arange(dim, device=device) for dim in shape_tensor])
    m = torch.stack(m, dim=-1)  # *original_values.shape[1:] x Ndims
    m = m.unsqueeze(0) # 1 x *original_values.shape[1:] x Ndims
    m = m.unsqueeze(-2) # 1 x *original_values.shape[1:] x 1 x Ndims

    # maps from integer coords to [0,1] coords to radians
    freqs = m/shape_tensor * 2 * torch.pi # 1 x *original_values.shape[1:] x 1 x Ndims

    # After broadcasting, there will be a copy of the sample points for every 
    # point in the fourier-transformed version of the original values
    sample_points = sample_points.view(batch_size, *[1 for _ in shape_tensor], num_output_points, ndims) # B x 1...1 x output_num_points x Ndims
    sinusoid_coords = freqs * sample_points # *original_values.shape x output_num_points x Ndims

    fourier_coeffs = torch.fft.fftn(
        original_values, dim=tuple(range(1, len(original_values.shape)))
    )
    fourier_magnitudes = torch.abs(fourier_coeffs) # *original_values.shape
    fourier_phases = torch.angle(fourier_coeffs) # *original_values.shape

    fourier_magnitudes = fourier_magnitudes.view(*original_values.shape, 1, 1) # *original_values.shape x 1 x 1
    fourier_phases = fourier_phases.view(*original_values.shape, 1, 1) # *original_values.shape x 1 x 1

    sinusoid_coords = sinusoid_coords + fourier_phases # *original_values.shape x output_num_points x Ndims
    sinusoids = torch.cos(sinusoid_coords) # *original_values.shape x output_num_points x Ndims
    sinusoids = torch.prod(sinusoids, dim=-1) # *original_values.shape x output_num_points
    sinusoids = sinusoids * fourier_magnitudes # *original_values.shape x output_num_points

    # Average over all sinusoids
    dims_to_collapse = tuple([i for i in range(1, len(original_values.shape) - 1)])
    interpolated = torch.mean(sinusoids, dim=dims_to_collapse) # B x output_num_points

    # Un-complexify them
    interpolated = interpolated.real

    return interpolated
    
    


    # implementation based off of https://brianmcfee.net/dstbook-site/content/ch07-inverse-dft/Synthesis.html#idft-as-synthesis

    for dim, N in enumerate(original_values.shape[1:]):

        

        # select only the coordinates for the dimension we're currently
        # doing IFFT for
        # map from [0,1] to the integer space that the FFT uses
        x = sample_points[..., dim : dim + 1] * N  # B x output_num_points x 1

        m = torch.arange(N, device=device)
        m = m.expand(*fourier_coeffs.shape, N)  # *original_values.shape x N

        freqs = 2 * torch.pi * (m.float() / N) * x  # B x output_num_points x N

        # sinusoids = amps * torch.cos(freqs + phases) # B x output_num_points x N


class FourierInterpolator(nn.Module):
    def __init__(self, input_shape: Tuple[int], output_shape: Tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_output_el = np.prod(output_shape)

    def forward(self, y: torch.Tensor, xnew: torch.Tensor) -> torch.Tensor:
        """
        :param y: The original values.
        :param xnew: The xnew points to which y shall be interpolated.
        :return: The interpolated values ynew at xnew.
        """
        batch_size = y.shape[0]

        xnew = xnew.reshape(batch_size, self.n_output_el, len(self.input_shape))
        ynew = n_fourier_interp(y, xnew)
        ynew = ynew.view(batch_size, *self.output_shape)

        return ynew
