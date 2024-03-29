import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

EPSILON = 1e-8

# Many thanks to user enrico-stauss on the PyTorch forums for this implementation,
# which I have butchered to fit my specific needs.
# https://discuss.pytorch.org/t/linear-interpolation-in-pytorch/66861/10

# Additional thanks to GitHub user aliutkus, whose implementation I used as a reference
# https://github.com/aliutkus/torchinterp1d/blob/master/torchinterp1d/interp1d.py


def unrolling_cartesian_product(cube: torch.Tensor) -> torch.Tensor:
    return torch.cartesian_prod(*cube)


double_batched_unrolling_cartesian_product = torch.vmap(
    torch.vmap(unrolling_cartesian_product)
)


def n_linear_interp(
    original_values: torch.Tensor, sample_points: torch.Tensor
) -> torch.Tensor:
    """Assumes a regular grid of original x values. Assumes sample points has B x n_sample_points x Ndims shape and are in the range [0,1]"""

    device = original_values.device
    assert (
        len(sample_points.shape) == 3
    ), "Sample points must be B x n_sample_points x Ndims"

    # what the dimension of the data is, ignoring the batch dim
    batch_size = original_values.shape[0]
    batchless_input_shape = original_values.shape[1:]
    n_dims = len(batchless_input_shape)
    n_sample_points = sample_points.shape[1]

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
    left_indices = left_indices.long()  # B x n_sample_points x Ndims

    # Cap right-indices so that they don't try to index out of bounds
    # This means any attempts to interpolate out of bounds will just result
    # in repeating the last value in that dimension
    right_indices = left_indices + 1
    input_shape_tensor = torch.tensor(batchless_input_shape, device=device).unsqueeze(
        0
    )  # 1 x Ndims
    right_indices = torch.where(
        right_indices >= input_shape_tensor, left_indices, right_indices
    )  # B x n_sample_points x Ndims

    discrete_indices = torch.stack(
        [left_indices, right_indices], dim=-1
    )  # B x n_sample_points x Ndims x 2

    ncube_corner_coords = double_batched_unrolling_cartesian_product(
        discrete_indices
    )  # B x n_sample_points x 2^Ndims x Ndims

    ncube_corner_coords: torch.Tensor = ncube_corner_coords.view(
        batch_size, n_sample_points, 2**n_dims, n_dims
    )  # B x n_sample_points x 2^Ndims x Ndims

    # Add the batch dimension to the corner coordinates
    batch_coordinates = (
        torch.arange(batch_size, device=device)
        .view(-1, 1, 1, 1)
        .expand(-1, n_sample_points, 2**n_dims, -1)
    )  # B x n_sample_points x 2^Ndims x 1

    ncube_corner_coords = torch.cat(
        [batch_coordinates, ncube_corner_coords], dim=-1
    )  # B x n_sample_points x 2^Ndims x (Ndims+1)

    ncube_corner_coords = ncube_corner_coords.flatten(
        start_dim=0, end_dim=-2
    )  # (B*n_sample_points*2^Ndims) x (Ndims+1)
    ncube_corner_coords = list(
        ncube_corner_coords.T
    )  # list with len=(B*n_sample_points*2^Ndims) of tensors with shape (Ndims)

    corner_values = original_values[
        ncube_corner_coords
    ]  # (B*n_sample_points*2^Ndims) x 1

    corner_values = corner_values.view(
        batch_size, n_sample_points, 2**n_dims
    )  # B x n_sample_points x 2^Ndims

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
        # offsets is of shape B x n_sample_points x Ndims, after indexing and unsqueezing is B x n_sample_points x 1.
        interpolated = a + slope * offsets[..., i].unsqueeze(-1)

    return interpolated.squeeze(-1)


class LinearInterpolator(nn.Module):
    def __init__(self, input_shape: Tuple[int], output_shape: Tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_output_el = np.prod(output_shape)

    def forward(
        self, y: torch.Tensor, sample_parameters: Tuple[torch.Tensor]
    ) -> torch.Tensor:
        """
        :param y: The original values.
        :param xnew: The xnew points to which y shall be interpolated.
        :return: The interpolated values ynew at xnew.
        """
        xnew = sample_parameters[0]
        batch_size = y.shape[0]

        xnew = xnew.expand(batch_size, *self.output_shape, len(self.input_shape))
        xnew = xnew.reshape(batch_size, self.n_output_el, len(self.input_shape))
        ynew = n_linear_interp(y, xnew)
        ynew = ynew.view(batch_size, *self.output_shape)

        return ynew


def n_binary_tree_linear_interp(
    values: torch.Tensor, sample_points: torch.Tensor
) -> torch.Tensor:
    """Assumes a regular grid of original x values. Assumes sample points has B x n_sample_points x Ndims shape and are in the range [0,1]"""

    # what the dimension of the data is, ignoring the batch dim
    batchless_input_shape = values.shape[1:]
    n_dims = len(batchless_input_shape)
    n_sample_points = sample_points.shape[0]

    assert (
        sample_points.shape[-1] == n_dims
    ), "Sample point coordinates must have Ndims values"

    assert all(
        [dim == 2 for dim in batchless_input_shape]
    ), "Binary tree interp only works for inputs where all dimensions are 2."

    values = values.unsqueeze(-1)
    values = values.expand(-1, *batchless_input_shape, n_sample_points)

    for i in reversed(range(n_dims)):

        # n_sample_points
        weights = sample_points[..., i]
        # a set of n_sample_points weights for every point in the values tensor
        # *[2 for _ in range(i)] x n_sample_points
        weights = weights.expand(*[2 for _ in range(i)], n_sample_points)

        # a, b, slope have shape B x *[2 for _ in range(i)] x n_sample_points
        a = values[..., 0, :]
        b = values[..., 1, :]
        slope = b - a

        values = a + (slope * weights)

    return values


class BinaryTreeLinearInterpolator(nn.Module):
    def __init__(self, input_shape: Tuple[int], output_shape: Tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_output_el = np.prod(output_shape)

    def forward(
        self, y: torch.Tensor, sample_parameters: Tuple[torch.Tensor]
    ) -> torch.Tensor:
        """
        :param y: The original values.
        :param xnew: The xnew points to which y shall be interpolated.
        :return: The interpolated values ynew at xnew.
        """
        xnew = sample_parameters[0]
        batch_size = y.shape[0]

        xnew = xnew.reshape(self.n_output_el, len(self.input_shape))
        ynew = n_binary_tree_linear_interp(y, xnew)
        ynew = ynew.view(batch_size, *self.output_shape)

        return ynew


def n_fourier_interp(
    original_values: torch.Tensor, sample_points: torch.Tensor
) -> torch.Tensor:
    """
    original_values has some arbitrary shape (B x ...)
    sample_points has shape (B x n_sample_points x Ndims) and all values are in the range [0,1]

    Below is a list of very useful resources and explainers that helped me unsmooth my brain and finally understand the math behind this:
    1. https://brianmcfee.net/dstbook-site/content/ch07-inverse-dft/Synthesis.html#idft-as-synthesis
    2. https://see.stanford.edu/materials/lsoftaee261/chap8.pdf
    3. https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
    4. https://gatiaher.github.io/projects/1d-and-2d-fourier-transforms/

    """

    device = original_values.device
    ndims = len(original_values.shape[1:])
    batch_size = original_values.shape[0]

    fft_shape = torch.tensor(original_values.shape[1:], device=device)  # Ndims
    n_sample_points = sample_points.shape[1]

    # AHAHAHAAHAHAHAHAAHAHAHAAA
    # I SPENT SO MANY HOURS TRYING TO FIGURE OUT WHY MY CODE WASN'T WORKING
    # AND IT WAS BECAUSE I WAS MAPPING THE SAMPLE POINTS FROM [0,1] TO [0, N]
    # INSTEAD OF [0,1] TO [0, N-1].
    sample_points = sample_points * (fft_shape - 1).clamp(min=0)

    # list of (*fft_shape) with length Ndims
    m = torch.meshgrid(
        *[torch.arange(dim, device=device, dtype=torch.float) for dim in fft_shape],
        indexing="ij"
    )
    # m is in the range [0,1]
    # *fft_shape x Ndims
    m = torch.stack(m, dim=-1) / fft_shape
    # 1 x *fft_shape x 1 x Ndims
    m = m.view(1, *fft_shape, 1, ndims)

    # After broadcasting, there will be a copy of the sample points for every
    # point in the fourier-transformed version of the original values:
    # B x *fft_shape x n_sample_points x Ndims
    sample_points = sample_points.view(
        batch_size, *[1 for _ in fft_shape], n_sample_points, ndims
    )

    # B x *fft_shape x n_sample_points x Ndims
    sinusoid_coords = m * sample_points

    # B x *fft_shape x n_sample_points
    sinusoid_coords = sinusoid_coords.sum(dim=-1)

    # [1]
    complex_j = torch.complex(
        torch.tensor(0, device=device, dtype=torch.float),
        torch.tensor(1, device=device, dtype=torch.float),
    )

    sinusoid_coords = 2 * torch.pi * sinusoid_coords

    # B x *fft_shape
    dims_to_fourier = tuple(range(1, ndims + 1))
    fourier_coeffs: torch.Tensor = torch.fft.fftn(original_values, dim=dims_to_fourier)

    sinusoids = torch.cos(sinusoid_coords) + complex_j * torch.sin(sinusoid_coords)

    # sinusoids = torch.exp(complex_j * sinusoid_coords)

    # B x *fft_shape x 1
    fourier_coeffs = fourier_coeffs.unsqueeze(-1)

    # B x *fft_shape x n_sample_points
    sinusoids = fourier_coeffs * sinusoids

    # Average over all sinusoids
    dims_to_collapse = tuple([i + 1 for i in range(len(fft_shape))])
    # B x n_sample_points
    interpolated = torch.mean(sinusoids, dim=dims_to_collapse)

    # Un-complexify them
    interpolated = interpolated.real

    return interpolated


class FourierInterpolator(nn.Module):
    def __init__(self, input_shape: Tuple[int], output_shape: Tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_output_el = np.prod(self.output_shape)

    def forward(
        self, y: torch.Tensor, sample_parameters: Tuple[torch.Tensor]
    ) -> torch.Tensor:
        """
        :param y: The original values.
        :param xnew: The xnew points to which y shall be interpolated.
        :return: The interpolated values ynew at xnew.
        """

        xnew = sample_parameters[0]
        batch_size = y.shape[0]

        xnew = xnew.expand(batch_size, *self.output_shape, len(self.input_shape))

        xnew = xnew.reshape(batch_size, self.n_output_el, len(self.input_shape))
        ynew = n_fourier_interp(y, xnew)
        ynew = ynew.view(batch_size, *self.output_shape)

        return ynew


# This torch Gaussian PDF implementation is adapted from my PainterBot project: https://github.com/mkaic/painterbot/blob/4b29f2d01b8941b3978a1690b09f0fcedd82ad53/painterbot/render.py#L12-L89
def n_anisotropic_gaussian_pdf(
    coordinates: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    original_values has some arbitrary shape (B x *input_shape)
    original_coordinates has the same input_shape, but wihtout the batch dim. are in the range [0,1]
    mu and sigma are of shape (n_sample_points x Ndims)
    """

    # Simplified N-dim Gaussian PDF function:
    # e ^ ( -1 * sum( ( coordinates - mu ) ^ 2 / ( 2 * sigma ^ 2 ) )

    # *input_shape x 1 x Ndims
    # ->
    # B x *input_shape x n_sample_points x Ndims
    # after broadcasting and subtraction

    coordinates = coordinates.unsqueeze(-2) - mu

    coordinates = torch.square(coordinates)
    coordinates = coordinates / (2 * torch.square(sigma))

    # B x *input_shape x n_sample_points
    pdf = torch.exp(-1 * (torch.sum(coordinates, dim=-1)))

    return pdf


class AnisotropicGaussianSampler(nn.Module):
    def __init__(self, input_shape: Tuple[int], output_shape: Tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.ndims = len(input_shape)

        # *input_shape x Ndims
        coordinate_grid = torch.meshgrid(
            *[torch.arange(dim, dtype=torch.float) for dim in self.input_shape],
            indexing="ij"
        )
        coordinate_grid = torch.stack(coordinate_grid, dim=-1) / torch.tensor(
            self.input_shape, dtype=torch.float
        )
        self.register_buffer("coordinate_grid", coordinate_grid)

    def forward(
        self,
        activations: torch.Tensor,
        sample_parameters: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size = activations.shape[0]

        mu, sigma = sample_parameters

        mu = mu.view(-1, len(self.input_shape))
        sigma = sigma.view(-1, len(self.input_shape))

        # B x *input_shape x n_sample_points
        pdfs = n_anisotropic_gaussian_pdf(self.coordinate_grid, mu, sigma)

        correlation = pdfs * activations.unsqueeze(-1)

        # B x n_sample_points
        dims_to_reduce = tuple(range(1, self.ndims + 1))
        correlation = torch.mean(correlation, dim=dims_to_reduce)

        correlation = correlation.view(batch_size, *self.output_shape)

        return correlation
