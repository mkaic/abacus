import torch
from ..src.interpolators import (
    LinearInterpolator,
    SciPyLinearInterpolator,
    FourierInterpolator,
)
import matplotlib.pyplot as plt
import itertools

BATCH_SIZE = 3

input_shapes = [(4,), (4, 4), (4, 4, 4), (4, 4, 4, 4)]
output_shapes = [(8,), (8, 8), (8, 8, 8), (8, 8, 8, 8)]

print("TESTING COMBINATIONS OF INPUT AND OUTPUT SHAPES\nAND VERIFYING LINEAR INTERP")
for input_shape, output_shape in itertools.product(input_shapes, output_shapes):
    print(input_shape, output_shape)

    linear_interpolator = LinearInterpolator(input_shape, output_shape)
    scipy_interpolator = SciPyLinearInterpolator(input_shape, output_shape)
    fourier_interpolator = FourierInterpolator(input_shape, output_shape)

    input_values = torch.rand(BATCH_SIZE, *input_shape)
    output_points = torch.rand(BATCH_SIZE, *output_shape, len(input_shape))

    linear_output = linear_interpolator(input_values, output_points)
    linear_reference = scipy_interpolator(input_values, output_points)
    fourier_output = fourier_interpolator(input_values, output_points)

    assert torch.allclose(
        linear_output, linear_reference, atol=1e-6
    ), f"Interpolation failed. Error: {torch.abs(linear_output - linear_reference).max()}"


print("VERIFYING FOURIER INTERP ON EVEN AND ODD SIZES")
# Test that Fourier output is lossless at identical resolutions in all dimensions
shapes = [
    (16,),
    (17,),
    (16, 16),
    (17, 17),
    (8, 8, 8),
    (9, 9, 9),
    (4, 4, 4, 4),
    (5, 5, 5, 5),
]
for shape in shapes:
    print(shape)

    fourier_interpolator = FourierInterpolator(shape, shape)

    input_values = torch.rand(BATCH_SIZE, *shape)
    output_points = torch.meshgrid(
        *[torch.linspace(0, 1, dim) for dim in shape],
        indexing="ij",
    )
    output_points = (
        torch.stack(output_points, dim=-1)
        .unsqueeze(0)
        .expand(BATCH_SIZE, *[-1 for _ in shape], -1)
    )
    fourier_output = fourier_interpolator(input_values, output_points)

    assert torch.allclose(
        input_values, fourier_output, atol=1e-4
    ), f"Interpolation failed. Error: {torch.abs(input_values - fourier_output).max()}"

print("VISUALIZING 2D-TO-2D INTERPS AS A SANITY CHECK")
# Now we will visually inspect the 2D to 2D case
input_shape = (4, 4)

for resolution in (4, 64):

    output_shape = (resolution, resolution)

    linear_interpolator = LinearInterpolator(input_shape, output_shape)
    scipy_interpolator = SciPyLinearInterpolator(input_shape, output_shape)
    fourier_interpolator = FourierInterpolator(input_shape, output_shape)

    input_values = torch.rand(BATCH_SIZE, *input_shape)

    output_points = torch.meshgrid(
        torch.linspace(0, 1, resolution),
        torch.linspace(0, 1, resolution),
        indexing="ij",
    )
    output_points = (
        torch.stack(output_points, dim=-1).unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1)
    )

    linear_output = linear_interpolator(input_values, output_points)
    linear_reference = scipy_interpolator(input_values, output_points)
    fourier_output = fourier_interpolator(input_values, output_points)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.colorbar(axes[0, 0].imshow(input_values[0]))
    plt.colorbar(axes[0, 1].imshow(linear_output[0]))
    plt.colorbar(axes[1, 0].imshow(linear_reference[0]))
    plt.colorbar(axes[1, 1].imshow(fourier_output[0]))

    plt.savefig(f"abacus/tests/images/interpn_test_{resolution}.png")

    fft = torch.fft.fftn(linear_output[0])
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(linear_output[0])
    axes[0, 1].imshow(torch.log(torch.abs(fft)))
    axes[1, 0].imshow(torch.fft.ifftn(fft).real)
    axes[1, 1].imshow(torch.log(torch.abs(torch.fft.fftshift(fft))))
    plt.savefig(f"abacus/tests/images/fft.png")

print("DONE")
