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
    This is a rudimentary implementation of numpy.interp for the 1D case only.
    I also made it break if you try to interpolate anything outside a 0-1 range.
    :param x: The original coordinates.
    :param y: The original values.
    :param xnew: The xnew points to which y shall be interpolated.
    """
    
    assert not (torch.any(xnew < 0) or torch.any(xnew > 1) or torch.any(x < 0) or torch.any(x > 1)), "All x and xnew values must be in [0,1]"
    
    
    # Evaluate the forward difference
    slope = ((y[1:] - y[:-1]) / (x[1:] - x[:-1]))

    # Get the indices of the closest point to the left for each xnew point
    xnew_closest_left_indices = torch.searchsorted(x, xnew)
    
    print(xnew)
    print(xnew_closest_left_indices)
    
    # Get the offset from the point to the left to the xnew point
    xnew_offset = xnew - x[xnew_closest_left_indices]
    
    # Calculate the value for the nonzero xnew: value of the point to the left plus slope times offset
    ynew = y[xnew_closest_left_indices] + slope[xnew_closest_left_indices - 1] * xnew_offset


    return ynew

# plt.plot(x_new, interp1d(x, y, x_new) - 0.02, "go", label="Custom interpolation")
# plt.plot(x_new, np.interp(x_new, x, y, left=0, right=0) + 0.02, "ro", label="np.interp")
# plt.plot(x, y, "b--", label="original values")
# plt.legend()
# plt.show()


def fuzzy_nand(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (1 - x) * (1 - y)


# -

class SparseAbacus(nn.Module):
    def __init__(self, n_in, n_out)
