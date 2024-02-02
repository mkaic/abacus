#! /opt/conda/bin/python

# %%
import time
import torch
from ..src.sparse_abacus import interp1d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mc
import matplotlib as mpl
from scipy.interpolate import interp1d as scipy_interp1d

start = time.time()

errors = []
for i in range(100):
    x = torch.sort(torch.rand(2, 256), dim=-1)[0]
    x = x - x.min(dim=-1).values.unsqueeze(-1)
    x = x / x.max(dim=-1).values.unsqueeze(-1)
    y = torch.rand_like(x)
    x_new = torch.sort(torch.rand(2, 256), dim=-1)[0]

    torchs = interp1d(x, y, x_new)
    numpys = torch.from_numpy(np.interp(x_new[1], x[1], y[1], left=0, right=0)).float()
    errors.append((torchs[1] - numpys).abs().max())
print(f"Time: {(time.time() - start):.5f}")
print(
    f"Error = {sum(errors)/len(errors):.5f} +- {np.std(errors):.5f} (max = {max(errors):.5f})"
)

# Sanity check
x = torch.sort(torch.rand(2, 4), dim=-1)[0]
x = x - x.min(dim=-1).values.unsqueeze(-1)
x = x / x.max(dim=-1).values.unsqueeze(-1)
y = torch.rand_like(x)
x_new = torch.linspace(0, 1, 1024).repeat(2, 1)

y_new = interp1d(x, y, x_new)

y_new_numpy_0 = (
    torch.from_numpy(np.interp(x_new[0], x[0], y[0], left=0, right=0))
    .float()
    .unsqueeze(0)
)
y_new_numpy_1 = (
    torch.from_numpy(np.interp(x_new[1], x[1], y[1], left=0, right=0))
    .float()
    .unsqueeze(0)
)

y = y.unsqueeze(1)
og_0 = scipy_interp1d(x[0], y[0], kind="nearest")(x_new[0].numpy())
og_1 = scipy_interp1d(x[1], y[1], kind="nearest")(x_new[1].numpy())

y_new = y_new.unsqueeze(1)

# Make a colormap out of the original data and the interpolation
# Plot them side by side to see if they look right.
# Do this twice to make sure batching works properly.

fig, axs = plt.subplots(3, 2, figsize=(12, 4))
axs[0, 0].imshow(og_0, aspect=128)
axs[0, 0].set_xticks(ticks=x[0] * 1024, labels=y[0, 0].numpy().round(2))
axs[1, 0].imshow(y_new[0, :], aspect=128)
axs[2, 0].imshow(y_new_numpy_0, aspect=128)
axs[0, 1].imshow(og_1, aspect=128)
axs[0, 1].set_xticks(ticks=x[1] * 1024, labels=y[1, 0].numpy().round(2))
axs[1, 1].imshow(y_new[1, :], aspect=128)
axs[2, 1].imshow(y_new_numpy_1, aspect=128)

plt.savefig("abacus/tests/interp1d.png", facecolor="white")

# %%
