#! /opt/conda/bin/python

# %%
import time
import torch
from src.sparse_abacus import interp1d
import numpy as np

start = time.time()

errors = []
for i in range(10000):
    x = torch.sort(torch.rand(2, 256), dim=-1)[0]
    x = x - x.min(dim=-1).values.unsqueeze(-1)
    x = x / x.max(dim=-1).values.unsqueeze(-1)
    y = torch.rand_like(x)
    x_new = torch.sort(torch.rand(2, 256), dim=-1)[0]

    torchs = interp1d(x, y, x_new)
    numpys = torch.from_numpy(np.interp(x_new[1], x[1], y[1], left=0, right=0)).float()
    errors.append((torchs[1] - numpys).abs().max())
print(f"Time: {(time.time() - start)}")
print(f"Error = {sum(errors)/len(errors)} +- {np.std(errors)} (max = {max(errors)})")   

# %%
