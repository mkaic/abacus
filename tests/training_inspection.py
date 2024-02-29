import torch
import matplotlib.pyplot as plt
import os
from pprint import pprint

weights = []

for fname in sorted(os.listdir("abacus/weights"))[::10]:
    if fname.endswith(".ckpt"):
        weights.append(torch.load(f"abacus/weights/{fname}"))


weights_over_time = {
    k: torch.stack([w[k] for w in weights], dim=0) for k in weights[0].keys()
}

pprint(weights_over_time.keys())

print(weights_over_time["_orig_mod.layers.0.sample_points"].shape)
print(weights_over_time["_orig_mod.layers.0.sample_points"][:, 0, 0, 0, 0])
