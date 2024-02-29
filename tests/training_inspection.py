import torch
import matplotlib.pyplot as plt
import os
from pprint import pprint

weights = []

for fname in os.listdir("abacus/weights"):
    if fname.endswith(".ckpt"):
        weights.append(torch.load(f"abacus/weights/{fname}"))


weights_over_time = {k: torch.stack([w[k] for w in weights]) for k in weights[0].keys()}

pprint(weights_over_time.keys())