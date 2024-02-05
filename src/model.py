import torch.nn as nn
import torch
from typing import List, Callable

from .sparse_abacus import SparseAbacusLayer, fuzzy_nand


class SparseAbacusModel(nn.Module):
    """
    A neural network where neurons are free to move their own sparse connections.
    :param activation_dims: How big the input vector should be at each layer. The first item in this list should be the size of the input vector.
    :param data_dependent: Whether the sampling points of each layer are predicted on-the-fly in a data-dependent manner, a la attention, or are simply directly trainable parameters. Default: False.
    :param degree: The number of inbound connections to each neuron. Default: 2. If any value more than 2 is chosen, you'll likely want to change the aggregation function used in SparseAbacusLayer, as it is designed for neurons with only 2 inputs.
    :return: None
    """

    def __init__(
        self,
        input_dim: int,
        layer_dims: List[int],
        output_dim: int,
        data_dependent: bool = False,
        degree: int = 2,
        aggregator: Callable[[torch.Tensor], torch.Tensor] = fuzzy_nand,
        lookbehind: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        activations_dims = [input_dim, *layer_dims, output_dim]
        for i in range(len(activations_dims) - 1):
            # If we want attention-style data dependence, we need to create a separate module which does the data-dependent prediction for the main layers.
            if data_dependent:
                sample_points_predictor = SparseAbacusLayer(
                    input_dims=activations_dims[i],
                    output_dims=activations_dims[i + 1] * 2,
                    degree=degree,
                    aggregator=aggregator,
                    sample_points_predictor=None,
                    lookbehind=lookbehind,
                )
            else:
                sample_points_predictor = None

            self.layers.append(
                SparseAbacusLayer(
                    input_dims=activations_dims[i],
                    output_dims=activations_dims[i + 1],
                    degree=degree,
                    aggregator=aggregator,
                    sample_points_predictor=sample_points_predictor,
                    lookbehind=lookbehind,
                )
            )

        param_count = sum(p.numel() for p in self.parameters())
        print(
            f"Initialized SparseAbacusModel with {param_count:,} total trainable parameters."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def clamp_params(self):
        for p in self.parameters():
            p.data.clamp_(0, 1)
