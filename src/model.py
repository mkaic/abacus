import torch.nn as nn
import torch
from typing import List, Tuple

from .sparse_abacus import SparseAbacusLayer


class SparseAbacusModel(nn.Module):
    """
    A neural network where neurons are free to move their own sparse connections.
    :param data_shapes: Shapes of the input tensor, the activations at each layer, and output tensor.
    :param data_dependent: Whether the sampling points of each layer are predicted on-the-fly in a data-dependent manner, a la attention, or are simply directly trainable parameters. Default: False.
    :param degree: The number of inbound connections to each neuron. Default: 2. If any value more than 2 is chosen, you'll likely want to change the aggregation function used in SparseAbacusLayer, as it is designed for neurons with only 2 inputs.
    :param aggregator: A function that takes the `degree` samples for each neuron and aggregates them into a single value. By default, this is the fuzzy NAND function because `degree` defaults to 2.
    :param lookbehind: The number of previous layers of activations the current layer can sample from. Defaults to 1. If set > 1, the network can theoretically learn skip-connections.
    :return: None
    """

    def __init__(
        self,
        data_shapes: List[Tuple[int]],
        data_dependent: bool = False,
        degree: int = 2,
        lookbehind: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(data_shapes) - 1):
            # If we want attention-style data dependence, we need to create a separate module which does the data-dependent prediction for the main layers.
            if data_dependent:
                sample_points_predictor = SparseAbacusLayer(
                    input_shape=data_shapes[i],
                    output_shape=data_shapes[i + 1],
                    degree=degree,
                    sample_points_predictor=None,
                    lookbehind=lookbehind,
                )
            else:
                sample_points_predictor = None

            self.layers.append(
                SparseAbacusLayer(
                    input_shape=data_shapes[i],
                    output_shape=data_shapes[i + 1],
                    degree=degree,
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
        for layer in self.layers:
            layer.clamp_params()
