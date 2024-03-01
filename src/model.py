import torch.nn as nn
import torch
from typing import List, Tuple

from .sparse_abacus import SparseAbacusLayer
from .interpolators import LinearInterpolator
from .aggregators import LinearFuzzyNAND


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
        input_shapes: List[Tuple[int]],
        mid_block_shapes: List[Tuple[int]],
        output_shapes: List[Tuple[int]],
        degree: int = 2,
        interpolator_class: nn.Module = LinearInterpolator,
        aggregator_class: nn.Module = LinearFuzzyNAND,
    ):
        super().__init__()
        self.input_shapes = input_shapes
        self.mid_block_shapes = mid_block_shapes
        self.output_shapes = output_shapes
        self.degree = degree
        self.interpolator_class = interpolator_class
        self.aggregator_class = aggregator_class
        self.n_mid_blocks = len(mid_block_shapes)

        self.layers = nn.ModuleList()

        data_shapes = input_shapes + mid_block_shapes + output_shapes

        # Input layers
        for i in range(len(input_shapes)):
            self.layers.append(
                self.build_layer(
                    input_shape=data_shapes[i],
                    output_shape=data_shapes[i + 1],
                )
            )

        # Mid-block layers
        for i in range(len(input_shapes), len(input_shapes) + len(mid_block_shapes)):
            self.layers.append(
                self.build_layer(
                    input_shape=data_shapes[i],
                    output_shape=data_shapes[i + 1],
                )
            )

        # Output layers
        for i in range(len(input_shapes) + len(mid_block_shapes), len(data_shapes) - 1):
            self.layers.append(
                self.build_layer(
                    input_shape=data_shapes[i],
                    output_shape=data_shapes[i + 1],
                )
            )

        param_count = sum(p.numel() for p in self.parameters())
        print(
            f"Initialized SparseAbacusModel with {param_count:,} total trainable parameters."
        )

    def build_layer(self, input_shape, output_shape, data_dependent=False):
        # If we want attention-style data dependence, we need to create a separate module which does the data-dependent prediction for the main layers.

        residual = input_shape == output_shape

        if data_dependent:
            sample_points_predictor = SparseAbacusLayer(
                input_shape=input_shape,
                output_shape=(*output_shape, self.degree, len(input_shape)),
                interpolator=self.interpolator_class,
                aggregator=self.aggregator_class,
                degree=self.degree,
                sample_points_predictor=None,
            )
        else:
            sample_points_predictor = None

        return SparseAbacusLayer(
            input_shape=input_shape,
            output_shape=output_shape,
            interpolator=self.interpolator_class,
            aggregator=self.aggregator_class,
            degree=self.degree,
            sample_points_predictor=sample_points_predictor,
            residual=residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def clamp_params(self):
        for layer in self.layers:
            layer.clamp_params()
