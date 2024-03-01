import torch.nn as nn
import torch
from typing import List, Tuple

from .sparse_abacus import SparseAbacusLayer
from .interpolators import LinearInterpolator, FourierInterpolator
from .aggregators import FuzzyNAND, FuzzyNOR, FuzzyNANDNOR, LinearCombination


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
        data_dependent: List[bool] = None,
        degree: int = 2,
        interpolator_class: nn.Module = LinearInterpolator,
        aggregator_class: nn.Module = LinearCombination,
        lookbehind: int = 1,
    ):
        super().__init__()
        self.input_shapes = input_shapes
        self.mid_block_shapes = mid_block_shapes
        self.output_shapes = output_shapes
        self.data_dependent = data_dependent
        self.degree = degree
        self.interpolator_class = interpolator_class
        self.aggregator_class = aggregator_class
        self.lookbehind = lookbehind
        self.lookbehinds_list = (
            [1 for _ in self.input_shapes]
            + [1]
            + [min(self.lookbehind, i) for i in range(1, len(mid_block_shapes))]
            + [1 for _ in self.output_shapes]
        )
        self.n_mid_blocks = len(mid_block_shapes)

        self.layers = nn.ModuleList()

        data_shapes = input_shapes + mid_block_shapes + output_shapes

        for i, (input_shape, output_shape, lookbehind, data_dependent) in enumerate(
            zip(
                data_shapes[:-1],
                data_shapes[1:],
                self.lookbehinds_list,
                self.data_dependent,
            )
        ):

            self.layers.append(
                self.build_layer(input_shape, output_shape, data_dependent, lookbehind)
            )

        param_count = sum(p.numel() for p in self.parameters())
        print(
            f"Initialized SparseAbacusModel with {param_count:,} total trainable parameters."
        )

    def build_layer(self, input_shape, output_shape, data_dependent, lookbehind):
        # If we want attention-style data dependence, we need to create a separate module which does the data-dependent prediction for the main layers.

        residual = input_shape == output_shape

        if lookbehind > 1:
            input_shape = [lookbehind, *input_shape]

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
            lookbehind=lookbehind,
            residual=residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.lookbehind > 1:
            activations = []

        for layer, lookbehind in zip(self.layers, self.lookbehinds_list):

            if lookbehind > 1:
                cache = torch.stack(activations[-lookbehind:], dim=1)
            else:
                cache = None

            x = layer(x, cache=cache)

            # self.lookbehind is different (it's global) from just lookbehind (which is local for this layer)
            if self.lookbehind > 1:
                activations.append(x)
        return x

    def clamp_params(self):
        for layer in self.layers:
            layer.clamp_params()
