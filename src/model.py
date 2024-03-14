import torch.nn as nn
import torch
from typing import List, Tuple


class SamplerModel(nn.Module):
    """
    A neural network where neurons are free to move their own sparse connections.
    :param data_shapes: Shapes of the input tensor, the activations at each layer, and output tensor.
    :param data_dependent: Whether the sampling points of each layer are predicted on-the-fly in a data-dependent manner, a la attention, or are simply directly trainable parameters. Default: False.
    :param degree: The number of inbound connections to each neuron. Default: 2. If any value more than 2 is chosen, you'll likely want to change the aggregation function used in BinaryTreeSparseAbacusLayer, as it is designed for neurons with only 2 inputs.
    :param lookbehind: The number of previous layers of activations the current layer can sample from. Defaults to 1. If set > 1, the network can theoretically learn skip-connections.
    :return: None
    """

    def __init__(
        self,
        input_shapes: List[Tuple[int]],
        mid_block_shapes: List[Tuple[int]],
        output_shapes: List[Tuple[int]],
        layer_class: nn.Module,
        first_layer_class: nn.Module = None,
        degree: int = 2,
    ):
        super().__init__()
        self.input_shapes = input_shapes
        self.mid_block_shapes = mid_block_shapes
        self.output_shapes = output_shapes
        self.degree = degree
        self.layer_class = layer_class
        self.first_layer_class = first_layer_class
        if self.first_layer_class is None:
            self.first_layer_class = layer_class
        self.n_mid_blocks = len(mid_block_shapes)

        self.layers = nn.ModuleList()

        data_shapes = input_shapes + mid_block_shapes + output_shapes

        # Input layers
        for i in range(len(input_shapes)):
            self.layers.append(
                self.build_layer(
                    input_shape=data_shapes[i],
                    output_shape=data_shapes[i + 1],
                    first_layer=True,
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
            f"Initialized SamplerModel with {param_count:,} total trainable parameters."
        )

    def build_layer(
        self, input_shape, output_shape, data_dependent=False, first_layer=False
    ):
        # If we want attention-style data dependence, we need to create a separate module which does the data-dependent prediction for the main layers.

        residual = input_shape == output_shape
        layer_class = self.layer_class if not first_layer else self.first_layer_class

        if data_dependent:
            sample_parameters_predictor = layer_class(
                input_shape=input_shape,
                output_shape=(*output_shape, self.degree, len(input_shape)),
                degree=self.degree,
                sample_parameters_predictor=None,
            )
        else:
            sample_parameters_predictor = None

        return layer_class(
            input_shape=input_shape,
            output_shape=output_shape,
            degree=self.degree,
            sample_parameters_predictor=sample_parameters_predictor,
            residual=residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x)
        return x

    def clamp_params(self):
        for layer in self.layers:
            layer.clamp_params()
