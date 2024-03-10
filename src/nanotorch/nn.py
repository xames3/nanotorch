"""\
NanoTorch Neural Networks API
=============================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Thursday, December 14 2023
Last updated on: Saturday, March 09 2024

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2023 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

import random

from nanotorch._tensor import relu
from nanotorch._tensor import tanh
from nanotorch._tensor import tensor
from nanotorch._types import Number


class Parameter:
    """Represents a trainable parameter within a neural network model.

    This class serves as a wrapper for tensors, specifically designed to
    hold weights, biases, or any other data that should be considered a
    parameter of the model.

    :param data: The initial data for the parameter, defaults to
                 ``None``.
    """

    def __init__(self, data: Number | None = None) -> None:
        """Initializes a new instance of the ``Parameter`` class,
        optionally with specified initial data.
        """
        if data is not None:
            self.data = tensor(data)
        else:
            self.data = tensor(random.uniform(-1.0, 1.0))

    def __repr__(self) -> str:
        """Representation of a parameter object."""
        return f"Parameter containing:\n{self.data}"


class Module:
    """Base class for all neural network modules.

    Your models should also subclass this class.
    """

    type: str = "module"

    def __repr__(self) -> str:
        """Representation of the module."""
        layers = "\n"
        for attr, layer in self.__dict__.items():
            if layer.type == "container":
                layers += f"  ({attr}): {layer.__repr__(4)}\n"
            else:
                layers += f"  ({attr}): {layer}\n"
        return (f"{type(self).__name__}({layers})")

    def __call__(self, input: tensor) -> tensor:
        """Override ``__call__`` with ``forward``."""
        return self.forward(input)

    def forward(self, input: tensor) -> tensor:
        """Define the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError(
            f"Module: {type(self).__name__} is missing the"
            ' required "forward" function'
        )

    def parameters(self) -> list[tensor]:
        """List of all the computable parameters."""
        raise NotImplementedError(
            f"Module: {type(self).__name__} is missing computable parameters"
        )


class ActivationLayer(Module):
    """Derived class for subclassing activation layers."""

    type: str = "activation"


class Neuron(Module):
    """Represents a single neuron, encapsulating the basic unit of
    computation in a neural network. Each neuron in this context is
    capable of performing a weighted sum of its input features,
    optionally adding a bias term, to produce an output signal. This
    output can then be used as input to subsequent layers or neurons in
    a network.

    The weights of the neuron are initialized randomly within the range
    [-1.0, 1.0], which is a common practice for initializing neural
    network parameters to break symmetry and ensure diverse paths of
    learning.

    If enabled, the bias term is also randomly initialized within the
    same range, providing the neuron with the ability to adjust its
    output independently of its input.

    :param in_features: Specifies the number of input features to the
                        neuron. This determines the size of the
                        input vector that the neuron expects and the
                        number of weights it will initialize to compute
                        its output.
    :param activation_fn: Activation layer for the neuron, defaults to
                          ``None``.
    """

    type: str = "neuron"

    def __init__(
        self,
        in_features: int,
        activation_fn: ActivationLayer | None = ActivationLayer,
    ) -> None:
        """Initializes a new instance of the ``Neuron`` class with
        specified input features and bias configuration. Weights and bias
        are initialized at this stage, with weights being set to random
        values for each input feature.
        """
        self.in_features = in_features
        self.activation_fn = activation_fn
        self.weights = [Parameter() for _ in range(self.in_features)]
        self.bias = Parameter().data

    def forward(self, input: tensor) -> tensor:
        """Computes the forward pass of the neuron using the provided
        input.

        This method calculates the weighted sum of the input features,
        optionally adds a bias term, and returns the result. It
        exemplifies the basic computation unit in neural networks,
        effectively representing a linear transformation followed by an
        optional translation (bias).

        :param input: A sequence of tensors containing input features.
                      The tensor's size should match the ``in_features``
                      of the neuron.
        :returns: The output of the neuron after applying the weighted
                  sum on the input and adding bias (if enabled).
        """
        z = sum((w.data * x for w, x in zip(self.weights, input)), self.bias)
        return self.activation_fn(z) if self.activation_fn else z

    def parameters(self) -> list[tensor]:
        """List of all the computable parameters in the neuron."""
        return self.weights + [self.bias]


class Linear(Module):
    """Represents a linear layer, a fundamental building block in neural
    networks, which applies a linear transformation to the incoming
    data. This transformation is defined as ``y = x*w + b``, where ``x``
    is the input data, ``w`` is the layer's weights, and ``b`` is the
    bias.

    Each ``Linear`` layer is essentially a collection of neurons, where
    each neuron performs a weighted sum of the input features,
    optionally adds a bias term, and outputs the result. This layer is
    crucial for modeling relationships between inputs and outputs that
    are assumed to be linearly correlated.

    The weights and biases of the layer are initialized to random
    values, which are then adjusted during the training process through
    backpropagation and optimization algorithms to minimize some loss
    function.

    :param in_features: Specifies the number of input features, i.e., the
                        size of each input sample.
    :param out_features: Specifies the number of output features, i.e.,
                         the size of each output sample. This determines
                         the number of neurons in the layer, each
                         producing one output feature.
    :param activation_fn: Activation layer for the neuron, defaults to
                          ``None``.
    """

    type: str = "layer"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: ActivationLayer | None = None,
    ) -> None:
        """Initializes a new instance of the ``Linear`` layer with
        specified input and output features and bias configuration.

        It creates a series of neurons (as many as ``out_features``),
        each configured to process ``in_features`` inputs.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.neurons = [
            Neuron(self.in_features, self.activation_fn)
            for _ in range(self.out_features)
        ]

    def __repr__(self) -> str:
        """Representation of linear layer."""
        activation_fn = (
            self.activation_fn.__class__.__name__
            if self.activation_fn
            else None
        )
        return (
            f"{type(self).__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"activation_fn={activation_fn})"
        )

    def forward(self, input: tensor) -> tensor | list[tensor]:
        """Computes the forward pass of the ``Linear`` layer using the
        provided input tensor.

        This method iterates over the layer's neurons, each computing
        its output based on the input tensor. The outputs from all
        neurons are then collected into a list (or a single tensor if
        the output size is 1), representing the linearly transformed
        data. This is the crux of the layer's functionality, enabling it
        to transform input data into a higher or lower dimensional
        space, depending on the configuration of ``in_features``
        and ``out_features``.

        :param input: A tensor containing the input data.
        :returns: The output of the layer after applying the linear
                  transformation. If the layer has only one output
                  feature, a single tensor is returned; otherwise, a
                  list of tensors is returned, each corresponding to an
                  output feature.
        """
        output = [neuron(input) for neuron in self.neurons]
        return output[0] if len(output) == 1 else output

    def parameters(self) -> list[tensor]:
        """List of all the computable parameters in the whole layer."""
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]


class Sequential(Module):
    """A sequential container that holds and manages a series of layers
    or modules defined in a neural network model. It is designed to
    facilitate the construction of models by allowing layers to be added
    in a sequential manner, where the output of one layer is
    automatically passed as the input to the next.
    This class simplifies model architecture definitions, making it
    straightforward to stack layers and create feed-forward networks
    without manually handling the data flow between layers.

    The ``Sequential`` class can contain any number of layers, and it
    works with any objects that implement a forward method, enabling a
    flexible and modular approach to model building. It is particularly
    useful for creating simple sequential models where the data flows
    linearly through layers, but it can also serve as a component in
    more complex architectures.

    :param args: An arbitrary number of layers (or modules) that are
                 instances of ``Module`` or have a compatible interface.
                 These are the layers that will be sequentially executed
                 when the container is called with input data.
    """

    type: str = "container"

    def __init__(self, *args: Module) -> None:
        """Initializes a new instance of the ``Sequential`` container
        with the provided layers.
        """
        self.args = args
        self.layers: list[Module] = []
        for layer, next_layer in zip(args, args[1:]):
            if next_layer.type == "activation":
                layer.activation_fn = next_layer
        self.layers = [layer for layer in args if layer.type == "layer"]

    def __repr__(self, spaces: int = 2) -> str:
        """Representation of sequential layers."""
        args = "\n"
        for idx, layer in enumerate(self.args):
            args += f"{' ' * spaces}({idx}): {layer}\n"
        return (f"{type(self).__name__}({args}{(spaces - 2) * ' '})")

    def forward(self, input: tensor) -> tensor:
        """Executes the forward pass of the model by sequentially
        passing the input through all layers in the container.

        When this method is called, it iterates over each layer stored
        in the ``Sequential`` object, applying the ``forward`` method of
        each layer to the input (or to the output of the previous
        layer), thereby propagating the data through the model. The
        output of the last layer is then returned as the final result.

        :param input: The input tensor to be processed by the model.
                      The size and shape of this tensor must be
                      compatible with the first layer in the model.
        :returns: The output tensor produced after processing the input
                  through all layers in the model sequentially.
        """
        for layer in self.layers:
            input = layer(input)
        return input

    def parameters(self) -> list[tensor]:
        """List of all the computable parameters in the entire sequence
        of layers.
        """
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]


class Tanh(ActivationLayer):
    """Applies the Hyperbolic Tangent (Tanh) function."""

    def __repr__(self) -> str:
        """Representation of the Tanh activation layer."""
        return f"{type(self).__name__}()"

    def forward(self, input: tensor) -> tensor:
        """Executes the forward pass of the model."""
        return tanh(input)


class ReLU(ActivationLayer):
    """Applies the ReLU function."""

    def __repr__(self) -> str:
        """Representation of the ReLU activation layer."""
        return f"{type(self).__name__}()"

    def forward(self, input: tensor) -> tensor:
        """Executes the forward pass of the model."""
        return relu(input)
