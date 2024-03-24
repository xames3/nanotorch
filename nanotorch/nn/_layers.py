"""\
NanoTorch Neural Networks Layers API
====================================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, March 09 2024
Last updated on: Saturday, March 16 2024

Neural Networks Layers.

The layers module of NanoTorch provides a foundational framework for
constructing neural networks, closely mirroring the design and
functionality of PyTorch. It offers a collection of classes and methods
designed to facilitate the easy assembly of neural network architectures
from basic building blocks. This module includes implementations of
common layers such as ``Linear``, ``Module``, among others, enabling
users to define custom neural network models for a wide range of tasks in
machine learning and deep learning.

The ``Module`` class serves as the base class for all neural network
modules in NanoTorch. It defines a common interface for all subsequent
layers and models, ensuring consistent behavior and interaction. Users
can derive their custom layers or models by subclassing ``Module``,
leveraging its built-in functionality for parameter management, module
hierarchy, and more.

The ``Linear`` layer, also known as a fully connected or dense layer,
applies a linear transformation to the incoming data. It is a
fundamental component in neural networks, used for tasks ranging from
simple linear regression to complex deep neural networks.

The module also includes various utility functions and helper classes to
assist in the creation and manipulation of layers. These utilities
include methods for parameter initialization, module registration, and
more, facilitating a more streamlined and efficient model-building
experience. The design of the layers module emphasizes extensibility,
allowing users to easily introduce new layer types and functionalities.
By inheriting from the Module class, users can create custom layers that
seamlessly integrate with the rest of the NanoTorch framework,
leveraging its parameter management and modular architecture.

Future updates may include advanced layers such as convolutional and
recurrent layers, as well as improvements to the underlying
infrastructure to support more complex models and training regimes.

This module is a cornerstone of the NanoTorch package, offering a
versatile and user-friendly framework for building neural network
models. Its design is inspired by PyTorch, providing a familiar and
intuitive experience for users while maintaining the flexibility to meet
a wide range of modeling requirements. Whether you are a beginner or an
experienced practitioner, the layers module equips you with the tools
needed to embark on your deep learning journey.

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2024 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

import random
import typing as t

from nanotorch._tensor import relu
from nanotorch._tensor import tanh
from nanotorch._tensor import tensor


class Module:
    """Base class for all neural network modules.

    Your models should also subclass this class.
    """

    type: str = "module"

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialize a module with some arguments."""
        self.args = args
        self.kwargs = kwargs
        self._modules: list[Module] = []

    def __repr__(self) -> str:
        """Representation of the module."""
        modules = "\n"
        for attr, module in self.__dict__.items():
            if attr in ("args", "kwargs", "_modules"):
                continue
            modules += f"  ({attr}): {module}\n"
        return f"{type(self).__name__}({modules})"

    def __setattr__(self, name: str, value: t.Any) -> None:
        """Override ``__setattr__`` to accumulate modules."""
        if isinstance(value, Module):
            if value.type != "activation":
                self._modules.append(value)
        super().__setattr__(name, value)

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
        return [
            parameter
            for module in self._modules
            for parameter in module.parameters()
        ]

    def zero_grad(self) -> None:
        """Resets the gradients of all optimized parameters to zero.

        This method should be called before a new optimization step or
        gradient calculation to avoid accumulating gradients from
        multiple forward passes.
        """
        for parameter in self.parameters():
            parameter.grad = 0.0


class Layer(Module):
    """Derived class for subclassing layers."""

    type: str = "layer"


class Loss(Module):
    """Derived class for subclassing losses."""

    type: str = "loss"

    def forward(self, input: tensor, target: tensor, reduction: str) -> tensor:
        """Define the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError(
            f"Module: {type(self).__name__} is missing the"
            ' required "forward" function'
        )


class ActivationLayer(Layer):
    """Derived class foe subclassing activation layers."""

    type: str = "activation"

    def __repr__(self) -> str:
        """Representation of the activation layer."""
        return f"{type(self).__name__}()"


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
        """Initializes a new instance of the neuron with specified input
        features and activation function configuration. Weights and bias
        are initialized at this stage, with weights being set to random
        values for each input feature.
        """
        self.in_features = in_features
        self.activation_fn = activation_fn
        self.weights = [
            tensor(random.uniform(-1, 1)) for _ in range(self.in_features)
        ]
        self.bias = tensor(0.0)

    def forward(self, input: tensor) -> tensor:
        """Computes the forward pass of the neuron."""
        if isinstance(input, tensor):
            input = input.data
        z = sum((w * x for w, x in zip(self.weights, input)), self.bias)
        return self.activation_fn(z) if self.activation_fn else z

    def parameters(self) -> list[tensor]:
        """List of all the computable parameters in the neuron."""
        return self.weights + [self.bias]


class Linear(Layer):
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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: ActivationLayer | None = None,
    ) -> None:
        """Initializes a new instance of a linear layer with
        specified input, output features and activation function.
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
        return (
            f"{type(self).__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features})"
        )

    def forward(self, input: tensor) -> tensor | list[tensor]:
        """Computes the forward pass using the input tensor."""
        output = [neuron(input) for neuron in self.neurons]
        return output[0] if len(output) == 1 else output

    def parameters(self) -> list[tensor]:
        """List of all the computable parameters in the whole layer."""
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]


class ReLU(ActivationLayer):
    """Applies the ReLU function."""

    def forward(self, input: tensor) -> tensor:
        """Executes the forward pass of the model."""
        return relu(input)


class Tanh(ActivationLayer):
    """Applies the Hyperbolic Tangent (Tanh) function."""

    def forward(self, input: tensor) -> tensor:
        """Executes the forward pass of the model."""
        return tanh(input)
