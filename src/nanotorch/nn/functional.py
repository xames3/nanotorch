"""\
NanoTorch Functional API
========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, March 09 2024
Last updated on: Saturday, March 16 2024

Functions.

This module in NanoTorch offers a comprehensive suite of stateless
functions that perform various tensor operations, mimicking the
functionality of PyTorch's functional API. This module is designed to
provide users with a wide array of operations, including activation
functions, loss computations, and other mathematical transformations,
which are essential building blocks for creating and training neural
network models. Unlike the object-oriented approach of the layers and
losses modules, the functional module delivers these operations in a
purely functional manner.

Activation functions are crucial for introducing non-linearity into
neural networks, enabling them to learn complex patterns in data. The
functional module includes implementations of popular activation
functions:

- ``relu``: Applies the Rectified Linear Unit function element-wise,
    returning zero for all negative inputs and the original value for
    all non-negative inputs.
- ``tanh``: Applies the hyperbolic tangent function element-wise,
    squashing the input values to be within the range (-1, 1).

Loss functions measure the difference between the model outputs and the
target values, guiding the optimization process. The functional module
provides implementations of commonly used loss functions:

- ``mse_loss``: Computes the Mean Squared Error loss between the
    predicted values and the actual targets, commonly used for
    regression tasks.

The modular and functional design of the functional module makes it
easy to extend with new operations as needed. Users can contribute
additional functions following the same pattern, enhancing the utility
and flexibility of the NanoTorch package. The development roadmap for the
functional module includes the addition of more advanced operations and
optimizations for performance and numerical stability.

The functional module is a versatile and essential part of the NanoTorch
package, providing users with a wide range of operations for building
and training neural network models. Its functional API design promotes
clarity and efficiency in code, making advanced neural network
construction accessible to both novice and experienced practitioners in
the field of deep learning.

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2024 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

from nanotorch._tensor import relu
from nanotorch._tensor import tanh
from nanotorch._tensor import tensor


def relu(input: tensor) -> tensor:
    """Applies the rectified linear unit function."""
    return relu(input)


def tanh(input: tensor) -> tensor:
    """Returns a new tensor with the hyperbolic tangent of the input."""
    return tanh(input)


def mse_loss(input: tensor, target: tensor) -> tensor:
    """Measures the element-wise mean squared error."""
    loss = [(i - t) ** 2 for i, t in zip(input, target)]
    return sum(loss) * (1.0 / len(loss))


def hinge_embedding_loss(input: tensor, target: tensor) -> tensor:
    """Measures the loss given an input tensor and a labels tensor."""
    loss = [(1 + t * -i).relu() for i, t in zip(input, target)]
    return sum(loss) * (1.0 / len(loss))
