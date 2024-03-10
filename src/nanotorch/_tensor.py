"""\
NanoTorch Tensor API
====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, December 03 2023
Last updated on: Friday, March 08 2024

Scalar-Level Tensor

The ``_tensor`` module in this educational rendition of PyTorch offers a
scalar-level implementation of tensors, providing a simplified yet
insightful perspective into the basic building blocks of neural network
libraries. This module is thoughtfully crafted to mimic core
functionalities of a PyTorch tensor, albeit at a scalar level rather
than as a fully-fledged tensor entity.

The essence of this module is to demonstrate the quintessential
properties and operations of tensors in a more fundamental and easily
graspable manner. Key features include:

    - Scalar Tensors: The module primarily deals with scalar values
                      wrapped as tensor objects, laying the groundwork
                      for understanding more complex tensor operations.
    - Common APIs: Emulating the PyTorch interface, the module supports
                   common tensor creation methods such as:

                   - zeros: Creates a tensor filled with zeros.
                   - ones: Generates a tensor filled with ones.
                   - rand: Initializes a tensor with random values.
    - Basic Operations: Essential tensor operations such as addition,
                        multiplication, and basic mathematical functions
                        like ``tanh`` are implemented. These operations
                        are pivotal in understanding the transformation
                        and manipulation of data within neural networks.
    - Automatic Differentiation Support: While simplistic, the module
                                         also ventures into the realm of
                                         automatic differentiation,
                                         essential for gradient-based
                                         optimization methods.

These methods serve to familiarize users with standard tensor
initialization techniques in deep learning libraries. This module is an
integral part of the educational toolkit aimed at demystifying the
complexities of tensor operations within neural networks. It is
particularly beneficial for:

    - Beginners who are taking their first steps into the world of deep
      learning and neural networks.
    - Intermediate learners and developers who wish to solidify their
      understanding of tensor operations and automatic differentiation.

The ``_tensor`` module is designed with simplicity and ease of use in
mind, closely mirroring PyTorch's intuitive interface. Comprehensive
documentation accompanies each function and method, ensuring that
learners can quickly grasp the concepts being demonstrated. Users are
encouraged to interact with the module, experiment with its features,
and explore the underlying mechanics.

As part of the broader miniature PyTorch project, this module plays a
crucial role in presenting a cohesive and comprehensive learning
experience. It stands as a foundational element that supports the
exploration and understanding of more complex modules and
functionalities within the project.

This module, with its focus on scalar-level tensor operations, invites
learners and enthusiasts to step into the fascinating world of tensors
and their applications in neural networks. Your feedback, suggestions,
and contributions to enhance and expand this module are highly valued
and eagerly anticipated.

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2024 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

import functools
import math
import random
import typing as t

from graphviz import Digraph

from nanotorch.exceptions import IncorrectRoundingModeError
from nanotorch.utils import colors

if t.TYPE_CHECKING:
    from nanotorch._types import Number
    from nanotorch._types import Size
    from nanotorch.utils import Generator


@functools.total_ordering
class tensor:
    """Class to represent tensors with no autograd history."""

    def __init__(self, data: Number) -> None:
        """Constructor, called when a new tensor is created.

        This will initialize a new ``tensor`` object with given data.
        """
        self.nodes: set[tensor] = set()
        self.data = data
        # NOTE(xames3): Initially, the gradient will be 0. This implies
        # that there is no effect during the initialization. Meaning, in
        # the beginning the inital values do not affect the output.
        # This also means that changing ``grad`` will not change the
        # loss function.
        self.grad = 0.0
        self.grad_fn = functools.partial(object)
        self.label = str()
        self.operator = str()

    def __repr__(self) -> str:
        """Representation of the tensor object."""
        if isinstance(self.data, float):
            data = f"{self.data:.4f}".rstrip("0")
        else:
            data = f"{self.data}"
        return f"tensor({data})"

    def __add__(self, other: tensor | Number) -> tensor:
        """Implementation of the addition operation.

        This method is called when using the ``+`` operator to add two
        tensors together.
        """
        other = other if isinstance(other, tensor) else tensor(other)
        result = tensor(self.data + other.data)
        result.nodes = {self, other}
        result.operator = "+"

        def grad_fn() -> None:
            if isinstance(other, tensor):
                self.grad += 1.0 * result.grad
                other.grad += 1.0 * result.grad

        result.grad_fn = grad_fn
        return result

    def __sub__(self, other: tensor | Number) -> tensor:
        """Implementation of difference/subtraction operation.

        This method is called when using the ``-`` operator to subtract
        two tensors.
        """
        return self + (-other)

    def __mul__(self, other: tensor | Number) -> tensor:
        """Implementation of the multiplication operation.

        This method is called when using the ``*`` operator to multiply
        two tensors together.
        """
        other = other if isinstance(other, tensor) else tensor(other)
        result = tensor(self.data * other.data)
        result.nodes = {self, other}
        result.operator = "×"

        def grad_fn() -> None:
            if isinstance(other, tensor):
                self.grad += other.data * result.grad
                other.grad += self.data * result.grad

        result.grad_fn = grad_fn
        return result

    def __floordiv__(self, other: tensor | Number) -> tensor:
        """Implementation of floor divison operation.

        This method is called when using the ``//`` operator to divide
        two tensors.
        """
        other = other if isinstance(other, tensor) else tensor(other)
        return tensor(math.floor(self.data / other.data))

    def __truediv__(self, other: tensor | Number) -> tensor:
        """Implementation of divison operation.

        This method is called when using the ``/`` operator to divide
        two tensors.
        """
        return self * other**-1

    def __radd__(self, other: tensor | Number) -> tensor:
        """Implementation of reverse addition."""
        return self + other

    def __rsub__(self, other: tensor | Number) -> tensor:
        """Implementation of reverse subtraction."""
        return -self + other

    def __rmul__(self, other: tensor | Number) -> tensor:
        """Implementation of reverse multiplication."""
        return self * other

    def __rfloordiv__(self, other: Number) -> tensor:
        """Implementation of reverse floor divison."""
        return tensor(math.floor(other / self.data))

    def __rtruediv__(self, other: tensor | Number) -> tensor:
        """Implementation of reverse divison."""
        return self**-1 * other

    def __neg__(self) -> tensor:
        """Implementation of unary negation operation."""
        return self * -1

    def __pow__(self, exponent: int | float) -> tensor:
        """Implementation of the exponentiation operation.

        This method is called when using the ``**`` operator to raise
        tensor to some power.
        """
        result = tensor(self.data**exponent)
        result.nodes = {self}
        result.operator = "^"

        def grad_fn() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * result.grad

        result.grad_fn = grad_fn
        return result

    def __rpow__(self, other: Number) -> Number:
        """Implementation of reverse exponentiation."""
        return other**self.data

    def __round__(self, ndigits: int) -> tensor:
        """Implementation of rounding operation.

        This method is called when using the ``round()`` function to
        round a tensor up to certain digits.
        """
        return tensor(round(self.data, ndigits))

    def __lt__(self, other: tensor | Number) -> bool:
        """Implementation of less than operation."""
        other = other if isinstance(other, tensor) else tensor(other)
        return self.data < other.data

    def _generate_nodes_and_edges(self) -> tuple[set[t.Any], ...]:
        """Build a set of nodes and edges from the tensor.

        This method traverses a graph starting from the target tensor or
        the target root node. It recursively explores the graph, adding
        each visited node to a set of nodes and each discovered edge to
        a set of edges.

        :returns: A pair consisting of two sets, first set containing all
                  nodes in the graph and the second set containing the
                  tuples representing the edges of the graph. Each tuple
                  is a pair of nodes indicating the start and end of an
                  edge.
        """
        nodes, edges = set(), set()

        def populate_nodes_and_edges(self: tensor) -> None:
            if self not in nodes:
                nodes.add(self)
                for node in self.nodes:
                    edges.add((node, self))
                    populate_nodes_and_edges(node)

        populate_nodes_and_edges(self)
        return nodes, edges

    def visualize(self) -> None:
        """Visualize a graph of tensors as a directed graph (Digraph).

        This method takes the target tensor-node of a graph and
        visualizes the entire graph. The graph is laid out from left to
        right. Each node is represented as a rectangle with its data,
        and each edge is represented as an arrow connecting the nodes.

        .. note::

            The graph is created with the "LR" (left-to-right) rank
            direction.

        .. seealso::

            [1] See the method ``_generate_nodes_and_edges()`` to
                understand how the nodes and edges of the graph are
                built.
        """
        graph = Digraph(name="Node Graph", graph_attr={"rankdir": "LR"})
        nodes, edges = self._generate_nodes_and_edges()
        for node in nodes:
            tensor = str(id(node))
            graph.node(
                name=tensor,
                label=(
                    f"node = {node.label}\n"
                    f"data = {node.data:.4f}\n"
                    f"grad = {node.grad:.4f}"
                ),
                fillcolor=random.choice(colors),
                shape="circle",
                style="filled",
                width="0.02",
            )
            if node.operator:
                operator = tensor + node.operator
                graph.node(name=operator, label=node.operator)
                graph.edge(operator, tensor)
        for node_1, node_2 in edges:
            graph.edge(str(id(node_1)), str(id(node_2)) + node_2.operator)
        graph.view(cleanup=True)

    def item(self) -> Number:
        """Returns the value of tensor as a standard Python object."""
        return self.data

    def _ndim(self, size: Size | Number) -> int:
        """Calculate the number of dimensions (depth) of a tensor.

        This method recursively determines the depth of a tensor, which
        is analogous to the number of dimensions in an array. The method
        assumes that the tensor is a regular list, meaning each sub-list
        at a particular level has the same dimension.

        :param size: A nested list or a tuple for which the number of
                     dimensions is to be calculated. The method handles
                     non-list inputs by returning zero dimensions.
        :returns: The number of dimensions of the tensor. For
                  the ``non-list`` inputs, the return value is 0. For an
                  empty list, it returns 1, representing a single
                  dimension. For nested lists, the method returns the
                  depth of nesting.

        .. warning::

            This method does not correctly handle irregular or
            mixed-type lists where elements of the list or sub-lists are
            of different types or dimensions.
        """
        if not isinstance(size, (list, tuple)):
            return 0
        if not size:
            return 1
        return 1 + self._ndim(size[0])

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of a tensor."""
        return self._ndim(self.data)

    def backward(self) -> None:
        """Computes the gradient of current tensor w.r.t graph leaves.

        The graph is differentiated using the chain rule. This method accumulates gradients in the leaves, thus one might need to zero
        ``.grad`` attributes or set it to ``None`` before calling it.

        .. note::

            When ``inputs`` are provided and a given input is not a
            leaf, the current implementation will call its grad_fn.
        """
        graph: list[tensor] = []
        visited: set[tensor] = set()

        def create_graph(node: tensor) -> None:
            if node not in visited:
                visited.add(node)
                for _node in node.nodes:
                    create_graph(_node)
                graph.append(node)

        create_graph(self)
        self.grad = 1.0
        for node in reversed(graph):
            node.grad_fn()


def add(
    input: tensor | Number, other: tensor | Number, *, alpha: Number = 1
) -> tensor | Number:
    """Adds other, scaled by alpha, to input.

    :param input: The input tensor or number.
    :param other: The tensor or number to add to input.
    :param alpha: The multiplier for other, defaults to ``1``.
    :returns: Output tensor.
    """
    return input + alpha * other


def sub(
    input: tensor | Number, other: tensor | Number, *, alpha: Number = 1
) -> tensor | Number:
    """Subtracts other, scaled by alpha, from input.

    :param input: The input tensor or number.
    :param other: The tensor or number to subtract from input.
    :param alpha: The multiplier for other, defaults to ``1``.
    :returns: Output tensor.
    """
    return input - alpha * other


subtract = sub


def mul(input: tensor | Number, other: tensor | Number) -> tensor | Number:
    """Multiplies input by other.

    :param input: The input tensor or number.
    :param other: The tensor or number to multiply to input.
    :returns: Output tensor.
    """
    return input * other


multiply = mul


def div(
    input: tensor | Number,
    other: tensor | Number,
    *,
    rounding_mode: str | None = None,
) -> tensor | Number:
    """Divides the input input by the corresponding element of other.

    :param input: The dividend.
    :param other: The divisor.
    :param rounding_mode: Type of rounding applied to the result,
                            default to ``None``.
    :returns: Output tensor.

    .. note:: By default, this performs a "true" division like Python 3.
    """
    if rounding_mode == "floor":
        return input // other
    elif rounding_mode is None:
        return input / other
    raise IncorrectRoundingModeError(
        "The function supports rounding mode as None or floor, but you passed "
        f"{rounding_mode!r}"
    )


divide = div


def true_divide(
    dividend: tensor | Number,
    divisor: tensor | Number,
) -> tensor | Number:
    """Alias for ``div()`` with ``rounding_mode=None``."""
    return div(dividend, divisor, rounding_mode=None)


def neg(input: tensor | Number) -> tensor | Number:
    """Returns a new tensor with the negative of the input."""
    return -1 * input


negative = neg


def rand(generator: Generator | None = None) -> tensor:
    """Returns a tensor filled with random number from a uniform
    distribution on the interval [0, 1].
    """
    if generator:
        random.seed(generator.seed)
    return tensor(random.random())


def zeros() -> tensor:
    """Returns a tensor filled with the scalar value 0."""
    return tensor(0.000)


def ones() -> tensor:
    """Returns a tensor filled with the scalar value 1."""
    return tensor(1.000)


def arange(end: Number, start: Number = 0, step: Number = 1) -> list[tensor]:
    """Returns a list of tensors with values from the interval [start,
    end) taken with common difference step beginning from start.

    :param end: The ending value for the set of points.
    :param start: The starting value for the set of points, defaults to
                  ``0``.
    :param step: The gap between each pair of adjacent points, defaults
                 to ``1``.
    :returns: List of output tensors.
    """
    if start:
        start, end = end, start
    out: list[tensor] = []
    while start < end:
        out.append(tensor(start))
        start += step
    return out


def tanh(input: tensor) -> tensor:
    """Returns a new tensor with the hyperbolic tangent of the input.

    :param input: The input tensor.
    :returns: Output tensor.
    """
    tanh = (math.exp(2 * input.data) - 1) / (math.exp(2 * input.data) + 1)
    result = tensor(tanh)
    result.nodes = {input}
    result.operator = "tanh"

    def grad_fn() -> None:
        input.grad += (1 - tanh**2) * result.grad

    result.grad_fn = grad_fn
    return result


def relu(input: tensor) -> tensor:
    """Applies the rectified linear unit function."""
    result = tensor(0.0 if input.data < 0.0 else input.data)
    result.nodes = {input}
    result.operator = "relu"

    def grad_fn() -> None:
        input.grad += (result > 0) * result.grad

    result.grad_fn = grad_fn
    return result
