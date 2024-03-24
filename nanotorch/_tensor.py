"""\
NanoTorch Tensor API
====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, December 03 2023
Last updated on: Saturday, March 16 2024

Tensor object.

The ``_tensor`` module in this educational rendition of PyTorch offers
an implementation of tensors, providing a simplified yet insightful
perspective into the basic building blocks of neural network libraries.
This module is thoughtfully crafted to mimic core functionalities of a
PyTorch tensor, albeit at a small level rather than as a fully-fledged
tensor entity.

The essence of this module is to demonstrate the quintessential
properties and operations of tensors in a more fundamental and easily
graspable manner. Key features include:

    - Numpy-based Tensors: The module primarily deals with scalar values
                           or arrays mimicking as numpy arrays wrapped
                           as tensor objects, laying the groundwork for
                           understanding more complex tensor operations.
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

import typing as t
from functools import partial
from functools import total_ordering

import numpy as np
from graphviz import Digraph

from nanotorch._types import Data
from nanotorch._types import NodesAndEdges

np.set_printoptions(formatter={"float": "{: 0.4f}".format})


@total_ordering
class tensor:
    """Class to represent tensors."""

    def __init__(self, data: Data, *, dtype: np.DTypeLike = None) -> None:
        """This will create a new tensor object with given data."""
        self.nodes: tuple[tensor] = ()
        if isinstance(data, (np.ndarray, list, tuple)):
            data = np.array(data)
        elif isinstance(data, tensor):
            data = data.data
        self.data = data
        self.dtype = dtype
        # NOTE(xames3): Initially, the gradient will be 0. This implies
        # that there is no effect during the initialization. Meaning, in
        # the beginning the inital values do not affect the output.
        # This also means that changing ``grad`` will not change the
        # loss function.
        self.grad = 0.0
        self.grad_fn = partial(object)
        self.label = str()
        self.operator = str()

    def __repr__(self) -> str:
        """Representation of the tensor object."""
        if isinstance(self.data, float):
            data = f'({f"{self.data:.4f}".rstrip("0")}'
        elif isinstance(self.data, np.ndarray):
            data = np.array_repr(self.data, precision=4).strip("array")
            data = data.replace("\n", "\n ").replace(")", "")
            data = data.replace("dtype=", "dtype=torch.")
        else:
            data = f"({self.data}"
        if self.dtype:
            dtype = self.dtype.__name__
            if dtype not in data:
                data += f", dtype=torch.{dtype}"
        return f"tensor{data})"

    def __add__(self, other: tensor | Data) -> tensor:
        """Implementation of addition operation."""
        other = other if isinstance(other, tensor) else tensor(other)
        result = tensor(self.data + other.data)
        result.nodes = (self, other)
        result.operator = "+"

        def grad_fn() -> None:
            self.grad += result.grad
            other.grad += result.grad

        result.grad_fn = grad_fn
        return result

    def __mul__(self, other: tensor | Data) -> tensor:
        """Implementation of multiplication operation."""
        other = other if isinstance(other, tensor) else tensor(other)
        result = tensor(self.data * other.data)
        result.nodes = (self, other)
        result.operator = "×"

        def grad_fn() -> None:
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result.grad_fn = grad_fn
        return result

    def __pow__(self, exponent: int | float) -> tensor:
        """Implementation of exponentiation operation."""
        result = tensor(self.data**exponent)
        result.nodes = (self,)
        result.operator = "^"

        def grad_fn() -> None:
            self.grad += (exponent * self.data ** (exponent - 1)) * result.grad

        result.grad_fn = grad_fn
        return result

    def __neg__(self) -> tensor:
        """Implementation of unary negation operation."""
        return self * -1

    def __sub__(self, other: tensor | Data) -> tensor:
        """Implementation of difference/subtraction operation."""
        return self + (-other)

    def __floordiv__(self, other: tensor | Data) -> tensor:
        """Implementation of floor divison operation."""
        other = other if isinstance(other, tensor) else tensor(other)
        return tensor(np.numpy.floor_divide(self.data / other.data))

    def __truediv__(self, other: tensor | Data) -> tensor:
        """Implementation of divison operation."""
        return self * other**-1

    def __round__(self, ndigits: int) -> tensor:
        """Implementation of rounding operation."""
        return tensor(round(self.data, ndigits))

    def __lt__(self, other: tensor | Data) -> bool:
        """Implementation of less-than operation."""
        other = other if isinstance(other, tensor) else tensor(other)
        return self.data < other.data

    def __radd__(self, other: tensor | Data) -> tensor:
        """Implementation of reverse addition."""
        return self + other

    def __rsub__(self, other: tensor | Data) -> tensor:
        """Implementation of reverse subtraction."""
        return -self + other

    def __rmul__(self, other: tensor | Data) -> tensor:
        """Implementation of reverse multiplication."""
        return self * other

    def __rfloordiv__(self, other: tensor | Data) -> tensor:
        """Implementation of reverse floor divison."""
        return tensor(np.floor_divide(other.data, self.data))

    def __rtruediv__(self, other: tensor | Data) -> tensor:
        """Implementation of reverse divison."""
        return self**-1 * other

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: t.Any, **kwargs: t.Any
    ) -> tensor | np.ndarray | int | float | complex | None:
        """Method to ensure the tensors supports numpy's universal
        functions.

        This method is called when a NumPy ``ufunc`` is applied to
        objects and allows for the tensor object to be compatible with
        NumPy's ufuncs, enabling operations like addition,
        multiplication, and more complex functions to be applied
        directly to tensor instances.

        :param ufunc: The universal function (np.ufunc) being applied.
        :param method: A string indicating how the ufunc was called.

        :returns: A tensor instance as the result of the ufunc
                  operation, or a scalar value if the ufunc operation
                  results in a scalar. The operation is applied to the
                  ``.data`` attribute of the tensor instance(s).

        .. note::

            The method gracefully handles cases where inputs include
            both tensor instances and other types supported by the
            NumPy ufunc. The result is wrapped in a tensor instance if
            it's an ``ndarray``, otherwise, the raw output
            (e.g., scalar) is returned.
        """
        other = [
            input.data if isinstance(input, tensor) else input
            for input in inputs
        ]
        result = getattr(ufunc, method)(*other, **kwargs)
        if isinstance(result, np.ndarray):
            return tensor(result)
        return result

    def _generate_nodes_and_edges(self) -> NodesAndEdges:
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
        """
        graph = Digraph(name="Computation Graph", graph_attr={"rankdir": "LR"})
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
                shape="circle",
                style="filled",
                width="0.02",
            )
            if node.operator:
                operator = tensor + node.operator
                graph.node(name=operator, label=node.operator)
                graph.edge(operator, tensor)
        for node_l, node_r in edges:
            graph.edge(str(id(node_l)), str(id(node_r)) + node_r.operator)
        graph.view(cleanup=True)

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

    def item(self) -> Data:
        """Returns the value of tensor as a standard Python object."""
        return self.data

    def relu(self) -> tensor:
        """Applies the rectified linear unit function."""
        return relu(self)

    def tanh(self) -> tensor:
        """Returns a new tensor with the hyperbolic tangent of the
        input.
        """
        return tanh(self)


def from_numpy(ndarray: np.ndarray) -> tensor:
    """Creates a tensor from a numpy.ndarray."""
    return tensor(ndarray)


def relu(input: tensor) -> tensor:
    """Applies the rectified linear unit function."""
    if isinstance(input, (np.ndarray, list, tuple)):
        result = tensor(np.maximum(input, 0.0))
    else:
        result = tensor(max(input, 0.0))
    result.nodes = (input,)
    result.operator = "relu"

    def grad_fn() -> None:
        input.grad += (result.data > 0) * result.grad

    result.grad_fn = grad_fn
    return result


def tanh(input: tensor) -> tensor:
    """Returns a new tensor with the hyperbolic tangent of the input."""
    if isinstance(input, (np.ndarray, list, tuple)):
        result = tensor(np.tanh(input))
    else:
        result = tensor(np.tanh(input.data))
    result.nodes = (input,)
    result.operator = "tanh"

    def grad_fn() -> None:
        input.grad += (1.0 - result.data**2) * result.grad

    result.grad_fn = grad_fn
    return result
