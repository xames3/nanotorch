"""\
NanoTorch Tensor API
====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, December 03 2023
Last updated on: Wednesday, December 06 2023

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2023 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

import random
import typing as t

from graphviz import Digraph

from nanotorch.utils import colors

if t.TYPE_CHECKING:
    from nanotorch import Generator

__all__ = ["rand", "ones", "tensor", "zeros"]

_Number = int | float
_Size = t.Sequence[int]


class _Tensor:
    """Class to represent tensors with no autograd history."""

    def __init__(
        self,
        data: _Number,
        *,
        label: str = None,
    ) -> None:
        """Constructor, called when a new tensor is created.

        This will initialize a new ``_Tensor`` object with given data.
        """
        self._data = data
        self._nodes = set()
        if label is None:
            label = ""
        self._operator = None
        self._label = label
        # NOTE(xames3): Initially, the gradient will be 0. This implies
        # that there is no effect during the initialization meaning, in
        # the beginning the inital values do not affect the output.
        # This also means that changing ``grad`` will not change the
        # loss function.
        self.grad = 0.0

    def __repr__(self) -> str:
        """Representation of the tensor object."""
        if isinstance(self._data, float):
            data = f"{self._data:.4f}".rstrip("0")
        else:
            data = self._data
        return f"tensor({data})"

    def __add__(self, __value: _Tensor) -> _Tensor:
        """Implementation of the addition operation.

        This method is called when using the ``+`` operator to add two
        tensors together.
        """
        result = _Tensor(
            self.item() + __value.item(),
        )
        result._nodes = {self, __value}
        result._operator = "+"
        return result

    def __mul__(self, __value: _Tensor) -> _Tensor:
        """Implementation of the multiplication operation.

        This method is called when using the ``*`` operator to multiply
        two tensors together.
        """
        result = _Tensor(
            self.item() * __value.item(),
        )
        result._nodes = {self, __value}
        result._operator = "×"
        return result

    def _generate_nodes_and_edges(self) -> tuple[set, ...]:
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

        def populate_nodes_and_edges(self) -> None:
            if self not in nodes:
                nodes.add(self)
                for child in self._nodes:
                    edges.add((child, self))
                    populate_nodes_and_edges(child)

        populate_nodes_and_edges(self)
        return nodes, edges

    def create_graph(self) -> None:
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
        graph = Digraph(
            name="_Tensor Graph",
            graph_attr={"rankdir": "LR"},
        )
        nodes, edges = self._generate_nodes_and_edges()
        for node in nodes:
            tensor = str(id(node))
            graph.node(
                name=tensor,
                label="{ %s | data %.4f | grad %.4f}"
                % (node._label, node.item(), node.grad),
                fillcolor=random.choice(colors),
                shape="record",
                style="filled",
            )
            if node._operator:
                operator = tensor + node._operator
                graph.node(name=operator, label=node._operator)
                graph.edge(operator, tensor)
        for node_1, node_2 in edges:
            graph.edge(str(id(node_1)), str(id(node_2)) + node_2._operator)
        graph.view(cleanup=True)

    def item(self) -> _Number:
        """Returns the value of tensor as a standard Python object."""
        return self._data

    def _ndim(self, size: _Size) -> int:
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
        return self._ndim(self._data)


def rand(generator: Generator | None = None) -> _Tensor:
    """Returns a tensor filled with random number from a uniform
    distribution on the interval [0, 1]
    """
    if generator:
        random.seed(generator.seed)
    return _Tensor(random.random())


def zeros() -> _Tensor:
    """Returns a tensor filled with the scalar value 0."""
    return _Tensor(0.000)


def ones() -> _Tensor:
    """Returns a tensor filled with the scalar value 1."""
    return _Tensor(1.000)


tensor = _Tensor

# Inputs
# x1 = tensor(2.0, label="x1")
# x2 = tensor(0.0, label="x2")

# # Weights
# w1 = tensor(-3.0, label="w1")
# w2 = tensor(1.0, label="w2")
# b = tensor(6.7, label="b")

# x1w1 = x1 * w1
# x1w1._label = "x1.w1"
# x2w2 = x2 * w2
# x2w2._label = "x2.w2"

# x1w1_x2w2 = x1w1 + x2w2
# x1w1_x2w2._label = "x1.w1 + x2.w2"

# n = x1w1_x2w2 + b
# n._label = "n"

# n.create_graph()
