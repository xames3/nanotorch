"""\
NanoTorch
=========

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, December 02 2023
Last updated on: Saturday, March 16 2024

Small-scale implementation of PyTorch from the ground up.

This project, a miniature implementation of the PyTorch library, is
crafted with the primary goal of elucidating the intricate workings of
neural network libraries. It serves as a pedagogical tool for those
seeking to unravel the mathematical complexities and the underlying
architecture that powers such sophisticated libraries.

The cornerstone of this endeavor is to provide a hands-on learning
experience by replicating key components of PyTorch, thereby granting
insights into its functional mechanisms. This bespoke implementation
focuses on the core aspects of neural network computation, including
tensor operations, automatic differentiation, and basic neural network
modules.

At the heart of this implementation lie tensor operations, which are the
building blocks of any neural network library. As of now, our tensors
support basic arithmetic functionalities found in PyTorch. A pivotal
feature of this project is a simplistic version of automatic
differentiation, akin to PyTorch's ``autograd``. It allows for the
computation of gradients automatically, which is essential for training
neural networks. The implementation includes rudimentary neural network
modules such as linear layers and activation functions. These modules
can be composed to construct simple neural network architectures. Basic
optimizers like SGD and common loss functions are included to facilitate
the training process of neural networks.

This project stands as a testament to the educational philosophy of
learning by doing. It is particularly beneficial for:

    - Students and enthusiasts who aspire to gain a profound
      understanding of the inner workings of neural network libraries.
    - Developers and researchers seeking to customize or extend the
      functionalities of existing deep learning libraries for their
      specific requirements.

The codebase is structured to be intuitive and mirrors the design
principles of PyTorch to a significant extent. Comprehensive docstrings
are provided for each module and function, ensuring clarity and ease of
understanding. Users are encouraged to delve into the code, experiment
with it, and modify it to suit their learning curve.

Contributions to this project are warmly welcomed. Whether it's refining
the code, enhancing the documentation, or extending the current feature
set, your input is highly valued. Feedback, whether constructive
criticism or commendation, is equally appreciated and will be
instrumental in the evolution of this educational tool.

This project is inspired by the remarkable work done by the PyTorch
development team. It is a tribute to their contributions to the field of
machine learning and the open-source community at large.

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2024 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

import typing as t

import numpy as np

from nanotorch import optim as optim
from nanotorch._tensor import from_numpy as from_numpy
from nanotorch._tensor import relu as relu
from nanotorch._tensor import tanh as tanh
from nanotorch._tensor import tensor as tensor
from nanotorch._types import FILE_LIKE as FILE_LIKE
from nanotorch._types import Data as Data
from nanotorch._types import NodesAndEdges as NodesAndEdges
from nanotorch.version import __version__ as __version__


def __getattr__(name: str) -> t.Any:
    """Delegate missing attributes to respective modules.

    This function is called when an attribute that does not exist in
    the ``nanotorch`` module is accessed.

    :param name: The name of the attribute being accessed.
    """
    try:
        attr = getattr(np, name)
        if isinstance(attr, np.dtype) or issubclass(attr, np.generic):
            return attr
    except AttributeError:
        pass
    raise AttributeError(f"module 'nanotorch' has no attribute {name!r}")
