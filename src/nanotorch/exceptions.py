"""\
NanoTorch Exceptions API
========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, December 02 2023
Last updated on: Wednesday, December 13 2023

All about Exceptions

In the realm of our miniature PyTorch project, the ``exceptions`` module
plays a pivotal role in defining and managing custom exceptions. This
module is a cornerstone for robust error handling, ensuring that the
library is not only instructive but also resilient and user-friendly.

The module's primary purpose is to encapsulate all the custom exceptions
that could arise during the operation of our miniature neural network
library. These exceptions are meticulously crafted to provide clear and
informative error messages, aiding users in quickly diagnosing and
rectifying issues.

Each exception class within this module is tailored to specific error
conditions that might occur in tensor operations, initialization, or
computational processes. Emphasis is placed on providing descriptive and
actionable error messages. This ensures that when an exception is
raised, it conveys the essence of the issue succinctly yet
comprehensively. By having dedicated exception classes, users can easily
pinpoint the source of errors, making the debugging process more
efficient and educational. These exceptions are seamlessly integrated
into the main library, ensuring that error handling is consistent and
logical across different modules.

Understanding error handling is crucial in software development, and
even more so in complex domains like machine learning.
The ``exceptions`` module serves as an educational tool in itself,
demonstrating how custom exceptions can be implemented and utilized
effectively in a library:

    - Beginners gain exposure to practical aspects of error handling in
      Python.
    - Advanced users can appreciate the nuances of creating meaningful
      and helpful custom exceptions.

The exceptions defined in this module are intended to be both
self-explanatory and instructive. Users are encouraged to explore the
various exception classes and understand the scenarios under which they
are raised. This exploration will enhance the user's ability to write
robust code and handle errors gracefully.

As an integral part of our educational PyTorch implementation, the
``exceptions`` module contributes significantly to the project's overall
robustness and user-friendliness. It exemplifies best practices in
exception handling tailored to the specific needs of a neural network
library.

We welcome contributions to this module, whether they involve refining
existing exceptions, adding new ones, or improving documentation. Your
insights and improvements are invaluable in making this module a more
effective educational resource.

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2024 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

import typing as t

from nanotorch.logger import get_logger

_logger = get_logger(__name__)


class NanoTorchException(Exception):
    """Base class for capturing all the exceptions raised by the
    NanoTorch module.

    This exception class serves as the primary entrypoint for capturing
    and logging exceptions related to all the operations.

    :param description: A human-readable description or message
                        explaining the reason for the exception.
    """

    def __init__(self, description: str | None, *args: t.Any) -> None:
        _logger.error(description)
        super().__init__(description, *args)


class IncorrectRoundingModeError(NanoTorchException):
    """Error to be raised when incorrect rounding mode is passed."""
