"""\
NanoTorch Exceptions API
========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, December 02 2023
Last updated on: Wednesday, December 06 2023

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2023 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

import typing as t

from nanotorch.logger import get_logger

__all__ = ["NanoTorchException"]

_logger = get_logger(__name__)


class NanoTorchException(Exception):
    """Base class for capturing all the exceptions raised by the
    NanoTorch module.

    This exception class serves as the primary entrypoint for capturing
    and logging exceptions related to all the operations.

    :var _description: A human-readable description or message
                       explaining the reason for the exception.
    """

    _description: str

    def __init__(self, description: str | None, *args: t.Any) -> None:
        if description:
            self._description = description
        _logger.error(self._description)
        super().__init__(self._description, *args)
