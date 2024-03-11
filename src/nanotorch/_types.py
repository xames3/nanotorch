"""\
NanoTorch Types
===============

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, December 09 2023
Last updated on: Saturday, December 09 2024

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2024 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

import os
import typing as t

FILE_LIKE: t.TypeAlias = str | os.PathLike | t.BinaryIO | t.IO[bytes]
Colors = tuple[str, ...]
Number = int | float | bool
Size = t.Sequence[int]
