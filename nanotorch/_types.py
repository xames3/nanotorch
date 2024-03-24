"""\
NanoTorch Types
===============

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, December 09 2023
Last updated on: Wednesday, March 13 2024

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2024 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

import os
import typing as t

import numpy as np

FILE_LIKE: t.TypeAlias = str | os.PathLike | t.BinaryIO | t.IO[bytes]

Data = np.ndarray[t.Any, np.dtype[t.Any]]
NodesAndEdges = tuple[set[t.Any], ...]
Size = t.Sequence[int]
