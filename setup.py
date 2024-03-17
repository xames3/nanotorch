"""\
NanoTorch Setup
===============

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, December 02 2023
Last updated on: Saturday, December 02 2023

This will install the ``nanotorch`` package in a python 3.10+
environment. Before proceeding, please ensure you have a virtual
environment setup & running.

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2024 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

import platform
import site
import sys

from pkg_resources import parse_version

try:
    import setuptools
except ImportError:
    raise RuntimeError(
        "Could not install package in the environment as setuptools is "
        "missing. Please create a new virtual environment before proceeding."
    )

_min_py_version: str = "3.10"
_current_py_version: str = platform.python_version()

if parse_version(_current_py_version) < parse_version(_min_py_version):
    raise SystemExit(
        "Could not install nanotorch! It requires python version 3.10+, "
        f"you are using {_current_py_version}..."
    )

# BUG: Cannot install into user directory with editable source.
# Using this solution: https://stackoverflow.com/a/68487739/14316408
# to solve the problem with installation. As of October, 2022 the issue
# is still open on GitHub: https://github.com/pypa/pip/issues/7953.

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

if __name__ == "__main__":
    setuptools.setup()
