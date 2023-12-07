"""\
NanoTorch Utilities API
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, December 03 2023
Last updated on: Wednesday, December 06 2023

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2023 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

import random

__all__ = ["Generator", "colors"]


class Generator:
    """Creates and returns a generator object that manages the state of
    the algorithm which produces pseudo random numbers.

    This class is a wrapper around Python's ``random.Random`` class,
    providing an interface to generate random numbers. It can be
    initialized with an optional seed to ensure reproducibility.
    """

    def __init__(self) -> None:
        """Constructor, called when a new generator is created.

        This will initialize a new ``Generator`` object with no seed.
        """
        self.seed = None

    def manual_seed(self, seed: int) -> int:
        """Sets the seed for generating random numbers.

        :param seed: An optional seed for the random number generator,
                     which initializes the generator in a random state.
        """
        self.seed = seed
        return random.seed(self.seed)


colors = [
    "lightblue",
    "lightblue1",
    "lightblue2",
    "lightblue3",
    "lightblue4",
    "lightcoral",
    "lightcyan",
    "lightcyan1",
    "lightcyan2",
    "lightcyan3",
    "lightcyan4",
    "lightgoldenrod",
    "lightgoldenrod1",
    "lightgoldenrod2",
    "lightgoldenrod3",
    "lightgoldenrod4",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgreen",
    "lightgrey",
    "lightpink",
    "lightpink1",
    "lightpink2",
    "lightpink3",
    "lightpink4",
    "lightsalmon",
    "lightsalmon1",
    "lightsalmon2",
    "lightsalmon3",
    "lightsalmon4",
    "lightseagreen",
    "lightskyblue",
    "lightskyblue1",
    "lightskyblue2",
    "lightskyblue3",
    "lightskyblue4",
    "lightslateblue",
    "lightslategray",
    "lightslategrey",
    "lightsteelblue",
    "lightsteelblue1",
    "lightsteelblue2",
    "lightsteelblue3",
    "lightsteelblue4",
    "lightyellow",
    "lightyellow1",
    "lightyellow2",
    "lightyellow3",
    "lightyellow4",
]
