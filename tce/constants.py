r"""
This module defines useful cutoffs for convenience.
"""

from typing import Literal
import logging

import numpy as np
from numpy.typing import NDArray


LOGGER = logging.getLogger(__name__)

CUTOFFS: dict[Literal["sc", "bcc", "fcc", "graphene"], NDArray[np.floating]] = {
    "sc": np.array([1.0, np.sqrt(2.0), np.sqrt(3.0), 2.0,
                    np.sqrt(5.0), np.sqrt(6.0), 2.0 * np.sqrt(2.0), 3.0,
                    np.sqrt(10.0), np.sqrt(11.0), 2 * np.sqrt(3.0), np.sqrt(13.0),
                    np.sqrt(14.0), 4.0, np.sqrt(17.0), 3.0 * np.sqrt(2.0)]),
    "bcc": np.array([0.5 * np.sqrt(3.0), 1.0, np.sqrt(2.0), 0.5 * np.sqrt(11.0),
                    np.sqrt(3.0), 2.0, 0.5 * np.sqrt(19.0), np.sqrt(5.0), np.sqrt(6.0),
                    1.5 * np.sqrt(3.0), 2.0 * np.sqrt(2.0), 0.5 * np.sqrt(35.0),
                    3.0, 0.5 * np.sqrt(43.0), 2.0 * np.sqrt(3.0), 0.5 * np.sqrt(51.0)]),
    "fcc": np.array([0.5 * np.sqrt(2.0), 1.0, np.sqrt(1.5), np.sqrt(2.0), np.sqrt(2.5), 
                    np.sqrt(3.0), np.sqrt(3.5), 2.0, 1.5 * np.sqrt(2.0), np.sqrt(5.0),
                    np.sqrt(0.5 * 11.0), np.sqrt(6.0), np.sqrt(0.5 * 13.0), np.sqrt(0.5 * 15.0),
                    2.0 * np.sqrt(2.0), np.sqrt(0.5 * 17.0), 3.0, np.sqrt(0.5 * 19.0)]),
    "graphene": np.array([1.0 / np.sqrt(3.0), 1.0, 2.0 / np.sqrt(3.0), np.sqrt(7.0 / 3.0),
                    np.sqrt(3.0), 2.0, np.sqrt(13.0 / 3.0), 4.0 / np.sqrt(3.0)
                    ])
}
r"""Mapping from lattice structure to neighbor cutoffs, in units of the lattice parameter $a$"""
