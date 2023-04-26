"""
This module contains joiners, used to match spatial data with regions used in analysis.

All methods for spatial embedding (available in `srai`) are designed to operate on regions, not on
individual spatial features. This means that we need to match spatial features with our given
regions. This can be done in different ways, all of which are available under a common `Joiner`
interface.
"""

from ._base import Joiner
from .intersection_joiner import IntersectionJoiner

__all__ = [
    "Joiner",
    "IntersectionJoiner",
]
