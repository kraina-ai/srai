"""
This module contains regionizers, which are used to divide space before analysis.

Embedding methods available in `srai` operate on a regions, which can be defined in many ways. In
this module, we aggregate different regionization methods under a common `Regionizer` interface. We
include both pre-defined spatial indexes (e.g. H3 or S2), data-based ones (e.g. Voronoi) and OSM-
based ones (e.g. administrative boundaries).
"""

from ._base import Regionizer
from .administrative_boundary_regionizer import AdministrativeBoundaryRegionizer
from .h3_regionizer import H3Regionizer
from .s2_regionizer import S2Regionizer
from .slippy_map_regionizer import SlippyMapRegionizer
from .voronoi_regionizer import VoronoiRegionizer

__all__ = [
    "Regionizer",
    "AdministrativeBoundaryRegionizer",
    "H3Regionizer",
    "S2Regionizer",
    "VoronoiRegionizer",
    "SlippyMapRegionizer",
]
