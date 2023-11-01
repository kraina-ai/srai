"""
This module contains regionalizers, which are used to divide space before analysis.

Embedding methods available in `srai` operate on a regions, which can be defined in many ways. In
this module, we aggregate different regionalization methods under a common `Regionalizer` interface.
We include both pre-defined spatial indexes (e.g. H3 or S2), data-based ones (e.g. Voronoi) and OSM-
based ones (e.g. administrative boundaries).
"""

from ._base import Regionalizer
from .administrative_boundary_regionalizer import AdministrativeBoundaryRegionalizer
from .geocode import geocode_to_region_gdf
from .h3_regionalizer import H3Regionalizer
from .s2_regionalizer import S2Regionalizer
from .slippy_map_regionalizer import SlippyMapRegionalizer
from .voronoi_regionalizer import VoronoiRegionalizer

__all__ = [
    "Regionalizer",
    "AdministrativeBoundaryRegionalizer",
    "H3Regionalizer",
    "S2Regionalizer",
    "VoronoiRegionalizer",
    "SlippyMapRegionalizer",
    "geocode_to_region_gdf",
]
