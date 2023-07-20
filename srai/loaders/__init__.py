"""
This module contains loaders, used to load spatial data from different sources.

We want to unify loading from different data sources into a single interface. Thanks to this, we
have a unified spatial data format, which makes it possible to feed them into any of the embedding
methods available in this library.
"""

from ._base import Loader
from .download import download_file
from .geoparquet_loader import GeoparquetLoader
from .gtfs_loader import GTFSLoader
from .osm_loaders import OSMLoader, OSMOnlineLoader, OSMPbfLoader, OSMTileLoader
from .osm_way_loader import OSMNetworkType, OSMWayLoader

__all__ = [
    "Loader",
    "GeoparquetLoader",
    "GTFSLoader",
    "OSMLoader",
    "OSMWayLoader",
    "OSMOnlineLoader",
    "OSMPbfLoader",
    "OSMTileLoader",
    "OSMNetworkType",
    "download_file",
]
