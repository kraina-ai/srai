"""Loaders."""

from ._base import Loader
from .geoparquet_loader import GeoparquetLoader
from .gtfs_loader import GTFSLoader
from .osm_way_loader import OSMWayLoader

__all__ = ["Loader", "GeoparquetLoader", "GTFSLoader", "OSMWayLoader"]
