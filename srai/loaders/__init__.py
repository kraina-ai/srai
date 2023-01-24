"""Loaders."""

from .base import BaseLoader
from .geoparquet_loader import GeoparquetLoader
from .osm_way_loader import OSMWayLoader

__all__ = ["BaseLoader", "GeoparquetLoader", "OSMWayLoader"]
