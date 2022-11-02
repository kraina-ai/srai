"""Loaders."""

from .base import BaseLoader
from .geoparquet_loader import GeoparquetLoader

__all__ = ["BaseLoader", "GeoparquetLoader"]
