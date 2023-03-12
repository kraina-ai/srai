"""Utilities."""

from .download import download_file
from .merge import _merge_disjointed_gdf_geometries, _merge_disjointed_polygons

__all__ = [
    "_merge_disjointed_gdf_geometries",
    "_merge_disjointed_polygons",
    "download_file",
]
