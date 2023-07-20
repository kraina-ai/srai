"""
Module containing different utility functions.

Those are either used internally by other modules, or can be used to simplify spatial data
processing.
"""
from .geometry import buffer_geometry, flatten_geometry, flatten_geometry_series, remove_interiors
from .merge import merge_disjointed_gdf_geometries, merge_disjointed_polygons

__all__ = [
    "buffer_geometry",
    "flatten_geometry",
    "flatten_geometry_series",
    "remove_interiors",
    "merge_disjointed_polygons",
    "merge_disjointed_gdf_geometries",
]
