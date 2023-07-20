"""
Module containing different utility functions.

Those are either used internally by other modules, or can be used to simplify spatial data
processing.
"""
from .geometry import buffer_geometry, flatten_geometry, flatten_geometry_series, remove_interiors

__all__ = [
    "buffer_geometry",
    "flatten_geometry",
    "flatten_geometry_series",
    "remove_interiors",
]
