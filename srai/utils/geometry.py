"""DOCSTRING TODO."""
from typing import List

import geopandas as gpd
from functional import seq
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry


def flatten_geometry_series(geometry_series: gpd.GeoSeries) -> List[BaseGeometry]:
    """Flatten all geometries from a series into a list of BaseGeometries."""
    geometries: List[BaseGeometry] = (
        seq([flatten_geometry(geometry) for geometry in geometry_series]).flatten().to_list()
    )
    return geometries


def flatten_geometry(geometry: BaseGeometry) -> List[BaseGeometry]:
    """Flatten all geometries into a list of BaseGeometries."""
    if isinstance(geometry, BaseMultipartGeometry):
        geometries: List[BaseGeometry] = (
            seq([flatten_geometry(sub_geom) for sub_geom in geometry.geoms]).flatten().to_list()
        )
        return geometries
    return [geometry]


# https://stackoverflow.com/a/70387141/7766101
def remove_interiors(polygon: Polygon) -> Polygon:
    """Close polygon holes by limitation to the exterior ring."""
    if polygon.interiors:
        return Polygon(list(polygon.exterior.coords))
    return polygon
