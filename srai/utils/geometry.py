"""Utility geometry operations functions."""
from typing import List

import geopandas as gpd
import pyproj
from functional import seq
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import transform as shapely_transform


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
    """
    Close polygon holes by limitation to the exterior ring.

    Args:
        polygon (Polygon): Polygon to close.

    Returns:
        Polygon: Closed polygon.
    """
    if polygon.interiors:
        return Polygon(list(polygon.exterior.coords))
    return polygon


def buffer_geometry(geometry: BaseGeometry, meters: float) -> BaseGeometry:
    """
    Buffer geometry by a given radius in meters.

    Projects geometry into azimuthal projection before applying buffer and then changes values
    back to WGS84 coordinates.

    Args:
        geometry (BaseGeometry): Geometry to buffer.
        meters (float): Radius distance in meters.

    Returns:
        BaseGeometry: Buffered geometry.
    """
    _lon, _lat = geometry.centroid.coords[0]

    aeqd_proj = pyproj.Proj(proj="aeqd", ellps="WGS84", datum="WGS84", lat_0=_lat, lon_0=_lon)
    wgs84_proj = pyproj.Proj(proj="latlong", ellps="WGS84")

    wgs84_to_aeqd = pyproj.Transformer.from_proj(wgs84_proj, aeqd_proj, always_xy=True).transform
    aeqd_to_wgs84 = pyproj.Transformer.from_proj(aeqd_proj, wgs84_proj, always_xy=True).transform

    projected_geometry = shapely_transform(wgs84_to_aeqd, geometry)
    bufferred_projected_geometry = projected_geometry.buffer(meters)

    return shapely_transform(aeqd_to_wgs84, bufferred_projected_geometry)
