"""Utility geometry operations functions."""

import hashlib
from typing import Union

import geopandas as gpd
import pyproj
import shapely.wkt as wktlib
from functional import seq
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

__all__ = [
    "flatten_geometry_series",
    "flatten_geometry",
    "remove_interiors",
    "buffer_geometry",
    "merge_disjointed_polygons",
    "merge_disjointed_gdf_geometries",
]


def flatten_geometry_series(geometry_series: gpd.GeoSeries) -> list[BaseGeometry]:
    """Flatten all geometries from a series into a list of BaseGeometries."""
    geometries: list[BaseGeometry] = seq(geometry_series).flat_map(flatten_geometry).to_list()
    return geometries


def flatten_geometry(geometry: BaseGeometry) -> list[BaseGeometry]:
    """Flatten all geometries into a list of BaseGeometries."""
    if isinstance(geometry, BaseMultipartGeometry):
        geometries: list[BaseGeometry] = seq(geometry.geoms).flat_map(flatten_geometry).to_list()
        return geometries
    return [geometry]


# https://stackoverflow.com/a/70387141/7766101
def remove_interiors(geometry: Union[Polygon, MultiPolygon]) -> Polygon:
    """
    Close polygon holes by limitation to the exterior ring.

    Args:
        geometry (Union[Polygon, MultiPolygon])): Polygon to close.

    Returns:
        Polygon: Closed polygon.
    """
    if isinstance(geometry, MultiPolygon):
        return unary_union([remove_interiors(sub_polygon) for sub_polygon in geometry.geoms])
    if geometry.interiors:
        return Polygon(list(geometry.exterior.coords))
    return geometry


def buffer_geometry(geometry: BaseGeometry, meters: float) -> BaseGeometry:
    """
    Buffer geometry by a given radius in meters.

    Projects geometry into azimuthal projection before applying buffer and then changes values
    back to WGS84 coordinates.

    Doesn't work with polygons covering the whole earth (from -180 to 180 longitude).

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


def merge_disjointed_polygons(polygons: list[Union[Polygon, MultiPolygon]]) -> MultiPolygon:
    """
    Merges all polygons into a single MultiPolygon.

    Input polygons are expected to be disjointed.

    Args:
        polygons (List[Union[Polygon, MultiPolygon]]): List of polygons to merge

    Returns:
        MultiPolygon: Merged polygon
    """
    single_polygons = []
    for geom in polygons:
        if isinstance(geom, Polygon):
            single_polygons.append(geom)
        else:
            single_polygons.extend(geom.geoms)
    return MultiPolygon(single_polygons)


def merge_disjointed_gdf_geometries(gdf: gpd.GeoDataFrame) -> MultiPolygon:
    """
    Merges geometries from a GeoDataFrame into a single MultiPolygon.

    Input geometries are expected to be disjointed.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with geometries to merge.

    Returns:
        MultiPolygon: Merged polygon
    """
    return merge_disjointed_polygons(list(gdf.geometry))


def get_geometry_hash(geometry: BaseGeometry) -> str:
    """Generate SHA256 hash based on WKT representation of the polygon."""
    wkt_string = wktlib.dumps(geometry)
    h = hashlib.new("sha256")
    h.update(wkt_string.encode())
    return h.hexdigest()
