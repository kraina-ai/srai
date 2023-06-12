"""Utility geometry operations functions."""
from typing import List

import geopandas as gpd
import pyproj
import topojson as tp
from functional import seq
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid

from srai.constants import WGS84_CRS

SIMPLIFICATION_TOLERANCE_VALUES = [
    1e-07,
    2e-07,
    5e-07,
    1e-06,
    2e-06,
    5e-06,
    1e-05,
    2e-05,
    5e-05,
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
]


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


def simplify_polygon_with_buffer(polygon: Polygon) -> Polygon:
    """
    Prepare polygon for download.

    Function buffers the polygon, closes internal holes and simplifies its boundary to 1000 points.

    Makes sure that the generated polygon with fully cover the original one by increasing the buffer
    size incrementally.
    """
    is_fully_covered = False
    buffer_size_meters = 50
    while not is_fully_covered:
        buffered_polygon = buffer_geometry(polygon, meters=buffer_size_meters)
        simplified_polygon = simplify_polygon(buffered_polygon)
        closed_polygon = remove_interiors(simplified_polygon)
        is_fully_covered = polygon.covered_by(closed_polygon)
        buffer_size_meters += 50
    return closed_polygon


def simplify_polygon(polygon: Polygon) -> Polygon:
    """Simplify a polygon boundary to up to 1000 points."""
    simplified_polygon = polygon

    for simplify_tolerance in SIMPLIFICATION_TOLERANCE_VALUES:
        simplified_polygon = (
            tp.Topology(
                polygon,
                toposimplify=simplify_tolerance,
                prevent_oversimplify=True,
            )
            .to_gdf(winding_order="CW_CCW", crs=WGS84_CRS, validate=True)
            .geometry[0]
        )
        simplified_polygon = make_valid(simplified_polygon)
        if len(simplified_polygon.exterior.coords) < 1000:
            break

    if len(simplified_polygon.exterior.coords) > 1000:
        simplified_polygon = polygon.convex_hull

    if len(simplified_polygon.exterior.coords) > 1000:
        simplified_polygon = polygon.minimum_rotated_rectangle

    return simplified_polygon
