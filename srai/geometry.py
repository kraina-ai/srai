"""Utility geometry operations functions."""

import hashlib
from collections.abc import Iterable
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
import pyproj
import shapely.wkt as wktlib
from functional import seq
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from srai.constants import (
    FEATURES_INDEX,
    FEATURES_INDEX_TYPE,
    REGIONS_INDEX,
    REGIONS_INDEX_TYPE,
    WGS84_CRS,
)

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


def convert_to_regions_gdf(
    geometry: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
    index_column: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Convert any geometry to a regions GeoDataFrame.

    Args:
        geometry (Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame]): Geo
            objects to convert.
        index_column (Optional[str], optional): Name of the column used to define the index.
            If None, will rename the existing index. Defaults to None.

    Returns:
        gpd.GeoDataFrame: Regions gdf with proper index definition.
    """
    return _convert_to_internal_format(
        geometry=geometry, destination_index_name=REGIONS_INDEX, index_column=index_column
    )


def convert_to_features_gdf(
    geometry: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
    index_column: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Convert any geometry to a features GeoDataFrame.

    Args:
        geometry (Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame]): Geo
            objects to convert.
        index_column (Optional[str], optional): Name of the column used to define the index.
            If None, will rename the existing index. Defaults to None.

    Returns:
        gpd.GeoDataFrame: Features gdf with proper index definition.
    """
    return _convert_to_internal_format(
        geometry=geometry, destination_index_name=FEATURES_INDEX, index_column=index_column
    )


def _convert_to_internal_format(
    geometry: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
    destination_index_name: Union[REGIONS_INDEX_TYPE, FEATURES_INDEX_TYPE],
    index_column: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Converts geometry into internal format with a proper index name."""
    if isinstance(geometry, gpd.GeoDataFrame):
        # Return a GeoDataFrame with changed index
        if isinstance(geometry.index, pd.MultiIndex):
            raise ValueError(
                "Cannot transform index of type pandas.MultiIndex. Please reset the index first."
            )

        if index_column is not None:
            if index_column not in geometry.columns:
                raise ValueError(f"Column {index_column} does not exist")

            geometry = geometry.set_index(index_column)

        geometry.index = geometry.index.rename(destination_index_name)

        if geometry.crs is None:
            geometry = geometry.set_crs(WGS84_CRS)
        else:
            geometry = geometry.to_crs(WGS84_CRS)

        return geometry
    elif isinstance(geometry, gpd.GeoSeries):
        # Create a GeoDataFrame with GeoSeries
        return _convert_to_internal_format(
            gpd.GeoDataFrame(geometry=geometry),
            destination_index_name=destination_index_name,
            index_column=index_column,
        )
    elif isinstance(geometry, Iterable):
        # Create a GeoSeries with a list of geometries
        return _convert_to_internal_format(
            gpd.GeoSeries(geometry),
            destination_index_name=destination_index_name,
            index_column=index_column,
        )
    # Wrap a single geometry with a list
    return _convert_to_internal_format(
        [geometry], destination_index_name=destination_index_name, index_column=index_column
    )
