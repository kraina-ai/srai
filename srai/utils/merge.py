"""Utility function for merging Shapely polygons."""

from typing import List, Union

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon


def _merge_disjointed_polygons(polygons: List[Union[Polygon, MultiPolygon]]) -> MultiPolygon:
    """
    Merges all polygons into a single MultiPolygon.

    Input polygons are expected to be disjointed.

    Args:
        polygons: List of polygons to merge

    Returns:
        MultiPolygon: Merged polygon
    """
    single_polygons = []
    for geom in polygons:
        if type(geom) is Polygon:
            single_polygons.append(geom)
        else:
            single_polygons.extend(geom.geoms)
    return MultiPolygon(single_polygons)


def _merge_disjointed_gdf_geometries(gdf: gpd.GeoDataFrame) -> MultiPolygon:
    """
    Merges geometries from a GeoDataFrame into a single MultiPolygon.

    Input geometries are expected to be disjointed.

    Args:
        gdf: GeoDataFrame with geometries to merge.

    Returns:
        MultiPolygon: Merged polygon
    """
    return _merge_disjointed_polygons(list(gdf.geometry))
