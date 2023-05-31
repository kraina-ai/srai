"""This module contains utilities for conversion between H3 and Shapely."""
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import h3
from shapely import geometry

from srai.constants import WGS84_CRS



def shapely_polygon_to_h3(polygon: geometry.Polygon) -> h3.Polygon:
    """
    Convert Shapely Polygon to H3 Polygon.

    Args:
        polygon (geometry.Polygon): Shapely polygon to be converted.

    Returns:
        h3.Polygon: Converted polygon.
    """
    exterior = [coord[::-1] for coord in list(polygon.exterior.coords)]
    interiors = [
        [coord[::-1] for coord in list(interior.coords)] for interior in polygon.interiors
    ]
    return h3.Polygon(exterior, *interiors)

def gdf_from_h3_indexes(h3_indexes: List[str]) -> gpd.GeoDataFrame:
    """
    Convert H3 Indexes to GeoDataFrame with geometries.

    Args:
        h3_indexes (List[str]): H3 Indexes.

    Returns:
        gpd.GeoDataFrame: H3 cells.
    """
    return gpd.GeoDataFrame(
        None,
        index=h3_indexes,
        geometry=[h3_index_to_shapely_polygon(h3_index) for h3_index in h3_indexes],
        crs=WGS84_CRS,
    )

def gdf_from_df_with_h3(df: pd.DataFrame, h3_column: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Convert H3 Indexes to GeoDataFrame with geometries.

    Args:
        h3_indexes (List[str]): H3 Indexes.

    Returns:
        gpd.GeoDataFrame: H3 cells.
    """
    if h3_column is None:
        h3_indexes = df.index
    else:
        h3_indexes = df[h3_column]
    geometries = [h3_index_to_shapely_polygon(h3_index) for h3_index in df[h3_column]]

    return gpd.GeoDataFrame(
        df,
        index=h3_indexes,
        geometry=geometries,
        crs=WGS84_CRS,
    )


def h3_index_to_shapely_polygon(h3_index: str) -> geometry.Polygon:
    """
    Convert H3 Index to Shapely polygon.

    Args:
        h3_index (str): H3 Index to be converted.

    Returns:
        geometry.Polygon: Converted polygon.
    """
    return h3_polygon_to_shapely(h3.cells_to_polygons([h3_index])[0])

def h3_polygon_to_shapely(polygon: h3.Polygon) -> geometry.Polygon:
    """
    Convert H3 Polygon to Shapely Polygon.

    Args:
        polygon (h3.Polygon): H3 Polygon to be converted.

    Returns:
        geometry.Polygon: Converted polygon.
    """
    return geometry.Polygon(
        shell=[coord[::-1] for coord in polygon.outer],
        holes=[[coord[::-1] for coord in hole] for hole in polygon.holes],
    )
