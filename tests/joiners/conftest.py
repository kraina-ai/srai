"""Conftest for Joiners."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely import geometry

from srai.constants import FEATURES_INDEX, REGIONS_INDEX, WGS84_CRS


@pytest.fixture  # type: ignore
def no_geometry_gdf() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame."""
    return gpd.GeoDataFrame()


@pytest.fixture  # type: ignore
def empty_gdf() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with no geometry."""
    return gpd.GeoDataFrame(geometry=[])


@pytest.fixture  # type: ignore
def regions_gdf() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with example regions."""
    regions = gpd.GeoDataFrame(
        geometry=[
            geometry.Polygon([(-1, 0), (-1, -1), (0, -1), (0, 0)]),
            geometry.Polygon([(1, 0), (1, 1), (0, 1), (0, 0)]),
            geometry.Polygon([(-2, -1), (-2, -2), (-1, -2), (-1, -1)]),
            geometry.Polygon([(-2, 0.5), (-2, -0.5), (-1, -0.5), (-1, 0.5)]),
        ],
        crs=WGS84_CRS,
    )
    return regions


@pytest.fixture  # type: ignore
def features_gdf() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with example features."""
    features = gpd.GeoDataFrame(
        [1, 2, 3, 4],
        geometry=[
            geometry.Polygon([(-1.5, 0.5), (-1.5, 0), (-0.5, 0), (-0.5, 0.5)]),
            geometry.Polygon([(-1.5, -1.5), (-1.5, -2.5), (-0.5, -2.5), (-0.5, -1.5)]),
            geometry.Point((0, 0)),
            geometry.Point((-0.5, -0.5)),
        ],
        crs=WGS84_CRS,
    )
    return features


@pytest.fixture  # type: ignore
def joint_multiindex() -> pd.MultiIndex:
    """Get MultiIndex for joint GeoDataFrame."""
    return pd.MultiIndex.from_tuples(
        [(0, 2), (0, 3), (1, 2), (0, 0), (3, 0), (2, 1)], names=[REGIONS_INDEX, FEATURES_INDEX]
    )
