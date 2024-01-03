"""Conftest for OSM loaders."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, WGS84_CRS


@pytest.fixture  # type: ignore
def empty_area_gdf() -> gpd.GeoDataFrame:
    """Get a gdf with no geometry."""
    return gpd.GeoDataFrame(crs=WGS84_CRS, geometry=[])


@pytest.fixture  # type: ignore
def area_with_no_objects_gdf() -> gpd.GeoDataFrame:
    """Get a gdf that contains no OSM objects."""
    return gpd.GeoDataFrame(crs=WGS84_CRS, geometry=[Polygon([(3, 5), (3, 10), (7, 10), (7, 5)])])


@pytest.fixture  # type: ignore
def empty_result_gdf() -> gpd.GeoDataFrame:
    """Get empty OSMOnlineLoader result gdf."""
    result_index = pd.Index(data=[], name=FEATURES_INDEX, dtype="object")
    return gpd.GeoDataFrame(index=result_index, crs=WGS84_CRS, geometry=[])


@pytest.fixture  # type: ignore
def single_polygon_area_gdf() -> gpd.GeoDataFrame:
    """Get an example area gdf with with one polygon."""
    polygon_1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [polygon_1]}, crs=WGS84_CRS)
    return gdf


@pytest.fixture  # type: ignore
def two_polygons_area_gdf() -> gpd.GeoDataFrame:
    """Get an example area gdf with with two polygons."""
    polygon_1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    polygon_2 = Polygon([(1, 1), (2, 2), (2, 1), (1, 0)])
    gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [polygon_1, polygon_2]}, crs=WGS84_CRS)
    return gdf


@pytest.fixture  # type: ignore
def amenities_gdf() -> gpd.GeoDataFrame:
    """Get an example gdf representing OSM objects with amenity tag."""
    return gpd.GeoDataFrame(
        {"amenity": ["restaurant", "restaurant", "bar"], "address": ["a", "b", "c"]},
        geometry=[Point(0, 0), Point(1, 1), Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        index=pd.MultiIndex.from_arrays(
            arrays=[
                ["node", "node", "way"],
                [1, 2, 3],
            ],
            names=["element_type", "osmid"],
        ),
        crs=WGS84_CRS,
    )


@pytest.fixture  # type: ignore
def building_gdf() -> gpd.GeoDataFrame:
    """Get an example gdf representing OSM objects with building tag."""
    return gpd.GeoDataFrame(
        {"building": ["commercial", "retail"], "address": ["a", "c"]},
        geometry=[Point(0, 0), Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        index=pd.MultiIndex.from_arrays(
            arrays=[
                ["node", "way"],
                [1, 3],
            ],
            names=["element_type", "osmid"],
        ),
        crs=WGS84_CRS,
    )


@pytest.fixture  # type: ignore
def expected_result_single_polygon() -> gpd.GeoDataFrame:
    """Get the expected result of a query with single polygon for testing."""
    return gpd.GeoDataFrame(
        {
            "amenity": ["restaurant"],
        },
        geometry=[Point(0, 0)],
        index=pd.Index(
            data=[
                "node/1",
            ],
            name=FEATURES_INDEX,
            dtype="object",
        ),
        crs=WGS84_CRS,
    )


@pytest.fixture  # type: ignore
def expected_result_gdf_simple() -> gpd.GeoDataFrame:
    """Get the expected result of a simple query for testing."""
    return gpd.GeoDataFrame(
        {
            "amenity": ["restaurant", "restaurant"],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        index=pd.Index(
            data=[
                "node/1",
                "node/2",
            ],
            name=FEATURES_INDEX,
            dtype="object",
        ),
        crs=WGS84_CRS,
    )


@pytest.fixture  # type: ignore
def expected_result_gdf_complex() -> gpd.GeoDataFrame:
    """Get the expected result of complex query for testing."""
    return gpd.GeoDataFrame(
        {
            "amenity": ["restaurant", "restaurant", "bar"],
            "building": ["commercial", None, "retail"],
        },
        geometry=[Point(0, 0), Point(1, 1), Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        index=pd.Index(
            data=[
                "node/1",
                "node/2",
                "way/3",
            ],
            name=FEATURES_INDEX,
            dtype="object",
        ),
        crs=WGS84_CRS,
    )
