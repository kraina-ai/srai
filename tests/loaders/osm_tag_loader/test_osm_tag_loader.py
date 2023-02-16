"""Tests for OSMTagLoader."""
from typing import Any, Dict, List, Union

import geopandas as gpd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from shapely.geometry import Point, Polygon

from srai.loaders.osm_tag_loader import OSMTagLoader
from srai.utils.constants import WGS84_CRS


@pytest.fixture  # type: ignore
def empty_area_gdf() -> gpd.GeoDataFrame:
    """Get empty OSMTagLoader result gdf."""
    return gpd.GeoDataFrame(crs=WGS84_CRS, geometry=[])


@pytest.fixture  # type: ignore
def empty_result_gdf() -> gpd.GeoDataFrame:
    """Get empty OSMTagLoader result gdf."""
    result_index = pd.MultiIndex.from_arrays(arrays=[[], []], names=["element_type", "osmid"])
    return gpd.GeoDataFrame(index=result_index, crs=WGS84_CRS, geometry=[])


@pytest.fixture  # type: ignore
def area_gdf() -> gpd.GeoDataFrame:
    """Get an example area gdf with with two polygons."""
    polygon_1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    polygon_2 = Polygon([(1, 1), (2, 2), (2, 1), (1, 0)])
    gdf = gpd.GeoDataFrame({"geometry": [polygon_1, polygon_2]}, crs=WGS84_CRS)
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
def mock_osmnx(mocker, area_gdf, amenities_gdf, building_gdf):
    """Monkeypatch `ox.geometries_from_polygon` to return data from predefined gdfs."""
    gdfs = {"amenity": amenities_gdf, "building": building_gdf}
    polygon_1, polygon_2 = area_gdf["geometry"]

    def mock_geometries_from_polygon(
        polygon: Polygon, tags: Dict[str, Union[List[str], str, bool]]
    ) -> gpd.GeoDataFrame:
        tag_key, tag_value = list(tags.items())[0]
        gdf = gdfs[tag_key]
        if tag_value is True:
            tag_res = gdf
        else:
            tag_res = gdf.loc[gdf[tag_key] == tag_value]
        if tag_res.empty:
            return tag_res
        if polygon == polygon_1:
            return tag_res.iloc[:1]
        elif polygon == polygon_2:
            return tag_res.iloc[1:]
        return None

    mocker.patch("osmnx.geometries_from_polygon", new=mock_geometries_from_polygon)


@pytest.fixture  # type: ignore
def expected_result_gdf_simple() -> gpd.GeoDataFrame:
    """Get the expected result of a simple query for testing."""
    return gpd.GeoDataFrame(
        {
            "amenity": ["restaurant", "restaurant"],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        index=pd.MultiIndex.from_arrays(
            arrays=[
                ["node", "node"],
                [1, 2],
            ],
            names=["element_type", "osmid"],
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
        index=pd.MultiIndex.from_arrays(
            arrays=[
                ["node", "node", "way"],
                [1, 2, 3],
            ],
            names=["element_type", "osmid"],
        ),
        crs=WGS84_CRS,
    )


@pytest.mark.parametrize(  # type: ignore
    "area_gdf_fixture,query,expected_result_gdf_fixture",
    [
        ("area_gdf", {"amenity": "restaurant"}, "expected_result_gdf_simple"),
        (
            "area_gdf",
            {"amenity": ["restaurant", "bar"], "building": True},
            "expected_result_gdf_complex",
        ),
        (
            "empty_area_gdf",
            {"amenity": ["restaurant", "bar"], "building": True},
            "empty_result_gdf",
        ),
    ],
)
def test_osm_tag_loader(
    area_gdf_fixture: str,
    query: Dict[str, Union[List[str], str, bool]],
    expected_result_gdf_fixture: str,
    request: Any,
):
    """Test `OSMTagLoader.load()`."""
    _ = request.getfixturevalue("mock_osmnx")
    area_gdf = request.getfixturevalue(area_gdf_fixture)
    expected_result_gdf = request.getfixturevalue(expected_result_gdf_fixture)
    loader = OSMTagLoader()
    res = loader.load(area_gdf, query)
    assert "address" not in res.columns
    assert_frame_equal(res, expected_result_gdf, check_like=True)
