"""Tests for OSMWayLoader."""
from contextlib import nullcontext as does_not_raise
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import pandas as pd
import pytest
import shapely.geometry as shpg
from pytest_check import check

from srai.loaders.exceptions import LoadedDataIsEmptyException
from srai.loaders.osm_way_loader import NetworkType, OSMWayLoader
from srai.utils.constants import WGS84_CRS


@pytest.fixture  # type: ignore
def empty_area_gdf() -> gpd.GeoDataFrame:
    """Get a gdf with no geometry."""
    return gpd.GeoDataFrame(crs=WGS84_CRS, geometry=[])


@pytest.fixture  # type: ignore
def no_crs_area_gdf() -> gpd.GeoDataFrame:
    """Get an example area gdf without crs defined."""
    polygon = shpg.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    gdf = gpd.GeoDataFrame({"geometry": [polygon]})
    return gdf


@pytest.fixture  # type: ignore
def bad_geometry_area_gdf() -> gpd.GeoDataFrame:
    """Get an example area gdf with unsupported geometry (not Polygon or Multipolygon)."""
    point = shpg.Point(0, 0)
    gdf = gpd.GeoDataFrame({"geometry": [point]}, crs=WGS84_CRS)
    return gdf


@pytest.fixture  # type: ignore
def empty_polygon_area_gdf() -> gpd.GeoDataFrame:
    """Get an example area gdf where there is no road infrastructure."""
    polygon = shpg.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs=WGS84_CRS)
    return gdf


@pytest.fixture  # type: ignore
def first_polygon_area_gdf() -> gpd.GeoDataFrame:
    """Get an example area gdf with one polygon."""
    polygon = shpg.Polygon(
        [
            (17.1005309, 51.1100158),
            (17.1020436, 51.1100427),
            (17.1021938, 51.1082509),
            (17.1006274, 51.1081027),
            (17.1005201, 51.1099956),
        ]
    )

    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs=WGS84_CRS)
    return gdf


@pytest.fixture  # type: ignore
def second_polygon_area_gdf() -> gpd.GeoDataFrame:
    """Get an example area gdf with one polygon."""
    polygon = shpg.Polygon(
        [
            (17.0994473, 51.1084126),
            (17.1023226, 51.1086551),
            (17.1023333, 51.1076312),
            (17.0994473, 51.1083722),
        ]
    )

    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs=WGS84_CRS)
    return gdf


@pytest.fixture  # type: ignore
def multiple_polygons_overlaping_area_gdf(
    first_polygon_area_gdf: gpd.GeoDataFrame, second_polygon_area_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Get an example area gdf with two polygons."""
    return pd.concat([first_polygon_area_gdf, second_polygon_area_gdf], axis=0)


@pytest.fixture  # type: ignore
def multipolygons_area_gdf() -> gpd.GeoDataFrame:
    """Get an example area gdf with a multipolygon."""
    polygon1 = shpg.Polygon(
        [
            (17.1005309, 51.1100158),
            (17.1020436, 51.1100427),
            (17.1021938, 51.1082509),
            (17.1006274, 51.1081027),
            (17.1005201, 51.1099956),
        ]
    )
    polygon2 = shpg.Polygon(
        [
            (17.0997584, 51.1049434),
            (17.0995009, 51.1044112),
            (17.1003485, 51.1043910),
            (17.0997584, 51.1049434),
        ]
    )
    multipolygon = shpg.MultiPolygon([polygon1, polygon2])
    return gpd.GeoDataFrame(geometry=[multipolygon], crs=WGS84_CRS)


@pytest.fixture  # type: ignore
def valid_and_empty_polygons_area_gdf(
    first_polygon_area_gdf: gpd.GeoDataFrame, empty_polygon_area_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Get an example area gdf with one valid polygon and one without road infrastructure."""
    return pd.concat([first_polygon_area_gdf, empty_polygon_area_gdf], axis=0)


@pytest.mark.parametrize(  # type: ignore
    "area_gdf_fixture,expected_result,loader_params,expectation",
    [
        (
            "empty_area_gdf",
            None,
            None,
            pytest.raises(ValueError),
        ),
        (
            "no_crs_area_gdf",
            None,
            None,
            pytest.raises(ValueError),
        ),
        (
            "bad_geometry_area_gdf",
            None,
            None,
            pytest.raises(TypeError),
        ),
        (
            "empty_polygon_area_gdf",
            None,
            None,
            pytest.raises(LoadedDataIsEmptyException),
        ),
        (
            "first_polygon_area_gdf",
            (7, 6),
            None,
            does_not_raise(),
        ),
        (
            "multiple_polygons_overlaping_area_gdf",
            (7, 6),
            None,
            does_not_raise(),
        ),
        (
            "multiple_polygons_overlaping_area_gdf",
            (9, 7),  # FIXME: shouldn't contain a node without an edge
            {"network_type": NetworkType.BIKE, "contain_within_area": True},
            does_not_raise(),
        ),
        (
            "multipolygons_area_gdf",
            (11, 9),
            None,
            does_not_raise(),
        ),
        (
            "valid_and_empty_polygons_area_gdf",
            (7, 6),
            None,
            does_not_raise(),
        ),
    ],
)
def test_osm_way_loader(
    area_gdf_fixture: str,
    expected_result: Optional[Tuple[int, int]],
    loader_params: Optional[Dict[str, Any]],
    expectation,
    request: pytest.FixtureRequest,
):
    """Test `OSMWayLoader.load()`."""
    if not loader_params:
        loader_params = {"network_type": NetworkType.DRIVE}

    loader = OSMWayLoader(**loader_params)

    area_gdf = request.getfixturevalue(area_gdf_fixture)
    nodes_expected_len, edges_expected_len = expected_result if expected_result else (None, None)

    with expectation:
        nodes, edges = loader.load(area_gdf)
        check.equal(len(nodes), nodes_expected_len)
        check.equal(len(edges), edges_expected_len)
        check.is_in("geometry", nodes.columns)
        check.is_in("geometry", edges.columns)
