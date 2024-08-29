"""Tests for OSMWayLoader."""

import pickle as pkl
from collections.abc import Sequence
from contextlib import nullcontext as does_not_raise
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union
from unittest import TestCase

import geopandas as gpd
import numpy as np
import osmnx
import pandas as pd
import pytest
import shapely.geometry as shpg
from parametrization import Parametrization as P
from pytest_check import check
from pytest_mock import MockerFixture

from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.exceptions import LoadedDataIsEmptyException
from srai.loaders import OSMNetworkType, OSMWayLoader

ut = TestCase()


@pytest.fixture  # type: ignore
def empty_area_gdf() -> gpd.GeoDataFrame:
    """Get a gdf with no geometry."""
    return gpd.GeoDataFrame(crs=WGS84_CRS, geometry=[])


@pytest.fixture  # type: ignore
def no_crs_area_gdf() -> gpd.GeoDataFrame:
    """Get an example area gdf without crs defined."""
    polygon = shpg.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [polygon]})
    return gdf


@pytest.fixture  # type: ignore
def bad_geometry_area_gdf() -> gpd.GeoDataFrame:
    """Get an example area gdf with unsupported geometry (not Polygon or Multipolygon)."""
    point = shpg.Point(0, 0)
    gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [point]}, crs=WGS84_CRS)
    return gdf


@pytest.fixture  # type: ignore
def empty_polygon_area_gdf() -> gpd.GeoDataFrame:
    """Get an example area gdf where there is no road infrastructure."""
    polygon = shpg.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [polygon]}, crs=WGS84_CRS)
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

    gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [polygon]}, crs=WGS84_CRS)
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

    gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [polygon]}, crs=WGS84_CRS)
    return gdf


@pytest.fixture  # type: ignore
def multiple_polygons_overlapping_area_gdf(
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


@P.parameters(  # type: ignore
    "area_gdf_fixture", "expected_result", "loader_params", "file_name", "expectation"
)
@P.case(  # type: ignore
    "Raise when no geometry", "empty_area_gdf", None, None, None, pytest.raises(ValueError)
)
@P.case(  # type: ignore
    "Raise when no CRS",
    "no_crs_area_gdf",
    None,
    None,
    None,
    pytest.raises(ValueError),
)
@P.case(  # type: ignore
    "Raise when invalid geometry",
    "bad_geometry_area_gdf",
    None,
    None,
    "type_error",
    pytest.raises(TypeError),
)
@P.case(  # type: ignore
    "Raise when no road infrastructure",
    "empty_polygon_area_gdf",
    None,
    None,
    "empty_overpass_response",
    pytest.raises(LoadedDataIsEmptyException),
)
@P.case(  # type: ignore
    "Return infrastructure when single polygon",
    "first_polygon_area_gdf",
    (7, 6),
    None,
    "graph_1",
    does_not_raise(),
)
@P.case(  # type: ignore
    "Return infrastructure when multiple overlapping polygons",
    "multiple_polygons_overlapping_area_gdf",
    (7, 7),  # FIXME: should be (7, 6)
    None,
    "graph_1",
    does_not_raise(),
)
@P.case(  # type: ignore
    "Return infrastructure when node without any edges",  # FIXME: shouldn't have a node w/o an edge
    "multiple_polygons_overlapping_area_gdf",
    (9, 7),
    {"network_type": OSMNetworkType.BIKE, "contain_within_area": True},
    "graph_2",
    does_not_raise(),
)
@P.case(  # type: ignore
    "Return infrastructure when multipolygon",
    "multipolygons_area_gdf",
    (11, 9),
    None,
    "graph_3",
    does_not_raise(),
)
@P.case(  # type: ignore
    "Return infrastructure when correct polygon and polygon with no road infrastructure",
    "valid_and_empty_polygons_area_gdf",
    (7, 6),
    None,
    "graph_1",
    does_not_raise(),
)
@P.case(  # type: ignore
    "Return infrastructure without preprocessing",
    "first_polygon_area_gdf",
    (7, 6),
    {"network_type": OSMNetworkType.DRIVE, "preprocess": False},
    "graph_1",
    does_not_raise(),
)
@P.case(  # type: ignore
    "Return infrastructure in long format",
    "first_polygon_area_gdf",
    (7, 6),
    {"network_type": OSMNetworkType.DRIVE, "wide": False},
    "graph_1",
    does_not_raise(),
)
@P.case(  # type: ignore
    "Return infrastructure when network_type supplied as a string",
    "first_polygon_area_gdf",
    (7, 6),
    {"network_type": "drive", "wide": False},
    "graph_4",
    does_not_raise(),
)
def test_contract(
    area_gdf_fixture: str,
    expected_result: Optional[tuple[int, int]],
    loader_params: Optional[dict[str, Any]],
    file_name: Optional[str],
    expectation,
    request: pytest.FixtureRequest,
    mocker: MockerFixture,
):
    """Test `OSMWayLoader.load`'s contract."""
    if not loader_params:
        loader_params = {"network_type": OSMNetworkType.DRIVE}

    loader = OSMWayLoader(**loader_params)

    area_gdf = request.getfixturevalue(area_gdf_fixture)
    nodes_expected_len, edges_expected_len = expected_result or (None, None)

    def patched_graph_from_polygon(f_name: Optional[str], *args, **kwargs) -> Any:  # type: ignore
        if f_name == "type_error":
            raise TypeError
        elif f_name == "empty_overpass_response":
            from packaging import version

            osmnx_new_api = version.parse(osmnx.__version__) >= version.parse("1.6.0")

            if osmnx_new_api:
                raise osmnx._errors.InsufficientResponseError
            else:
                raise osmnx._errors.EmptyOverpassResponse

        files_path = Path(__file__).parent / "test_files"
        file_name = (f_name or "") + ".pkl"
        file_path = files_path / file_name

        with file_path.open("rb") as f:
            res = pkl.load(f)

        return res

    with expectation:
        mocker.patch(
            "osmnx.graph_from_polygon",
            wraps=partial(patched_graph_from_polygon, file_name),
        )
        nodes, edges = loader.load(area_gdf)

        check.equal(len(nodes), nodes_expected_len)
        check.equal(len(edges), edges_expected_len)
        check.is_in(GEOMETRY_COLUMN, nodes.columns)
        check.is_in(GEOMETRY_COLUMN, edges.columns)


@P.parameters("column_name", "input", "expected")  # type: ignore
@P.case(
    "Return None when `oneway` is missing",
    "oneway",
    ("", "none", "None", np.nan, "nan", "NaN", None),
    None,
)  # type: ignore
@P.case(
    "Return None when `lanes` is missing",
    "lanes",
    ("", "none", "None", np.nan, "nan", "NaN", None),
    None,
)  # type: ignore
@P.case(
    "Return None when `highway` is missing",
    "highway",
    ("", "none", "None", np.nan, "nan", "NaN", None),
    None,
)  # type: ignore
@P.case(
    "Test lanes",
    "lanes",
    [-1, 0, 1, 14, 15, 16, 17, 100, 3.7, "a", "2"],
    [-1, 0, 1, 14, 15, 15, 15, 15, 3, None, 2],
)  # type: ignore
@P.case(  # type: ignore
    "Test maxspeed",
    "maxspeed",
    [
        -1,
        0,
        1,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        14,
        15,
        16,
        17,
        44,
        45,
        46,
        199,
        200,
        201,
        300,
        6.7,
        "3",
        "4.5 km/h",
        "4.5km/h",
        "3mph",
        "4 mph",
        "signals",
        "variable",
        "CZ:pedestrian_zone",
    ],
    [
        0,
        0,
        5,
        5,
        5,
        7,
        7,
        10,
        10,
        10,
        15,
        15,
        15,
        20,
        20,
        40,
        40,
        50,
        200,
        200,
        200,
        200,
        7,
        5,
        5,
        5,
        5,
        7,
        None,
        None,
        20,
    ],
)
@P.case(
    "Test width",
    "width",
    [
        -1.3,
        -1,
        0,
        1,
        2,
        1.0,
        4.4,
        4.5,
        4.6,
        29,
        30,
        31,
        40,
        100,
        "4.6",
        "12m",
        "12 m",
        "12meter",
        "12 meter",
        "130.4'",
        "130.4 '",
        "25.4ft",
        "25.4 ft",
    ],
    [
        -1.5,
        -1.0,
        0.0,
        1.0,
        2.0,
        1.0,
        4.5,
        4.5,
        4.5,
        29.0,
        30.0,
        30.0,
        30.0,
        30.0,
        4.5,
        12.0,
        12.0,
        12.0,
        12.0,
        3.5,
        3.5,
        7.5,
        7.5,
    ],
)  # type: ignore
def test_preprocessing(
    column_name: str, input: Union[Any, Sequence[Any]], expected: Union[Any, Sequence[Any]]
) -> None:
    """Test `OSMWayLoader._sanitize_and_normalize()` preprocessing."""
    loader = OSMWayLoader(network_type=OSMNetworkType.DRIVE)

    input = list(input) if isinstance(input, Sequence) else [input]
    expected = list(expected) if isinstance(expected, Sequence) else [expected]

    if len(expected) == 1:
        expected = expected * len(input)

    if len(expected) != len(input):
        raise ValueError(
            f"Mismatch in length between input {len(input)} and expected {len(expected)}."
        )

    for x, y in zip(input, expected):
        result = loader._sanitize_and_normalize(x, column_name)
        check.equal(result, str(y))
