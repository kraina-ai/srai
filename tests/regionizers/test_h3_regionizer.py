"""Tests for H3Regionizer."""
from contextlib import nullcontext as does_not_raise
from typing import Any, List
from unittest import TestCase

import geopandas as gpd
import pytest

from srai.regionizers import H3Regionizer

ut = TestCase()
H3_RESOLUTION = 3


@pytest.fixture  # type: ignore
def expected_h3_indexes() -> List[str]:
    """Get expected h3 indexes."""
    return [
        "837559fffffffff",
        "83754efffffffff",
        "83754cfffffffff",
        "837541fffffffff",
        "83755dfffffffff",
        "837543fffffffff",
        "83754afffffffff",
    ]


@pytest.mark.parametrize(  # type: ignore
    "gdf_fixture,expected_h3_indexes_fixture,resolution,expectation",
    [
        ("gdf_polygons", "expected_h3_indexes", H3_RESOLUTION, does_not_raise()),
        ("gdf_multipolygon", "expected_h3_indexes", H3_RESOLUTION, does_not_raise()),
        ("gdf_empty", "gdf_empty", H3_RESOLUTION, pytest.raises(AttributeError)),
        ("gdf_polygons", "expected_h3_indexes", -1, pytest.raises(ValueError)),
        ("gdf_polygons", "expected_h3_indexes", 16, pytest.raises(ValueError)),
        ("gdf_no_crs", "gdf_no_crs", H3_RESOLUTION, pytest.raises(ValueError)),
    ],
)
def test_transform(
    gdf_fixture: str,
    expected_h3_indexes_fixture: str,
    resolution: int,
    expectation: Any,
    request: Any,
) -> None:
    """Test data structure."""
    gdf: gpd.GeoDataFrame = request.getfixturevalue(gdf_fixture)
    h3_indexes: List[str] = request.getfixturevalue(expected_h3_indexes_fixture)
    with expectation:
        gdf_h3 = H3Regionizer(resolution).transform(gdf)

        ut.assertCountEqual(first=gdf_h3.index.to_list(), second=h3_indexes)
        assert "geometry" in gdf_h3
