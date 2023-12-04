"""Tests for H3Regionalizer."""

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING, Any
from unittest import TestCase

import pytest

from srai.constants import GEOMETRY_COLUMN
from srai.regionalizers import S2Regionalizer

if TYPE_CHECKING:  # pragma: no cover
    import geopandas as gpd

ut = TestCase()
S2_RESOLUTION = 7


@pytest.fixture  # type: ignore
def expected_s2_indexes() -> list[str]:
    """Get expected s2 indexes."""
    return [
        "0555c",
        "0ffe4",
        "0fff4",
        "0fffc",
        "10004",
        "1000c",
        "10014",
        "1001c",
        "1aaa4",
        "05554",
        "1aaac",
    ]


@pytest.mark.parametrize(  # type: ignore
    "gdf_fixture,expected_s2_indexes_fixture,resolution,expectation",
    [
        ("gdf_polygons", "expected_s2_indexes", S2_RESOLUTION, does_not_raise()),
        ("gdf_multipolygon", "expected_s2_indexes", S2_RESOLUTION, does_not_raise()),
        ("gdf_empty", "expected_s2_indexes", S2_RESOLUTION, pytest.raises(AttributeError)),
        ("gdf_polygons", "expected_s2_indexes", -1, pytest.raises(ValueError)),
        ("gdf_polygons", "expected_s2_indexes", 31, pytest.raises(ValueError)),
        ("gdf_no_crs", "expected_s2_indexes", S2_RESOLUTION, pytest.raises(ValueError)),
    ],
)
def test_transform(
    gdf_fixture: str,
    expected_s2_indexes_fixture: str,
    resolution: int,
    expectation: Any,
    request: Any,
) -> None:
    """Test transform of H3Regionalizer."""
    gdf: gpd.GeoDataFrame = request.getfixturevalue(gdf_fixture)
    s2_indexes: list[str] = request.getfixturevalue(expected_s2_indexes_fixture)
    with expectation:
        gdf_s2 = S2Regionalizer(resolution).transform(gdf)

        ut.assertCountEqual(first=gdf_s2.index.to_list(), second=s2_indexes)
        assert GEOMETRY_COLUMN in gdf_s2
