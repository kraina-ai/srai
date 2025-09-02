"""Tests for H3Regionalizer."""

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING, Any
from unittest import TestCase

import pytest

from srai.constants import GEOMETRY_COLUMN
from srai.regionalizers import H3Regionalizer
from srai.regionalizers.geocode import geocode_to_region_gdf

if TYPE_CHECKING:  # pragma: no cover
    import geopandas as gpd


ut = TestCase()
H3_RESOLUTION = 3


@pytest.fixture  # type: ignore
def expected_h3_indexes() -> list[int]:
    """Get expected h3 indexes."""
    return [
        592036021705637887,
        592035265791393791,
        592035128352440319,
        592034372438196223,
        592036296583544831,
        592034509877149695,
        592034990913486847,
    ]


@pytest.fixture  # type: ignore
def expected_unbuffered_h3_indexes() -> list[int]:
    """Get expected h3 index for the unbuffered case."""
    return [592035265791393791]


@pytest.mark.parametrize(  # type: ignore
    "gdf_fixture,expected_h3_indexes_fixture,resolution,buffer,expectation",
    [
        ("gdf_polygons", "expected_h3_indexes", H3_RESOLUTION, True, does_not_raise()),
        ("gdf_polygons", "expected_unbuffered_h3_indexes", H3_RESOLUTION, False, does_not_raise()),
        ("gdf_multipolygon", "expected_h3_indexes", H3_RESOLUTION, True, does_not_raise()),
        ("gdf_empty", "expected_h3_indexes", H3_RESOLUTION, True, pytest.raises(AttributeError)),
        ("gdf_polygons", "expected_h3_indexes", -1, True, pytest.raises(ValueError)),
        ("gdf_polygons", "expected_h3_indexes", 16, True, pytest.raises(ValueError)),
        ("gdf_no_crs", "expected_h3_indexes", H3_RESOLUTION, True, pytest.raises(ValueError)),
    ],
)
def test_transform(
    gdf_fixture: str,
    expected_h3_indexes_fixture: str,
    resolution: int,
    buffer: bool,
    expectation: Any,
    request: Any,
) -> None:
    """Test transform of H3Regionalizer."""
    gdf: gpd.GeoDataFrame = request.getfixturevalue(gdf_fixture)
    h3_indexes: list[str] = request.getfixturevalue(expected_h3_indexes_fixture)
    with expectation:
        gdf_h3 = H3Regionalizer(resolution, buffer=buffer).transform(gdf).to_geodataframe()

        ut.assertCountEqual(first=gdf_h3.index.to_list(), second=h3_indexes)
        assert GEOMETRY_COLUMN in gdf_h3


def test_wroclaw_edge_case() -> None:
    """Test edge case from H3Neighbourhood example error."""
    gdf_wro = geocode_to_region_gdf("Wrocław, PL")
    regions_gdf = H3Regionalizer(8).transform(gdf_wro).to_geodataframe()

    edge_region_id = 613019535601041407

    assert edge_region_id in regions_gdf.index, "Edge cell is not in the regions."
