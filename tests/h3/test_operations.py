"""Module for testing srai.h3 module functionality."""

from typing import Any, Callable

import geopandas as gpd
import pytest

from srai.h3 import (
    ring_buffer_geometry,
    ring_buffer_h3_regions_gdf,
)
from srai.regionalizers.geocode import geocode_to_region_gdf
from srai.regionalizers.h3_regionalizer import H3Regionalizer
from tests.h3.conftest import _gdf_noop, _gdf_to_geometry_list, _gdf_to_geoseries
from tests.regionalizers.test_h3_regionalizer import H3_RESOLUTION

# TODO: add expected values, now only checks if runs without errors


@pytest.mark.parametrize(
    "geometry_fixture, resolution, distance",
    [
        ("gdf_single_point", 10, 10),
        ("gdf_multipolygon", H3_RESOLUTION, 2),
        ("gdf_polygons", H3_RESOLUTION, 2),
    ],
)  # type: ignore
@pytest.mark.parametrize(
    "geometry_parser_function",
    [_gdf_noop, _gdf_to_geoseries, _gdf_to_geometry_list],
)  # type: ignore
def test_ring_buffer_geometry(
    geometry_fixture: str,
    resolution: int,
    distance: int,
    geometry_parser_function: Callable[[gpd.GeoDataFrame], Any],
    request: pytest.FixtureRequest,
) -> None:
    """Test checks if ring_buffer_geometry function works."""
    geometry = request.getfixturevalue(geometry_fixture)

    parsed_geometry = geometry_parser_function(geometry)
    ring_buffer_geometry(parsed_geometry, h3_resolution=resolution, distance=distance)


def test_ring_buffer_h3_regions_gdf() -> None:
    """Test checks if ring_buffer_h3_regions_gdf function works."""
    gdf_wro = geocode_to_region_gdf("Wrocław, PL")
    regions_gdf = H3Regionalizer(8).transform(gdf_wro)

    ring_buffer_h3_regions_gdf(regions_gdf, distance=10)
