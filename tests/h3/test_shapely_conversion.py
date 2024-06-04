"""H3 shapely conversion tests."""

from pathlib import Path
from typing import Any, Callable
from unittest import TestCase

import geopandas as gpd
import pytest
from shapely.geometry.base import BaseGeometry

from srai.constants import GEOMETRY_COLUMN
from srai.h3 import shapely_geometry_to_h3
from tests.regionalizers.test_h3_regionalizer import H3_RESOLUTION

ut = TestCase()


def _gdf_noop(gdf_fixture: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf_fixture


def _gdf_to_geoseries(gdf_fixture: gpd.GeoDataFrame) -> gpd.GeoSeries:  # noqa: FURB118
    return gdf_fixture[GEOMETRY_COLUMN]


def _gdf_to_geometry_list(gdf_fixture: gpd.GeoDataFrame) -> list[BaseGeometry]:
    return list(gdf_fixture[GEOMETRY_COLUMN])


def _gdf_to_single_geometry(gdf_fixture: gpd.GeoDataFrame) -> BaseGeometry:
    return gdf_fixture[GEOMETRY_COLUMN][0]


@pytest.mark.parametrize(
    "geometry_fixture, resolution, expected_h3_cells_fixture",
    [
        ("gdf_single_point", 10, "expected_point_h3_index"),
        ("gdf_multipolygon", H3_RESOLUTION, "expected_unbuffered_h3_indexes"),
        ("gdf_polygons", H3_RESOLUTION, "expected_unbuffered_h3_indexes"),
    ],
)  # type: ignore
@pytest.mark.parametrize(
    "geometry_parser_function",
    [_gdf_noop, _gdf_to_geoseries, _gdf_to_geometry_list],
)  # type: ignore
def test_shapely_geometry_to_h3_unbuffered(
    geometry_fixture: str,
    resolution: int,
    expected_h3_cells_fixture: str,
    geometry_parser_function: Callable[[gpd.GeoDataFrame], Any],
    request: pytest.FixtureRequest,
) -> None:
    """Test checks if conversion from shapely to h3 works."""
    geometry = request.getfixturevalue(geometry_fixture)
    expected_h3_cells = request.getfixturevalue(expected_h3_cells_fixture)

    parsed_geometry = geometry_parser_function(geometry)
    h3_cells = shapely_geometry_to_h3(
        geometry=parsed_geometry, h3_resolution=resolution, buffer=False
    )
    ut.assertCountEqual(h3_cells, expected_h3_cells)


@pytest.mark.parametrize(
    "geometry_fixture, resolution, expected_h3_cells_fixture",
    [
        ("gdf_single_point", 10, "expected_point_h3_index"),
        ("gdf_multipolygon", H3_RESOLUTION, "expected_h3_indexes"),
        ("gdf_polygons", H3_RESOLUTION, "expected_h3_indexes"),
    ],
)  # type: ignore
@pytest.mark.parametrize(
    "geometry_parser_function",
    [_gdf_noop, _gdf_to_geoseries, _gdf_to_geometry_list],
)  # type: ignore
def test_shapely_geometry_to_h3_buffered(
    geometry_fixture: str,
    resolution: int,
    expected_h3_cells_fixture: str,
    geometry_parser_function: Callable[[gpd.GeoDataFrame], Any],
    request: pytest.FixtureRequest,
) -> None:
    """Test checks if conversion from shapely to h3 with buffer works."""
    geometry = request.getfixturevalue(geometry_fixture)
    expected_h3_cells = request.getfixturevalue(expected_h3_cells_fixture)

    parsed_geometry = geometry_parser_function(geometry)
    h3_cells = shapely_geometry_to_h3(
        geometry=parsed_geometry, h3_resolution=resolution, buffer=True
    )
    ut.assertCountEqual(h3_cells, expected_h3_cells)


def test_full_coverage() -> None:
    """Test if h3 coverage works if geometry is smaller than single cell."""
    gdf = gpd.read_file(Path(__file__).parent / "test_files" / "buildings.geojson")

    intersections = {}
    for geom, osm_id in zip(gdf.geometry, gdf.id):
        intersections[osm_id] = shapely_geometry_to_h3(geom, h3_resolution=10)

    assert all(len(value) > 0 for value in intersections.values())
    assert len(intersections["way/843232154"]) == 1
    assert "8a2ab5760167fff" in intersections["way/843232154"]
