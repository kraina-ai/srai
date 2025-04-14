"""H3 shapely conversion tests."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Union
from unittest import TestCase

import geopandas as gpd
import h3
import numpy as np
import pytest
from shapely.geometry import Point

from srai.constants import WGS84_CRS
from srai.h3 import h3_to_geoseries, h3_to_shapely_geometry, shapely_geometry_to_h3
from tests.h3.conftest import _gdf_noop, _gdf_to_geometry_list, _gdf_to_geoseries
from tests.regionalizers.test_h3_regionalizer import H3_RESOLUTION

ut = TestCase()


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


@pytest.mark.parametrize(
    "h3_index",
    [
        "8a1f52711897fff",
        h3.str_to_int("8a1f52711897fff"),
    ],
)  # type: ignore
def test_h3_to_shapely_geometry_consistency(
    h3_index: Union[int, str],
) -> None:
    """Test checks whether conversion from h3 to shapely is consistent."""
    geometry = h3_to_shapely_geometry(h3_index)
    geometries = h3_to_shapely_geometry([h3_index])

    assert geometry == geometries[0]


CENTROID_8a1f52711897fff = Point(20.0005, 51.9997)
CENTROID_8758128deffffff = Point(3.9683, 11.5235)


@pytest.mark.parametrize(
    "h3_index,expected_centroids",
    [
        ("8a1f52711897fff", [CENTROID_8a1f52711897fff]),
        (["8a1f52711897fff"], [CENTROID_8a1f52711897fff]),
        (
            ["8a1f52711897fff", "8758128deffffff"],
            [CENTROID_8a1f52711897fff, CENTROID_8758128deffffff],
        ),
        (h3.str_to_int("8a1f52711897fff"), [CENTROID_8a1f52711897fff]),
        ([h3.str_to_int("8a1f52711897fff")], [CENTROID_8a1f52711897fff]),
        (
            [h3.str_to_int("8a1f52711897fff"), h3.str_to_int("8758128deffffff")],
            [CENTROID_8a1f52711897fff, CENTROID_8758128deffffff],
        ),
        (
            ["8a1f52711897fff", h3.str_to_int("8758128deffffff")],
            [CENTROID_8a1f52711897fff, CENTROID_8758128deffffff],
        ),
    ],
)  # type: ignore
def test_h3_to_geoseries(
    h3_index: Union[int, str, Iterable[Union[int, str]]], expected_centroids: Iterable[Point]
) -> None:
    """Test checks whether conversion from h3 to geoseries works."""
    gs = h3_to_geoseries(h3_index)
    assert gs.crs == WGS84_CRS

    centroids = gs.centroid.values.tolist()

    np.testing.assert_almost_equal(
        [c.x for c in centroids], [p.x for p in expected_centroids], decimal=4
    )

    np.testing.assert_almost_equal(
        [c.y for c in centroids], [p.y for p in expected_centroids], decimal=4
    )
