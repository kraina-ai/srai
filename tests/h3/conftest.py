"""Conftest for H3 tests."""

import geopandas as gpd
import pytest
from shapely import geometry
from shapely.geometry.base import BaseGeometry

from srai.constants import GEOMETRY_COLUMN, WGS84_CRS


@pytest.fixture  # type: ignore
def gdf_single_point() -> gpd.GeoDataFrame:
    """Get the point case."""
    return gpd.GeoDataFrame(geometry=[geometry.Point(17.9261, 50.6696)], crs=WGS84_CRS)


@pytest.fixture  # type: ignore
def expected_point_h3_index() -> list[int]:
    """Get expected h3 index for the point case."""
    return [622026972032532479]


@pytest.fixture  # type: ignore
def gdf_polygons() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with two polygons."""
    return gpd.GeoDataFrame(
        geometry=[
            geometry.Polygon(
                shell=[
                    (-1, 0),
                    (0, 0.5),
                    (1, 0),
                    (1, 1),
                    (0, 1),
                ],
                holes=[
                    [
                        (0.8, 0.9),
                        (0.9, 0.55),
                        (0.8, 0.3),
                        (0.5, 0.4),
                    ]
                ],
            ),
            geometry.Polygon(shell=[(-0.25, 0), (0.25, 0), (0, 0.2)]),
        ],
        crs=WGS84_CRS,
    )


@pytest.fixture  # type: ignore
def gdf_multipolygon() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with multipolygon."""
    return gpd.GeoDataFrame(
        geometry=[
            geometry.MultiPolygon(
                [
                    (
                        [
                            (-1, 0),
                            (0, 0.5),
                            (1, 0),
                            (1, 1),
                            (0, 1),
                        ],
                        (
                            [
                                [
                                    (0.8, 0.9),
                                    (0.9, 0.55),
                                    (0.8, 0.3),
                                    (0.5, 0.4),
                                ]
                            ]
                        ),
                    ),
                    (
                        [(-0.25, 0), (0.25, 0), (0, 0.2)],
                        (),
                    ),
                ]
            )
        ],
        crs=WGS84_CRS,
    )


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


def _gdf_noop(gdf_fixture: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf_fixture


def _gdf_to_geoseries(gdf_fixture: gpd.GeoDataFrame) -> gpd.GeoSeries:  # noqa: FURB118
    return gdf_fixture[GEOMETRY_COLUMN]


def _gdf_to_geometry_list(gdf_fixture: gpd.GeoDataFrame) -> list[BaseGeometry]:
    return list(gdf_fixture[GEOMETRY_COLUMN])


def _gdf_to_single_geometry(gdf_fixture: gpd.GeoDataFrame) -> BaseGeometry:
    return gdf_fixture[GEOMETRY_COLUMN][0]
