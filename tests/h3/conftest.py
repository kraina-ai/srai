"""Conftest for H3 tests."""

import geopandas as gpd
import pytest
from shapely import geometry

from srai.constants import WGS84_CRS


@pytest.fixture  # type: ignore
def gdf_single_point() -> gpd.GeoDataFrame:
    """Get the point case."""
    return gpd.GeoDataFrame(geometry=[geometry.Point(17.9261, 50.6696)], crs=WGS84_CRS)


@pytest.fixture  # type: ignore
def expected_point_h3_index() -> list[str]:
    """Get expected h3 index for the point case."""
    return [
        "8a1e23c44b5ffff",
    ]


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
def expected_h3_indexes() -> list[str]:
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


@pytest.fixture  # type: ignore
def expected_unbuffered_h3_indexes() -> list[str]:
    """Get expected h3 index for the unbuffered case."""
    return [
        "83754efffffffff",
    ]
