"""Conftest for Regionizers."""

import geopandas as gpd
import pytest
from shapely import geometry


@pytest.fixture  # type: ignore
def gdf_empty() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame."""
    return gpd.GeoDataFrame()


@pytest.fixture  # type: ignore
def gdf_no_crs() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with no crs."""
    return gpd.GeoDataFrame(
        geometry=[
            geometry.Polygon(
                shell=[
                    (-1, 0),
                    (0, 0.5),
                    (1, 0),
                    (1, 1),
                    (0, 1),
                ]
            )
        ]
    )


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
        crs="epsg:4326",
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
        crs="epsg:4326",
    )
