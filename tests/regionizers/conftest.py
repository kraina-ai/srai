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


@pytest.fixture  # type: ignore
def gdf_earth_poles() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with 6 Earth poles."""
    return gpd.GeoDataFrame(
        {
            "geometry": [
                geometry.Point(0, 0),
                geometry.Point(90, 0),
                geometry.Point(180, 0),
                geometry.Point(-90, 0),
                geometry.Point(0, 90),
                geometry.Point(0, -90),
            ]
        },
        index=[1, 2, 3, 4, 5, 6],
        crs="EPSG:4326",
    )


@pytest.fixture  # type: ignore
def gdf_poland() -> gpd.GeoDataFrame:
    """Get Poland GeoDataFrame."""
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    area = world[world.name == "Poland"]
    return area


@pytest.fixture  # type: ignore
def earth_bbox() -> geometry.Polygon:
    """Get full bounding box GeoDataFrame."""
    return geometry.box(minx=-180, maxx=180, miny=-90, maxy=90)


@pytest.fixture  # type: ignore
def gdf_earth_bbox(earth_bbox) -> gpd.GeoDataFrame:
    """Get full bounding box GeoDataFrame."""
    return gpd.GeoDataFrame({"geometry": [earth_bbox]}, crs="EPSG:4326")
