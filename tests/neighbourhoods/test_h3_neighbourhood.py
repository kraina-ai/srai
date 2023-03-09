from typing import Any, Set

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from srai.neighbourhoods import H3Neighbourhood


@pytest.fixture  # type: ignore
def empty_gdf() -> gpd.GeoDataFrame:
    """Fixture for an empty GeoDataFrame."""
    return gpd.GeoDataFrame()


@pytest.fixture  # type: ignore
def single_hex_gdf() -> gpd.GeoDataFrame:
    """Fixture for a GeoDataFrame with a single hexagon."""
    return gpd.GeoDataFrame(
        {"index": ["811e3ffffffffff"], "geometry": [Polygon([(0, 0), (0, 1), (1, 0), (1, 1)])]}
    )


@pytest.fixture  # type: ignore
def hex_without_one_neighbour_gdf() -> gpd.GeoDataFrame:
    """Fixture for a GeoDataFrame with a single hexagon."""
    return gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
        index=[
            "811e3ffffffffff",
            "811f3ffffffffff",
            "811fbffffffffff",
            "811ebffffffffff",
            "811efffffffffff",
            "811e7ffffffffff",
        ],
    )


@pytest.mark.parametrize(  # type: ignore
    "regions_gdf_fixture,expected",
    [
        (
            "empty_gdf",
            set(),
        ),
        (
            "single_hex_gdf",
            set(),
        ),
        (
            "hex_without_one_neighbour_gdf",
            {
                "811f3ffffffffff",
                "811fbffffffffff",
                "811ebffffffffff",
                "811efffffffffff",
                "811e7ffffffffff",
            },
        ),
    ],
)
def test_get_neighbours_with_regions_gdf(
    regions_gdf_fixture: str, expected: Set[str], request: Any
) -> None:
    """Test get_neighbours of H3Neighbourhood with a specified regions GeoDataFrame."""
    regions_gdf = request.getfixturevalue(regions_gdf_fixture)
    assert H3Neighbourhood(regions_gdf).get_neighbours("811e3ffffffffff") == expected


@pytest.mark.parametrize(  # type: ignore
    "index,expected",
    [
        (
            "811e3ffffffffff",
            {
                "811f3ffffffffff",
                "811fbffffffffff",
                "811ebffffffffff",
                "811efffffffffff",
                "811e7ffffffffff",
                "811f7ffffffffff",
            },
        ),
        (
            "831f0bfffffffff",
            {
                "831f08fffffffff",
                "831f0afffffffff",
                "831e24fffffffff",
                "831e25fffffffff",
                "831f56fffffffff",
                "831f09fffffffff",
            },
        ),
        (
            "882baa7b69fffff",
            {
                "882baa4e97fffff",
                "882baa7b6dfffff",
                "882baa7b61fffff",
                "882baa7b6bfffff",
                "882baa7b45fffff",
                "882baa4e93fffff",
            },
        ),
    ],
)
def test_get_neighbours(index: str, expected: Set[str]) -> None:
    """Test get_neighbours of H3Neighbourhood."""
    assert H3Neighbourhood().get_neighbours(index) == expected


@pytest.mark.parametrize(  # type: ignore
    "index,distance,expected",
    [
        ("811e3ffffffffff", -2, set()),
        ("811e3ffffffffff", -1, set()),
        ("811e3ffffffffff", 0, set()),
        (
            "861f09b27ffffff",
            1,
            {
                "861f09b07ffffff",
                "861f09b2fffffff",
                "861f09b37ffffff",
                "861f7248fffffff",
                "861f7249fffffff",
                "861f724d7ffffff",
            },
        ),
        (
            "861f09b27ffffff",
            2,
            {
                "861f0984fffffff",
                "861f0986fffffff",
                "861f09b07ffffff",
                "861f09b0fffffff",
                "861f09b17ffffff",
                "861f09b1fffffff",
                "861f09b2fffffff",
                "861f09b37ffffff",
                "861f09b77ffffff",
                "861f72487ffffff",
                "861f7248fffffff",
                "861f72497ffffff",
                "861f7249fffffff",
                "861f724afffffff",
                "861f724c7ffffff",
                "861f724d7ffffff",
                "861f724dfffffff",
                "861f724f7ffffff",
            },
        ),
    ],
)
def test_get_neighbours_up_to_distance(index: str, distance: int, expected: Set[str]) -> None:
    """Test get_neighbours_up_to_distance of H3Neighbourhood."""
    assert H3Neighbourhood().get_neighbours_up_to_distance(index, distance) == expected


@pytest.mark.parametrize(  # type: ignore
    "index,distance,expected",
    [
        ("811e3ffffffffff", -2, set()),
        ("811e3ffffffffff", -1, set()),
        ("811e3ffffffffff", 0, set()),
        (
            "862bac507ffffff",
            1,
            {
                "862bac50fffffff",
                "862bac517ffffff",
                "862bac51fffffff",
                "862bac527ffffff",
                "862bac52fffffff",
                "862bac537ffffff",
            },
        ),
        (
            "862bac507ffffff",
            2,
            {
                "862ba124fffffff",
                "862ba126fffffff",
                "862bac427ffffff",
                "862bac437ffffff",
                "862bac557ffffff",
                "862bac577ffffff",
                "862bac5a7ffffff",
                "862bac5afffffff",
                "862bacc8fffffff",
                "862bacc9fffffff",
                "862baccd7ffffff",
                "862baccdfffffff",
            },
        ),
    ],
)
def test_get_neighbours_at_distance(index: str, distance: int, expected: Set[str]) -> None:
    """Test get_neighbours_up_to_distance of H3Neighbourhood."""
    assert H3Neighbourhood().get_neighbours_at_distance(index, distance) == expected
