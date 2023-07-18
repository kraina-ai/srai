"""Tests for neigbourhoods with regions."""
from typing import Any, Set

import geopandas as gpd
import pytest

from srai.neighbourhoods import H3Neighbourhood


@pytest.fixture  # type: ignore
def empty_gdf() -> gpd.GeoDataFrame:
    """Fixture for an empty GeoDataFrame."""
    return gpd.GeoDataFrame()


@pytest.fixture  # type: ignore
def single_hex_gdf() -> gpd.GeoDataFrame:
    """Fixture for a GeoDataFrame with a single hexagon."""
    return gpd.GeoDataFrame(
        index=["811e3ffffffffff"],
    )


@pytest.fixture  # type: ignore
def hex_without_one_neighbour_gdf() -> gpd.GeoDataFrame:
    """Fixture for a GeoDataFrame with a single hexagon, and its neighbours except for one."""
    return gpd.GeoDataFrame(
        index=[
            "811e3ffffffffff",
            "811f3ffffffffff",
            "811fbffffffffff",
            "811ebffffffffff",
            "811efffffffffff",
            "811e7ffffffffff",
        ],
    )


@pytest.fixture  # type: ignore
def two_rings_regions_some_missing() -> gpd.GeoDataFrame:
    """
    Fixture for a GeoDataFrame with regions up to two rings around initial one.

    Some regions are missing.
    """
    initial_region = ["862bac507ffffff"]
    first_ring = [
        "862bac50fffffff",
        "862bac517ffffff",
        "862bac51fffffff",
        "862bac527ffffff",
        "862bac52fffffff",
        "862bac537ffffff",
    ]
    second_ring = [
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
    ]
    # This is the result of running h3.grid_disk("862bac507ffffff", 2)
    indices = initial_region + first_ring + second_ring

    # Explicitly remove the 'missing' regions so it's clear what's going on
    # Remove two regions from first ring
    indices.remove("862bac52fffffff")
    indices.remove("862bac537ffffff")
    # Remove three regions from second ring
    indices.remove("862bacc9fffffff")
    indices.remove("862baccd7ffffff")
    indices.remove("862baccdfffffff")
    return gpd.GeoDataFrame(index=indices)


@pytest.mark.parametrize(  # type: ignore
    "regions_gdf_fixture,include_center,expected",
    [
        (
            "empty_gdf",
            False,
            set(),
        ),
        (
            "empty_gdf",
            True,
            set(),
        ),
        (
            "single_hex_gdf",
            False,
            set(),
        ),
        (
            "single_hex_gdf",
            True,
            {"811e3ffffffffff"},
        ),
        (
            "hex_without_one_neighbour_gdf",
            False,
            {
                "811f3ffffffffff",
                "811fbffffffffff",
                "811ebffffffffff",
                "811efffffffffff",
                "811e7ffffffffff",
            },
        ),
        (
            "hex_without_one_neighbour_gdf",
            True,
            {
                "811f3ffffffffff",
                "811fbffffffffff",
                "811ebffffffffff",
                "811efffffffffff",
                "811e7ffffffffff",
                "811e3ffffffffff",
            },
        ),
    ],
)
def test_get_neighbours_with_regions_gdf(
    regions_gdf_fixture: str, include_center: bool, expected: Set[str], request: Any
) -> None:
    """Test get_neighbours of H3Neighbourhood with a specified regions GeoDataFrame."""
    regions_gdf = request.getfixturevalue(regions_gdf_fixture)
    assert (
        H3Neighbourhood(regions_gdf, include_center).get_neighbours("811e3ffffffffff") == expected
    )


@pytest.mark.parametrize(  # type: ignore
    "distance,include_center,expected",
    [
        (
            0,
            False,
            set(),
        ),
        (
            0,
            True,
            {"862bac507ffffff"},
        ),
        (
            1,
            False,
            {
                "862bac50fffffff",
                "862bac517ffffff",
                "862bac51fffffff",
                "862bac527ffffff",
            },
        ),
        (
            1,
            True,
            {
                "862bac50fffffff",
                "862bac517ffffff",
                "862bac51fffffff",
                "862bac527ffffff",
                "862bac507ffffff",
            },
        ),
        (
            2,
            False,
            {
                "862bac50fffffff",
                "862bac517ffffff",
                "862bac51fffffff",
                "862bac527ffffff",
                "862ba124fffffff",
                "862ba126fffffff",
                "862bac427ffffff",
                "862bac437ffffff",
                "862bac557ffffff",
                "862bac577ffffff",
                "862bac5a7ffffff",
                "862bac5afffffff",
                "862bacc8fffffff",
            },
        ),
        (
            2,
            True,
            {
                "862bac50fffffff",
                "862bac517ffffff",
                "862bac51fffffff",
                "862bac527ffffff",
                "862ba124fffffff",
                "862ba126fffffff",
                "862bac427ffffff",
                "862bac437ffffff",
                "862bac557ffffff",
                "862bac577ffffff",
                "862bac5a7ffffff",
                "862bac5afffffff",
                "862bacc8fffffff",
                "862bac507ffffff",
            },
        ),
    ],
)
def test_get_neighbours_up_to_distance_with_regions_gdf(
    distance: int, include_center: bool, expected: Set[str], request: Any
) -> None:
    """Test get_neighbours_up_to_distance of H3Neighbourhood with a specified regions."""
    regions_gdf = request.getfixturevalue("two_rings_regions_some_missing")
    neighbourhood = H3Neighbourhood(regions_gdf, include_center)
    initial_region_index = "862bac507ffffff"
    assert neighbourhood.get_neighbours_up_to_distance(initial_region_index, distance) == expected


@pytest.mark.parametrize(  # type: ignore
    "distance,include_center,expected",
    [
        (
            0,
            False,
            set(),
        ),
        (
            0,
            True,
            {"862bac507ffffff"},
        ),
        (
            1,
            False,
            {
                "862bac50fffffff",
                "862bac517ffffff",
                "862bac51fffffff",
                "862bac527ffffff",
            },
        ),
        (
            1,
            True,
            {
                "862bac50fffffff",
                "862bac517ffffff",
                "862bac51fffffff",
                "862bac527ffffff",
            },
        ),
        (
            2,
            False,
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
            },
        ),
        (
            2,
            True,
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
            },
        ),
    ],
)
def test_get_neighbours_at_distance_with_regions_gdf(
    distance: int, include_center: bool, expected: Set[str], request: Any
) -> None:
    """Test get_neighbours_at_distance of H3Neighbourhood with a specified regions GeoDataFrame."""
    regions_gdf = request.getfixturevalue("two_rings_regions_some_missing")
    neighbourhood = H3Neighbourhood(regions_gdf, include_center)
    initial_region_index = "862bac507ffffff"
    assert neighbourhood.get_neighbours_at_distance(initial_region_index, distance) == expected
