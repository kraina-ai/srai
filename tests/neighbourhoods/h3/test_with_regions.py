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
    "distance,expected",
    [
        (
            0,
            set(),
        ),
        (
            1,
            {
                "862bac50fffffff",
                "862bac517ffffff",
                "862bac51fffffff",
                "862bac527ffffff",
            },
        ),
        (
            2,
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
    ],
)
def test_get_neighbours_up_to_distance_with_regions_gdf(
    distance: int, expected: Set[str], request: Any
) -> None:
    """Test get_neighbours_up_to_distance of H3Neighbourhood with a specified regions
    GeoDataFrame.
    """
    regions_gdf = request.getfixturevalue("two_rings_regions_some_missing")
    neighbourhood = H3Neighbourhood(regions_gdf)
    initial_region_index = "862bac507ffffff"
    assert neighbourhood.get_neighbours_up_to_distance(initial_region_index, distance) == expected


@pytest.mark.parametrize(  # type: ignore
    "distance,expected",
    [
        (
            0,
            set(),
        ),
        (
            1,
            {
                "862bac50fffffff",
                "862bac517ffffff",
                "862bac51fffffff",
                "862bac527ffffff",
            },
        ),
        (
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
            },
        ),
    ],
)
def test_get_neighbours_at_distance_with_regions_gdf(
    distance: int, expected: Set[str], request: Any
) -> None:
    """Test get_neighbours_at_distance of H3Neighbourhood with a specified regions GeoDataFrame."""
    regions_gdf = request.getfixturevalue("two_rings_regions_some_missing")
    neighbourhood = H3Neighbourhood(regions_gdf)
    initial_region_index = "862bac507ffffff"
    assert neighbourhood.get_neighbours_at_distance(initial_region_index, distance) == expected
