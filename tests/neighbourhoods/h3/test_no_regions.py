"""Tests for neigbourhoods with no regions."""

import pytest

from srai.neighbourhoods import H3Neighbourhood


@pytest.mark.parametrize(  # type: ignore
    "index,expected,expected_with_include_center",
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
            {
                "811f3ffffffffff",
                "811fbffffffffff",
                "811ebffffffffff",
                "811efffffffffff",
                "811e7ffffffffff",
                "811f7ffffffffff",
                "811e3ffffffffff",
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
            {
                "831f08fffffffff",
                "831f0afffffffff",
                "831e24fffffffff",
                "831e25fffffffff",
                "831f56fffffffff",
                "831f09fffffffff",
                "831f0bfffffffff",
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
            {
                "882baa4e97fffff",
                "882baa7b6dfffff",
                "882baa7b61fffff",
                "882baa7b6bfffff",
                "882baa7b45fffff",
                "882baa4e93fffff",
                "882baa7b69fffff",
            },
        ),
    ],
)
def test_get_neighbours(
    index: str, expected: set[str], expected_with_include_center: set[str]
) -> None:
    """Test get_neighbours of H3Neighbourhood."""
    neighbourhood = H3Neighbourhood()
    assert neighbourhood.get_neighbours(index) == expected
    assert neighbourhood.get_neighbours(index, include_center=True) == expected_with_include_center

    neighbourhood_with_include_center = H3Neighbourhood(include_center=True)
    assert neighbourhood_with_include_center.get_neighbours(index) == expected_with_include_center
    assert neighbourhood_with_include_center.get_neighbours(index, include_center=False) == expected


@pytest.mark.parametrize(  # type: ignore
    "index,distance,expected,expected_with_include_center",
    [
        ("811e3ffffffffff", -2, set(), set()),
        ("811e3ffffffffff", -1, set(), set()),
        ("811e3ffffffffff", 0, set(), {"811e3ffffffffff"}),
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
            {
                "861f09b07ffffff",
                "861f09b2fffffff",
                "861f09b37ffffff",
                "861f7248fffffff",
                "861f7249fffffff",
                "861f724d7ffffff",
                "861f09b27ffffff",
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
                "861f09b27ffffff",
            },
        ),
    ],
)
def test_get_neighbours_up_to_distance(
    index: str, distance: int, expected: set[str], expected_with_include_center: set[str]
) -> None:
    """Test get_neighbours_up_to_distance of H3Neighbourhood."""
    neighbourhood = H3Neighbourhood()
    assert neighbourhood.get_neighbours_up_to_distance(index, distance) == expected
    assert (
        neighbourhood.get_neighbours_up_to_distance(index, distance, include_center=True)
        == expected_with_include_center
    )

    neighbourhood_with_include_center = H3Neighbourhood(include_center=True)
    assert (
        neighbourhood_with_include_center.get_neighbours_up_to_distance(index, distance)
        == expected_with_include_center
    )
    assert (
        neighbourhood_with_include_center.get_neighbours_up_to_distance(
            index, distance, include_center=False
        )
        == expected
    )


@pytest.mark.parametrize(  # type: ignore
    "index,distance,expected,expected_with_include_center",
    [
        ("811e3ffffffffff", -2, set(), set()),
        ("811e3ffffffffff", -1, set(), set()),
        ("811e3ffffffffff", 0, set(), {"811e3ffffffffff"}),
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
def test_get_neighbours_at_distance(
    index: str, distance: int, expected: set[str], expected_with_include_center: set[str]
) -> None:
    """Test get_neighbours_at_distance of H3Neighbourhood."""
    neighbourhood = H3Neighbourhood()
    assert neighbourhood.get_neighbours_at_distance(index, distance) == expected
    assert (
        neighbourhood.get_neighbours_at_distance(index, distance, include_center=True)
        == expected_with_include_center
    )

    neighbourhood_with_include_center = H3Neighbourhood(include_center=True)
    assert (
        neighbourhood_with_include_center.get_neighbours_at_distance(index, distance)
        == expected_with_include_center
    )
    assert (
        neighbourhood_with_include_center.get_neighbours_at_distance(
            index, distance, include_center=False
        )
        == expected
    )
