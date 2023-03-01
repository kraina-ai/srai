from typing import Set

import pytest

from srai.neighbourhood import H3Neighbourhood


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
        # TODO: proper cases
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
        # TODO: proper cases
    ],
)
def test_get_neighbours_at_distance(index: str, distance: int, expected: Set[str]) -> None:
    """Test get_neighbours_up_to_distance of H3Neighbourhood."""
    assert H3Neighbourhood().get_neighbours_at_distance(index, distance) == expected
