"""Tests for LookupNeighbourhood."""
from typing import Dict, Set, TypeVar

import pytest

from srai.neighbourhoods import Neighbourhood

T = TypeVar("T")


class LookupNeighbourhood(Neighbourhood[T]):
    """LookupNeighbourhood."""

    def __init__(self, lookup: Dict[T, Set[T]]) -> None:
        """
        LookupNeighbourhood constructor.

        Args:
            lookup (Dict[T, Set[T]]): Mapping of region to its neighbours.
        """
        self.lookup = lookup

    def get_neighbours(self, index: T) -> Set[T]:
        """
        Get neighbours for region at index.

        Args:
            index (T): Index of region in mapping.
        """
        return self.lookup[index]


@pytest.fixture  # type: ignore
def grid_3_by_3_neighbourhood() -> Dict[int, Set[int]]:
    """
    Get grid neighbourhood.

    This dict represents a simple 3 by 3 grid-like neighbourhood. The tiles are numbered from 1 to
    9, from left to right, top to bottom. The tiles are considered neighbours if they are adjacent
    by edge, not by vertex. Visually it looks like this:

    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

    Returns:
        Dict[int, Set[int]]: A dict representing 3 by 3 grid neighbourhood.
    """
    return {
        1: {2, 4},
        2: {1, 3, 5},
        3: {2, 6},
        4: {1, 5, 7},
        5: {2, 4, 6, 8},
        6: {3, 5, 9},
        7: {4, 8},
        8: {5, 7, 9},
        9: {6, 8},
    }


@pytest.fixture  # type: ignore
def grid_3_by_3_irrregular_neighbourhood() -> Dict[int, Set[int]]:
    """
    Get irregular grid neighbourhood.

    This dict represents a simple 3 by 3 grid-like neighbourhood. The tiles are numbered from 1 to
    4, from left to right, top to bottom. The tiles are considered neighbours if they are adjacent
    by edge, not by vertex. Tiles are irregular, not single squares 1 by 1. Visually it looks like
    this:

    [[1, 1, 2],
     [1, 1, 2],
     [3, 4, 4]]

    Returns:
        Dict[int, Set[int]]: A dict representing 3 by 3 grid neighbourhood.
    """
    return {
        1: {2, 3, 4},
        2: {1, 4},
        3: {1, 4},
        4: {1, 2, 3},
    }


@pytest.mark.parametrize(  # type: ignore
    "index, distance, expected",
    [
        (5, -2, set()),
        (5, -1, set()),
        (5, 0, set()),
        (5, 1, {2, 4, 6, 8}),
        (5, 2, {1, 3, 7, 9}),
        (5, 3, set()),
        (4, 1, {1, 5, 7}),
        (4, 2, {2, 6, 8}),
        (4, 3, {3, 9}),
        (4, 4, set()),
        (1, 1, {2, 4}),
        (1, 2, {3, 5, 7}),
        (1, 3, {6, 8}),
        (1, 4, {9}),
        (1, 5, set()),
    ],
)
def test_get_neighbours_at_distance(
    index: str, distance: int, expected: Set[str], grid_3_by_3_neighbourhood: Dict[str, Set[str]]
) -> None:
    """Test neighbours at distance."""
    neighbourhood = LookupNeighbourhood(grid_3_by_3_neighbourhood)
    neighbours = neighbourhood.get_neighbours_at_distance(index, distance)
    assert neighbours == expected


@pytest.mark.parametrize(  # type: ignore
    "index, distance, expected",
    [
        (5, -2, set()),
        (5, -1, set()),
        (5, 0, set()),
        (5, 1, {2, 4, 6, 8}),
        (5, 2, {1, 2, 3, 4, 6, 7, 8, 9}),
        (5, 3, {1, 2, 3, 4, 6, 7, 8, 9}),
        (4, 1, {1, 5, 7}),
        (4, 2, {1, 2, 5, 6, 7, 8}),
        (4, 3, {1, 2, 3, 5, 6, 7, 8, 9}),
        (4, 4, {1, 2, 3, 5, 6, 7, 8, 9}),
        (1, 1, {2, 4}),
        (1, 2, {2, 3, 4, 5, 7}),
        (1, 3, {2, 3, 4, 5, 6, 7, 8}),
        (1, 4, {2, 3, 4, 5, 6, 7, 8, 9}),
        (1, 5, {2, 3, 4, 5, 6, 7, 8, 9}),
    ],
)
def test_get_neighbours_up_to_distance(
    index: str, distance: int, expected: Set[str], grid_3_by_3_neighbourhood: Dict[str, Set[str]]
) -> None:
    """Test neighbours up to a distance."""
    neighbourhood = LookupNeighbourhood(grid_3_by_3_neighbourhood)
    neighbours = neighbourhood.get_neighbours_up_to_distance(index, distance)
    assert neighbours == expected


@pytest.mark.parametrize(  # type: ignore
    "index, distance, expected",
    [
        (1, -2, set()),
        (1, -1, set()),
        (1, 0, set()),
        (1, 1, {2, 3, 4}),
        (1, 2, set()),
        (2, 1, {1, 4}),
        (2, 2, {3}),
        (2, 3, set()),
        (3, 1, {1, 4}),
        (3, 2, {2}),
        (3, 3, set()),
        (4, 1, {1, 2, 3}),
        (4, 2, set()),
    ],
)
def test_get_neighbours_at_distance_irregular(
    index: str,
    distance: int,
    expected: Set[str],
    grid_3_by_3_irrregular_neighbourhood: Dict[str, Set[str]],
) -> None:
    """Test neighbours at distance."""
    neighbourhood = LookupNeighbourhood(grid_3_by_3_irrregular_neighbourhood)
    neighbours = neighbourhood.get_neighbours_at_distance(index, distance)
    assert neighbours == expected


@pytest.mark.parametrize(  # type: ignore
    "index, distance, expected",
    [
        (1, -2, set()),
        (1, -1, set()),
        (1, 0, set()),
        (1, 1, {2, 3, 4}),
        (1, 2, {2, 3, 4}),
        (2, 1, {1, 4}),
        (2, 2, {1, 3, 4}),
        (2, 3, {1, 3, 4}),
        (3, 1, {1, 4}),
        (3, 2, {1, 2, 4}),
        (3, 3, {1, 2, 4}),
        (4, 1, {1, 2, 3}),
        (4, 2, {1, 2, 3}),
    ],
)
def test_get_neighbours_up_to_distance_irregular(
    index: str,
    distance: int,
    expected: Set[str],
    grid_3_by_3_irrregular_neighbourhood: Dict[str, Set[str]],
) -> None:
    """Test neighbours up to a distance."""
    neighbourhood = LookupNeighbourhood(grid_3_by_3_irrregular_neighbourhood)
    neighbours = neighbourhood.get_neighbours_up_to_distance(index, distance)
    assert neighbours == expected
