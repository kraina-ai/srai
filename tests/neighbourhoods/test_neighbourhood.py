"""Tests for LookupNeighbourhood."""
from typing import Dict, Optional, Set, TypeVar

import pytest

from srai.neighbourhoods import Neighbourhood

T = TypeVar("T")


class LookupNeighbourhood(Neighbourhood[T]):
    """LookupNeighbourhood."""

    def __init__(self, lookup: Dict[T, Set[T]], include_center: bool = False) -> None:
        """
        LookupNeighbourhood constructor.

        Args:
            lookup (Dict[T, Set[T]]): Mapping of region to its neighbours.
            include_center (bool): Whether to include the region itself in the neighbours.
        """
        super().__init__(include_center)
        self.lookup = lookup

    def get_neighbours(self, index: T, include_center: Optional[bool] = None) -> Set[T]:
        """
        Get neighbours for region at index.

        Args:
            index (T): Index of region in mapping.
            include_center (Optional[bool]): Whether to include the region itself in the neighbours.
            If None, the value set in __init__ is used. Defaults to None.
        """
        neighbours = self.lookup[index]
        return self._handle_center(
            index, 1, neighbours, at_distance=False, include_center_override=include_center
        )


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
    "index, expected",
    [
        (1, {2, 4}),
        (2, {1, 3, 5}),
        (3, {2, 6}),
        (4, {1, 5, 7}),
        (5, {2, 4, 6, 8}),
        (6, {3, 5, 9}),
        (7, {4, 8}),
        (8, {5, 7, 9}),
        (9, {6, 8}),
    ],
)
def test_get_neighbours(
    index: str, expected: Set[str], grid_3_by_3_neighbourhood: Dict[str, Set[str]]
) -> None:
    """Test get_neighbours."""
    neighbourhood = LookupNeighbourhood(grid_3_by_3_neighbourhood)
    neighbours = neighbourhood.get_neighbours(index)
    assert neighbours == expected


@pytest.mark.parametrize(  # type: ignore
    "index, expected",
    [
        (1, {2, 4, 1}),
        (2, {1, 3, 5, 2}),
        (3, {2, 6, 3}),
        (4, {1, 5, 7, 4}),
        (5, {2, 4, 6, 8, 5}),
        (6, {3, 5, 9, 6}),
        (7, {4, 8, 7}),
        (8, {5, 7, 9, 8}),
        (9, {6, 8, 9}),
    ],
)
def test_get_neighbours_include_center(
    index: str, expected: Set[str], grid_3_by_3_neighbourhood: Dict[str, Set[str]]
) -> None:
    """Test get_neighbours with include_center=True."""
    neighbourhood = LookupNeighbourhood(grid_3_by_3_neighbourhood, include_center=True)
    neighbours = neighbourhood.get_neighbours(index)
    assert neighbours == expected


@pytest.mark.parametrize(  # type: ignore
    "index, expected",
    [
        (1, {2, 3, 4}),
        (2, {1, 4}),
        (3, {1, 4}),
        (4, {1, 2, 3}),
    ],
)
def test_get_neighbours_irregular(
    index: str,
    expected: Set[str],
    grid_3_by_3_irrregular_neighbourhood: Dict[str, Set[str]],
) -> None:
    """Test get_neighbours with irregular neighbourhood."""
    neighbourhood = LookupNeighbourhood(grid_3_by_3_irrregular_neighbourhood)
    neighbours = neighbourhood.get_neighbours(index)
    assert neighbours == expected


@pytest.mark.parametrize(  # type: ignore
    "index, expected",
    [
        (1, {2, 3, 4, 1}),
        (2, {1, 4, 2}),
        (3, {1, 4, 3}),
        (4, {1, 2, 3, 4}),
    ],
)
def test_get_neighbours_irregular_include_center(
    index: str,
    expected: Set[str],
    grid_3_by_3_irrregular_neighbourhood: Dict[str, Set[str]],
) -> None:
    """Test get_neighbours with irregular neighbourhood and include_center=True."""
    neighbourhood = LookupNeighbourhood(grid_3_by_3_irrregular_neighbourhood, include_center=True)
    neighbours = neighbourhood.get_neighbours(index)
    assert neighbours == expected


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
        (5, 0, {5}),
        (5, 1, {2, 4, 6, 8}),
        (5, 2, {1, 3, 7, 9}),
        (5, 3, set()),
        (4, 0, {4}),
        (4, 1, {1, 5, 7}),
        (4, 2, {2, 6, 8}),
        (4, 3, {3, 9}),
        (4, 4, set()),
        (1, 0, {1}),
        (1, 1, {2, 4}),
        (1, 2, {3, 5, 7}),
        (1, 3, {6, 8}),
        (1, 4, {9}),
        (1, 5, set()),
    ],
)
def test_get_neighbours_at_distance_include_center(
    index: str, distance: int, expected: Set[str], grid_3_by_3_neighbourhood: Dict[str, Set[str]]
) -> None:
    """Test neighbours at distance."""
    neighbourhood = LookupNeighbourhood(grid_3_by_3_neighbourhood, include_center=True)
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
        (5, -2, set()),
        (5, -1, set()),
        (5, 0, {5}),
        (5, 1, {5, 2, 4, 6, 8}),
        (5, 2, {5, 1, 2, 3, 4, 6, 7, 8, 9}),
        (5, 3, {5, 1, 2, 3, 4, 6, 7, 8, 9}),
        (4, 0, {4}),
        (4, 1, {4, 1, 5, 7}),
        (4, 2, {4, 1, 2, 5, 6, 7, 8}),
        (4, 3, {4, 1, 2, 3, 5, 6, 7, 8, 9}),
        (4, 4, {4, 1, 2, 3, 5, 6, 7, 8, 9}),
        (1, 0, {1}),
        (1, 1, {1, 2, 4}),
        (1, 2, {1, 2, 3, 4, 5, 7}),
        (1, 3, {1, 2, 3, 4, 5, 6, 7, 8}),
        (1, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9}),
        (1, 5, {1, 2, 3, 4, 5, 6, 7, 8, 9}),
    ],
)
def test_get_neighbours_up_to_distance_include_center(
    index: str, distance: int, expected: Set[str], grid_3_by_3_neighbourhood: Dict[str, Set[str]]
) -> None:
    """Test neighbours up to a distance."""
    neighbourhood = LookupNeighbourhood(grid_3_by_3_neighbourhood, include_center=True)
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
        (1, 0, {1}),
        (1, 1, {2, 3, 4}),
        (1, 2, set()),
        (2, 0, {2}),
        (2, 1, {1, 4}),
        (2, 2, {3}),
        (2, 3, set()),
        (3, 0, {3}),
        (3, 1, {1, 4}),
        (3, 2, {2}),
        (3, 3, set()),
        (4, 0, {4}),
        (4, 1, {1, 2, 3}),
        (4, 2, set()),
    ],
)
def test_get_neighbours_at_distance_irregular_include_center(
    index: str,
    distance: int,
    expected: Set[str],
    grid_3_by_3_irrregular_neighbourhood: Dict[str, Set[str]],
) -> None:
    """Test neighbours at distance."""
    neighbourhood = LookupNeighbourhood(grid_3_by_3_irrregular_neighbourhood, include_center=True)
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


@pytest.mark.parametrize(  # type: ignore
    "index, distance, expected",
    [
        (1, -2, set()),
        (1, -1, set()),
        (1, 0, {1}),
        (1, 1, {1, 2, 3, 4}),
        (1, 2, {1, 2, 3, 4}),
        (2, 0, {2}),
        (2, 1, {2, 1, 4}),
        (2, 2, {2, 1, 3, 4}),
        (2, 3, {2, 1, 3, 4}),
        (3, 0, {3}),
        (3, 1, {3, 1, 4}),
        (3, 2, {3, 1, 2, 4}),
        (3, 3, {3, 1, 2, 4}),
        (4, 0, {4}),
        (4, 1, {4, 1, 2, 3}),
        (4, 2, {4, 1, 2, 3}),
    ],
)
def test_get_neighbours_up_to_distance_irregular_include_center(
    index: str,
    distance: int,
    expected: Set[str],
    grid_3_by_3_irrregular_neighbourhood: Dict[str, Set[str]],
) -> None:
    """Test neighbours up to a distance."""
    neighbourhood = LookupNeighbourhood(grid_3_by_3_irrregular_neighbourhood, include_center=True)
    neighbours = neighbourhood.get_neighbours_up_to_distance(index, distance)
    assert neighbours == expected


@pytest.mark.parametrize(  # type: ignore
    "index, expected, expected_with_include_center",
    [
        (1, {2, 4}, {2, 4, 1}),
        (2, {1, 3, 5}, {1, 3, 5, 2}),
        (3, {2, 6}, {2, 6, 3}),
        (4, {1, 5, 7}, {1, 5, 7, 4}),
        (5, {2, 4, 6, 8}, {2, 4, 6, 8, 5}),
        (6, {3, 5, 9}, {3, 5, 9, 6}),
        (7, {4, 8}, {4, 8, 7}),
        (8, {5, 7, 9}, {5, 7, 9, 8}),
        (9, {6, 8}, {6, 8, 9}),
    ],
)
def test_get_neighbours_include_center_override(
    index: str,
    expected: Set[str],
    expected_with_include_center: Set[str],
    grid_3_by_3_neighbourhood: Dict[str, Set[str]],
) -> None:
    """Test get_neighbours with overriding include_center."""
    # Test with class include_center=False
    neighbourhood = LookupNeighbourhood(grid_3_by_3_neighbourhood)
    neighbours = neighbourhood.get_neighbours(index)
    assert neighbours == expected

    neighbours = neighbourhood.get_neighbours(index, include_center=True)
    assert neighbours == expected_with_include_center

    # Test with class include_center=True
    neighbourhood = LookupNeighbourhood(grid_3_by_3_neighbourhood, include_center=True)
    neighbours = neighbourhood.get_neighbours(index)
    assert neighbours == expected_with_include_center

    neighbours = neighbourhood.get_neighbours(index, include_center=False)
    assert neighbours == expected
