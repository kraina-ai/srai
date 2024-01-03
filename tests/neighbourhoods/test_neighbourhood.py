"""Tests for LookupNeighbourhood."""

from typing import Any, Optional, TypeVar

import pytest

from srai.neighbourhoods import Neighbourhood

T = TypeVar("T")


class LookupNeighbourhood(Neighbourhood[T]):
    """LookupNeighbourhood."""

    def __init__(self, lookup: dict[T, set[T]], include_center: bool = False) -> None:
        """
        LookupNeighbourhood constructor.

        Args:
            lookup (Dict[T, Set[T]]): Mapping of region to its neighbours.
            include_center (bool): Whether to include the region itself in the neighbours.
        """
        super().__init__(include_center)
        self.lookup = lookup

    def get_neighbours(self, index: T, include_center: Optional[bool] = None) -> set[T]:
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
def grid_3_by_3_neighbourhood() -> dict[int, set[int]]:
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
def grid_3_by_3_irrregular_neighbourhood() -> dict[int, set[int]]:
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
    "neighbourhood_fixture,index,expected",
    [
        ("grid_3_by_3_neighbourhood", 1, {2, 4}),
        ("grid_3_by_3_neighbourhood", 2, {1, 3, 5}),
        ("grid_3_by_3_neighbourhood", 3, {2, 6}),
        ("grid_3_by_3_neighbourhood", 4, {1, 5, 7}),
        ("grid_3_by_3_neighbourhood", 5, {2, 4, 6, 8}),
        ("grid_3_by_3_neighbourhood", 6, {3, 5, 9}),
        ("grid_3_by_3_neighbourhood", 7, {4, 8}),
        ("grid_3_by_3_neighbourhood", 8, {5, 7, 9}),
        ("grid_3_by_3_neighbourhood", 9, {6, 8}),
        ("grid_3_by_3_irrregular_neighbourhood", 1, {2, 3, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 2, {1, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 3, {1, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 4, {1, 2, 3}),
    ],
)
def test_get_neighbours(
    neighbourhood_fixture: str,
    index: str,
    expected: set[str],
    request: Any,
) -> None:
    """Test get_neighbours with overriding include_center."""
    # Test with class include_center=False
    neighbourhood_data = request.getfixturevalue(neighbourhood_fixture)
    expected_with_include_center = expected | {index}

    neighbourhood = LookupNeighbourhood(neighbourhood_data)
    neighbours = neighbourhood.get_neighbours(index)
    assert neighbours == expected

    neighbours = neighbourhood.get_neighbours(index, include_center=True)
    assert neighbours == expected_with_include_center

    # Test with class include_center=True
    neighbourhood = LookupNeighbourhood(neighbourhood_data, include_center=True)
    neighbours = neighbourhood.get_neighbours(index)
    assert neighbours == expected_with_include_center

    neighbours = neighbourhood.get_neighbours(index, include_center=False)
    assert neighbours == expected


@pytest.mark.parametrize(  # type: ignore
    "neighbourhood_fixture, index, distance, expected, expected_with_include_center",
    [
        ("grid_3_by_3_neighbourhood", 5, -2, set(), set()),
        ("grid_3_by_3_neighbourhood", 5, -1, set(), set()),
        ("grid_3_by_3_neighbourhood", 5, 0, set(), {5}),
        ("grid_3_by_3_neighbourhood", 5, 1, {2, 4, 6, 8}, {2, 4, 6, 8}),
        ("grid_3_by_3_neighbourhood", 5, 2, {1, 3, 7, 9}, {1, 3, 7, 9}),
        ("grid_3_by_3_neighbourhood", 5, 3, set(), set()),
        ("grid_3_by_3_neighbourhood", 4, 0, set(), {4}),
        ("grid_3_by_3_neighbourhood", 4, 1, {1, 5, 7}, {1, 5, 7}),
        ("grid_3_by_3_neighbourhood", 4, 2, {2, 6, 8}, {2, 6, 8}),
        ("grid_3_by_3_neighbourhood", 4, 3, {3, 9}, {3, 9}),
        ("grid_3_by_3_neighbourhood", 4, 4, set(), set()),
        ("grid_3_by_3_neighbourhood", 1, 0, set(), {1}),
        ("grid_3_by_3_neighbourhood", 1, 1, {2, 4}, {2, 4}),
        ("grid_3_by_3_neighbourhood", 1, 2, {3, 5, 7}, {3, 5, 7}),
        ("grid_3_by_3_neighbourhood", 1, 3, {6, 8}, {6, 8}),
        ("grid_3_by_3_neighbourhood", 1, 4, {9}, {9}),
        ("grid_3_by_3_neighbourhood", 1, 5, set(), set()),
        ("grid_3_by_3_irrregular_neighbourhood", 1, -2, set(), set()),
        ("grid_3_by_3_irrregular_neighbourhood", 1, -1, set(), set()),
        ("grid_3_by_3_irrregular_neighbourhood", 1, 0, set(), {1}),
        ("grid_3_by_3_irrregular_neighbourhood", 1, 1, {2, 3, 4}, {2, 3, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 1, 2, set(), set()),
        ("grid_3_by_3_irrregular_neighbourhood", 2, 0, set(), {2}),
        ("grid_3_by_3_irrregular_neighbourhood", 2, 1, {1, 4}, {1, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 2, 2, {3}, {3}),
        ("grid_3_by_3_irrregular_neighbourhood", 2, 3, set(), set()),
        ("grid_3_by_3_irrregular_neighbourhood", 3, 0, set(), {3}),
        ("grid_3_by_3_irrregular_neighbourhood", 3, 1, {1, 4}, {1, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 3, 2, {2}, {2}),
        ("grid_3_by_3_irrregular_neighbourhood", 3, 3, set(), set()),
        ("grid_3_by_3_irrregular_neighbourhood", 4, 0, set(), {4}),
        ("grid_3_by_3_irrregular_neighbourhood", 4, 1, {1, 2, 3}, {1, 2, 3}),
        ("grid_3_by_3_irrregular_neighbourhood", 4, 2, set(), set()),
    ],
)
def test_get_neighbours_at_distance(
    neighbourhood_fixture: str,
    index: str,
    distance: int,
    expected: set[str],
    expected_with_include_center: set[str],
    request: Any,
) -> None:
    """Test neighbours at distance."""
    neighbourhood_data = request.getfixturevalue(neighbourhood_fixture)
    neighbourhood = LookupNeighbourhood(neighbourhood_data)
    neighbours = neighbourhood.get_neighbours_at_distance(index, distance)
    assert neighbours == expected

    neighbours = neighbourhood.get_neighbours_at_distance(index, distance, include_center=True)
    assert neighbours == expected_with_include_center

    neighbourhood = LookupNeighbourhood(neighbourhood_data, include_center=True)
    neighbours = neighbourhood.get_neighbours_at_distance(index, distance)
    assert neighbours == expected_with_include_center

    neighbours = neighbourhood.get_neighbours_at_distance(index, distance, include_center=False)
    assert neighbours == expected


@pytest.mark.parametrize(  # type: ignore
    "neighbourhood_fixture, index, distance, expected",
    [
        ("grid_3_by_3_neighbourhood", 5, -2, set()),
        ("grid_3_by_3_neighbourhood", 5, -1, set()),
        ("grid_3_by_3_neighbourhood", 5, 0, set()),
        ("grid_3_by_3_neighbourhood", 5, 1, {2, 4, 6, 8}),
        ("grid_3_by_3_neighbourhood", 5, 2, {1, 2, 3, 4, 6, 7, 8, 9}),
        ("grid_3_by_3_neighbourhood", 5, 3, {1, 2, 3, 4, 6, 7, 8, 9}),
        ("grid_3_by_3_neighbourhood", 4, 0, set()),
        ("grid_3_by_3_neighbourhood", 4, 1, {1, 5, 7}),
        ("grid_3_by_3_neighbourhood", 4, 2, {1, 2, 5, 6, 7, 8}),
        ("grid_3_by_3_neighbourhood", 4, 3, {1, 2, 3, 5, 6, 7, 8, 9}),
        ("grid_3_by_3_neighbourhood", 4, 4, {1, 2, 3, 5, 6, 7, 8, 9}),
        ("grid_3_by_3_neighbourhood", 1, 0, set()),
        ("grid_3_by_3_neighbourhood", 1, 1, {2, 4}),
        ("grid_3_by_3_neighbourhood", 1, 2, {2, 3, 4, 5, 7}),
        ("grid_3_by_3_neighbourhood", 1, 3, {2, 3, 4, 5, 6, 7, 8}),
        ("grid_3_by_3_neighbourhood", 1, 4, {2, 3, 4, 5, 6, 7, 8, 9}),
        ("grid_3_by_3_neighbourhood", 1, 5, {2, 3, 4, 5, 6, 7, 8, 9}),
        ("grid_3_by_3_irrregular_neighbourhood", 1, -2, set()),
        ("grid_3_by_3_irrregular_neighbourhood", 1, -1, set()),
        ("grid_3_by_3_irrregular_neighbourhood", 1, 0, set()),
        ("grid_3_by_3_irrregular_neighbourhood", 1, 1, {2, 3, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 1, 2, {2, 3, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 2, 0, set()),
        ("grid_3_by_3_irrregular_neighbourhood", 2, 1, {1, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 2, 2, {1, 3, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 2, 3, {1, 3, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 3, 0, set()),
        ("grid_3_by_3_irrregular_neighbourhood", 3, 1, {1, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 3, 2, {1, 2, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 3, 3, {1, 2, 4}),
        ("grid_3_by_3_irrregular_neighbourhood", 4, 0, set()),
        ("grid_3_by_3_irrregular_neighbourhood", 4, 1, {1, 2, 3}),
        ("grid_3_by_3_irrregular_neighbourhood", 4, 2, {1, 2, 3}),
    ],
)
def test_get_neighbours_up_to_distance(
    neighbourhood_fixture: str, index: str, distance: int, expected: set[str], request: Any
) -> None:
    """Test neighbours up to a distance."""
    neighbourhood_data = request.getfixturevalue(neighbourhood_fixture)
    expected_with_include_center = expected.copy()
    if distance >= 0:
        expected_with_include_center.add(index)
    neighbourhood = LookupNeighbourhood(neighbourhood_data)
    neighbours = neighbourhood.get_neighbours_up_to_distance(index, distance)
    assert neighbours == expected

    neighbours = neighbourhood.get_neighbours_up_to_distance(index, distance, include_center=True)
    assert neighbours == expected_with_include_center

    neighbourhood = LookupNeighbourhood(neighbourhood_data, include_center=True)
    neighbours = neighbourhood.get_neighbours_up_to_distance(index, distance)
    assert neighbours == expected_with_include_center

    neighbours = neighbourhood.get_neighbours_up_to_distance(index, distance, include_center=False)
    assert neighbours == expected
