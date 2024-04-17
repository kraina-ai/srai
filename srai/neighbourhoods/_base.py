"""Neighbourhood interface."""

import operator
from abc import ABC, abstractmethod
from queue import Queue
from typing import Generic, Optional, TypeVar

from functional import seq

IndexType = TypeVar("IndexType")


class Neighbourhood(ABC, Generic[IndexType]):
    """
    Neighbourhood interface.

    This class abstracts away getting the neighbours of a region. It allows to get the neighbours at
    a certain distance or up to a certain distance. It is worth noting, that the distance here is
    not a metric distance, but a number of hops. This definition makes most sense semantically for
    grid systems such as H3 or S2 but should work for arbitrary neighbourhoods as well.

    The subclasses only need to implement the `get_neighbours` method, but can also override the
    `get_neighbours_up_to_distance` and `get_neighbours_at_distance` methods for performance
    reasons.  See the `H3Neighbourhood` class for an example. The class also provides a
    `_handle_center` method, which can be used to handle including/excluding the center region.
    """

    def __init__(self, include_center: bool = False) -> None:
        """
        Initializes the Neighbourhood.

        Args:
            include_center (bool): Whether to include the region itself in the neighbours.
            This is the default value used for all the methods of the class,
            unless overridden in the function call.
        """
        super().__init__()
        self.include_center = include_center

    @abstractmethod
    def get_neighbours(
        self, index: IndexType, include_center: Optional[bool] = None
    ) -> set[IndexType]:
        """
        Get the direct neighbours of a region using its index.

        Args:
            index (IndexType): Unique identifier of the region.
                Dependant on the implementation.
            include_center (Optional[bool]): Whether to include the region itself in the neighbours.
            If None, the value set in __init__ is used. Defaults to None.

        Returns:
            Set[IndexType]: Indexes of the neighbours.
        """

    def get_neighbours_up_to_distance(
        self, index: IndexType, distance: int, include_center: Optional[bool] = None
    ) -> set[IndexType]:
        """
        Get the neighbours of a region up to a certain distance.

        Args:
            index (IndexType): Unique identifier of the region.
                Dependant on the implementation.
            distance (int): Maximum distance to the neighbours.
            include_center (Optional[bool]): Whether to include the region itself in the neighbours.
            If None, the value set in __init__ is used. Defaults to None.

        Returns:
            Set[IndexType]: Indexes of the neighbours.
        """
        neighbours_with_distances = self._get_neighbours_with_distances(index, distance)
        neighbours: set[IndexType] = (
            seq(neighbours_with_distances).map(operator.itemgetter(0)).to_set()
        )
        neighbours = self._handle_center(
            index,
            distance,
            neighbours,
            at_distance=False,
            include_center_override=include_center,
        )
        return neighbours

    def get_neighbours_at_distance(
        self, index: IndexType, distance: int, include_center: Optional[bool] = None
    ) -> set[IndexType]:
        """
        Get the neighbours of a region at a certain distance.

        Args:
            index (IndexType): Unique identifier of the region.
                Dependant on the implementation.
            distance (int): Distance to the neighbours.
            include_center (Optional[bool]): Whether to include the region itself in the neighbours.
            If None, the value set in __init__ is used. Defaults to None.

        Returns:
            Set[IndexType]: Indexes of the neighbours.
        """
        neighbours_up_to_distance = self._get_neighbours_with_distances(index, distance)
        neighbours_at_distance: set[IndexType] = (
            seq(neighbours_up_to_distance)
            .filter(lambda x: x[1] == distance)
            .map(operator.itemgetter(0))
            .to_set()
        )
        neighbours_at_distance = self._handle_center(
            index,
            distance,
            neighbours_at_distance,
            at_distance=True,
            include_center_override=include_center,
        )
        return neighbours_at_distance

    def _get_neighbours_with_distances(
        self, index: IndexType, distance: int
    ) -> set[tuple[IndexType, int]]:
        visited_indexes: dict[IndexType, int] = {}
        to_visit: Queue[tuple[IndexType, int]] = Queue()

        to_visit.put((index, 0))
        while not to_visit.empty():
            current_index, current_distance = to_visit.get()

            visited_indexes[current_index] = min(
                current_distance, visited_indexes.get(current_index, distance)
            )
            if current_distance < distance:
                current_neighbours = self.get_neighbours(current_index)
                for neighbour in current_neighbours:
                    if neighbour not in visited_indexes:
                        to_visit.put((neighbour, current_distance + 1))

        return set(visited_indexes.items())

    def _handle_center(
        self,
        index: IndexType,
        distance: int,
        neighbours: set[IndexType],
        at_distance: bool,
        include_center_override: Optional[bool],
    ) -> set[IndexType]:
        if include_center_override is None:
            include_center = self.include_center
        else:
            include_center = include_center_override

        if distance < 0:
            return set()
        elif distance == 0:
            if include_center:
                neighbours.add(index)
            else:
                neighbours.discard(index)
        else:
            if at_distance:
                neighbours.discard(index)
            else:
                if include_center:
                    neighbours.add(index)
                else:
                    neighbours.discard(index)
        return neighbours
