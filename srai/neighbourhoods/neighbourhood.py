"""Neighbourhood interface."""

from abc import ABC, abstractmethod
from queue import Queue
from typing import Generic, Set, Tuple, TypeVar

from functional import seq

IndexType = TypeVar("IndexType")


class Neighbourhood(ABC, Generic[IndexType]):
    """
    Neighbourhood interface.

    This class abstracts away getting the neighbours of a region.
    It allows to get the neighbours at a certain distance or up to a certain distance.
    It is worth noting, that the distance here is not a metric distance, but a number of hops.
    This definition makes most sense semantically for grid systems such as H3 or S2 but should work
    for arbitrary neighbourhoods as well.

    The subclasses only need to implement the `get_neighbours` method, but can also override the
    `get_neighbours_up_to_distance` and `get_neighbours_at_distance` methods for performance
    reasons.
    See the `H3Neighbourhood` class for an example.
    """

    @abstractmethod
    def get_neighbours(self, index: IndexType) -> Set[IndexType]:
        """
        Get the direct neighbours of a region using its index.

        Args:
            index (IndexType): Unique identifier of the region.
                Dependant on the implementation.

        Returns:
            Set[IndexType]: Indexes of the neighbours.
        """
        pass

    def get_neighbours_up_to_distance(self, index: IndexType, distance: int) -> Set[IndexType]:
        """
        Get the neighbours of a region up to a certain distance.

        Args:
            index (IndexType): Unique identifier of the region.
                Dependant on the implementation.
            distance (int): Maximum distance to the neighbours.

        Returns:
            List[IndexType]: Indexes of the neighbours.
        """
        neighbours_with_distances = self._get_neighbours_with_distances(index, distance)
        neighbours: Set[IndexType] = seq(neighbours_with_distances).map(lambda x: x[0]).to_set()
        neighbours.discard(index)
        return neighbours

    def get_neighbours_at_distance(self, index: IndexType, distance: int) -> Set[IndexType]:
        """
        Get the neighbours of a region at a certain distance.

        Args:
            index (IndexType): Unique identifier of the region.
                Dependant on the implementation.
            distance (int): Distance to the neighbours.

        Returns:
            List[IndexType]: Indexes of the neighbours.
        """
        neighbours_up_to_distance = self._get_neighbours_with_distances(index, distance)
        neighbours_at_distance: Set[IndexType] = (
            seq(neighbours_up_to_distance)
            .filter(lambda x: x[1] == distance)
            .map(lambda x: x[0])
            .to_set()
        )
        neighbours_at_distance.discard(index)
        return neighbours_at_distance

    def _get_neighbours_with_distances(
        self, index: IndexType, distance: int
    ) -> Set[Tuple[IndexType, int]]:
        visited = set()
        visited_with_distances = set()
        to_visit: Queue[Tuple[IndexType, int]] = Queue()

        to_visit.put((index, 0))

        while not to_visit.empty():
            current, current_distance = to_visit.get()

            visited.add(current)
            visited_with_distances.add((current, current_distance))
            if current_distance < distance:
                current_neighbours = self.get_neighbours(current)
                for neighbour in current_neighbours:
                    if neighbour not in visited:
                        to_visit.put((neighbour, current_distance + 1))

        return visited_with_distances
