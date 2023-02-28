"""Neighbourhood interface."""

from abc import ABC, abstractmethod
from queue import Queue
from typing import Generic, Set, Tuple, TypeVar

from functional import seq

IndexType = TypeVar("IndexType")


class Neighbourhood(ABC, Generic[IndexType]):
    """Neighbourhood interface."""

    @abstractmethod
    def get_neighbours(self, index: IndexType) -> Set[IndexType]:
        """
        Get the direct neighbours of a region using its index.

        Args:
            index (Any): Unique identifier of the region.
                Dependant on the implementation.

        Returns:
            Set[Any]: Indexes of the neighbours.
        """
        pass

    def get_neighbours_up_to_distance(self, index: IndexType, distance: int) -> Set[IndexType]:
        """
        Get the neighbours of a region up to a certain distance.

        Args:
            index (Any): Unique identifier of the region.
                Dependant on the implementation.
            distance (int): Maximum distance to the neighbours.

        Returns:
            List[Any]: Indexes of the neighbours.
        """
        neighbours_with_distances = self._get_neighbours_with_distances(index, distance)
        neighbours: Set[IndexType] = seq(neighbours_with_distances).map(lambda x: x[0]).to_set()
        if index in neighbours:
            neighbours.remove(index)
        return neighbours

    def get_neighbours_at_distance(self, index: IndexType, distance: int) -> Set[IndexType]:
        """
        Get the neighbours of a region at a certain distance.

        Args:
            index (Any): Unique identifier of the region.
                Dependant on the implementation.
            distance (int): Distance to the neighbours.

        Returns:
            List[Any]: Indexes of the neighbours.
        """
        neighbours_up_to_distance = self._get_neighbours_with_distances(index, distance)
        neighbours_at_distance: Set[IndexType] = (
            seq(neighbours_up_to_distance)
            .filter(lambda x: x[1] == distance)
            .map(lambda x: x[0])
            .to_set()
        )
        if index in neighbours_at_distance:
            neighbours_at_distance.remove(index)
        return neighbours_at_distance

    def _get_neighbours_with_distances(
        self, index: IndexType, distance: int
    ) -> Set[Tuple[IndexType, int]]:
        visited = set()
        visited_with_distances = set()
        to_visit: Queue[Tuple[IndexType, int]] = Queue()

        visited.add(index)
        visited_with_distances.add((index, 0))
        to_visit.put((index, 0))

        while not to_visit.empty():
            current, current_distance = to_visit.get()

            visited.add(current)
            if current_distance < distance:
                current_neighbours = self.get_neighbours(current)
                for neighbour in current_neighbours:
                    if neighbour not in visited:
                        to_visit.put((neighbour, current_distance + 1))
                        visited_with_distances.add((neighbour, current_distance + 1))
                        visited.add(neighbour)

        return visited_with_distances
