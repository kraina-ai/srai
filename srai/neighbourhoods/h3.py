"""
H3 neighbourhood.

This module contains the H3Neighbourhood class, that allows to get the neighbours of an H3 region.
"""
from typing import Set

import h3

from .neighbourhood import Neighbourhood


class H3Neighbourhood(Neighbourhood[str]):
    """
    H3 Neighbourhood.

    This class allows to get the neighbours of an H3 region.
    """

    def get_neighbours(self, index: str) -> Set[str]:
        """
        Get the direct neighbours of an H3 region using its index.

        Args:
            index (str): H3 index of the region.

        Returns:
            Set[str]: Indexes of the neighbours.
        """
        return self.get_neighbours_up_to_distance(index, 1)

    def get_neighbours_up_to_distance(self, index: str, distance: int) -> Set[str]:
        """
        Get the neighbours of an H3 region up to a certain distance.

        Args:
            index (str): H3 index of the region.
            distance (int): Distance to the neighbours.

        Returns:
            Set[str]: Indexes of the neighbours up to the given distance.
        """
        if self._distance_incorrect(distance):
            return set()

        neighbours: Set[str] = h3.grid_disk(index, distance)
        neighbours.discard(index)
        return neighbours

    def get_neighbours_at_distance(self, index: str, distance: int) -> Set[str]:
        """
        Get the neighbours of an H3 region at a certain distance.

        Args:
            index (str): H3 index of the region.
            distance (int): Distance to the neighbours.

        Returns:
            Set[str]: Indexes of the neighbours at the given distance.
        """
        if self._distance_incorrect(distance):
            return set()

        neighbours: Set[str] = h3.grid_ring(index, distance)
        neighbours.discard(index)
        return neighbours

    def _distance_incorrect(self, distance: int) -> bool:
        return distance <= 0
