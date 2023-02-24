"""
H3 neighbourhood.

This module contains the H3Neighbourhood class, that allows to get the neighbours of an H3 region.
"""
from typing import List, Set

import h3

from .neighbourhood import Neighbourhood


class H3Neighbourhood(Neighbourhood[str]):
    """
    H3 Neighbourhood.

    This class allows to get the neighbours of an H3 region.
    """

    def get_neighbours(self, index: str, exclude_anchor: bool = True) -> List[str]:
        """
        Get the direct neighbours of an H3 region using its index.

        Args:
            index (str): H3 index of the region.
            exclude_anchor (bool): Whether to exclude the anchor index from the neighbours.
                Defaults to True.

        Returns:
            List[str]: Indexes of the neighbours.
        """
        return self.get_neighbours_up_to_distance(index, 1, exclude_anchor)

    def get_neighbours_up_to_distance(
        self, index: str, distance: int, exclude_anchor: bool = True
    ) -> List[str]:
        """
        Get the neighbours of an H3 region up to a certain distance.

        Args:
            index (str): H3 index of the region.
            distance (int): Distance to the neighbours.
            exclude_anchor (bool): Whether to exclude the anchor index from the neighbours.

        Returns:
            List[str]: Indexes of the neighbours.
        """
        neighbours: Set[str] = h3.grid_disk(index, distance)
        if exclude_anchor and index in neighbours:
            neighbours.remove(index)
        return list(neighbours)

    def get_neighbours_at_distance(self, index: str, distance: int) -> List[str]:
        """
        Get the neighbours of an H3 region at a certain distance.

        Args:
            index (str): _description_
            distance (int): _description_

        Returns:
            List[str]: _description_
        """
        neighbours: List[str] = h3.grid_ring(index, distance)
        return neighbours
