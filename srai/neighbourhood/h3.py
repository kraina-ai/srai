"""
H3 neighbourhood.

This module contains the H3Neighbourhood class, that allows to get the neighbours of an H3 region.
"""
from typing import List

import h3

from .neighbourhood import Neighbourhood


class H3Neighbourhood(Neighbourhood):
    """
    H3 Neighbourhood.

    This class allows to get the neighbours of an H3 region.
    """

    def get_neighbours(self, index: str, exclude_anchor: bool = True) -> List[str]:
        """
        Get the neighbours of an H3 region using its index.

        Args:
            index (str): Unique identifier of the region.
                Depends on the implementation.
            exclude_anchor (bool): Whether to exclude the anchor index from the neighbours.

        Returns:
            List[str]: Indexes of the neighbours.
        """
        if exclude_anchor:
            neighbour_function = h3.grid_ring
        else:
            neighbour_function = h3.grid_disk
        neighbours: List[str] = neighbour_function(index, 1)
        return neighbours
