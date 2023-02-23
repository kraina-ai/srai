"""Neighbourhood interface."""

from abc import ABC, abstractmethod
from typing import Any, List


class Neighbourhood(ABC):
    """Neighbourhood interface."""

    @abstractmethod
    def get_neighbours(self, index: Any) -> List[Any]:
        """
        Get the direct neighbours of a region using its index.

        Args:
            index (Any): Unique identifier of the region.
                Dependant on the implementation.

        Returns:
            List[Any]: Indexes of the neighbours.
        """
        pass
