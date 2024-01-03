"""
This module contains neighbourhood calculation methods.

Some embedding methods require a neighbourhood to be defined. This module contains
neighbourhood calculation methods - both dedicated to a specific regionalization method
and general ones.
"""

from ._base import Neighbourhood
from .adjacency_neighbourhood import AdjacencyNeighbourhood
from .h3_neighbourhood import H3Neighbourhood

__all__ = ["Neighbourhood", "AdjacencyNeighbourhood", "H3Neighbourhood"]
