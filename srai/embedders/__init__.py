"""
This module contains embedders, used to convert spatial data to their vector representations.

Embedders are designed to unify different types of spatial data embedding methods, such as hex2vec
or gtfs2vec into a single interface. This allows to easily switch between different embedding
methods without changing the rest of the code.
"""

from ._base import Embedder
from .count_embedder import CountEmbedder
from .gtfs2vec import GTFS2VecEmbedder
from .highway2vec import Highway2VecEmbedder

__all__ = ["Embedder", "CountEmbedder", "GTFS2VecEmbedder", "Highway2VecEmbedder"]
