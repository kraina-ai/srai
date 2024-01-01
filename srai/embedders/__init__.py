"""
This module contains embedders, used to convert spatial data to their vector representations.

Embedders are designed to unify different types of spatial data embedding methods, such as hex2vec
or gtfs2vec into a single interface. This allows to easily switch between different embedding
methods without changing the rest of the code.
"""

from ._base import Embedder, Model, ModelT
from .contextual_count_embedder import ContextualCountEmbedder
from .count_embedder import CountEmbedder
from .geovex import GeoVexEmbedder
from .gtfs2vec import GTFS2VecEmbedder
from .hex2vec import Hex2VecEmbedder
from .highway2vec import Highway2VecEmbedder

__all__ = [
    "Embedder",
    "CountEmbedder",
    "ContextualCountEmbedder",
    "GTFS2VecEmbedder",
    "Hex2VecEmbedder",
    "Highway2VecEmbedder",
    "GeoVexEmbedder",
    "Model",
    "ModelT",
]
