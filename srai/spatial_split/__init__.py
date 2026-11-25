"""
Module for spatially splitting input data.

Points are grouped together by the region they are in, and whole regions are assigned to the splits.
This way the splits don't cover the same areas. Additionally, the distribution of the points or of
the target is kept similar in all splits.
"""

from .kfold import spatial_kfold
from .ratio_split import spatial_split
from .train_test_split import spatial_split_points, train_test_spatial_split

__all__ = [
    "train_test_spatial_split",
    "spatial_split_points",
    "spatial_split",
    "spatial_kfold",
]
