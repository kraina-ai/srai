"""
S2VecDataset.

This dataset is used to train a S2 masked autoencoder model.
As defined in S2Vec paper[1].

References:
    [1] https://arxiv.org/abs/2504.16942
"""

from typing import TYPE_CHECKING, Any, Generic, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from srai._optional import import_optional_dependencies
from srai.constants import FORCE_TERMINAL

if TYPE_CHECKING:  # pragma: no cover
    import torch

try:  # pragma: no cover
    from torch.utils.data import Dataset

except ImportError:
    from srai.embedders._pytorch_stubs import Dataset

T = TypeVar("T")

# define a type for the dataset item
Ij_Index = tuple[int, int]
Neighbors = list[tuple[int, Ij_Index]]
CellInfo = tuple[str, int, Neighbors]


class S2VecDataset(Dataset["torch.Tensor"], Generic[T]):  # type: ignore
    """
    Dataset for the S2 masked autoencoder.

    It works by returning a 3d tensor of square S2 regions.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        img_patch_joint_gdf: gpd.GeoDataFrame,
    ):
        """
        Initialize the S2VecDataset.

        Args:
            data (pd.DataFrame): Data to use for training. Raw counts of features in regions.
            img_patch_joint_gdf (gpd.GeoDataFrame): GeoDataFrame with the images and patches
            S2 indices.
        """
        import_optional_dependencies(dependency_group="torch", modules=["torch"])
        import torch

        # number of columns in the dataset
        self._N: int = data.shape[1]
        # store the data as a torch tensor
        self._data_torch = torch.Tensor(data.to_numpy(dtype=np.float32))

        self.patch_s2_ids = data.index.tolist()

        self._input_ids = [
            [data.index.get_loc(index) for index in group.index.get_level_values(1)]
            for _, group in tqdm(img_patch_joint_gdf.groupby(level=0), disable=FORCE_TERMINAL)
        ]

    def __len__(self) -> int:
        """
        Returns the number of inputs in the dataset.

        Returns:
            int: Number of inputs in the dataset.
        """
        return len(self._input_ids)

    def __getitem__(self, index: Any) -> "torch.Tensor":
        """
        Return a single item from the dataset.

        Args:
            index (Any): The index of dataset item to return

        Returns:
            torch.Tensor: The dataset item
        """
        patch_idxs = self._input_ids[index]
        return self._data_torch[patch_idxs]
