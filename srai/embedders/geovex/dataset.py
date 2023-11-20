"""
HexagonalDataset.

This dataset is used to train a hexagonal encoder model.
As defined in GeoVex paper[1].

References:
    [1] https://openreview.net/forum?id=7bvWopYY1H
"""

from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import pandas as pd
from tqdm import tqdm

from srai._optional import import_optional_dependencies
from srai.h3 import get_local_ij_index
from srai.neighbourhoods import H3Neighbourhood

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


class HexagonalDataset(Dataset["torch.Tensor"], Generic[T]):  # type: ignore
    """
    Dataset for the hexagonal encoder model.

    It works by returning a 3d tensor of hexagonal regions. The tensor is a cube with the target
    hexagonal region in the center, and the rings of neighbors around surrounding it.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        neighbourhood: H3Neighbourhood,
        neighbor_k_ring: int = 6,
    ):
        """
        Initialize the HexagonalDataset.

        Args:
            data (pd.DataFrame): Data to use for training. Raw counts of features in regions.
            neighbourhood (H3Neighbourhood): H3Neighbourhood to use for training.
                It has to be initialized with the same data as the data argument.
            neighbor_k_ring (int, optional): The hexagonal rings of neighbors to include
                in the tensor. Defaults to 6.
        """
        import_optional_dependencies(dependency_group="torch", modules=["torch"])
        import torch

        self._assert_k_ring_correct(neighbor_k_ring)
        self._assert_h3_neighbourhood(neighbourhood)
        # store the desired k
        self._k: int = neighbor_k_ring
        # number of columns in the dataset
        self._N: int = data.shape[1]
        # store the list of valid h3 indices (have all the neighbors in the dataset)
        self._valid_cells: list[CellInfo] = []
        # store the data as a torch tensor
        self._data_torch = torch.Tensor(data.to_numpy(dtype=np.float32))
        # iterate over the data and build the valid h3 indices
        self._invalid_cells, self._valid_cells = self._seperate_valid_invalid_cells(
            data, neighbourhood, neighbor_k_ring, set(data.index)
        )

    def _seperate_valid_invalid_cells(
        self,
        data: pd.DataFrame,
        neighbourhood: H3Neighbourhood,
        neighbor_k_ring: int,
        all_indices: set[str],
    ) -> tuple[set[str], list[CellInfo]]:
        invalid_h3s = set()
        valid_h3s = []

        for h3_index in tqdm(data.index, total=len(data)):
            neighbors = neighbourhood.get_neighbours_up_to_distance(
                h3_index, neighbor_k_ring, include_center=False, unchecked=True
            )
            # check if all the neighbors are in the dataset
            if len(neighbors.intersection(all_indices)) == len(neighbors):
                # add the h3_index to the valid h3 indices, with the ring of neighbors
                valid_h3s.append(
                    (
                        h3_index,
                        data.index.get_loc(h3_index),
                        [
                            # get the index of the h3 in the dataset
                            (data.index.get_loc(_h), get_local_ij_index(h3_index, _h))
                            for _h in neighbors
                        ],
                    )
                )
            else:
                # some of the neighbors are not in the dataset, add the h3_index to the invalid h3s
                invalid_h3s.add(h3_index)
        return invalid_h3s, valid_h3s

    def __len__(self) -> int:
        """
        Returns the number of valid h3 indices in the dataset.

        Returns:
            int: Number of valid h3 indices in the dataset.
        """
        return len(self._valid_cells)

    def __getitem__(self, index: Any) -> "torch.Tensor":
        """
        Return a single item from the dataset.

        Args:
            index (Any): The index of dataset item to return

        Returns:
            HexagonalDatasetItem: The dataset item
        """
        _, target_idx, neighbors_idxs = self._valid_cells[index]
        return self._build_tensor(target_idx, neighbors_idxs)

    def _build_tensor(self, target_idx: int, neighbors_idxs: Neighbors) -> "torch.Tensor":
        import torch

        # build the 3d tensor
        # it is a tensor with diagonals of length neighbor_k_ring
        # the diagonals are the neighbors of the target h3
        # the target h3 is in the center of the tensor
        # the tensor is 2*neighbor_k_ring + 1 x 2*neighbor_k_ring + 1 x 2*neighbor_k_ring + 1
        # make a tensor of zeros, padded by 1 zero all around to make it even for the convolutions
        tensor: torch.Tensor = torch.zeros(
            (
                self._N,
                2 * self._k + 2,
                2 * self._k + 2,
            )
        )

        # set the target h3 to the center of the tensor
        tensor[
            :,
            self._k,
            self._k,
        ] = self._data_torch[target_idx]

        # set the neighbors of the target h3 to the diagonals of the tensor
        for neighbor_idx, (i, j) in neighbors_idxs:
            tensor[:, self._k + i, self._k - j] = self._data_torch[neighbor_idx]

        # return the tensor and the target (which is same as the tensor)
        # should we return the target as a copy of the tensor?
        return tensor

    def _assert_k_ring_correct(self, k_ring: int) -> None:
        if k_ring < 2:
            raise ValueError(f"k_ring must be at least 2, but was {k_ring}")

    def _assert_h3_neighbourhood(self, neighbourhood: H3Neighbourhood) -> None:
        # force that the neighbourhood is an H3Neighbourhood,
        # because we need the get_ij_index method
        if not isinstance(neighbourhood, H3Neighbourhood):
            raise ValueError(
                f"neighbourhood has to be an H3Neighbourhood, but was {type(neighbourhood)}"
            )

    def get_valid_cells(self) -> list[str]:
        """
        Returns the list of valid h3 indices in the dataset.

        Returns:
            List[str]: List of valid h3 indices in the dataset.
        """
        return [h3_index for h3_index, _, _ in self._valid_cells]

    def get_invalid_cells(self) -> list[str]:
        """
        Returns the list of invalid h3 indices in the dataset.

        Returns:
            List[str]: List of invalid h3 indices in the dataset.
        """
        return list(self._invalid_cells)
