"""
NeighbourDataset.

This dataset is used to train a model to predict whether regions are neighbours or not.
As defined in Hex2Vec paper[1].

References:
    [1] https://dl.acm.org/doi/10.1145/3486635.3491076
"""

from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar

import numpy as np
import pandas as pd
from tqdm import tqdm

from srai._optional import import_optional_dependencies
from srai.neighbourhoods import Neighbourhood

if TYPE_CHECKING:  # pragma: no cover
    import torch

try:  # pragma: no cover
    from torch.utils.data import Dataset

except ImportError:
    from srai.embedders._pytorch_stubs import Dataset


T = TypeVar("T")


class NeighbourDatasetItem(NamedTuple):
    """
    Neighbour dataset item.

    Attributes:
        X_anchor (torch.Tensor): Anchor regions.
        X_positive (torch.Tensor): Positive regions.
            Data for the regions that are neighbours of regions in X_anchor.
        X_negative (torch.Tensor): Negative regions.
            Data for the regions that are NOT neighbours of the regions in X_anchor.
    """

    X_anchor: "torch.Tensor"
    X_positive: "torch.Tensor"
    X_negative: "torch.Tensor"


class NeighbourDataset(Dataset[NeighbourDatasetItem], Generic[T]):  # type: ignore
    """
    Dataset for training a model to predict neighbours.

    It works by returning triplets of regions: anchor, positive and negative. A model can be trained
    to predict that the anchor region is a neighbour of the positive region, and that it is not a
    neighbour of the negative region.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        neighbourhood: Neighbourhood[T],
        negative_sample_k_distance: int = 2,
    ):
        """
        Initialize NeighbourDataset.

        Args:
            data (pd.DataFrame): Data to use for training. Raw counts of features in regions.
            neighbourhood (Neighbourhood[T]): Neighbourhood to use for training.
                It has to be initialized with the same data as the data argument.
            negative_sample_k_distance (int): How many neighbours away to sample negative regions.
                For example, if k=2, then the negative regions will be sampled from regions that are
                at least 3 hops away from the anchor region. Has to be >= 2.

        Raises:
            ValueError: If negative_sample_k_distance < 2.
        """
        import_optional_dependencies(dependency_group="torch", modules=["torch"])
        import torch

        self._data = torch.Tensor(data.to_numpy())
        self._assert_negative_sample_k_distance_correct(negative_sample_k_distance)
        self._negative_sample_k_distance = negative_sample_k_distance

        self._anchor_df_locs_lookup: np.ndarray
        self._positive_df_locs_lookup: np.ndarray
        self._excluded_from_negatives: dict[int, set[int]] = {}

        self._region_index_to_df_loc: dict[T, int] = {
            region_index: i for i, region_index in enumerate(data.index)
        }
        self._df_loc_to_region_index: dict[int, T] = {
            i: region_index for region_index, i in self._region_index_to_df_loc.items()
        }

        self._build_lookup_tables(data, neighbourhood)

    def _build_lookup_tables(self, data: pd.DataFrame, neighbourhood: Neighbourhood[T]) -> None:
        anchor_df_locs_lookup: list[int] = []
        positive_df_locs_lookup: list[int] = []

        for region_df_loc, region_index in tqdm(enumerate(data.index), total=len(data)):
            region_direct_neighbours = neighbourhood.get_neighbours(region_index)
            neighbours_df_locs: set[int] = {
                self._region_index_to_df_loc[neighbour_index]
                for neighbour_index in region_direct_neighbours
            }
            anchor_df_locs_lookup.extend([region_df_loc] * len(neighbours_df_locs))
            positive_df_locs_lookup.extend(neighbours_df_locs)

            indices_excluded_from_negatives = neighbourhood.get_neighbours_up_to_distance(
                region_index, self._negative_sample_k_distance
            )
            self._excluded_from_negatives[region_df_loc] = {
                self._region_index_to_df_loc[excluded_index]
                for excluded_index in indices_excluded_from_negatives
            }

        self._anchor_df_locs_lookup = np.array(anchor_df_locs_lookup)
        self._positive_df_locs_lookup = np.array(positive_df_locs_lookup)

    def __len__(self) -> int:
        """
        Return the number of anchor-positive pairs available in the dataset.

        Returns:
            int: The number of pairs.
        """
        return len(self._anchor_df_locs_lookup)

    def __getitem__(self, data_row_index: Any) -> NeighbourDatasetItem:
        """
        Return a single dataset item (anchor, positive, negative).

        Args:
            data_row_index (Any): The index of the dataset item to return.

        Returns:
            NeighbourDatasetItem: The dataset item.
                This includes the anchor region, positive region
                and arandomly sampled negative region.
        """
        anchor_df_loc = self._anchor_df_locs_lookup[data_row_index]
        positive_df_loc = self._positive_df_locs_lookup[data_row_index]
        negative_df_loc = self._get_random_negative_df_loc(anchor_df_loc)

        anchor_region = self._data[anchor_df_loc]
        positive_region = self._data[positive_df_loc]
        negative_region = self._data[negative_df_loc]

        return NeighbourDatasetItem(anchor_region, positive_region, negative_region)

    def _get_random_negative_df_loc(self, input_df_loc: int) -> int:
        excluded_df_locs = self._excluded_from_negatives[input_df_loc]
        negative_candidate: int = np.random.randint(0, len(self._data))  # noqa: NPY002
        while negative_candidate in excluded_df_locs:
            negative_candidate = np.random.randint(0, len(self._data))  # noqa: NPY002
        return negative_candidate

    def _assert_negative_sample_k_distance_correct(self, negative_sample_k_distance: int) -> None:
        if negative_sample_k_distance < 2:
            raise ValueError(
                "negative_sample_k_distance must be at least 2, "
                f"but was {negative_sample_k_distance}"
            )
