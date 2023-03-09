"""TODO: Docstring for neighbour_dataset.py."""
from typing import Any, Dict, Generic, List, Set, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from srai.neighbourhoods import Neighbourhood

T = TypeVar("T")

NeighbourDatasetItem = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]


class NeighbourDataset(Dataset[NeighbourDatasetItem], Generic[T]):  # type: ignore
    """TODO: Docstring for NeighbourDataset."""

    def __init__(
        self,
        data: pd.DataFrame,
        neighbourhood: Neighbourhood[T],
        negative_sample_k_distance: int = 2,
    ):
        """TODO: Docstring for __init__."""
        self._data = torch.Tensor(data.to_numpy())
        self._negative_sample_k_distance = negative_sample_k_distance

        self._input_df_locs_lookup: np.ndarray
        self._context_df_locs_lookup: np.ndarray
        self._excluded_from_negatives: Dict[int, Set[int]] = {}

        self.region_index_to_df_loc: Dict[T, int] = {
            region_index: i for i, region_index in enumerate(data.index)
        }
        self.df_loc_to_region_index: Dict[int, T] = {
            i: region_index for region_index, i in self.region_index_to_df_loc.items()
        }

        self._build_lookup_tables(data, neighbourhood)

    def _build_lookup_tables(self, data: pd.DataFrame, neighbourhood: Neighbourhood[T]) -> None:
        available_regions_indices: Set[T] = set(data.index)
        input_df_locs_lookup: List[int] = []
        context_df_locs_lookup: List[int] = []

        for region_df_loc, region_index in tqdm(enumerate(data.index), total=len(data)):
            region_direct_neighbours = neighbourhood.get_neighbours(region_index)
            neighbours_available_in_data = region_direct_neighbours.intersection(
                available_regions_indices
            )
            neighbours_df_locs: Set[int] = {
                self.region_index_to_df_loc[neighbour_index]
                for neighbour_index in neighbours_available_in_data
            }
            input_df_locs_lookup.extend([region_df_loc] * len(neighbours_df_locs))
            context_df_locs_lookup.extend(neighbours_df_locs)

            indices_excluded_from_negatives = neighbourhood.get_neighbours_up_to_distance(
                region_index, self._negative_sample_k_distance
            )
            available_excluded = indices_excluded_from_negatives.intersection(
                available_regions_indices
            )
            self._excluded_from_negatives[region_df_loc] = {
                self.region_index_to_df_loc[excluded_index] for excluded_index in available_excluded
            }

        self._input_df_locs_lookup = np.array(input_df_locs_lookup)
        self._context_df_locs_lookup = np.array(context_df_locs_lookup)

    def __len__(self) -> int:
        """
        Return the number of input-context pairs available in the dataset.

        Returns:
            int: The number of pairs.
        """
        return len(self._input_df_locs_lookup)

    def __getitem__(
        self, data_row_index: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """TODO: Docstring for __getitem__."""
        input_df_loc = self._input_df_locs_lookup[data_row_index]
        context_df_loc = self._context_df_locs_lookup[data_row_index]
        negative_df_loc = self._get_random_negative_df_loc(input_df_loc)

        input = self._data[input_df_loc]
        context = self._data[context_df_loc]
        negative = self._data[negative_df_loc]

        y_pos = 1.0
        y_neg = 0.0

        return input, context, negative, y_pos, y_neg

    def _get_random_negative_df_loc(self, input_df_loc: int) -> int:
        excluded_df_locs = self._excluded_from_negatives[input_df_loc]
        negative_candidate: int = np.random.randint(0, len(self._data))
        while negative_candidate in excluded_df_locs:
            negative_candidate = np.random.randint(0, len(self._data))
        return negative_candidate
