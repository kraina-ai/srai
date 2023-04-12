"""
Contextual Count Embedder.

This module contains contextual count embedder implementation from ARIC@SIGSPATIAL 2021 paper[1].

References:
    [1] https://doi.org/10.1145/3486626.3493434
    [1] https://arxiv.org/abs/2111.00990
"""

from typing import List, Optional, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from srai.embedders import CountEmbedder
from srai.neighbourhoods import Neighbourhood

T = TypeVar("T")


class ContextualCountEmbedder(CountEmbedder):
    """ContextualCountEmbedder."""

    def __init__(
        self,
        neighbourhood: Neighbourhood[T],
        neighbourhood_distance: int,
        squash_vectors: bool = True,
        expected_output_features: Optional[List[str]] = None,
        count_subcategories: bool = False,
    ) -> None:
        """TODO."""
        # """
        # Init ContextualCountEmbedder.

        # Args:
        #     expected_output_features (List[str], optional): The features that are expected
        #         to be found in the resulting embedding. If not None, the missing features are
        # added
        #         and filled with 0. The unexpected features are removed.
        #         The resulting columns are sorted accordingly. Defaults to None.
        #     count_subcategories (bool, optional): Whether to count all subcategories individually
        #         or count features only on the highest level based on features column name.
        #         Defaults to True.
        # """
        super().__init__(expected_output_features, count_subcategories)

        self.neighbourhood = neighbourhood
        self.neighbourhood_distance = neighbourhood_distance
        self.squash_vectors = squash_vectors

    def transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """TODO."""
        counts_df = super().transform(regions_gdf, features_gdf, joint_gdf)

        if self.squash_vectors:
            return self._get_squashed_embeddings(counts_df)
        else:
            return self._get_concatenated_embeddings(counts_df)

    def _get_squashed_embeddings(self, counts_df: pd.DataFrame) -> pd.DataFrame:
        base_columns = list(counts_df.columns)

        result_array = counts_df.values.astype(float)
        for idx, region_id in tqdm(enumerate(counts_df.index), desc="Generating embeddings"):
            for distance in range(1, self.neighbourhood_distance + 1):
                neighbours = self.neighbourhood.get_neighbours_at_distance(region_id, distance)
                matching_neighbours = counts_df.index.intersection(neighbours)
                if not matching_neighbours.empty:
                    values = counts_df.loc[matching_neighbours].values
                    flattened_values = np.average(values, axis=0)
                    result_array[idx, :] += flattened_values / ((distance + 1) ** 2)
        return pd.DataFrame(data=result_array, index=counts_df.index, columns=base_columns)

    def _get_concatenated_embeddings(self, counts_df: pd.DataFrame) -> pd.DataFrame:
        base_columns = list(counts_df.columns)
        no_base_columns = len(base_columns)

        columns = [
            f"{column}_{distance}"
            for distance in range(self.neighbourhood_distance + 1)
            for column in base_columns
        ]
        result_array = np.zeros(shape=(len(counts_df.index), len(columns)))
        result_array[:, 0:no_base_columns] = counts_df.values
        for idx, region_id in tqdm(enumerate(counts_df.index), desc="Generating embeddings"):
            for distance in range(1, self.neighbourhood_distance + 1):
                neighbours = self.neighbourhood.get_neighbours_at_distance(region_id, distance)
                matching_neighbours = counts_df.index.intersection(neighbours)
                if not matching_neighbours.empty:
                    values = counts_df.loc[matching_neighbours].values
                    flattened_values = np.average(values, axis=0)
                    result_array[
                        idx, no_base_columns * distance : no_base_columns * (distance + 1)
                    ] = flattened_values
        return pd.DataFrame(data=result_array, index=counts_df.index, columns=columns)
