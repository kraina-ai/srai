"""
Contextual Count Embedder.

This module contains contextual count embedder implementation from ARIC@SIGSPATIAL 2021 paper [1].

References:
    1. https://doi.org/10.1145/3486626.3493434
    1. https://arxiv.org/abs/2111.00990
"""

from collections.abc import Collection, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from math import ceil
from multiprocessing import cpu_count
from typing import Any, Literal, Optional, Union

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from srai.constants import FORCE_TERMINAL
from srai.embedders.count_embedder import CountEmbedder
from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter, OsmTagsFilter
from srai.neighbourhoods import Neighbourhood
from srai.neighbourhoods._base import IndexType


class ContextualCountEmbedder(CountEmbedder):
    """ContextualCountEmbedder."""

    def __init__(
        self,
        neighbourhood: Neighbourhood[IndexType],
        neighbourhood_distance: int,
        concatenate_vectors: bool = False,
        expected_output_features: Optional[
            Union[list[str], OsmTagsFilter, GroupedOsmTagsFilter]
        ] = None,
        count_subcategories: bool = False,
        aggregation_function: Literal["average", "median", "sum", "min", "max"] = "average",
        num_of_multiprocessing_workers: int = -1,
        multiprocessing_activation_threshold: Optional[int] = None,
    ) -> None:
        """
        Init ContextualCountEmbedder.

        Args:
            neighbourhood (Neighbourhood[T]): Neighbourhood object used to get neighbours for
                the contextualization.
            neighbourhood_distance (int): How many neighbours levels should be included in
                the embedding.
            concatenate_vectors (bool, optional): Whether to sum all neighbours into a single vector
                with the same width as `CountEmbedder`, or to concatenate them to the wide format
                and keep all neighbour levels separate. Defaults to False.
            expected_output_features
                (Union[List[str], OsmTagsFilter, GroupedOsmTagsFilter], optional):
                The features that are expected to be found in the resulting embedding.
                If not None, the missing features are added and filled with 0.
                The unexpected features are removed. The resulting columns are sorted accordingly.
                Defaults to None.
            count_subcategories (bool, optional): Whether to count all subcategories individually
                or count features only on the highest level based on features column name.
                Defaults to False.
            aggregation_function (Literal["average", "median", "sum", "min", "max"], optional):
                Function used to aggregate data from the neighbours. Defaults to average.
            num_of_multiprocessing_workers (int, optional): Number of workers used for
                multiprocessing. Defaults to -1 which results in a total number of available
                cpu threads. `0` and `1` values disable multiprocessing.
                Similar to `n_jobs` parameter from `scikit-learn` library.
            multiprocessing_activation_threshold (int, optional): Number of seeds required to start
                processing on multiple processes. Activating multiprocessing for a small
                amount of points might not be feasible. Defaults to 100.

        Raises:
            ValueError: If `neighbourhood_distance` is negative.
        """
        super().__init__(expected_output_features, count_subcategories)

        self.neighbourhood = neighbourhood
        self.neighbourhood_distance = neighbourhood_distance
        self.concatenate_vectors = concatenate_vectors
        self.aggregation_function = aggregation_function

        if self.neighbourhood_distance < 0:
            raise ValueError("Neighbourhood distance must be positive.")

        self.num_of_multiprocessing_workers = _parse_num_of_multiprocessing_workers(
            num_of_multiprocessing_workers
        )
        self.multiprocessing_activation_threshold = _parse_multiprocessing_activation_threshold(
            multiprocessing_activation_threshold
        )

    def transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Embed a given GeoDataFrame.

        Creates region embeddings by counting the frequencies of each feature value and applying
        a contextualization based on neighbours of regions. For each region, features will be
        altered based on the neighbours either by adding aggregated values dimished based on
        distance, or by adding new separate columns with neighbour distance postfix.
        Expects features_gdf to be in wide format with each column being a separate type of
        feature (e.g. amenity, leisure) and rows to hold values of these features for each object.
        The rows will hold numbers of this type of feature in each region. Numbers can be
        fractional because neighbourhoods are aggregated to represent a single value from
        all neighbours on a given level.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.

        Returns:
            pd.DataFrame: Embedding for each region in regions_gdf.

        Raises:
            ValueError: If features_gdf is empty and self.expected_output_features is not set.
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
        """
        counts_df = super().transform(regions_gdf, features_gdf, joint_gdf)

        result_df: pd.DataFrame
        if self.concatenate_vectors:
            result_df = self._get_concatenated_embeddings(counts_df)
        else:
            result_df = self._get_squashed_embeddings(counts_df)

        return result_df

    def _get_squashed_embeddings(self, counts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate embeddings for regions by summing all neighbourhood levels.

        Creates embedding by getting an average of a neighbourhood at a given distance and adding it
        to the base values with weight equal to 1 / (distance + 1) squared. This way, farther
        neighbourhoods have lower impact on feature values.

        Args:
            counts_df (pd.DataFrame): Calculated features from CountEmbedder.

        Returns:
            pd.DataFrame: Embedding for each region in regions_gdf with number of features equal to
                the same as returned by the CountEmbedder.
        """
        base_columns = list(counts_df.columns)

        result_array = counts_df.values.astype(float)

        for distance, aggregated_values in self._get_aggregated_values_for_distances(counts_df):
            result_array += aggregated_values / ((distance + 1) ** 2)

        return pd.DataFrame(data=result_array, index=counts_df.index, columns=base_columns)

    def _get_concatenated_embeddings(self, counts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate embeddings for regions by concatenating different neighbourhood levels.

        Creates embedding by getting an average of a neighbourhood at a given distance and adding
        those features as separate columns with a postfix added to feature name. This way,
        all neighbourhoods can be analyzed separately, but number of columns grows linearly with
        a distance.

        Args:
            counts_df (pd.DataFrame): Calculated features from CountEmbedder.

        Returns:
            pd.DataFrame: Embedding for each region in regions_gdf with number of features equal to
                a number of features returned by the CountEmbedder multiplied
                by (neighbourhood distance + 1).
        """
        base_columns = list(counts_df.columns)
        number_of_base_columns = len(base_columns)
        columns = [
            f"{column}_{distance}"
            for distance in range(self.neighbourhood_distance + 1)
            for column in base_columns
        ]

        result_array = np.zeros(shape=(len(counts_df.index), len(columns)))
        result_array[:, 0:number_of_base_columns] = counts_df.values

        for distance, aggregated_values in self._get_aggregated_values_for_distances(counts_df):
            result_array[
                :,
                number_of_base_columns * distance : number_of_base_columns * (distance + 1),
            ] = aggregated_values

        return pd.DataFrame(data=result_array, index=counts_df.index, columns=columns)

    def _get_aggregated_values_for_distances(
        self, counts_df: pd.DataFrame
    ) -> Iterator[tuple[int, npt.NDArray[np.float32]]]:
        """
        Generate aggregated values for neighbours at given distances.

        Function will yield tuples of distances and aggregated values arrays
        calculated based on neighbours at a given distance.

        Distance 0 is skipped.
        If embedder has `neighbourhood_distance` set to 0, nothing will be returned.

        Args:
            counts_df (pd.DataFrame): Calculated features from CountEmbedder.

        Yields:
            Iterator[Tuple[int, npt.NDArray[np.float32]]]: Iterator of distances and values.
        """
        if self.neighbourhood_distance == 0:
            return

        number_of_base_columns = len(counts_df.columns)

        activate_multiprocessing = (
            self.num_of_multiprocessing_workers > 1
            and len(counts_df.index) >= self.multiprocessing_activation_threshold
        )

        with (
            tqdm(
                total=self.neighbourhood_distance * len(counts_df.index) * 2,
                desc="Generating embeddings for neighbours",
                disable=FORCE_TERMINAL,
            ) as pbar,
            ProcessPoolExecutor(max_workers=self.num_of_multiprocessing_workers) as executor,
        ):
            for distance in range(1, self.neighbourhood_distance + 1):
                pbar.set_postfix_str(f"Distance: {distance}", refresh=True)
                if len(counts_df.index) == 0:
                    continue

                neighbours_series = []

                if activate_multiprocessing:
                    fn_neighbours = partial(
                        _get_existing_neighbours_at_distance,
                        neighbour_distance=distance,
                        counts_index=counts_df.index,
                        neighbourhood=self.neighbourhood,
                    )

                    for result in executor.map(
                        fn_neighbours,
                        counts_df.index,
                        chunksize=ceil(
                            len(counts_df.index) / (4 * self.num_of_multiprocessing_workers)
                        ),
                    ):
                        neighbours_series.append(result)
                        pbar.update()
                else:
                    for result in counts_df.index.map(
                        lambda region_id, neighbour_distance=distance: counts_df.index.intersection(
                            self.neighbourhood.get_neighbours_at_distance(
                                region_id, neighbour_distance, include_center=False
                            )
                        ).values
                    ):
                        neighbours_series.append(result)
                        pbar.update()

                if not neighbours_series:
                    continue

                values_to_stack = []

                if activate_multiprocessing:
                    fn_embeddings = partial(
                        _get_embeddings_for_neighbours,
                        counts_df=counts_df,
                        aggregation_function=self.aggregation_function,
                        number_of_base_columns=number_of_base_columns,
                    )

                    for result in executor.map(
                        fn_embeddings,
                        neighbours_series,
                        chunksize=ceil(
                            len(neighbours_series) / (4 * self.num_of_multiprocessing_workers)
                        ),
                    ):
                        values_to_stack.append(result)
                        pbar.update()
                else:
                    for neighbours in neighbours_series:
                        values_to_stack.append(
                            _get_embeddings_for_neighbours(
                                region_ids=neighbours,
                                counts_df=counts_df,
                                aggregation_function=self.aggregation_function,
                                number_of_base_columns=number_of_base_columns,
                            )
                        )
                        pbar.update()

                aggregated_values_stacked = np.stack(values_to_stack)

                yield distance, aggregated_values_stacked


def _parse_num_of_multiprocessing_workers(num_of_multiprocessing_workers: int) -> int:
    if num_of_multiprocessing_workers == 0:
        num_of_multiprocessing_workers = 1
    elif num_of_multiprocessing_workers < 0:
        num_of_multiprocessing_workers = cpu_count()

    return num_of_multiprocessing_workers


def _parse_multiprocessing_activation_threshold(
    multiprocessing_activation_threshold: Optional[int],
) -> int:
    if not multiprocessing_activation_threshold:
        multiprocessing_activation_threshold = 100

    return multiprocessing_activation_threshold


def _get_existing_neighbours_at_distance(
    region_id: IndexType,
    neighbour_distance: int,
    neighbourhood: Neighbourhood[IndexType],
    counts_index: pd.Index,
) -> Any:
    return counts_index.intersection(
        neighbourhood.get_neighbours_at_distance(
            region_id, neighbour_distance, include_center=False
        )
    ).values


def _get_embeddings_for_neighbours(
    region_ids: Collection[IndexType],
    counts_df: pd.DataFrame,
    aggregation_function: Literal["average", "median", "sum", "min", "max"],
    number_of_base_columns: int,
) -> Any:
    if len(region_ids) == 0:
        return np.zeros((number_of_base_columns,))

    if aggregation_function == "average":
        aggregation = np.nanmean(counts_df.loc[region_ids].values, axis=0)
    elif aggregation_function == "median":
        aggregation = np.nanmedian(counts_df.loc[region_ids].values, axis=0)
    elif aggregation_function == "sum":
        aggregation = np.sum(counts_df.loc[region_ids].values, axis=0)
    elif aggregation_function == "min":
        aggregation = np.min(counts_df.loc[region_ids].values, axis=0)
    elif aggregation_function == "max":
        aggregation = np.max(counts_df.loc[region_ids].values, axis=0)
    else:
        raise ValueError(f"Unknown aggregation function: {aggregation_function}")

    return np.nan_to_num(aggregation)
