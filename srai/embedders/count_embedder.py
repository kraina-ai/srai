"""
Count Embedder.

This module contains count embedder implementation.
"""
from typing import List, Optional

import geopandas as gpd
import pandas as pd

from srai.embedders import Embedder


class CountEmbedder(Embedder):
    """Simple Embedder that counts occurences of feature values."""

    def __init__(
        self, expected_output_features: Optional[List[str]] = None, count_subcategories: bool = True
    ) -> None:
        """
        Init CountEmbedder.

        Args:
            expected_output_features (List[str], optional): The features that are expected
                to be found in the resulting embedding. If not None, the missing features are added
                and filled with 0. The unexpected features are removed.
                The resulting columns are sorted accordingly. Defaults to None.
            count_subcategories (bool, optional): Whether to count all subcategories individually
                or count features only on the highest level based on features column name.
                Defaults to True.
        """
        self.expected_output_features: Optional[pd.Series] = (
            None if expected_output_features is None else pd.Series(expected_output_features)
        )

        self.count_subcategories = count_subcategories

    def transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Embed a given GeoDataFrame.

        Creates region embeddings by counting the frequencies of each feature value.
        Expects features_gdf to be in wide format with each column
        being a separate type of feature (e.g. amenity, leisure)
        and rows to hold values of these features for each object.
        The resulting DataFrame will have columns made by combining
        the feature name (column) and value (row) e.g. amenity_fuel or type_0.
        The rows will hold numbers of this type of feature in each region.

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
        self._validate_indexes(regions_gdf, features_gdf, joint_gdf)
        if features_gdf.empty:
            if self.expected_output_features is not None:
                return pd.DataFrame(
                    0, index=regions_gdf.index, columns=self.expected_output_features
                )
            else:
                raise ValueError(
                    "Cannot embed with empty features_gdf and no expected_output_features."
                )

        regions_df = self._remove_geometry_if_present(regions_gdf)
        features_df = self._remove_geometry_if_present(features_gdf)
        joint_df = self._remove_geometry_if_present(joint_gdf)

        if self.count_subcategories:
            feature_encodings = pd.get_dummies(features_df)
        else:
            feature_encodings = features_df.notna().astype(int)
        joint_with_encodings = joint_df.join(feature_encodings)
        region_embeddings = joint_with_encodings.groupby(level=0).sum()

        region_embeddings = self._maybe_filter_to_expected_features(region_embeddings)
        region_embedding_df = regions_df.join(region_embeddings, how="left").fillna(0).astype(int)

        return region_embedding_df

    def _maybe_filter_to_expected_features(self, region_embeddings: pd.DataFrame) -> pd.DataFrame:
        """
        Add missing and remove excessive columns from embeddings.

        Args:
            region_embeddings (pd.DataFrame): Counted frequencies of each feature value.

        Returns:
            pd.DataFrame: Embeddings with expected columns only.
        """
        if self.expected_output_features is None:
            return region_embeddings

        missing_features = self.expected_output_features[
            ~self.expected_output_features.isin(region_embeddings.columns)
        ]
        missing_features_df = pd.DataFrame(
            0, index=region_embeddings.index, columns=missing_features
        )
        region_embeddings = pd.concat([region_embeddings, missing_features_df], axis=1)
        region_embeddings = region_embeddings[self.expected_output_features]
        return region_embeddings
