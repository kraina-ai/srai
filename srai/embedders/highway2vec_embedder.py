"""Highway2Vec Embedder."""
from typing import List, Optional

import geopandas as gpd
import pandas as pd


class Highway2VecEmbedder:
    """Simple Embedder that counts occurences of feature values."""

    def __init__(self, expected_output_features: Optional[List[str]] = None) -> None:
        """
        Init CountEmbedder.

        Args:
            expected_output_features (List[str], optional): The features that are expected
                to be found in the resulting embedding. If not None, the missing features are added
                and filled with 0. The unexpected features are removed.
                The resulting columns are sorted accordingly. Defaults to None.

        """
        self.expected_output_features = (
            pd.Series(expected_output_features) if expected_output_features is not None else None
        )

    def embed(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Embed a given GeoDataFrame.

        Creates region embeddings by counting the frequencies of each feature value.
        Expects features_gdf to be in wide format with each column
        being a separate type of feature (e.g. amenity, leisure)
        and rows to hold values of these features for each object.
        The resulting GeoDataFrame will have columns made by combining
        the feature name (column) and value (row) e.g. amenity_fuel or type_0.
        The rows will hold numbers of this type of feature in each region.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.

        Returns:
            gpd.GeoDataFrame: Embedding and geometry for each region in regions_gdf.

        """
        joint_with_features = joint_gdf.join(features_gdf.drop("geometry", axis=1))
        region_embeddings = (
            pd.get_dummies(joint_with_features.drop("geometry", axis=1)).groupby(level=0).sum()
        )
        region_embeddings = self._maybe_filter_to_expected_features(region_embeddings)
        region_embedding_gdf = regions_gdf.join(region_embeddings, how="left").fillna(0)

        return region_embedding_gdf

    def _maybe_filter_to_expected_features(self, region_embeddings: pd.DataFrame) -> pd.DataFrame:
        """
        Filter columns if expected_output_features provided.

        Args:
            region_embeddings (pd.DataFrame): Counted frequencies of each feature value.

        Returns:
            pd.DataFrame: region_embeddings either unchanged
                or with expected columns only.

        """
        if self.expected_output_features is not None:
            return self._filter_to_expected_features(region_embeddings)
        return region_embeddings

    def _filter_to_expected_features(self, region_embeddings: pd.DataFrame) -> pd.DataFrame:
        """
        Add missing and remove excessive columns from embeddings.

        Args:
            region_embeddings (pd.DataFrame): Counted frequencies of each feature value.

        Returns:
            pd.DataFrame: Embeddings with expected columns only.

        """
        missing_features = self.expected_output_features[
            ~self.expected_output_features.isin(region_embeddings.columns)
        ]
        region_embeddings[missing_features] = 0
        region_embeddings = region_embeddings[self.expected_output_features]
        return region_embeddings
