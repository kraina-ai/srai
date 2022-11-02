"""
Count Embedder.

This module contains count embedder implementation.

"""
import geopandas as gpd
import pandas as pd


class CountEmbedder:
    """Simple Embedder that counts features."""

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
        and rows holding values of these features for each object.
        The resulting GeoDataFrame will have columns made by combining
        the feature name (column) and value (row) e.g. amenity_fuel or type_0.
        The rows will hold numbers of this type of feature in each region.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.

        Returns:
            gpd.GeoDataFrame with embedding for each region.

        """
        joint_with_features = joint_gdf.join(features_gdf.drop("geometry", axis=1))
        region_embeddings = (
            pd.get_dummies(joint_with_features.drop("geometry", axis=1)).groupby(level=0).sum()
        )
        result_gdf = regions_gdf.join(region_embeddings, how="left").fillna(0)
        return result_gdf
