"""
Count Embedder.

This module contains count embedder implementation.

"""
import geopandas as gpd
import pandas as pd


class CountEmbedder:
    """Simple Embedder that counts features."""

    def embed(
        self, regions_gdf: gpd.GeoDataFrame, features_gdf: gpd.GeoDataFrame, joint: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Embed a given GeoDataFrame by counting features.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be embedded.

        Returns:
            gpd.GeoDataFrame with embedding for each region.

        """
        joint_with_features = joint.join(features_gdf.drop("geometry", axis=1))
        region_embeddings = (
            pd.get_dummies(joint_with_features.drop("geometry", axis=1)).groupby(level=0).sum()
        )
        result_gdf = regions_gdf.join(region_embeddings, how="left").fillna(0)
        return result_gdf
