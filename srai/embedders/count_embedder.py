"""
Count Embedder.

This module contains count embedder implementation.

"""
import geopandas as gpd


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
        regions_geometry = regions_gdf[["geometry"]]
        joint_features = joint.drop(columns="geometry")
        joint_index_names = joint_features.index.names
        joint_long = (
            joint_features.reset_index()
            .melt(id_vars=joint_index_names)
            .dropna(axis=0)
            .set_index(joint_index_names)
        )
        joint_long["val"] = joint_long["variable"] + "_" + joint_long["value"]
        feature_counts = joint_long.groupby(level=0)["val"].value_counts().unstack(fill_value=0)
        result = regions_geometry.join(feature_counts, how="left")
        return result
