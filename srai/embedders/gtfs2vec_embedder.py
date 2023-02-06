"""
gtfs2vec embedder.

This module contains embedder from gtfs2vec paper [1].

References:
    [1] https://doi.org/10.1145/3486640.3491392

"""


import geopandas as gpd
import pandas as pd

from srai.embedders import BaseEmbedder


class GTFS2VecEmbedder(BaseEmbedder):
    """GTFS2Vec Embedder."""

    def __init__(self) -> None:
        """Init GTFS2VecEmbedder."""

    def transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Embed a given GeoDataFrame.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.

        Returns:
            pd.DataFrame: Embedding and geometry index for each region in regions_gdf.

        Raises:
            ValueError: If features_gdf is empty and self.expected_output_features is not set.
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.

        """
        self._validate_indexes(regions_gdf, features_gdf, joint_gdf)
