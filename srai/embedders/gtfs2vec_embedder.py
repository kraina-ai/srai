"""
gtfs2vec embedder.

This module contains embedder from gtfs2vec paper [1].

References:
    [1] https://doi.org/10.1145/3486640.3491392

"""


from functools import reduce
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd

from srai.embedders import BaseEmbedder
from srai.models import GTFS2VecModel
from srai.utils.exceptions import ModelNotFitException


class GTFS2VecEmbedder(BaseEmbedder):
    """GTFS2Vec Embedder."""

    TRIP_COLUMNS_PREFIX = "trip_count_at_"
    DIRECTIONS_COLUMNS_PREFIX = "directions_at_"

    def __init__(self) -> None:
        """Init GTFS2VecEmbedder."""
        self._model: Optional[GTFS2VecModel] = None

    def transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Embed a given data.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.

        Returns:
            pd.DataFrame: Embedding and geometry index for each region in regions_gdf.

        Raises:
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
            ValueError: If number of features is incosistent with the model.
            ModelNotFitException: If model is not fit.

        """
        self._validate_indexes(regions_gdf, features_gdf, joint_gdf)
        if self._model is None:
            raise ModelNotFitException("Model not fit! Run fit() or fit_transform() first.")

        if len(features_gdf.columns) != self._model.n_features:
            raise ValueError(
                f"Number of features in features_gdf ({len(features_gdf.columns)}) is "
                f"incosistent with the model ({self._model.n_features})."
            )

        regions_gdf = self._remove_geometry_if_present(regions_gdf)
        features_gdf = self._remove_geometry_if_present(features_gdf)
        joint_gdf = self._remove_geometry_if_present(joint_gdf)

        joint_features = (
            joint_gdf.join(features_gdf, on="feature_id")
            .groupby("region_id")
            .agg(self._get_columns_aggregation(features_gdf.columns))
        )

        regions_features = regions_gdf.join(joint_features, on="region_id").fillna(0).astype(int)

        return regions_features

    def fit(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> None:
        """
        Fit model to a given data.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.

        Raises:
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.

        """
        self._validate_indexes(regions_gdf, features_gdf, joint_gdf)
        self._model = GTFS2VecModel(n_features=len(features_gdf.columns))

    def fit_transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Fit model and transform a given data.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.

        Returns:
            pd.DataFrame: Embedding and geometry index for each region in regions_gdf.

        Raises:
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.

        """
        self.fit(regions_gdf=regions_gdf, features_gdf=features_gdf, joint_gdf=joint_gdf)
        return self.transform(
            regions_gdf=regions_gdf, features_gdf=features_gdf, joint_gdf=joint_gdf
        )

    def _get_columns_aggregation(self, columns: List[str]) -> Dict[str, Any]:
        """
        Get aggregation dict for given columns.

        Args:
            columns (list): List of columns.

        Returns:
            dict: Aggregation dict.

        """
        agg_dict: Dict[str, Any] = {}

        for column in columns:
            if column.startswith(self.TRIP_COLUMNS_PREFIX):
                agg_dict[column] = "sum"
            elif column.startswith(self.DIRECTIONS_COLUMNS_PREFIX):
                agg_dict[column] = lambda x: len(reduce(set.union, x))
        return agg_dict
