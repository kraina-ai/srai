"""
gtfs2vec embedder.

This module contains embedder from gtfs2vec paper [1].

References:
    [1] https://doi.org/10.1145/3486640.3491392
"""


from functools import reduce
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from srai.embedders import BaseEmbedder
from srai.models import GTFS2VecModel
from srai.utils.constants import GTFS2VEC_DIRECTIONS_PREFIX, GTFS2VEC_TRIPS_PREFIX
from srai.utils.exceptions import ModelNotFitException


class GTFS2VecEmbedder(BaseEmbedder):
    """GTFS2Vec Embedder."""

    def __init__(
        self, hidden_size: int = 48, embedding_size: int = 64, skip_embedding: bool = False
    ) -> None:
        """
        Init GTFS2VecEmbedder.

        Args:
            hidden_size (int, optional): Hidden size in encoder and decoder. Defaults to 48.
            embedding_size (int, optional): Embedding size. Defaults to 64.
            skip_embedding (bool, optional): Skip embedding and return aggregated features instead.
            Defaults to False.
        """
        self._model: Optional[GTFS2VecModel] = None
        self._hidden_size = hidden_size
        self._embedding_size = embedding_size
        self._skip_embedding = skip_embedding

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
        features = self._prepare_features(regions_gdf, features_gdf, joint_gdf)

        if not self._skip_embedding:
            model = self._maybe_get_model()
            if len(features.columns) != model.n_features:
                raise ValueError(
                    f"Number of features in features_gdf ({len(features_gdf.columns)}) is "
                    f"incosistent with the model ({model.n_features})."
                )

        if self._skip_embedding:
            return features
        else:
            return self._embedd(features)

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
        features = self._prepare_features(regions_gdf, features_gdf, joint_gdf)

        if not self._skip_embedding:
            self._model = self._train_model_unsupervised(features)

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
        self._validate_indexes(regions_gdf, features_gdf, joint_gdf)
        features = self._prepare_features(regions_gdf, features_gdf, joint_gdf)

        if self._skip_embedding:
            return features
        else:
            self._model = self._train_model_unsupervised(features)
            return self._embedd(features)

    def _maybe_get_model(self) -> GTFS2VecModel:
        """Check if model is fit and return it."""
        if self._model is None:
            raise ModelNotFitException("Model not fit! Run fit() or fit_transform() first.")
        return self._model

    def _prepare_features(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Prepare features for embedding.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
        """
        regions_gdf = self._remove_geometry_if_present(regions_gdf)
        features_gdf = self._remove_geometry_if_present(features_gdf)
        joint_gdf = self._remove_geometry_if_present(joint_gdf)

        joint_features = (
            joint_gdf.join(features_gdf, on=features_gdf.index.name)
            .groupby(regions_gdf.index.name)
            .agg(self._get_columns_aggregation(features_gdf.columns))
        )

        regions_features = (
            regions_gdf.join(joint_features, on=regions_gdf.index.name).fillna(0).astype(int)
        )
        regions_features = self._normalize_features(regions_features)
        return regions_features

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
            if column.startswith(GTFS2VEC_TRIPS_PREFIX):
                agg_dict[column] = "sum"
            elif column.startswith(GTFS2VEC_DIRECTIONS_PREFIX):
                agg_dict[column] = lambda x: len(reduce(set.union, x))
        return agg_dict

    def _normalize_columns_group(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Normalize given columns in df.

        Args:
            df (pd.DataFrame): DataFrame to normalize.
            columns (list): List of columns.
        """
        df[columns] = (df[columns] - df[columns].min().min()) / (
            df[columns].max().max() - df[columns].min().min()
        )
        return df

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features.

        Args:
            features (pd.DataFrame): Features.

        Returns:
            pd.DataFrame: Normalized features.
        """
        norm_columns = [
            [
                column
                for column in features.columns
                if column.startswith(GTFS2VEC_DIRECTIONS_PREFIX)
            ],
            [column for column in features.columns if column.startswith(GTFS2VEC_TRIPS_PREFIX)],
        ]

        for columns in norm_columns:
            features = self._normalize_columns_group(features, columns)

        return features

    def _train_model_unsupervised(self, features: pd.DataFrame) -> GTFS2VecModel:
        """
        Train model unsupervised.

        Args:
            features (pd.DataFrame): Features.
        """
        model = GTFS2VecModel(
            n_features=len(features.columns),
            n_hidden=self._hidden_size,
            emb_size=self._embedding_size,
        )
        X = features.to_numpy().astype(np.float32)
        x_dataloader = DataLoader(X, batch_size=24, shuffle=True, num_workers=4)
        trainer = pl.Trainer(max_epochs=10)

        trainer.fit(model, x_dataloader)

        return model

    def _embedd(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Embedd features.

        Args:
            features (pd.DataFrame): Features.

        Returns:
            pd.DataFrame: Embeddings.
        """
        model = self._maybe_get_model()
        if len(features.columns) != model.n_features:
            raise ValueError(
                f"Features must have {model.n_features} columns but has {len(features.columns)}."
            )

        embeddings = (
            self._maybe_get_model()(torch.Tensor(features.to_numpy().astype(np.float32)))
            .detach()
            .numpy()
        )

        return pd.DataFrame(
            data=embeddings,
            index=features.index,
        )
