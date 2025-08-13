"""
Gtfs2vec embedder.

This module contains embedder from gtfs2vec paper [1].

References:
    1. https://doi.org/10.1145/3486640.3491392
"""

import json
from collections.abc import Iterable
from functools import reduce
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from srai._optional import import_optional_dependencies
from srai.constants import GEOMETRY_COLUMN
from srai.embedders import Embedder, ModelT
from srai.embedders.gtfs2vec.model import GTFS2VecModel
from srai.exceptions import ModelNotFitException
from srai.geodatatable import VALID_DATA_INPUT, ParquetDataTable, prepare_data_input
from srai.loaders.gtfs_loader import GTFS2VEC_DIRECTIONS_PREFIX, GTFS2VEC_TRIPS_PREFIX


class GTFS2VecEmbedder(Embedder):
    """GTFS2Vec Embedder."""

    def __init__(
        self,
        hidden_size: int = 48,
        embedding_size: int = 64,
        skip_autoencoder: bool = False,
    ) -> None:
        """
        Init GTFS2VecEmbedder.

        Args:
            hidden_size (int, optional): Hidden size in encoder and decoder. Defaults to 48.
            embedding_size (int, optional): Embedding size. Defaults to 64.
            skip_autoencoder (bool, optional): Skip using autoencoder as part of embedding.
            Defaults to False.
        """
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )
        self._model: Optional[GTFS2VecModel] = None
        self._hidden_size = hidden_size
        self._embedding_size = embedding_size
        self._skip_autoencoder = skip_autoencoder
        self._is_fitted = False

    def transform(
        self,
        regions: VALID_DATA_INPUT,
        features: VALID_DATA_INPUT,
        joint: VALID_DATA_INPUT,
    ) -> ParquetDataTable:
        """
        Embed a given data.

        Args:
            regions (VALID_DATA_INPUT): Region indexes and geometries.
            features (VALID_DATA_INPUT): Feature indexes, geometries and feature values.
            joint (VALID_DATA_INPUT): Joiner result with region-feature multi-index.

        Returns:
            ParquetDataTable: Embedding and geometry index for each region in regions.

        Raises:
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
            ValueError: If number of features is incosistent with the model.
            ModelNotFitException: If model is not fit.
        """
        regions_pdt = prepare_data_input(regions).drop_columns([GEOMETRY_COLUMN], missing_ok=True)
        features_pdt = prepare_data_input(features).drop_columns([GEOMETRY_COLUMN], missing_ok=True)
        joint_pdt = prepare_data_input(joint).drop_columns([GEOMETRY_COLUMN], missing_ok=True)

        self._validate_indexes(regions_pdt, features_pdt, joint_pdt)
        gtfs_features = self._prepare_features(
            regions_pdt.to_dataframe(),
            features_pdt.to_dataframe(),
            joint_pdt.to_dataframe(),
        )

        result_file_name = ParquetDataTable.generate_filename()
        result_parquet_path = (
            ParquetDataTable.get_directory() / f"{result_file_name}_embeddings.parquet"
        )

        if self._skip_autoencoder:
            return ParquetDataTable.from_dataframe(gtfs_features, result_parquet_path)

        embedded_gtfs_features = self._embed(gtfs_features)
        return ParquetDataTable.from_dataframe(embedded_gtfs_features, result_parquet_path)

    def fit(
        self,
        regions: VALID_DATA_INPUT,
        features: VALID_DATA_INPUT,
        joint: VALID_DATA_INPUT,
        trainer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Fit model to a given data.

        Args:
            regions (VALID_DATA_INPUT): Region indexes and geometries.
            features (VALID_DATA_INPUT): Feature indexes, geometries and feature values.
            joint (VALID_DATA_INPUT): Joiner result with region-feature multi-index.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs. Defaults to None.

        Raises:
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint.index doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
        """
        regions_pdt = prepare_data_input(regions).drop_columns([GEOMETRY_COLUMN], missing_ok=True)
        features_pdt = prepare_data_input(features).drop_columns([GEOMETRY_COLUMN], missing_ok=True)
        joint_pdt = prepare_data_input(joint).drop_columns([GEOMETRY_COLUMN], missing_ok=True)

        self._validate_indexes(regions_pdt, features_pdt, joint_pdt)
        gtfs_features = self._prepare_features(
            regions_pdt.to_dataframe(),
            features_pdt.to_dataframe(),
            joint_pdt.to_dataframe(),
        )

        if not self._skip_autoencoder:
            self._model = self._train_model_unsupervised(gtfs_features, trainer_kwargs)

    def fit_transform(
        self,
        regions: VALID_DATA_INPUT,
        features: VALID_DATA_INPUT,
        joint: VALID_DATA_INPUT,
        trainer_kwargs: Optional[dict[str, Any]] = None,
    ) -> ParquetDataTable:
        """
        Fit model and transform a given data.

        Args:
            regions (VALID_DATA_INPUT): Region indexes and geometries.
            features (VALID_DATA_INPUT): Feature indexes, geometries and feature values.
            joint (VALID_DATA_INPUT): Joiner result with region-feature multi-index.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs. Defaults to None.

        Returns:
            ParquetDataTable: Embedding and geometry index for each region in regions.

        Raises:
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
        """
        regions_pdt = prepare_data_input(regions).drop_columns([GEOMETRY_COLUMN], missing_ok=True)
        features_pdt = prepare_data_input(features).drop_columns([GEOMETRY_COLUMN], missing_ok=True)
        joint_pdt = prepare_data_input(joint).drop_columns([GEOMETRY_COLUMN], missing_ok=True)

        self._validate_indexes(regions_pdt, features_pdt, joint_pdt)
        gtfs_features = self._prepare_features(
            regions_pdt.to_dataframe(),
            features_pdt.to_dataframe(),
            joint_pdt.to_dataframe(),
        )

        result_file_name = ParquetDataTable.generate_filename()
        result_parquet_path = (
            ParquetDataTable.get_directory() / f"{result_file_name}_embeddings.parquet"
        )

        if self._skip_autoencoder:
            return ParquetDataTable.from_dataframe(gtfs_features, result_parquet_path)

        self._model = self._train_model_unsupervised(gtfs_features, trainer_kwargs)
        embedded_gtfs_features = self._embed(gtfs_features)
        return ParquetDataTable.from_dataframe(embedded_gtfs_features, result_parquet_path)

    def _maybe_get_model(self) -> GTFS2VecModel:
        """Check if model is fit and return it."""
        if self._model is None:
            raise ModelNotFitException("Model not fit! Run fit() or fit_transform() first.")
        return self._model

    def _prepare_features(
        self,
        regions_df: pd.DataFrame,
        features_df: pd.DataFrame,
        joint_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare features for embedding.

        Args:
            regions_df (pd.DataFrame): Region indexes and geometries.
            features_df (pd.DataFrame): Feature indexes, geometries and feature values.
            joint_df (pd.DataFrame): Joiner result with region-feature multi-index.
        """
        joint_features = (
            joint_df.join(features_df, on=features_df.index.name)
            .groupby(regions_df.index.name)
            .agg(self._get_columns_aggregation(features_df.columns))
        )

        regions_features = (
            regions_df.join(joint_features, on=regions_df.index.name).fillna(0).astype(int)
        )
        regions_features = self._normalize_features(regions_features)
        return regions_features

    def _get_columns_aggregation(self, columns: list[str]) -> dict[str, Any]:
        """
        Get aggregation dict for given columns.

        Args:
            columns (list): List of columns.

        Returns:
            dict: Aggregation dict.
        """
        agg_dict: dict[str, Any] = {}

        for column in columns:
            if column.startswith(GTFS2VEC_TRIPS_PREFIX):
                agg_dict[column] = "sum"
            elif column.startswith(GTFS2VEC_DIRECTIONS_PREFIX):
                agg_dict[column] = lambda x: len(
                    reduce(set.union, (set(val) for val in x if isinstance(val, Iterable)), set())
                )
        return agg_dict

    def _normalize_columns_group(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
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

    def _train_model_unsupervised(
        self,
        features: pd.DataFrame,
        trainer_kwargs: Optional[dict[str, Any]] = None,
    ) -> GTFS2VecModel:
        """
        Train model unsupervised.

        Args:
            features (pd.DataFrame): Features.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs. Defaults to None.
        """
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader

        model = GTFS2VecModel(
            n_features=len(features.columns),
            n_hidden=self._hidden_size,
            n_embed=self._embedding_size,
        )
        X = features.to_numpy().astype(np.float32)
        x_dataloader = DataLoader(X, batch_size=24, shuffle=True, num_workers=4)

        trainer_kwargs = trainer_kwargs or {}
        if "max_epochs" not in trainer_kwargs:
            trainer_kwargs["max_epochs"] = 10
        trainer = pl.Trainer(**trainer_kwargs)

        trainer.fit(model, x_dataloader)

        return model

    def _embed(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Embed features.

        Args:
            features (pd.DataFrame): Features.

        Returns:
            pd.DataFrame: Embeddings.
        """
        import torch

        model = self._maybe_get_model()
        if len(features.columns) != model.n_features:
            raise ValueError(
                f"Features must have {model.n_features} columns but has {len(features.columns)}."
            )

        embeddings = model(torch.Tensor(features.to_numpy().astype(np.float32))).detach().numpy()

        return pd.DataFrame(
            data=embeddings,
            index=features.index,
        )

    def _save(self, path: Union[Path, str], embedder_config: dict[str, Any]) -> None:
        if isinstance(path, str):
            path = Path(path)

        model = self._maybe_get_model()

        path.mkdir(parents=True, exist_ok=True)

        model.save(path / "model.pt")

        config = {
            "model_config": model.get_config(),
            "embedder_config": embedder_config,
        }
        with (path / "config.json").open("w") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    def save(self, path: Union[Path, str]) -> None:
        """
        Save the model to a directory.

        Args:
            path (Path): Path to the directory.
        """
        embedder_config = {
            "hidden_size": self._hidden_size,
            "embedding_size": self._embedding_size,
            "skip_autoencoder": self._skip_autoencoder,
        }
        self._save(path, embedder_config)

    @classmethod
    def _load(cls, path: Union[Path, str], model_module: type[ModelT]) -> "GTFS2VecEmbedder":
        if isinstance(path, str):
            path = Path(path)

        with (path / "config.json").open("r") as f:
            config = json.load(f)
        embedder = cls(**config["embedder_config"])
        model_path = path / "model.pt"
        model = model_module.load(model_path, **config["model_config"])
        embedder._model = model
        embedder._is_fitted = True
        return embedder

    @classmethod
    def load(cls, path: Union[Path, str]) -> "GTFS2VecEmbedder":
        """
        Load the model from a directory.

        Args:
            path (Path): Path to the directory.

        Returns:
            Hex2VecEmbedder: The loaded embedder.
        """
        return cls._load(path, GTFS2VecModel)
