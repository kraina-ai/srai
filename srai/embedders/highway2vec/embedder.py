"""
Highway2Vec embedder.

This module contains the embedder from the `highway2vec` paper [1].

References:
    1. https://doi.org/10.1145/3557918.3565865
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

import geopandas as gpd
import pandas as pd

from srai._optional import import_optional_dependencies
from srai.embedders import Embedder, ModelT
from srai.exceptions import ModelNotFitException

from .model import Highway2VecModel


class Highway2VecEmbedder(Embedder):
    """Highway2Vec Embedder."""

    def __init__(self, hidden_size: int = 64, embedding_size: int = 30) -> None:
        """
        Init Highway2Vec Embedder.

        Args:
            hidden_size (int, optional): Hidden size in encoder and decoder. Defaults to 64.
            embedding_size (int, optional): Embedding size. Defaults to 30.
        """
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )

        self._model: Optional[Highway2VecModel] = None
        self._hidden_size = hidden_size
        self._embedding_size = embedding_size
        self._is_fitted = False

    def transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:  # pragma: no cover
        """
        Embed regions using features.

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
        import torch

        self._validate_indexes(regions_gdf, features_gdf, joint_gdf)
        self._check_is_fitted()
        features_df = self._remove_geometry_if_present(features_gdf)

        self._model.eval()  # type: ignore
        embeddings = self._model(torch.Tensor(features_df.values)).detach().numpy()  # type: ignore
        embeddings_df = pd.DataFrame(embeddings, index=features_df.index)
        embeddings_joint = joint_gdf.join(embeddings_df)
        embeddings_aggregated = embeddings_joint.groupby(level=[0]).mean()

        return embeddings_aggregated

    def fit(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        trainer_kwargs: Optional[dict[str, Any]] = None,
        dataloader_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Fit the model to the data.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs. Defaults to None.
            dataloader_kwargs (Optional[Dict[str, Any]], optional): Dataloader kwargs.
                Defaults to None.

        Raises:
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
        """
        import pytorch_lightning as pl
        import torch
        from torch.utils.data import DataLoader

        self._validate_indexes(regions_gdf, features_gdf, joint_gdf)
        features_df = self._remove_geometry_if_present(features_gdf)

        num_features = len(features_df.columns)
        self._model = Highway2VecModel(
            n_features=num_features, n_hidden=self._hidden_size, n_embed=self._embedding_size
        )

        dataloader_kwargs = dataloader_kwargs or {}
        if "batch_size" not in dataloader_kwargs:
            dataloader_kwargs["batch_size"] = 128

        dataloader = DataLoader(torch.Tensor(features_df.values), **dataloader_kwargs)

        trainer_kwargs = trainer_kwargs or {}
        if "max_epochs" not in trainer_kwargs:
            trainer_kwargs["max_epochs"] = 10

        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self._model, dataloader)
        self._is_fitted = True

    def fit_transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        trainer_kwargs: Optional[dict[str, Any]] = None,
        dataloader_kwargs: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Fit the model to the data and return the embeddings.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs. Defaults to None.
            dataloader_kwargs (Optional[Dict[str, Any]], optional): Dataloader kwargs.
                Defaults to None.

        Returns:
            pd.DataFrame: Region embeddings.

        Raises:
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
        """
        self.fit(regions_gdf, features_gdf, joint_gdf, trainer_kwargs, dataloader_kwargs)
        return self.transform(regions_gdf, features_gdf, joint_gdf)

    def _check_is_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise ModelNotFitException("Model not fitted. Call fit() or fit_transform() first.")

    def _save(self, path: Union[Path, str], embedder_config: dict[str, Any]) -> None:
        if isinstance(path, str):
            path = Path(path)

        self._check_is_fitted()

        path.mkdir(parents=True, exist_ok=True)

        self._model.save(path / "model.pt")  # type: ignore

        config = {
            "model_config": self._model.get_config(),  # type: ignore
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
        embedder_config = {"hidden_size": self._hidden_size, "embedding_size": self._embedding_size}
        self._save(path, embedder_config)

    @classmethod
    def _load(cls, path: Union[Path, str], model_module: type[ModelT]) -> "Highway2VecEmbedder":
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
    def load(cls, path: Union[Path, str]) -> "Highway2VecEmbedder":
        """
        Load the model from a directory.

        Args:
            path (Path): Path to the directory.

        Returns:
            Hex2VecEmbedder: The loaded embedder.
        """
        return cls._load(path, Highway2VecModel)
