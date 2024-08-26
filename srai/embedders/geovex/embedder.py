"""
GeoVex Embedder.

This module contains embedder from GeoVex paper[1].

References:
    [1] https://openreview.net/forum?id=7bvWopYY1H
"""

import json
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from srai._optional import import_optional_dependencies
from srai.constants import REGIONS_INDEX
from srai.embedders import CountEmbedder, ModelT
from srai.embedders.geovex.dataset import HexagonalDataset
from srai.embedders.geovex.model import GeoVexModel
from srai.exceptions import ModelNotFitException
from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter, OsmTagsFilter
from srai.neighbourhoods import H3Neighbourhood

T = TypeVar("T")

try:  # pragma: no cover
    from torch.utils.data import DataLoader

except ImportError:
    from srai.embedders._pytorch_stubs import DataLoader


class GeoVexEmbedder(CountEmbedder):
    """GeoVex Embedder."""

    def __init__(
        self,
        target_features: Union[list[str], OsmTagsFilter, GroupedOsmTagsFilter],
        batch_size: Optional[int] = 32,
        neighbourhood_radius: int = 4,
        convolutional_layers: int = 2,
        embedding_size: int = 32,
        convolutional_layer_size: int = 256,
    ) -> None:
        """
        Initialize GeoVex Embedder.

        Args:
            target_features (Union[List[str], OsmTagsFilter, GroupedOsmTagsFilter]): The features
                that are to be used in the embedding. Should be in "flat" format,
                i.e. "<super-tag>_<sub-tag>", or use OsmTagsFilter object.
            batch_size (int, optional): Batch size. Defaults to 32.
            convolutional_layers (int, optional): Number of convolutional layers. Defaults to 2.
            neighbourhood_radius (int, optional): Radius of the neighbourhood. Defaults to 4.
            embedding_size (int, optional): Size of the embedding. Defaults to 32.
            convolutional_layer_size (int, optional): Size of the first convolutional layer.
        """
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )

        super().__init__(
            expected_output_features=target_features,
        )

        self._assert_feature_length(self.expected_output_features, convolutional_layer_size)

        self._model: Optional[GeoVexModel] = None
        self._is_fitted = False
        self._r = neighbourhood_radius
        self._embedding_size = embedding_size
        self._convolutional_layers = convolutional_layers
        self._convolutional_layer_size = convolutional_layer_size

        self._batch_size = batch_size

        # save invalid h3s for later
        self._invalid_cells: list[str] = []
        self._dataset: DataLoader = None

    @property
    def invalid_cells(self) -> list[str]:
        """List of invalid h3s."""
        return self._invalid_cells

    def transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Create region embeddings.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.

        Returns:
            pd.DataFrame: Region embeddings.
        """
        self._check_is_fitted()

        neighbourhood = H3Neighbourhood(
            regions_gdf=regions_gdf,
        )

        _, dataloader, self._dataset = self._prepare_dataset(
            regions_gdf,
            features_gdf,
            joint_gdf,
            neighbourhood,
            self._batch_size,
            shuffle=False,
        )

        return self._transform(dataset=self._dataset, dataloader=dataloader)

    def _transform(
        self,
        dataset: HexagonalDataset[T],
        dataloader: Optional[DataLoader] = None,
    ) -> pd.DataFrame:
        if dataloader is None:
            dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False)

        embeddings = [
            self._model.encoder(batch).detach().numpy()  # type: ignore
            for batch in dataloader
        ]
        if dataset.get_invalid_cells():
            print(
                "Warning: Some regions were not able to be encoded, as they don't have"
                f" r={self._r} neighbors."
            )
        df = pd.DataFrame(np.concatenate(embeddings), index=dataset.get_valid_cells())
        df.index.name = REGIONS_INDEX
        return df

    def fit(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        neighbourhood: H3Neighbourhood,
        learning_rate: float = 0.001,
        trainer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Fit the model to the data.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
            neighbourhood (H3Neighbourhood): The neighbourhood to use.
                Should be intialized with the same regions.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs.
                This is where the number of epochs can be set. Defaults to None.
        """
        import pytorch_lightning as pl

        trainer_kwargs = self._prepare_trainer_kwargs(trainer_kwargs)
        counts_df, dataloader, dataset = self._prepare_dataset(  # type: ignore
            regions_gdf, features_gdf, joint_gdf, neighbourhood, self._batch_size, shuffle=True
        )

        self._prepare_model(counts_df, learning_rate)

        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self._model, dataloader)
        self._is_fitted = True
        self._dataset = dataset

    def _prepare_model(self, counts_df: pd.DataFrame, learning_rate: float) -> None:
        if self._model is None:
            self._model = GeoVexModel(
                k_dim=len(counts_df.columns),
                radius=self._r,
                conv_layers=self._convolutional_layers,
                emb_size=self._embedding_size,
                learning_rate=learning_rate,
                conv_layer_size=self._convolutional_layer_size,
            )

    def _prepare_dataset(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        neighbourhood: H3Neighbourhood,
        batch_size: Optional[int],
        shuffle: bool = True,
    ) -> tuple[pd.DataFrame, DataLoader, HexagonalDataset[T]]:
        counts_df = self._get_raw_counts(regions_gdf, features_gdf, joint_gdf)
        dataset: HexagonalDataset[T] = HexagonalDataset(
            counts_df,
            neighbourhood,
            neighbor_k_ring=self._r,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return counts_df, dataloader, dataset

    def fit_transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        neighbourhood: H3Neighbourhood,
        learning_rate: float = 0.001,
        trainer_kwargs: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Fit the model to the data and create region embeddings.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
            neighbourhood (H3Neighbourhood): The neighbourhood to use.
                Should be intialized with the same regions.
            negative_sample_k_distance (int, optional): Distance of negative samples. Defaults to 2.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs. This is where the
                number of epochs can be set. Defaults to None.
        """
        self.fit(
            regions_gdf=regions_gdf,
            features_gdf=features_gdf,
            joint_gdf=joint_gdf,
            neighbourhood=neighbourhood,
            learning_rate=learning_rate,
            trainer_kwargs=trainer_kwargs,
        )
        assert self._dataset is not None  # for mypy
        return self._transform(dataset=self._dataset)

    def _get_raw_counts(
        self, regions_gdf: pd.DataFrame, features_gdf: pd.DataFrame, joint_gdf: pd.DataFrame
    ) -> pd.DataFrame:
        return super().transform(regions_gdf, features_gdf, joint_gdf).astype(np.float32)

    def _check_is_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise ModelNotFitException("Model not fitted. Call fit() or fit_transform() first.")

    def _prepare_trainer_kwargs(self, trainer_kwargs: Optional[dict[str, Any]]) -> dict[str, Any]:
        if trainer_kwargs is None:
            trainer_kwargs = {}
        if "max_epochs" not in trainer_kwargs:
            trainer_kwargs["max_epochs"] = 3
        return trainer_kwargs

    def _assert_feature_length(self, target_features: list[str], conv_layer_size: int) -> None:
        if len(target_features) < conv_layer_size:
            raise ValueError(
                f"The convolutional layers in GeoVex expect >= {conv_layer_size} features."
            )

    def save(self, path: Union[str, Any]) -> None:
        """
        Save the model to a directory.

        Args:
            path (Union[str, Any]): Path to the directory.
        """
        # embedder_config must match the constructor signature:
        # target_features: Union[list[str], OsmTagsFilter, GroupedOsmTagsFilter],
        # batch_size: Optional[int] = 32,
        # neighbourhood_radius: int = 4,
        # convolutional_layers: int = 2,
        # embedding_size: int = 32,
        # convolutional_layer_size: int = 256,
        embedder_config = {
            "target_features": self.expected_output_features.to_json(),
            "batch_size": self._batch_size,
            "neighbourhood_radius": self._r,
            "convolutional_layers": self._convolutional_layers,
            "embedding_size": self._embedding_size,
            "convolutional_layer_size": self._convolutional_layer_size,
        }
        self._save(path, embedder_config)

    def _save(self, path: Union[str, Any], embedder_config: dict[str, Any]) -> None:
        if isinstance(path, str):
            path = Path(path)

        self._check_is_fitted()

        path.mkdir(parents=True, exist_ok=True)

        # save model and config
        self._model.save(path / "model.pt")  # type: ignore
        # combine model config and embedder config
        model_config = self._model.get_config()  # type: ignore

        config = {
            "model_config": model_config,
            "embedder_config": embedder_config,
        }

        with (path / "config.json").open("w") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "GeoVexEmbedder":
        """
        Load the model from a directory.

        Args:
            path (Union[Path, str]): Path to the directory.
            model_module (type[ModelT]): Model class.

        Returns:
            GeoVexEmbedder: GeoVexEmbedder object.
        """
        return cls._load(path, GeoVexModel)

    @classmethod
    def _load(cls, path: Union[Path, str], model_module: type[ModelT]) -> "GeoVexEmbedder":
        if isinstance(path, str):
            path = Path(path)
        with (path / "config.json").open("r") as f:
            config = json.load(f)

        config["embedder_config"]["target_features"] = json.loads(
            config["embedder_config"]["target_features"]
        )
        embedder = cls(**config["embedder_config"])
        model_path = path / "model.pt"
        model = model_module.load(model_path, **config["model_config"])
        embedder._model = model
        embedder._is_fitted = True
        return embedder
