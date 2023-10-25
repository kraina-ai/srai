"""
GeoVex Embedder.

This module contains embedder from GeoVex paper[1].

References:
    [1] https://openreview.net/forum?id=7bvWopYY1H
"""
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd

from srai._optional import import_optional_dependencies
from srai.constants import REGIONS_INDEX
from srai.embedders import CountEmbedder
from srai.embedders.geovex.dataset import HexagonalDataset
from srai.embedders.geovex.model import GeoVexModel
from srai.exceptions import ModelNotFitException
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
        target_features: List[str],
        batch_size: Optional[int] = 32,
        neighbourhood_radius: int = 4,
        convolutional_layers: int = 2,
        embedding_size: int = 32,
        convolutional_layer_size: int = 256,
        dataset: Optional[HexagonalDataset[T]] = None,
    ) -> None:
        """
        Initialize GeoVex Embedder.

        Args:
            target_features (List[str]): The features that are to be used in the embedding.
                Should be in "flat" format, i.e. "<super-tag>_<sub-tag>".
            batch_size (int, optional): Batch size. Defaults to 32.
            convolutional_layers (int, optional): Number of convolutional layers. Defaults to 2.
            neighbourhood_radius (int, optional): Radius of the neighbourhood. Defaults to 4.
            embedding_size (int, optional): Size of the embedding. Defaults to 32.
            convolutional_layer_size (int, optional): Size of the first convolutional layer.
            dataset (Optional[HexagonalDataset], optional): Dataset to use. Defaults to None.
        """
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )

        self._assert_feature_length(target_features, convolutional_layer_size)

        super().__init__(
            expected_output_features=target_features,
        )

        self._model: Optional[GeoVexModel] = None
        self._is_fitted = False
        self._r = neighbourhood_radius
        self._embedding_size = embedding_size
        self._convolutional_layers = convolutional_layers
        self._convolutional_layer_size = convolutional_layer_size

        self._batch_size = batch_size
        self._dataset = dataset

        # save invalid h3s for later
        self._invalid_cells: List[str] = []

    @property
    def invalid_cells(self) -> List[str]:
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
            self._model.encoder(batch).detach().numpy() for batch in dataloader  # type: ignore
        ]
        if len(dataset.get_invalid_cells()) > 0:
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
        trainer_kwargs: Optional[Dict[str, Any]] = None,
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
        self._model = GeoVexModel(
            k_dim=len(counts_df.columns),
            radius=self._r,
            conv_layers=self._convolutional_layers,
            emb_size=self._embedding_size,
            learning_rate=learning_rate,
            conv_layer_size=self._convolutional_layer_size,
        )
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self._model, dataloader)
        self._is_fitted = True
        self._dataset = dataset

    def _prepare_dataset(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        neighbourhood: H3Neighbourhood,
        batch_size: Optional[int],
        shuffle: bool = True,
    ) -> Tuple[pd.DataFrame, DataLoader, HexagonalDataset[T]]:
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
        trainer_kwargs: Optional[Dict[str, Any]] = None,
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

    def _prepare_trainer_kwargs(self, trainer_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if trainer_kwargs is None:
            trainer_kwargs = {}
        if "max_epochs" not in trainer_kwargs:
            trainer_kwargs["max_epochs"] = 3
        return trainer_kwargs

    def _assert_feature_length(self, target_features: List[str], conv_layer_size: int) -> None:
        if len(target_features) < conv_layer_size:
            raise ValueError(
                f"The convolutional layers in GeoVex expect >= {conv_layer_size} features."
            )