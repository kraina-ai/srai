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

from srai.embedders import CountEmbedder
from srai.embedders.geovex.dataset import HexagonalDataset
from srai.embedders.geovex.model import GeoVexModel
from srai.exceptions import ModelNotFitException
from srai.neighbourhoods import H3Neighbourhood
from srai.utils._optional import import_optional_dependencies

T = TypeVar("T")

try:  # pragma: no cover
    from torch.utils.data import DataLoader

except ImportError:
    from srai.utils._pytorch_stubs import DataLoader


class GeoVexEmbedder(CountEmbedder):
    """Hex2Vec Embedder."""

    def __init__(
        self,
        target_features: List[str],
        neighbourhood: H3Neighbourhood,
        batch_size: Optional[int] = 32,
        neighbourhood_radius: int = 4,
        convolutional_layers: int = 2,
        embedding_size: int = 32,
        dataset: Optional[HexagonalDataset[T]] = None,
    ) -> None:
        """
        Initialize GeoVex Embedder.

        Args:
            target_features (List[str]): The features that are to be used in the embedding.
                Should be in "flat" format, i.e. "<super tag>_<sub_tag>".
            neighbourhood (H3Neighbourhood): The neighbourhood to use.
                Should be intialized with the same regions.
            batch_size (int, optional): Batch size. Defaults to 32.
            convolutional_layers (int, optional): Number of convolutional layers. Defaults to 2.
            neighbourhood_radius (int, optional): Radius of the neighbourhood. Defaults to 4.
            embedding_size (int, optional): Size of the embedding. Defaults to 32.
            dataset (Optional[HexagonalDataset], optional): Dataset to use. Defaults to None.
        """
        import_optional_dependencies(
            dependency_group="torch", modules=["torch", "pytorch_lightning"]
        )

        super().__init__(
            expected_output_features=target_features,
        )

        self._model: Optional[GeoVexModel] = None
        self._is_fitted = False
        self._r = neighbourhood_radius
        self._embedding_size = embedding_size
        self._convolutional_layers = convolutional_layers

        self._neighbourhood = neighbourhood
        self._batch_size = batch_size
        self._neighbourhood = neighbourhood
        self._dataset = dataset

        # save invalid h3s for later
        self._invalid_h3s: List[str] = []

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
        from torch.utils.data import DataLoader

        self._check_is_fitted()

        if self._dataset is None:
            # build the dataset
            _, dataloader, self._dataset = self._prepare_dataset(
                regions_gdf,
                features_gdf,
                joint_gdf,
                self._neighbourhood,
                self._batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(self._dataset, batch_size=self._batch_size, shuffle=False)
        embeddings = [
            self._model.encoder(batch).detach().numpy() for batch in dataloader  # type: ignore
        ]

        if len(self._dataset.get_invalid_h3s()) > 0:
            print(
                "Warning: Some regions were not able to be encoded, as they don't have"
                f" r={self._r} neighbors."
            )

        return pd.DataFrame(np.concatenate(embeddings), index=self._dataset.get_ordered_index())

    def fit(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        neighbourhood: H3Neighbourhood,
        batch_size: int = 32,
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
            batch_size (int, optional): Batch size. Defaults to 32.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs.
                This is where the number of epochs can be set. Defaults to None.
        """
        import pytorch_lightning as pl

        trainer_kwargs = self._prepare_trainer_kwargs(trainer_kwargs)
        counts_df, dataloader, dataset = self._prepare_dataset(  # type: ignore
            regions_gdf, features_gdf, joint_gdf, neighbourhood, batch_size, shuffle=True
        )
        self._model = GeoVexModel(
            k_dim=len(counts_df.columns),
            R=self._r,
            conv_layers=self._convolutional_layers,
            emb_size=self._embedding_size,
            learning_rate=learning_rate,
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
        batch_size: int = 32,
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
            batch_size (int, optional): Batch size. Defaults to 32.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs. This is where the
                number of epochs can be set. Defaults to None.
        """
        self.fit(
            regions_gdf=regions_gdf,
            features_gdf=features_gdf,
            joint_gdf=joint_gdf,
            neighbourhood=neighbourhood,
            batch_size=batch_size,
            learning_rate=learning_rate,
            trainer_kwargs=trainer_kwargs,
        )
        return self.transform(
            regions_gdf=regions_gdf,
            features_gdf=features_gdf,
            joint_gdf=joint_gdf,
        )

    def _get_raw_counts(
        self, regions_gdf: pd.DataFrame, features_gdf: pd.DataFrame, joint_gdf: pd.DataFrame
    ) -> pd.DataFrame:
        return super().transform(regions_gdf, features_gdf, joint_gdf).astype(np.float32)

    def _check_is_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise ModelNotFitException("Model not fitted. Call fit() or fit_transform() first.")

    @property
    def invalid_h3s(self) -> List[str]:
        """List of invalid h3s."""
        return self._invalid_h3s

    def _prepare_trainer_kwargs(self, trainer_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # TODO: this is copy pasted from Hex2VecEmbedder, should be refactored
        # to a common base class, util, or something
        if trainer_kwargs is None:
            trainer_kwargs = {}
        if "max_epochs" not in trainer_kwargs:
            trainer_kwargs["max_epochs"] = 3
        return trainer_kwargs
