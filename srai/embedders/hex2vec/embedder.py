"""
Hex2Vec Embedder.

This module contains embedder from Hex2Vec paper[1].

References:
    [1] https://dl.acm.org/doi/10.1145/3486635.3491076
"""
from typing import Any, Dict, List, Optional, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from srai.embedders import CountEmbedder
from srai.embedders.hex2vec.model import Hex2VecModel
from srai.embedders.hex2vec.neighbour_dataset import NeighbourDataset
from srai.exceptions import ModelNotFitException
from srai.neighbourhoods import Neighbourhood

T = TypeVar("T")


class Hex2VecEmbedder(CountEmbedder):
    """Hex2Vec Embedder."""

    def __init__(
        self,
        encoder_sizes: Optional[List[int]] = None,
        expected_output_features: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize Hex2VecEmbedder.

        Args:
            encoder_sizes (Optional[List[int]], optional): Sizes of the encoder layers.
                The input layer size shouldn't be included - it's inferred from the data.
                The last element is the embedding size. Defaults to [150, 75, 50].
            expected_output_features (Optional[List[str]], optional): List of expected output
                features. Defaults to None.
        """
        super().__init__(expected_output_features)
        if encoder_sizes is None:
            encoder_sizes = [150, 75, 50]
        self._assert_encoder_sizes_correct(encoder_sizes)
        self._encoder_sizes = encoder_sizes
        self._model: Optional[Hex2VecModel] = None
        self._is_fitted = False

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
            pd.DataFrame: Embedding and geometry index for each region in regions_gdf.

        Raises:
            ValueError: If features_gdf is empty and self.expected_output_features is not set.
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
        """
        self._check_is_fitted()
        counts_df = self._get_raw_counts(regions_gdf, features_gdf, joint_gdf)
        counts_tensor = torch.from_numpy(counts_df.values)
        embeddings = self._model(counts_tensor).detach().numpy()  # type: ignore
        return pd.DataFrame(embeddings, index=counts_df.index)

    def fit(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        neighbourhood: Neighbourhood[T],
        negative_sample_k_distance: int = 2,
        batch_size: int = 32,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Fit the model to the data.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
            neighbourhood (Neighbourhood[T]): The neighbourhood to use.
                Should be intialized with the same regions.
            negative_sample_k_distance (int, optional): When sampling negative samples,
                sample from a distance > k. Defaults to 2.
            batch_size (int, optional): Batch size. Defaults to 32.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs. Defaults to None.

        Raises:
            ValueError: If features_gdf is empty and self.expected_output_features is not set.
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
            ValueError: If negative_sample_k_distance < 2.
        """
        if trainer_kwargs is None:
            trainer_kwargs = {}
        counts_df = self._get_raw_counts(regions_gdf, features_gdf, joint_gdf)
        num_features = len(counts_df.columns)
        self._model = Hex2VecModel(layer_sizes=[num_features, *self._encoder_sizes])
        dataset = NeighbourDataset(counts_df, neighbourhood, negative_sample_k_distance)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        neighbourhood: Neighbourhood[T],
        negative_sample_k_distance: int = 2,
        batch_size: int = 32,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Fit the model to the data and return the embeddings.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and feature values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
            neighbourhood (Neighbourhood[T]): The neighbourhood to use.
                Should be intialized with the same regions.
            negative_sample_k_distance (int, optional): When sampling negative samples,
                sample from a distance > k. Defaults to 2.
            batch_size (int, optional): Batch size. Defaults to 32.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Trainer kwargs. Defaults to None.

        Returns:
            pd.DataFrame: Region embeddings.

        Raises:
            ValueError: If features_gdf is empty and self.expected_output_features is not set.
            ValueError: If any of the gdfs index names is None.
            ValueError: If joint_gdf.index is not of type pd.MultiIndex or doesn't have 2 levels.
            ValueError: If index levels in gdfs don't overlap correctly.
            ValueError: If negative_sample_k_distance < 2.
        """
        self.fit(
            regions_gdf,
            features_gdf,
            joint_gdf,
            neighbourhood,
            negative_sample_k_distance,
            batch_size,
            trainer_kwargs,
        )
        return self.transform(regions_gdf, features_gdf, joint_gdf)

    def _get_raw_counts(
        self, regions_gdf: pd.DataFrame, features_gdf: pd.DataFrame, joint_gdf: pd.DataFrame
    ) -> pd.DataFrame:
        return super().transform(regions_gdf, features_gdf, joint_gdf).astype(np.float32)

    def _check_is_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise ModelNotFitException("Model not fitted. Call fit() or fit_transform() first.")

    def _assert_encoder_sizes_correct(self, encoder_sizes: List[int]) -> None:
        if len(encoder_sizes) < 1:
            raise ValueError("Encoder sizes must have at least one element - embedding size.")
        if any(size <= 0 for size in encoder_sizes):
            raise ValueError("Encoder sizes must be positive integers.")
