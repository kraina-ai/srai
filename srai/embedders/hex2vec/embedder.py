"""TODO."""
from typing import Any, Dict, List, Optional, TypeVar

import geopandas as gpd
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from srai.embedders import CountEmbedder
from srai.embedders.hex2vec.model import Hex2VecModel
from srai.embedders.hex2vec.neighbour_dataset import NeighbourDataset
from srai.neighbourhoods import Neighbourhood
from srai.utils.exceptions import ModelNotFitException

T = TypeVar("T")


class Hex2VecEmbedder(CountEmbedder):
    """TODO."""

    def __init__(
        self,
        encoder_sizes: Optional[List[int]] = None,
        expected_output_features: Optional[List[str]] = None,
    ) -> None:
        """TODO."""
        super().__init__(expected_output_features)
        if encoder_sizes is None:
            encoder_sizes = [150, 75, 50]
        self._model = Hex2VecModel(encoder_sizes)
        self._is_fitted = False

    def transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """TODO."""
        self._check_is_fitted()
        counts_df = super().transform(regions_gdf, features_gdf, joint_gdf)
        counts_tensor = torch.from_numpy(counts_df.values)
        embeddings = self._model(counts_tensor).detach().numpy()
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
        """TODO."""
        if trainer_kwargs is None:
            trainer_kwargs = {}
        counts_df = super().transform(regions_gdf, features_gdf, joint_gdf)
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
        """TODO."""
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

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise ModelNotFitException("Model not fitted. Call fit() or fit_transform() first.")
