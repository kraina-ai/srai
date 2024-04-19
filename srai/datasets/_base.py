"""
Base class of Dataset.

This module contains Base class of Dataset.
"""

import abc
from typing import Optional

import geopandas as gpd
from omegaconf import OmegaConf

from srai.loaders import HFLoader


class Dataset(abc.ABC):
    """Abstract class for datasets."""

    def __init__(self, config_path: str) -> None:
        self.conf = OmegaConf.load(config_path)

    @abc.abstractmethod
    def _preprocessing(
        self, data: gpd.GeoDataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Load named datase.

        Args:
            data: GeoDataFrame of Dataset to preprocess
            data_version_name: version of dataset
            *args: Positional arguments dependating on a specific dataset.
            **kwargs: Keyword arguments dependating on a specific dataset.

        Returns:
            GeoDataFrame with the downloaded dataset.
        """
        raise NotImplementedError

    def load(self, hf_token: Optional[str] = None) -> gpd.GeoDataFrame:
        """Method to load dataset."""
        dataset_name = self.conf["dataset_name"]
        name = self.conf.get("name")
        data = HFLoader(hf_token=hf_token).load(dataset_name=dataset_name, name=name)
        processed_data = self._preprocessing(data)

        return processed_data
