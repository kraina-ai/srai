"""Base classes for Datasets."""

import abc
from typing import Optional

import geopandas as gpd
import pandas as pd

from srai.loaders import HuggingFaceLoader


class HuggingFaceDataset(abc.ABC):
    """Abstract class for HuggingFace datasets."""

    def __init__(self, path: str, version: Optional[str] = None) -> None:
        self.path = path
        self.version = version

    @abc.abstractmethod
    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocess the dataset from HuggingFace.

        Args:
            data (pd.DataFrame): a dataset to preprocess
            version (str, optional): version of dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        raise NotImplementedError

    def load(
        self, hf_token: Optional[str] = None, version: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str, optional): version of a dataset

        Returns:
            gpd.GeoDataFrame: Loaded data.
        """
        dataset_name = self.path
        version = version or self.version
        data = HuggingFaceLoader(hf_token=hf_token).load(dataset_name=dataset_name, name=version)
        processed_data = self._preprocessing(data)

        return processed_data
