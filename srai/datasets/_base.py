"""Base classes for Datasets."""

import abc
import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from omegaconf import OmegaConf

from srai.loaders import HuggingFaceLoader


class HuggingFaceDataset(abc.ABC):
    """Abstract class for HuggingFace datasets."""

    def __init__(self, config_path: Path | None = None) -> None:
        path_to_subclass = sys.modules[self.__module__].__file__ or "."
        config_path = config_path or Path(path_to_subclass).parent / "config.yaml"

        if not config_path.exists() or not config_path.is_file():
            raise ValueError(f"Missing config file at {config_path} for a dataset.")

        self.conf = OmegaConf.load(config_path)

    @abc.abstractmethod
    def _preprocessing(
        self, data: pd.DataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Load named datase.

        Args:
            data (gpd.GeoDataFrame): a dataset to preprocess
            data_version_name (str): version of a dataset

        Returns:
            GeoDataFrame with the downloaded dataset.
        """
        raise NotImplementedError

    def load(self, hf_token: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed a User Access Token needed to authenticate to
                the Hugging Face Hub. Defaults to None.

        Returns:
            gpd.GeoDataFrame: Loaded data.
        """
        dataset_name = self.conf["dataset_name"]
        name = self.conf.get("name")
        data = HuggingFaceLoader(hf_token=hf_token).load(dataset_name=dataset_name, name=name)
        processed_data = self._preprocessing(data)

        return processed_data
