import abc
from typing import Any
from omegaconf import OmegaConf
from srai.loaders import HFLoader
import geopandas as gpd
import os
%load_ext dotenv
%dotenv

class Dataset(abc.ABC):
    """Abstract class for datasets."""
    def __init__(self, config_path: str) -> None:
        self.conf = OmegaConf.load(config_path)
        

    @abc.abstractmethod
    def _preprocessing(self, data):
        """
        Load named datase.

        Args:
            *args: Positional arguments dependating on a specific dataset.
            **kwargs: Keyword arguments dependating on a specific dataset.

        Returns:
            GeoDataFrame with the downloaded dataset.
        """
        raise NotImplementedError
    
    def load(self):
        """
        Method to load dataset
        """
        dataset_name = self.conf["dataset_name"]
        name = self.conf.get("name")
        data = HFLoader(os.environ["HF_access_token"]).load(dataset_name=dataset_name, name=name)
        processed_data = self._preprocessing(data)

        return processed_data
