"""
Hugging Face loader.

This module contains Hugging Face loader.
"""

from typing import Optional

import geopandas as gpd
from datasets import load_dataset

from srai.loaders import Loader


class HFLoader(Loader):
    """
    Hugging Face loader.

    Loader to download dataset from HuggingFace and return GeoDataFrame.
    """

    def __init__(self, hf_token: Optional[str] = None) -> None:
        """
        Initialize loader.

        Args:
            hf_token (str): Token from KrainAI HF account.
        """
        self.hf_token = hf_token

    def load(self, dataset_name: str = "geolife", name: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Method to load data files from Hugging Face.

        Args:
            dataset_name (str, optional): Name of dataset provided by KrainaAI. \
                Defaults to "geolife".
            name (Optional[str], optional): Name of version of dataset e.g. "nyc_bike_2013". \
                Defaults to None.

        Returns:
            gpd.GeoDataFrame: _description_
        """
        dataset = load_dataset(
            f"kraina/{dataset_name}", name=name, token=self.hf_token, trust_remote_code=True
        )  # download dataset from HF

        df = dataset["train"].to_pandas()

        return gpd.GeoDataFrame(df)
