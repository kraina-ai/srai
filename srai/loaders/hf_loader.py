from typing import Optional

import geopandas as gpd
from datasets import load_dataset

from srai.loaders import Loader


class HFLoader(Loader):
    """Loader to download dataset from HuggingFace and return GeoDataFrame."""

    def __init__(self, hf_token: str) -> None:
        self.hf_token = hf_token

    def load(self, dataset_name: str = "geolife", name: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Parameters
        ----------
        dataset_name : str
            Name of dataset provided by kraina e.g. "nyc_bike".
        name : Optional[str]
            Name of version of dataset e.g. "nyc_bike_2013"
        """
        dataset = load_dataset(
            f"kraina/{dataset_name}", name=name, token=self.hf_token, trust_remote_code=True
        )  # download dataset from HF

        df = dataset["train"].to_pandas()

        return gpd.GeoDataFrame(df)
