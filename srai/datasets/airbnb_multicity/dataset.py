"""
AirbnbMulticity dataset loader.

This module contains AirbnbMulticity dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd

from srai.constants import WGS84_CRS
from srai.datasets import HuggingFaceDataset


class AirbnbMulticityDataset(HuggingFaceDataset):
    """
    AirbnbMulticity dataset.

    Dataset description will be added.
    """

    def __init__(self) -> None:
        """Create the dataset."""
        super().__init__("kraina/airbnb_multicity")

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data (pd.DataFrame): Data of AirbnbMulticity dataset.
            version (str, optional): version of a dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        gdf = gpd.GeoDataFrame(
            data.drop(["latitude", "longitude"], axis=1),
            geometry=gpd.points_from_xy(x=data["longitude"], y=data["latitude"]),
            crs=WGS84_CRS,
        )

        return gdf
