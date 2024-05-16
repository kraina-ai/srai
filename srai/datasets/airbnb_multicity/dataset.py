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
        categorical_columns = ["name", "host_name", "neighborhood", "room_type", "city"]
        numerical_columns = [
            "number_of_reviews",
            "minimum_nights",
            "availability_365",
            "calculated_host_listings_count",
            "number_of_reviews_ltm",
        ]
        target = "price"

        super().__init__(
            "kraina/airbnb_multicity",
            task="regression",
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target=target,
        )

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data (pd.DataFrame): Data of AirbnbMulticity dataset.
            version (str, optional): version of a dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["latitude", "longitude"], axis=1),
            geometry=gpd.points_from_xy(x=df["longitude"], y=df["latitude"]),
            crs=WGS84_CRS,
        )

        return gdf

    @staticmethod
    def train_dev_test_split_spatial() -> None:
        """
        Train, dev, test split based on spatial approach.

        Returns:
            _type_: NotImplementedError
        """
        raise NotImplementedError
