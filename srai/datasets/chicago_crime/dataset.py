"""
Chicago Crime dataset loader.

This module contains Chicago Crime Dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd

from srai.datasets import HuggingFaceDataset
from srai.loaders import HuggingFaceLoader


class ChicagoCrime(HuggingFaceDataset):
    """
    Chicago Crime dataset.

    This dataset reflects reported incidents of crime (with the exception of murders where data
    exists for each victim) that occurred in the City of Chicago. Data is extracted from the Chicago
    Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system.
    """

    def _preprocessing(
        self, data: pd.DataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of Chicago Crime dataset.
            data_version_name: version of dataset

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["Latitude", "Longitude", "X Coordinate", "Y Coordinate"], axis=1),
            geometry=gpd.points_from_xy(x=df["Longitude"], y=df["Latitude"]),
            crs="EPSG:4326",
        )
        return gdf

    def load(
        self, hf_token: Optional[str] = None, dataset_version_name: str = "2020"
    ) -> gpd.GeoDataFrame:
        """
        Method to load dataset.

        Args:
            dataset_version_name: Version name of dataset, e.g. "2020". \
                Available: 2020, 2021, 2022.
            hf_token: Token from Hugging Face

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        dataset_name = self.conf["dataset_name"]
        data = HuggingFaceLoader(hf_token=hf_token).load(
            dataset_name=dataset_name, name=dataset_version_name
        )
        processed_data = self._preprocessing(data)

        return processed_data
