"""
Philadelphia Crime dataset loader.

This module contains Philadelphia Crime Dataset.
"""

import os
from typing import Optional  # , override

import geopandas as gpd

from srai.datasets import Dataset
from srai.loaders import HFLoader


class PhiladelphiaCrime(Dataset):
    """
    Philadelphia Crime dataset.

    Crime incidents from the Philadelphia Police Department. Part I crimes include violent offenses
    such as aggravated assault, rape, arson, among others. Part II crimes include simple assault,
    prostitution, gambling, fraud, and other non-violent offenses.
    """

    def __init__(self) -> None:
        """Initialize Philadelphia Crime dataset."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "philadelphia_crime.yaml")
        super().__init__(config_path)

    def _preprocessing(
        self, data: gpd.GeoDataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of Philadelphia Crime dataset.
            data_version_name: version of dataset

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["lng", "lat"], axis=1),
            geometry=gpd.points_from_xy(df["lng"], df["lat"]),
            crs="EPSG:4326",
        )
        return gdf

    # @override
    def load(self, dataset_version_name: str = "incidents_2023") -> gpd.GeoDataFrame:
        """
        Method to load dataset.

        Args:
            dataset_version_name: Version name of dataset, e.g. "incidents_2013". \
                Available: incidents_2013 - incidents_2023.

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        dataset_name = self.conf["dataset_name"]
        data = HFLoader(os.environ["HF_access_token"]).load(
            dataset_name=dataset_name, name=dataset_version_name
        )
        processed_data = self._preprocessing(data)

        return processed_data
