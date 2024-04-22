"""
Philadelphia Crime dataset loader.

This module contains Philadelphia Crime Dataset.
"""

from typing import Optional

import geopandas as gpd

from srai.datasets import HuggingFaceDataset
from srai.loaders import HuggingFaceLoader


class PhiladelphiaCrime(HuggingFaceDataset):
    """
    Philadelphia Crime dataset.

    Crime incidents from the Philadelphia Police Department. Part I crimes include violent offenses
    such as aggravated assault, rape, arson, among others. Part II crimes include simple assault,
    prostitution, gambling, fraud, and other non-violent offenses.
    """

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

    def load(
        self, hf_token: Optional[str] = None, dataset_version_name: str = "incidents_2023"
    ) -> gpd.GeoDataFrame:
        """
        Method to load dataset.

        Args:
            dataset_version_name: Version name of dataset, e.g. "incidents_2013". \
                Available: incidents_2013 - incidents_2023.
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
