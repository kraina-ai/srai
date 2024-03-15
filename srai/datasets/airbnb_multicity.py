"""
AirbnbMulticity dataset loader.

This module contains AirbnbMulticity dataset.
"""

import os
from typing import Optional

import geopandas as gpd

from srai.datasets import Dataset


class AirbnbMulticity(Dataset):
    """
    AirbnbMulticity dataset.

    Dataset description will be added.
    """

    def __init__(self) -> None:
        """Initialize AirbnbMulticity dataset."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "airbnb_multicity.yaml")
        super().__init__(config_path)

    def _preprocessing(
        self, data: gpd.GeoDataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of AirbnbMulticity dataset.
            data_version_name: version of dataset

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["latitude", "longitude"], axis=1),
            geometry=gpd.points_from_xy(x=df["longitude"], y=df["latitude"]),
            crs="EPSG:4326",
        )

        return gdf
