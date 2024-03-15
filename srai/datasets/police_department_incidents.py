"""
The San Francisco Police Department's (SFPD) Incident Report dataset loader.

This module contains The San Francisco Police Department's (SFPD) Incident Report Datatset.
"""

import os
from typing import Optional

import geopandas as gpd

from srai.datasets import Dataset


class PoliceDepartmentIncidents(Dataset):
    """
    The San Francisco Police Department's (SFPD) Incident Report Datatset.

    This dataset includes incident reports that have been filed as of January 1, 2018 till March,
    2024. These reports are filed by officers or self-reported by members of the public using SFPD’s
    online reporting system.
    """

    def __init__(self) -> None:
        """Initialize  The San Francisco Police Department's (SFPD) Incident Report dataset."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "police_department_incidents.yaml")
        super().__init__(config_path)

    def _preprocessing(
        self, data: gpd.GeoDataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of The San Francisco Police Department's (SFPD) Incident Report dataset.
            data_version_name: version of dataset

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["Latitude", "Longitude"], axis=1),
            geometry=gpd.points_from_xy(x=df["Longitude"], y=df["Latitude"]),
            crs="EPSG:4326",
        )
        return gdf
