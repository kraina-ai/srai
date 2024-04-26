"""
The San Francisco Police Department's (SFPD) Incident Report dataset loader.

This module contains The San Francisco Police Department's (SFPD) Incident Report Datatset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd

from srai.constants import WGS84_CRS
from srai.datasets import HuggingFaceDataset


class PoliceDepartmentIncidentsDataset(HuggingFaceDataset):
    """
    The San Francisco Police Department's (SFPD) Incident Report Datatset.

    This dataset includes incident reports that have been filed as of January 1, 2018 till March,
    2024. These reports are filed by officers or self-reported by members of the public using SFPD’s
    online reporting system.
    """

    def __init__(self) -> None:
        """Create the dataset."""
        super().__init__("kraina/police_department_incidents")

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocess the dataset from HuggingFace.

        Args:
            data (pd.DataFrame): a dataset to preprocess
            version (str, optional): version of dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        gdf = gpd.GeoDataFrame(
            data.drop(["Latitude", "Longitude"], axis=1),
            geometry=gpd.points_from_xy(x=data["Longitude"], y=data["Latitude"]),
            crs=WGS84_CRS,
        )
        return gdf
