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
        numerical_columns = None
        categorical_columns = [
            "Incdident Year",
            "Incident Day of Week",
            "Police District",
            "Analysis Neighborhood",
            "Incident Description",
            "Incident Time",
            "Incident Code",
            "Report Type Code",
            "Police District",
            "Analysis Neighborhood",
        ]
        type = "point"
        # target = "Incident Category"
        target = None
        super().__init__(
            "kraina/police_department_incidents",
            type=type,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target=target,
        )

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data (pd.DataFrame): a dataset to preprocess
            version (str, optional): version of dataset.

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["Latitude", "Longitude"], axis=1),
            geometry=gpd.points_from_xy(x=df["Longitude"], y=df["Latitude"]),
            crs=WGS84_CRS,
        )
        return gdf

    def load(self, hf_token: Optional[str] = None, version: Optional[str] = "9") -> None:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str, optional): version of a dataset.
                Available: '8', '9', '10', where number is a h3 resolution used in train-test \
                    split. Defaults to '9'. Raw, full data available as 'all'.

        Returns:
            None
        """
        super().load(hf_token, version)
