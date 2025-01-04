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

    # TODO: add target and task
    def __init__(self) -> None:
        """Create the dataset."""
        numerical_columns = None
        categorical_columns = [
            "Incdident Year",
            "Incident Day of Week",
            "Incident Category",
            "Incident Subcategory",
            "Police District",
            "Analysis Neighborhood",
            "Incident Description",
            "Incident Time",
            "Incident Code",
        ]
        type = None
        super().__init__(
            "kraina/police_department_incidents",
            type=type,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )

    def _preprocessing(
        self, data: pd.DataFrame, version: Optional[str] = "res_9"
    ) -> gpd.GeoDataFrame:
        """
        Preprocess the dataset from HuggingFace.

        Args:
            data (pd.DataFrame): a dataset to preprocess
            version (str, optional): version of dataset.
            Available: Official spatial train-test split in chosen h3 resolution:
            'res_8', 'res_9, 'res_10'. Defaults to 'res_9'. All data available
            as 'all'.

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
