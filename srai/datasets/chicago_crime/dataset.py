"""
Chicago Crime dataset loader.

This module contains Chicago Crime Dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd

from srai.constants import WGS84_CRS
from srai.datasets import HuggingFaceDataset


class ChicagoCrimeDataset(HuggingFaceDataset):
    """
    Chicago Crime dataset.

    This dataset reflects reported incidents of crime (with the exception of murders where data
    exists for each victim) that occurred in the City of Chicago. Data is extracted from the Chicago
    Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system.
    """

    available_versions: list[str] = ["2020", "2021", "2022"]  # TODO: use

    def __init__(self) -> None:
        """Create the dataset."""
        numerical_columns = ["Ward", "Community Area"]
        categorical_columns = [
            "Primary Type",
            "Description",
            "Location Description",
            "Arrest",
            "Domestic",
            "Year",
        ]
        type = "point"
        target = "Primary Type"
        super().__init__(
            "kraina/chicago_crime",
            type=type,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target=target,
        )

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of Chicago Crime dataset.
            version: version of a dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["Latitude", "Longitude", "X Coordinate", "Y Coordinate"], axis=1),
            geometry=gpd.points_from_xy(x=df["Longitude"], y=df["Latitude"]),
            crs=WGS84_CRS,
        )
        return gdf

    def load(
        self, hf_token: Optional[str] = None, version: str | None = "2020"
    ) -> gpd.GeoDataFrame:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str, optional): version of a dataset.
                Available values: `2020`, `2021`, `2022`. Defaults to `2020`.

        Returns:
            gpd.GeoDataFrame: Loaded data.
        """
        return super().load(hf_token, version)
