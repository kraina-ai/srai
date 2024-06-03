"""
Philadelphia Crime dataset loader.

This module contains Philadelphia Crime Dataset.
"""

from typing import Optional

import geopandas as gpd

from srai.constants import WGS84_CRS
from srai.datasets import HuggingFaceDataset


class PhiladelphiaCrimeDataset(HuggingFaceDataset):
    """
    Philadelphia Crime dataset.

    Crime incidents from the Philadelphia Police Department. Part I crimes include violent offenses
    such as aggravated assault, rape, arson, among others. Part II crimes include simple assault,
    prostitution, gambling, fraud, and other non-violent offenses.
    """

    years_previous: list[int] = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
    years_current: list[int] = [2020, 2021, 2022, 2023]

    def __init__(self) -> None:
        """Create the dataset."""
        numerical_columns = None
        categorical_columns = None
        type = "point"
        target = "text_general_code"
        super().__init__(
            "kraina/philadelphia_crime",
            type=type,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target=target,
        )

    def _preprocessing(
        self, data: gpd.GeoDataFrame, version: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocess the dataset from HuggingFace.

        Args:
            data (pd.DataFrame): a dataset to preprocess
            version (str, optional): version of dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["lng", "lat"], axis=1),
            geometry=gpd.points_from_xy(df["lng"], df["lat"]),
            crs=WGS84_CRS,
        )
        return gdf

    def load(
        self, hf_token: Optional[str] = None, version: str | None = "res_8"
    ) -> gpd.GeoDataFrame:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str, optional): version of a dataset.
                Available: 'res_8', 'res_9, 'res_10. Defaults to `res_8`.

        Returns:
            gpd.GeoDataFrame: Loaded data.
        """
        return super().load(hf_token, version)
