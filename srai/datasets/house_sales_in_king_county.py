"""
House Sales in King County dataset loader.

This module contains House Sales in King County Dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd

from srai.constants import WGS84_CRS
from srai.datasets import HuggingFaceDataset


class HouseSalesInKingCountyDataset(HuggingFaceDataset):
    """
    House Sales in King County dataset.

    This dataset contains house sale prices for King County, which includes Seattle. It includes
    homes sold between May 2014 and May 2015.

    It's a great dataset for evaluating simple regression models.
    """

    def __init__(self) -> None:
        """Create the dataset."""
        numerical_columns = [
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "condition",
            "grade",
            "sqft_above",
            "sqft_basement",
            "sqft_living15",
            "sqft_lot15",
        ]
        categorical_columns = ["view", "yr_built", "yr_renovated", "waterfront"]
        type = "point"
        target = "price"
        super().__init__(
            "kraina/house_sales_in_king_county",
            type=type,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target=target,
        )

    def _preprocessing(
        self, data: pd.DataFrame, version: Optional[str] = "res_8"
    ) -> gpd.GeoDataFrame:
        """
        Preprocess the dataset from HuggingFace.

        Args:
            data (pd.DataFrame): a dataset to preprocess
            version (str, optional): version of dataset.
            Available: 'res_8', 'res_9', 'res_10'. Defaults to 'res_8'.
                    Raw data available as 'all'.

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        gdf = gpd.GeoDataFrame(
            data.drop(["lat", "long"], axis=1),
            geometry=gpd.points_from_xy(x=data["long"], y=data["lat"]),
            crs=WGS84_CRS,
        )
        return gdf

    def load(
        self, hf_token: Optional[str] = None, version: Optional[str] = "res_8"
    ) -> tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str, optional): version of a dataset.
                Available: 'res_8', 'res_9', 'res_10'. Defaults to 'res_8'. \
                    Raw, full data available as 'all'.

        Returns:
            gpd.GeoDataFrame, gpd.Geodataframe | None : Loaded train data and test data if exist.
        """
        return super().load(hf_token, version)
