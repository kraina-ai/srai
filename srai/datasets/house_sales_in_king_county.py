"""
House Sales in King County dataset loader.

This module contains House Sales in King County Dataset.
"""

from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from srai.constants import WGS84_CRS
from srai.datasets import PointDataset


class HouseSalesInKingCountyDataset(PointDataset):
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

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data (pd.DataFrame): a dataset to preprocess
            version (str, optional): version of dataset.


        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        gdf = gpd.GeoDataFrame(
            data.drop(["lat", "long"], axis=1),
            geometry=gpd.points_from_xy(x=data["long"], y=data["lat"]),
            crs=WGS84_CRS,
        )

        lower = np.percentile(gdf[self.target], 10)
        upper = np.percentile(gdf[self.target], 90)

        # Filter out outlier prices based on aggregated value
        gdf = gdf[(gdf[self.target] >= lower) & (gdf[self.target] <= upper)]
        return gdf

    def load(
        self, version: Optional[Union[int, str]] = 8, hf_token: Optional[str] = None
    ) -> dict[str, gpd.GeoDataFrame]:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str or int, optional): version of a dataset.
                Available: '8', '9', '10', where number is a h3 resolution used in train-test \
                    split. Defaults to '8'. Raw, full data available as 'all'.

        Returns:
            dict[str, gpd.GeoDataFrame]: Dictionary with all splits loaded from the dataset. Will
                contain keys "train" and "test" if available.
        """
        return super().load(hf_token=hf_token, version=version)
