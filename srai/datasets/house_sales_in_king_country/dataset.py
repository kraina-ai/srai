"""
House Sales in King Country dataset loader.

This module contains House Sales in King Country Dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd

from srai.constants import WGS84_CRS
from srai.datasets import HuggingFaceDataset


class HouseSalesInKingCountryDataset(HuggingFaceDataset):
    """
    House Sales in King Country dataset.

    This dataset contains house sale prices for King County, which includes Seattle. It includes
    homes sold between May 2014 and May 2015.

    It's a great dataset for evaluating simple regression models.
    """

    def __init__(self) -> None:
        """Create the dataset."""
        numerical_columns = None
        categorical_columns = None
        super().__init__(
            "kraina/house_sales_in_king_country",
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )

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
            data.drop(["lat", "long"], axis=1),
            geometry=gpd.points_from_xy(x=data["long"], y=data["lat"]),
            crs=WGS84_CRS,
        )
        return gdf
