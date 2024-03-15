"""
House Sales in King Country dataset loader.

This module contains House Sales in King Country Dataset.
"""

import os
from typing import Optional

import geopandas as gpd

from srai.datasets import Dataset


class HouseSalesInKingCountry(Dataset):
    """
    House Sales in King Country dataset.

    This dataset contains house sale prices for King County, which includes Seattle. It includes
    homes sold between May 2014 and May 2015.

    It's a great dataset for evaluating simple regression models.
    """

    def __init__(self) -> None:
        """Initialize House Sales in King Country dataset."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "house_sales_in_king_country.yaml")
        super().__init__(config_path)

    def _preprocessing(
        self, data: gpd.GeoDataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of House Sales In King Country dataset.
            data_version_name: version of dataset

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        gdf = gpd.GeoDataFrame(
            data.drop(["lat", "long"], axis=1),
            geometry=gpd.points_from_xy(x=data["long"], y=data["lat"]),
            crs="EPSG:4326",
        )
        return gdf
