"""
House Sales in King Country dataset loader.

This module contains House Sales in King Country Dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd

from srai.datasets import HuggingFaceDataset


class HouseSalesInKingCountry(HuggingFaceDataset):
    """
    House Sales in King Country dataset.

    This dataset contains house sale prices for King County, which includes Seattle. It includes
    homes sold between May 2014 and May 2015.

    It's a great dataset for evaluating simple regression models.
    """

    def _preprocessing(
        self, data: pd.DataFrame, data_version_name: Optional[str] = None
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
