"""
AirbnbMulticity dataset loader.

This module contains AirbnbMulticity dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd

from srai.datasets import HFDataset


class AirbnbMulticity(HFDataset):
    """
    AirbnbMulticity dataset.

    Dataset description will be added.
    """

    def _preprocessing(
        self, data: pd.DataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of AirbnbMulticity dataset.
            data_version_name: version of dataset

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["latitude", "longitude"], axis=1),
            geometry=gpd.points_from_xy(x=df["longitude"], y=df["latitude"]),
            crs="EPSG:4326",
        )

        return gdf
