"""
The San Francisco Police Department's (SFPD) Incident Report dataset loader.

This module contains The San Francisco Police Department's (SFPD) Incident Report Datatset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd

from srai.datasets import HFDataset


class PoliceDepartmentIncidents(HFDataset):
    """
    The San Francisco Police Department's (SFPD) Incident Report Datatset.

    This dataset includes incident reports that have been filed as of January 1, 2018 till March,
    2024. These reports are filed by officers or self-reported by members of the public using SFPD’s
    online reporting system.
    """

    def _preprocessing(
        self, data: pd.DataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of The San Francisco Police Department's (SFPD) Incident Report dataset.
            data_version_name: version of dataset

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["Latitude", "Longitude"], axis=1),
            geometry=gpd.points_from_xy(x=df["Longitude"], y=df["Latitude"]),
            crs="EPSG:4326",
        )
        return gdf
