"""
Geolife dataset loader.

This module contains Geolife Dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from srai.constants import WGS84_CRS
from srai.datasets import HuggingFaceDataset


class GeolifeDataset(HuggingFaceDataset):
    """
    Geolife Dataset.

    This GPS trajectory dataset was collected in (Microsoft Research Asia) Geolife project by 182
    users in a period of over five years  (from April 2007 to August 2012). A GPS trajectory of this
    dataset is represented by sequence of time-stamped points each of which contains the information
    of altitude, longitude, latitude. This dataset contains 17,784 trajectories, ~25M Points with a
    total distance of 1,292,951 kilometers and a total duration of 50,176 hours.
    """

    def __init__(self) -> None:
        """Create the dataset."""
        super().__init__("kraina/geolife")

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
            data.drop(["arrays_geometry"], axis=1),
            geometry=gpd.GeoSeries(data["arrays_geometry"].map(LineString)),
            crs=WGS84_CRS,
        )

        return gdf
