"""
Brightkite dataset loader.

This module contains Brightkite Dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from tqdm.contrib.concurrent import process_map

from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.datasets import HuggingFaceDataset


class BrightkiteDataset(HuggingFaceDataset):
    """
    Brightkite dataset.

    Brightkite was once a location-based social networking service provider where users shared their
    locations by checking-in. The friendship network was collected using their public API, and
    consists of 58,228 nodes and 214,078 edges.

    The network is originally directed but we have constructed a network with undirected edges when
    there is a friendship in both ways. We have also collected a total of 4,491,143 checkins of
    these users over the period of Apr. 2008 - Oct. 2010.
    """

    def __init__(self) -> None:
        """Create the dataset."""
        super().__init__("kraina/brightkite")

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data (pd.DataFrame): Data of Brightkite trajectories dataset.
            version (str, optional): version of dataset.

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        df = data.copy()
        df[GEOMETRY_COLUMN] = process_map(LineString, df[GEOMETRY_COLUMN], chunksize=1000)
        gdf = gpd.GeoDataFrame(data=df, geometry=GEOMETRY_COLUMN, crs=WGS84_CRS)

        return gdf
