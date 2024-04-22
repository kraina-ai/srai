"""
Brightkite dataset loader.

This module contains Brightkite Dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from tqdm.contrib.concurrent import process_map

from srai.datasets import HuggingFaceDataset


class Brightkite(HuggingFaceDataset):
    """
    Brightkite dataset.

    Brightkite was once a location-based social networking service provider where users shared their
    locations by checking-in. The friendship network was collected using their public API, and
    consists of 58,228 nodes and 214,078 edges.

    The network is originally directed but we have constructed a network with undirected edges when
    there is a friendship in both ways. We have also collected a total of 4,491,143 checkins of
    these users over the period of Apr. 2008 - Oct. 2010.
    """

    def _preprocessing(
        self, data: pd.DataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of Brightkite trajectories dataset.
            data_version_name: version of dataset

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        df = data.copy()
        df["geometry"] = process_map(LineString, df["geometry"], chunksize=1000, max_workers=20)
        gdf = gpd.GeoDataFrame(data=df, geometry="geometry", crs="EPSG:4326")

        return gdf
