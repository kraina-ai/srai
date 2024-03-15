"""
Brightkite dataset loader.

This module contains Brightkite Dataset.
"""

import os
from typing import Optional

import geopandas as gpd
from shapely.geometry import LineString
from tqdm.contrib.concurrent import process_map

from srai.datasets import Dataset


class Brightkite(Dataset):
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
        """Initialize Brightkite dataset."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "brightkite.yaml")
        super().__init__(config_path)

    def _preprocessing(
        self, data: gpd.GeoDataFrame, data_version_name: Optional[str] = None
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
