"""
Gowalla dataset loader.

This module contains Geolife Dataset.
"""

import os
from typing import Optional

import geopandas as gpd
from shapely.geometry import LineString
from tqdm.contrib.concurrent import process_map

from ._base import Dataset


class Gowalla(Dataset):
    """
    Gowalla Dataset.

    Gowalla is a location-based social networking website where users share their locations
    by checking-in. The friendship network is undirected and was collected using their
    public API, and consists of 196,591 nodes and 950,327 edges. We have collected a total
    of 6,442,890 check-ins of these users over the period of Feb. 2009 - Oct. 2010.
    """

    def __init__(self) -> None:
        """Initialize Gowalla dataset."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "gowalla.yaml")
        super().__init__(config_path)

    def _preprocessing(
        self, data: gpd.GeoDataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data.

        Args:
            data (gpd.GeoDataFrame): Data of Gowalla trajectories dataset.
            data_version_name (Optional[str], optional): Version of dataset. Defaults to None.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of dataset, contatins location data.
        """
        df = data.copy()
        df["geometry"] = process_map(LineString, df["geometry"], chunksize=1000, max_workers=20)
        gdf = gpd.GeoDataFrame(data=df, geometry="geometry", crs="EPSG:4326")

        return gdf
