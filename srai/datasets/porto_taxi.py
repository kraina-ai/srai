"""
Porto Taxi dataset loader.

This module contains Porto Taxi Dataset.
"""

import os
from typing import Optional

import geopandas as gpd
from shapely.geometry import LineString
from tqdm.contrib.concurrent import process_map

from srai.datasets import Dataset


class PortoTaxi(Dataset):
    """
    Porto Taxi dataset.

    The dataset covers a year of trajectory data for taxis in Porto, Portugal
    Each ride is categorized as:
    A) taxi central based,
    B) stand-based or
    C) non-taxi central based.
    Each data point represents a completed trip initiated through
    the dispatch central, a taxi stand, or a random street.
    """

    def __init__(self) -> None:
        """Initialize  Porto Taxi dataset."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "porto_taxi.yaml")
        super().__init__(config_path)

    def _preprocessing(
        self, data: gpd.GeoDataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of Porto Taxi trajectories dataset.
            data_version_name: version of dataset

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        df = data.copy()
        df["geometry"] = process_map(LineString, df["geometry"], chunksize=1000, max_workers=20)
        gdf = gpd.GeoDataFrame(data=df, geometry="geometry", crs="EPSG:4326")

        return gdf
