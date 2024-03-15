"""
Foursquare Checkins dataset loader.

This module contains Foursquare Checkins Dataset.
"""

import os
from typing import Optional

import geopandas as gpd
from shapely.geometry import LineString, Point

from srai.datasets import Dataset


class FoursquareCheckins(Dataset):
    """
    Foursquare Checkins dataset.

    The Foursquare dataset consists of check-in data for different cities. One subset contains
    check-ins in NYC and Tokyo collected for about 10 month (from 12 April 2012 to 16 February
    2013). It contains 227,428 check-ins in New York city and 573,703 check-ins in Tokyo. Each
    check-in is associated with its time stamp, its GPS coordinates and its semantic meaning
    (represented by fine-grained venue-categories).
    """

    def __init__(self) -> None:
        """Initialize Foursquare Checkins dataset."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "foursquare_checkins.yaml")
        super().__init__(config_path)

    def _preprocessing(
        self, data: gpd.GeoDataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of Geolife trajectories dataset.
            data_version_name: version of dataset

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            data=df.drop(columns=["latitude", "longitude"]),
            geometry=df.apply(
                lambda row: LineString(
                    [Point(lon, lat) for lon, lat in zip(row["longitude"], row["latitude"])]
                ),
                axis=1,
            ),
            crs="EPSG:4326",
        )

        return gdf
