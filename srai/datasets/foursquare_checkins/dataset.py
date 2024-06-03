"""
Foursquare Checkins dataset loader.

This module contains Foursquare Checkins Dataset.
"""

import itertools
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from srai.constants import WGS84_CRS
from srai.datasets import HuggingFaceDataset


class FoursquareCheckinsDataset(HuggingFaceDataset):
    """
    Foursquare Checkins dataset.

    The Foursquare dataset consists of check-in data for different cities. One subset contains
    check-ins in NYC and Tokyo collected for about 10 month (from 12 April 2012 to 16 February
    2013). It contains 227,428 check-ins in New York city and 573,703 check-ins in Tokyo. Each
    check-in is associated with its time stamp, its GPS coordinates and its semantic meaning
    (represented by fine-grained venue-categories).
    """

    def __init__(self) -> None:
        """Create the dataset."""
        super().__init__("kraina/foursquare_checkins", "tokyo_newyork")

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
            data=data.drop(columns=["latitude", "longitude"]),
            geometry=data.apply(
                lambda row: LineString(
                    list(itertools.starmap(Point, zip(row["longitude"], row["latitude"])))
                ),
                axis=1,
            ),
            crs=WGS84_CRS,
        )

        return gdf
