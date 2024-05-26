"""
Porto Taxi dataset loader.

This module contains Porto Taxi Dataset.
"""

from datetime import datetime
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely import Point

from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.datasets import HuggingFaceDataset


class PortoTaxiDataset(HuggingFaceDataset):
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
        """Create the dataset."""
        numerical_columns = None
        categorical_columns = None
        type = None
        super().__init__(
            "kraina/porto_taxi",
            type=type,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocess the dataset from HuggingFace.

        Args:
            data (pd.DataFrame): a dataset to preprocess
            version (str, optional): version of dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        df = data.copy()
        df["timestamp"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x))
        df[GEOMETRY_COLUMN] = df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        df.drop(["longitude", "latitude"], axis=1, inplace=True)

        gdf = gpd.GeoDataFrame(data=df, geometry=GEOMETRY_COLUMN, crs=WGS84_CRS)
        return gdf
