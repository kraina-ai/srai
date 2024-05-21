"""
Gowalla dataset loader.

This module contains Geolife Dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from tqdm.contrib.concurrent import process_map

from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.datasets import HuggingFaceDataset


class GowallaDataset(HuggingFaceDataset):
    """
    Gowalla Dataset.

    Gowalla is a location-based social networking website where users share their locations
    by checking-in. The friendship network is undirected and was collected using their
    public API, and consists of 196,591 nodes and 950,327 edges. We have collected a total
    of 6,442,890 check-ins of these users over the period of Feb. 2009 - Oct. 2010.
    """

    def __init__(self) -> None:
        """Create the dataset."""
        numerical_columns = None
        categorical_columns = None
        type = None
        super().__init__(
            "kraina/gowalla",
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
        df[GEOMETRY_COLUMN] = process_map(LineString, df[GEOMETRY_COLUMN], chunksize=1000)
        gdf = gpd.GeoDataFrame(data=df, geometry=GEOMETRY_COLUMN, crs=WGS84_CRS)

        return gdf
