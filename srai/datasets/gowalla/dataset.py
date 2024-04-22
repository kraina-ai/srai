"""
Gowalla dataset loader.

This module contains Geolife Dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from tqdm.contrib.concurrent import process_map

from srai.datasets import HuggingFaceDataset


class Gowalla(HuggingFaceDataset):
    """
    Gowalla Dataset.

    Gowalla is a location-based social networking website where users share their locations
    by checking-in. The friendship network is undirected and was collected using their
    public API, and consists of 196,591 nodes and 950,327 edges. We have collected a total
    of 6,442,890 check-ins of these users over the period of Feb. 2009 - Oct. 2010.
    """

    def _preprocessing(
        self, data: pd.DataFrame, data_version_name: Optional[str] = None
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
