"""File contains dataset definition of NYC Bike data."""
from srai.datasets import Dataset
import geopandas as gpd
import os


class NYCBike(Dataset):
    def _preprocessing(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame, based on GEO_EDA files.
        """
        raise NotImplementedError

