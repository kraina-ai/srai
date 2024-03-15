"""File contains dataset definition of NYC Bike data."""

import geopandas as gpd

from srai.datasets import Dataset


class NYCBike(Dataset):
    def _preprocessing(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Preprocessing to get GeoDataFrame, based on GEO_EDA files."""
        raise NotImplementedError
