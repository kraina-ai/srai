"""File contains dataset definition of NYC Bike data."""

import geopandas as gpd
from shapely.geometry import LineString

from srai.datasets import Dataset


class Geolife(Dataset):
    def _preprocessing(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of Geolife trajectories dataset.

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        gdf = gpd.GeoDataFrame(
            data.drop(["arrays_geometry"], axis=1),
            geometry=gpd.GeoSeries(data["arrays_geometry"].map(LineString)),
            crs="EPSG:4326",
        )

        return gdf
