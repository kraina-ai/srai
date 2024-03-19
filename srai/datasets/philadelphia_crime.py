import geopandas as gpd
import os
from srai.loaders import HFLoader

from srai.datasets import Dataset
from typing import override


class PhiladelphiaCrime(Dataset):
    def _preprocessing(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of Philadelphia Crime dataset.

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["lng", "lat"], axis=1),
            geometry=gpd.points_from_xy(df["lng"], df["lat"]),
            crs="EPSG:4326",
        )
        return gdf

    @override
    def load(self, dataset_version_name: str = "incidents_2023") -> gpd.GeoDataFrame:
        """
        Method to load dataset.

        Args:
            dataset_version_name: Version name of dataset, e.g. "incidents_2013". Available: incidents_2013 - incidents_2023.

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        dataset_name = self.conf["dataset_name"]
        data = HFLoader(os.environ["HF_access_token"]).load(
            dataset_name=dataset_name, name=dataset_version_name
        )
        processed_data = self._preprocessing(data)

        return processed_data
