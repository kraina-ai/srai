"""
New York City Bike dataset loader.

This module contains New York City Bike Dataset.
"""

import os
from typing import Optional  # , override

import geopandas as gpd
from shapely.geometry import MultiPoint

from srai.datasets import Dataset
from srai.loaders import HFLoader


class NYCBike(Dataset):
    """
    New York City Bike dataset.

    Where do Citi Bikers ride? When do they ride? How far do they go? Which stations are most
    popular? What days of the week are most rides taken on? We've heard all of these questions and
    more from you, and we're happy to provide the data to help you discover the answers to these
    questions and more. We invite developers, engineers, statisticians, artists, academics and other
    interested members of the public to use the data we provide for analysis, development,
    visualization and whatever else moves you. This data is provided according to the NYCBS Data Use
    Policy.
    """

    def __init__(self) -> None:
        """Initialize New York City Bike dataset."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "nyc_bike.yaml")
        super().__init__(config_path)

    def _preprocessing(
        self, data: gpd.GeoDataFrame, dataset_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of NYC dataset.
            dataset_version_name: Version name of dataset, e.g. "nyc_bike_2013". \
                Available: nyc_bike_2013 - nyc_bike_2023.

        Returns:
            GeoDataFrame of dataset, contatins location data - Multipoint(StartStation, EndStation).
        """
        if dataset_version_name:
            dataset_year = int(dataset_version_name[-4:])
        else:
            raise ValueError("Dataset version name is not valid.")
        if dataset_year in self.conf["years_previous"]:  # get a year from dataset version name
            start_station_geometry = gpd.points_from_xy(
                x=data["start station longitude"], y=data["start station latitude"]
            )
            end_station_geometry = gpd.points_from_xy(
                x=data["end station longitude"], y=data["end station latitude"]
            )
            multi_point_stations_geometries = [
                MultiPoint([start, end])
                for start, end in zip(start_station_geometry, end_station_geometry)
            ]
            gdf = gpd.GeoDataFrame(
                data.drop(
                    [
                        "start station latitude",
                        "start station longitude",
                        "end station latitude",
                        "end station longitude",
                    ],
                    axis=1,
                ),
                geometry=multi_point_stations_geometries,
                crs="EPSG:4326",
            )

        elif dataset_year in self.conf["years_current"]:
            start_station_geometry = gpd.points_from_xy(x=data["start_lng"], y=data["start_lat"])
            end_station_geometry = gpd.points_from_xy(x=data["end_lng"], y=data["end_lat"])
            multi_point_stations_geometries = [
                MultiPoint([start, end])
                for start, end in zip(start_station_geometry, end_station_geometry)
            ]
            gdf = gpd.GeoDataFrame(
                data.drop(
                    [
                        "start_lng",
                        "start_lat",
                        "end_lng",
                        "end_lat",
                    ],
                    axis=1,
                ),
                geometry=multi_point_stations_geometries,
                crs="EPSG:4326",
            )

        else:
            raise ValueError("Dataset version name is not valid.")

        return gdf

    # @override
    def load(self, dataset_version_name: str = "nyc_bike_2023") -> gpd.GeoDataFrame:
        """
        Method to load dataset.

        Args:
            dataset_version_name: Version name of dataset, e.g. "nyc_bike_2013". \
                Available: nyc_bike_2013 - nyc_bike_2023.

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        dataset_name = self.conf["dataset_name"]
        data = HFLoader(os.environ["HF_access_token"]).load(
            dataset_name=dataset_name, name=dataset_version_name
        )
        processed_data = self._preprocessing(data, dataset_version_name)

        return processed_data
