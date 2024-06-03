"""
New York City Bike dataset loader.

This module contains New York City Bike Dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint

from srai.constants import WGS84_CRS
from srai.datasets import HuggingFaceDataset


class NYCBikeDataset(HuggingFaceDataset):
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

    years_previous: list[int] = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    years_current: list[int] = [2021, 2022, 2023]

    def __init__(self) -> None:
        """Create the dataset."""
        super().__init__("kraina/nyc_bike")

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocess the dataset from HuggingFace.

        Args:
            data (pd.DataFrame): a dataset to preprocess
            version (str, optional): version of dataset.
                Available: from `nyc_bike_2013` to `nyc_bike_2023`.

        Returns:
            gpd.GeoDataFrame: preprocessed data with  Multipoint(StartStation, EndStation)
        """
        if version:
            dataset_year = int(version[-4:])
        else:
            raise ValueError("Dataset version name can't be None")

        if dataset_year in self.years_previous:
            c_start_lng = "start station longitude"
            c_start_lat = "start station latitude"
            c_end_lng = "end station longitude"
            c_end_lat = "end station latitude"
        elif dataset_year in self.years_current:
            c_start_lng = "start_lng"
            c_start_lat = "start_lat"
            c_end_lng = "end_lng"
            c_end_lat = "end_lat"
        else:
            raise ValueError(f"Dataset version name `{version}` is not valid.")

        start_station_geometry = gpd.points_from_xy(x=data[c_start_lng], y=data[c_start_lat])
        end_station_geometry = gpd.points_from_xy(x=data[c_end_lng], y=data[c_end_lat])
        multi_point_stations_geometries = [
            MultiPoint([start, end])
            for start, end in zip(start_station_geometry, end_station_geometry)
        ]
        gdf = gpd.GeoDataFrame(
            data.drop(
                [
                    c_start_lng,
                    c_start_lat,
                    c_end_lng,
                    c_end_lat,
                ],
                axis=1,
            ),
            geometry=multi_point_stations_geometries,
            crs=WGS84_CRS,
        )

        return gdf

    def load(
        self, hf_token: Optional[str] = None, version: Optional[str] = "nyc_bike_2023"
    ) -> gpd.GeoDataFrame:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str, optional): version of a dataset.
                Available: from `nyc_bike_2013` to `nyc_bike_2023`. Defaults to `nyc_bike_2023`.

        Returns:
            gpd.GeoDataFrame: Loaded data.
        """
        return super().load(hf_token, version)
