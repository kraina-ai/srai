"""File contains dataset definition of NYC Bike data."""

import geopandas as gpd
from shapely.geometry import MultiPoint

from srai.datasets import Dataset


class NYCBike(Dataset):
    def _preprocessing(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Preprocessing to get GeoDataFrame, based on GEO_EDA files."""
        if (
            int(self.conf["name"][-4:]) in self.conf["years_previous"]
        ):  # get a year from dataset version name
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

        elif int(self.conf["name"][-4:]) in self.conf["years_current"]:

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
            raise ValueError("Version name is not valid.")

        return gdf
