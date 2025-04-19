"""
Porto Taxi dataset loader.

This module contains Porto Taxi Dataset.
"""

from typing import Optional

import geopandas as gpd
import h3
import pandas as pd
from tqdm import tqdm

from srai.constants import WGS84_CRS
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
        numerical_columns = ["speed"]
        categorical_columns = ["call_type", "origin_call", "origin_stand", "day_type"]
        type = "trajectory"
        target = "trip_id"
        # target = None
        super().__init__(
            "kraina/porto_taxi",
            type=type,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target=target,
        )

    def _aggregate_trajectories_to_hexes(
        self,
        gdf: gpd.GeoDataFrame,
        resolution: int,
        version: str,
    ) -> gpd.GeoDataFrame:
        """
        Preprocess the gdf with linestring trajectories to h3 trajectories.

        Args:
            gdf (gpd.DataFrame): a gdf with prepared linestring.
            resolution (int) : h3 resolution to regionalize data.
            version (str): version of dataset.

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        tqdm.pandas(desc="Building h3 trajectories")
        _gdf = gdf.copy()
        # if version == "TTE":

        def process_row(row: pd.Series) -> pd.Series:
            """
            Process a single row containing a trajectory LineString and associated metadata.

            Converting to H3 hexagons.

            Args:
                row (pd.Series): A row from a GeoDataFrame with at least the following fields:
                    - geometry (shapely.geometry.LineString)
                    - speed (list[float])
                    - timestamp (list[pd.Timestamp])
                    - day_type (str)
                    - call_type (str)
                    - taxi_id (int or str)

            Returns:
                pd.Series: A new series containing:
                    - duration (float): Duration of the trajectory in seconds.
                    - h3_sequence (list[str]): List of H3 hex IDs visited.
                    - avg_speed_per_hex (list[float]): Average speed per H3 hex.
                    - day_type, call_type, taxi_id, geometry: Carried over from input.
            """
            coords = row.geometry.coords
            speeds = row["speed"]
            timestamps = row["timestamp"]

            duration = (timestamps[-1] - timestamps[0]).total_seconds()

            hex_speed_map: dict[str, float] = {}
            hex_point_counts: dict[str, int] = {}

            for (lon, lat), speed in zip(coords, speeds):
                hex_id = h3.latlng_to_cell(lat, lon, resolution)
                hex_speed_map[hex_id] = hex_speed_map.get(hex_id, 0) + speed
                hex_point_counts[hex_id] = hex_point_counts.get(hex_id, 0) + 1

            h3_sequence = list(hex_speed_map)
            avg_speed_per_hex = [hex_speed_map[h] / hex_point_counts[h] for h in h3_sequence]

            if version == "TTE":
                return pd.Series(
                    {
                        "trip_id": row["trip_id"],
                        "duration": duration,
                        "h3_sequence": h3_sequence,
                        "avg_speed_per_hex": avg_speed_per_hex,
                        "day_type": row["day_type"],
                        "call_type": row["call_type"],
                        "taxi_id": row["taxi_id"],
                        "geometry": row.geometry,
                        "timestamp": row["timestamp"],
                    }
                )
            elif version == "HMC":
                raise NotImplementedError
            elif version == "all":
                return pd.Series(
                    {
                        "trip_id": row["trip_id"],
                        "h3_sequence": h3_sequence,
                        "avg_speed_per_hex": avg_speed_per_hex,
                        "day_type": row["day_type"],
                        "call_type": row["call_type"],
                        "taxi_id": row["taxi_id"],
                        "geometry": row.geometry,
                        "timestamp": row["timestamp"],
                    }
                )

        # Apply with progress bar
        hexes_df = _gdf.progress_apply(process_row, axis=1)
        hexes_gdf = gpd.GeoDataFrame(hexes_df, geometry="geometry", crs=WGS84_CRS)

        return hexes_gdf

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of Chicago Crime dataset.
            version: version of a dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        df = data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])  # <- fix here

        gdf = gpd.GeoDataFrame(
            df.drop(["longitude", "latitude"], axis=1),
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs=WGS84_CRS,
        )

        assert self.target is not None
        assert self.resolution is not None
        assert self.version is not None
        trajectory_gdf = self._agg_points_to_trajectories(gdf=gdf, target_column=self.target)

        hexes_gdf = self._aggregate_trajectories_to_hexes(
            gdf=trajectory_gdf, resolution=self.resolution, version=self.version
        )

        return hexes_gdf

    def load(
        self,
        hf_token: Optional[str] = None,
        version: Optional[str] = "TTE",
        resolution: Optional[int] = None,
    ) -> dict[str, gpd.GeoDataFrame]:
        """
        Method to load dataset.

        Args:
            hf_token (Optional[str]): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (Optional[str]): version of a dataset.
                Available: Official train-test split for Travel Time Estimation task (TTE) and
                Human Mobility Classification task (HMC). Raw data from available as: 'all'.
            resolution (Optional[int]): H3 resolution for hex trajectories.
                Neccessary if using 'all' split.

        Returns:
            dict[str, gpd.GeoDataFrame]: Dictionary with all splits loaded from the dataset. Will
                contain keys "train" and "test" if available.
        """
        if version == "TTE":
            self.resolution = 9
        elif version == "HMC":
            self.resolution = 9
        elif version == "all":
            if resolution is None:
                raise ValueError("Pass the resolution parameter to generate h3 trajectories.")
            else:
                self.resolution = resolution
        return super().load(hf_token, version)
