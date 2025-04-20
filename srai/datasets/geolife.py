"""
Geolife dataset loader.

This module contains Geolife Dataset.
"""

from collections import Counter
from typing import Optional

import geopandas as gpd
import h3
import pandas as pd
from tqdm import tqdm

from srai.constants import WGS84_CRS
from srai.datasets import HuggingFaceDataset


class GeolifeDataset(HuggingFaceDataset):
    """
    Geolife dataset.

    GPS trajectories that were collected in (Microsoft Research Asia) Geolife Project by 182 users
    """

    def __init__(self) -> None:
        """Create the dataset."""
        numerical_columns = ["altitude"]
        categorical_columns = ["mode"]
        type = "trajectory"
        target = "trajectory_id"
        # target = None
        super().__init__(
            "kraina/geolife",
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
                    - timestamp (list[pd.Timestamp])
                    - mode (list[str])
                    - altitude (list[float])

            Returns:
                pd.Series: A new series containing:
                    - duration (float): Duration of the trajectory in seconds.
                    - h3_sequence (list[str]): List of H3 hex IDs visited.
                    - mode
                    - avg_altitude
            """
            coords = row.geometry.coords
            altitudes = row["altitude"]
            timestamps = row["timestamp"]
            mode = row["mode"]

            duration = (timestamps[-1] - timestamps[0]).total_seconds()

            raw_h3_seq = []
            hex_metadata = []

            for (lon, lat), alt, ts, m in zip(coords, altitudes, timestamps, mode):
                hex_id = h3.latlng_to_cell(lat, lon, resolution)
                raw_h3_seq.append(hex_id)
                hex_metadata.append(
                    {
                        "hex_id": hex_id,
                        "altitude": alt,
                        "timestamp": ts,
                        "mode": m,
                    }
                )

            # Remove consecutive duplicates
            cleaned_seq = [hex_metadata[0]]
            for meta in hex_metadata[1:]:
                if meta["hex_id"] != cleaned_seq[-1]["hex_id"]:
                    cleaned_seq.append(meta)

            # Interpolate missing hexes and propagate metadata
            full_seq: list[str] = []
            altitude_interp: list[float] = []
            timestamps_interp: list[pd.Timestamp] = []
            mode_interp: list[str] = []

            for i in range(len(cleaned_seq) - 1):
                start = cleaned_seq[i]
                end = cleaned_seq[i + 1]
                last_known_altitude = start["altitude"]
                last_known_ts = start["timestamp"]
                last_known_mode = start["mode"]

                try:
                    path = h3.grid_path_cells(start["hex_id"], end["hex_id"])
                    if path:
                        # Skip first to avoid duplication
                        if full_seq and path[0] == full_seq[-1]:
                            path = path[1:]
                        for h in path:
                            full_seq.append(h)
                            altitude_interp.append(last_known_altitude)
                            timestamps_interp.append(last_known_ts)
                            mode_interp.append(last_known_mode)
                except Exception:
                    full_seq.append(start["hex_id"])
                    altitude_interp.append(last_known_altitude)
                    timestamps_interp.append(last_known_ts)
                    mode_interp.append(last_known_mode)

            # Add last
            last = cleaned_seq[-1]
            full_seq.append(last["hex_id"])
            altitude_interp.append(last["altitude"])
            timestamps_interp.append(last["timestamp"])
            mode_interp.append(last["mode"])

            res = {
                "user_id": row["user_id"],
                "trajectory_id": row["trajectory_id"],
                "h3_sequence": full_seq,
                "avg_altitude_per_hex": altitude_interp,
                "timestamp": timestamps_interp,
                "geometry": row.geometry,
            }

            if version == "TTE":
                res["duration"] = duration
                res["mode"] = [Counter([m]).most_common(1)[0][0] for m in mode_interp]
            elif version == "HMC":
                split_idx = int(len(full_seq) * 0.85)
                if split_idx == len(full_seq):
                    split_idx = len(full_seq) - 1
                del res["h3_sequence"]
                res["h3_sequence_x"] = full_seq[:split_idx]
                res["h3_sequence_y"] = full_seq[split_idx:]
                res["mode"] = mode_interp

            return pd.Series(res)

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
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  # <- fix here

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
        version: Optional[str] = "HMC",
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
        if version == "TTE" or version == "HMC":
            self.resolution = 9
        elif version == "all":
            if resolution is None:
                raise ValueError("Pass the resolution parameter to generate h3 trajectories.")
            else:
                self.resolution = resolution
        return super().load(hf_token, version)
