"""
Geolife dataset loader.

This module contains Geolife Dataset.
"""

import csv
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union, cast
from zipfile import ZipFile

import duckdb
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from tqdm import tqdm

from srai.constants import FORCE_TERMINAL, WGS84_CRS
from srai.datasets import TrajectoryDataset
from srai.loaders.download import download_file

LNG_MIN = 73.33
LNG_MAX = 135.05
LAT_MIN = 20.14
LAT_MAX = 53.55


class GeolifeDataset(TrajectoryDataset):
    """
    Geolife dataset.

    GPS trajectories that were collected in (Microsoft Research Asia) Geolife Project by 182 users
    """

    RAW_ZIP_DOWNLOAD_URL = (
        "https://download.microsoft.com/download"
        "/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
    )

    def __init__(self) -> None:
        """Create the dataset."""
        numerical_columns = ["altitude"]
        categorical_columns = ["mode"]
        type = "trajectory"
        target = "trajectory_id"

        self.cache_root = self._get_global_dataset_cache_path()
        self.raw_data_path = self.cache_root / "raw"
        self.processed_path = self.cache_root / "preprocessed"
        self.prepared_path = self.cache_root / "prepared"

        # target = None
        super().__init__(
            "kraina/geolife",
            type=type,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target=target,
        )

    def _download_geolife(self) -> None:
        """
        Download and extract the Geolife dataset if not already cached.

        - Downloads a ZIP archive from the official Microsoft Geolife source.
        - Shows tqdm progress bars for both the download and extraction steps.
        - Skips download if the dataset already exists in the cache.
        """
        zip_path = self.raw_data_path / "geolife.zip"
        if not zip_path.exists():
            zip_path.parent.mkdir(exist_ok=True, parents=True)

            download_file(url=GeolifeDataset.RAW_ZIP_DOWNLOAD_URL, fname=zip_path)

            # --- Extract with progress ---
            with ZipFile(zip_path) as zfile:
                members = zfile.namelist()
                for member in tqdm(members, desc="Extracting Geolife", disable=FORCE_TERMINAL):
                    zfile.extract(member, self.raw_data_path)

        else:
            print("Geolife data already exists in cache. Download skipped.")

    def _geolife_clean_plt(
        self, root: Path, user_id: str, input_filepath: str, traj_id: int
    ) -> list[list[str]]:
        """
        Clean header of a Geolife .plt file and prepend metadata.

        Args:
            root (str): Root path of the Geolife dataset directory.
            user_id (str): Identifier of the user (subdirectory name).
            input_filepath (str): File name of the trajectory (.plt) to process.
            traj_id (int): Numeric trajectory identifier to prepend.

        Returns:
            list[list[str]]: Rows of the .plt file with user_id and trajectory_id prepended.
        """
        with (root / user_id / "Trajectory" / input_filepath).open() as fin:
            cr = csv.reader(fin)
            filecontents = [line for line in cr][6:]
            for f in filecontents:
                f.insert(0, str(traj_id))
                f.insert(0, user_id)
        return filecontents

    def _preprocess_geolife_trajectories_df(self, df: pd.DataFrame, result_path: Path) -> None:
        df["datetime"] = pd.to_datetime(df.date + " " + df.time)
        df.drop(["date", "time"], inplace=True, axis=1)

        # fix datetime timezone
        df["datetime"] = (
            df["datetime"].dt.tz_localize("GMT").dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        )
        df["timestamp"] = df["datetime"].astype("int64") // 10**9

        df.latitude = df.latitude.astype(float)
        df.longitude = df.longitude.astype(float)

        df = df[
            (df.latitude > LAT_MIN)
            & (df.latitude < LAT_MAX)
            & (df.longitude > LNG_MIN)
            & (df.longitude < LNG_MAX)
        ]

        df.to_parquet(result_path, compression="zstd")

    def _geolife_preprocess(self) -> None:
        """
        Preprocess the Geolife dataset into a single Parquet file in the cache.

        - Converts date and time into a unified datetime column.
        - Normalizes timezone to Asia/Shanghai and removes timezone info.
        - Converts datetime to UNIX timestamps.
        - Filters trajectories outside valid latitude/longitude bounds.
        - Saves as an Apache Parquet file for efficient reuse.
        """
        geolife_dir = self.raw_data_path / "Geolife Trajectories 1.3" / "Data"
        col_names = [
            "user_id",
            "trajectory_id",
            "latitude",
            "longitude",
            "-",
            "altitude",
            "dayNo",
            "date",
            "time",
        ]
        user_id_dirs = [
            name
            for name in os.listdir(geolife_dir)
            if os.path.isdir(os.path.join(geolife_dir, name))
        ]

        self.processed_path.mkdir(exist_ok=True, parents=True)

        with tqdm(
            total=len(user_id_dirs),
            desc="Loading Geolife user trajectories",
            disable=FORCE_TERMINAL,
        ) as pbar:
            for user_id in np.sort(user_id_dirs):
                user_result_path = self.processed_path / f"{user_id}.parquet"
                if user_result_path.exists():
                    pbar.update()
                    continue

                data = []
                subdirs = [
                    item
                    for item in os.listdir(geolife_dir / user_id / "Trajectory")
                    if not item.endswith(".DS_Store")
                ]
                traj_id = 0
                for subdir in subdirs:
                    data += self._geolife_clean_plt(geolife_dir, user_id, subdir, traj_id)
                    traj_id += 1

                user_df = pd.DataFrame(data, columns=col_names)
                self._preprocess_geolife_trajectories_df(user_df, result_path=user_result_path)
                pbar.update()

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
            # mode = row["mode"]

            duration = (timestamps[-1] - timestamps[0]).total_seconds()

            raw_h3_seq = []
            hex_metadata = []

            for (lon, lat), alt, ts in zip(coords, altitudes, timestamps):
                hex_id = h3.latlng_to_cell(lat, lon, resolution)
                raw_h3_seq.append(hex_id)
                hex_metadata.append(
                    {
                        "hex_id": hex_id,
                        "altitude": alt,
                        "timestamp": ts,
                        # "mode": m,
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
            # mode_interp: list[str] = []

            for i in range(len(cleaned_seq) - 1):
                start = cleaned_seq[i]
                end = cleaned_seq[i + 1]
                last_known_altitude = start["altitude"]
                last_known_ts = start["timestamp"]
                # last_known_mode = start["mode"]

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
                            # mode_interp.append(last_known_mode)
                except Exception:
                    full_seq.append(start["hex_id"])
                    altitude_interp.append(last_known_altitude)
                    timestamps_interp.append(last_known_ts)
                    # mode_interp.append(last_known_mode)

            # Add last
            last = cleaned_seq[-1]
            full_seq.append(last["hex_id"])
            altitude_interp.append(last["altitude"])
            timestamps_interp.append(last["timestamp"])
            # mode_interp.append(last["mode"])

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
                # res["mode"] = [Counter([m]).most_common(1)[0][0] for m in mode_interp]
            elif version == "HMP":
                split_idx = int(len(full_seq) * 0.85)
                if split_idx == len(full_seq):
                    split_idx = len(full_seq) - 1
                del res["h3_sequence"]
                res["h3_sequence_x"] = full_seq[:split_idx]
                res["h3_sequence_y"] = full_seq[split_idx:]
                # res["mode"] = mode_interp

            return pd.Series(res)

        # Apply with progress bar
        hexes_df = _gdf.apply(process_row, axis=1)

        if version == "HMP":
            hexes_df = hexes_df[
                hexes_df["h3_sequence_x"].apply(lambda x: len(x) > 0)
                & hexes_df["h3_sequence_y"].apply(lambda y: len(y) > 0)
            ].reset_index(drop=True)
        elif version == "TTE":
            hexes_df = hexes_df[hexes_df["h3_sequence"].apply(lambda x: len(x) > 0)].reset_index(
                drop=True
            )
            hexes_df = hexes_df[hexes_df["duration"] > 0.0].reset_index(drop=True)
        hexes_gdf = gpd.GeoDataFrame(hexes_df, geometry="geometry", crs=WGS84_CRS)

        return hexes_gdf

    def _transform_single_user_data(
        self, data: pd.DataFrame, input_parquet_path: Path
    ) -> gpd.GeoDataFrame:
        df_original = pd.read_parquet(input_parquet_path)
        # df_original["unique_id"] = df_original["user_id"].astype(str) + df_original[
        #     "trajectory_id"
        # ].astype(str)
        # valid_ids = set(data["trajectory_id"].unique())

        # df = df_original[df_original["unique_id"].isin(valid_ids)]
        # df = df.drop_duplicates()
        # df = df.rename(
        #     columns={"trajectory_id": "trajectory_id_old", "unique_id": "trajectory_id"}
        # )

        df_original["trajectory_id"] = df_original["user_id"].astype(str) + df_original[
            "trajectory_id"
        ].astype(str)
        valid_ids = set(data["trajectory_id"].unique())

        df = df_original[df_original["trajectory_id"].isin(valid_ids)]
        df = df.drop_duplicates()
        # df = df.rename(
        #     columns={"trajectory_id": "trajectory_id_old", "unique_id": "trajectory_id"}
        # )

        # df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  # <- fix here

        gdf = gpd.GeoDataFrame(
            df.drop(["longitude", "latitude"], axis=1),
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs=WGS84_CRS,
        )

        if gdf.empty:
            return gpd.GeoDataFrame(columns=["trajectory_id"], geometry=[], crs=WGS84_CRS)

        assert self.target is not None
        assert self.version is not None

        trajectory_gdf = self._agg_points_to_trajectories(
            gdf=gdf, target_column=self.target, progress_bar=False
        )

        if self.resolution is not None:
            hexes_gdf = self._aggregate_trajectories_to_hexes(
                gdf=trajectory_gdf, resolution=self.resolution, version=self.version
            )

            return hexes_gdf

        return trajectory_gdf

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of Geolife dataset with trajectory indexes.
            version: version of a dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        hasher = hashlib.new("sha256")
        hasher.update(str(data.values).encode())
        data_hash = hasher.hexdigest()
        parquet_path = (
            self.prepared_path / f"geolife_{self.version}_{self.resolution}_{data_hash}.parquet"
        )

        if not parquet_path.exists():
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            user_trajectories_parquet_files = sorted(self.processed_path.glob("*.parquet"))
            with (
                tempfile.TemporaryDirectory(dir=self.prepared_path.resolve()) as tmp_dir_name,
                duckdb.connect() as db_conn,
            ):
                tmp_dir_path = Path(tmp_dir_name)
                transformed_file_paths = []
                for user_trajectories_parquet_file in tqdm(
                    user_trajectories_parquet_files,
                    desc="Transforming Geolife user trajectories",
                    disable=FORCE_TERMINAL,
                ):
                    save_file_path = tmp_dir_path / user_trajectories_parquet_file.name
                    hexes_gdf = self._transform_single_user_data(
                        data, input_parquet_path=user_trajectories_parquet_file
                    )
                    if not hexes_gdf.empty:
                        hexes_gdf.to_parquet(save_file_path)
                        transformed_file_paths.append(str(save_file_path))

                db_conn.install_extension("spatial")
                db_conn.load_extension("spatial")

                # Filter outlier linestrings
                percentiles = cast(
                    "tuple[list[float]]",
                    db_conn.sql(
                        f"""
                    SELECT quantile_cont(ST_Length(geometry), [0.05, 0.95])
                    FROM read_parquet({transformed_file_paths})
                    """
                    ).fetchone(),
                )[0]
                lower = percentiles[0]
                upper = percentiles[1]

                all_trajectories = db_conn.sql(
                    f"""
                    SELECT *
                    FROM read_parquet({transformed_file_paths})
                    WHERE ST_Length(geometry) BETWEEN {lower} AND {upper}
                    """
                )

                all_trajectories.to_parquet(str(parquet_path), compression="zstd")

        return gpd.read_parquet(parquet_path)

    def load(
        self,
        version: Optional[Union[int, str]] = "HMP",
        hf_token: Optional[str] = None,
        resolution: Optional[int] = None,
    ) -> dict[str, gpd.GeoDataFrame]:
        """
        Method to load dataset.

        Args:
            hf_token (Optional[str]): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (Optional[str, int]): version of a dataset.
                Available: Official train-test split for Travel Time Estimation task (TTE) and
                Human Mobility Prediction task (HMP). Raw data from available as: 'all'.
            resolution (Optional[int]): H3 resolution for hex trajectories.
                Neccessary if using 'all' split.

        Returns:
            dict[str, gpd.GeoDataFrame]: Dictionary with all splits loaded from the dataset. Will
                contain keys "train" and "test" if available.
        """
        self.version = str(version)
        if self.version in ("TTE", "HMP"):
            self.resolution = 9
        elif self.version == "all":
            self.resolution = resolution if resolution is not None else None
        else:
            raise NotImplementedError("Version not implemented")

        if not self.prepared_path.exists():
            self._download_geolife()
            self._geolife_preprocess()

        # Remove raw downloaded data from cache and keep only preprocessed files
        shutil.rmtree(self.raw_data_path, ignore_errors=True)

        return super().load(hf_token=hf_token, version=version)
