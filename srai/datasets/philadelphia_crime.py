"""
Philadelphia Crime dataset loader.

This module contains Philadelphia Crime Dataset.
"""

from contextlib import suppress
from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote

import duckdb
import geopandas as gpd
import pandas as pd
from datasets import Dataset

from srai.constants import WGS84_CRS
from srai.datasets import PointDataset

years_previous: list[int] = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
years_current: list[int] = [2020, 2021, 2022, 2023]


class PhiladelphiaCrimeDataset(PointDataset):
    """
    Philadelphia Crime dataset.

    Crime incidents from the Philadelphia Police Department. Part I crimes include violent offenses
    such as aggravated assault, rape, arson, among others. Part II crimes include simple assault,
    prostitution, gambling, fraud, and other non-violent offenses.
    """

    def __init__(self) -> None:
        """Create the dataset."""
        numerical_columns = None
        categorical_columns = [
            "hour",
            "dispatch_date",
            "dispatch_time",
            "dc_dist",
            "psa",
        ]
        type = "point"
        # target = "text_general_code"
        target = "count"

        super().__init__(
            "kraina/philadelphia_crime",
            type=type,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target=target,
        )

    def _get_cache_file(self, version: str) -> Path:
        """
        Get the path to the Parquet cache file for a given dataset version (year).

        Args:
            version (str): Year of the dataset or the resolution (e.g., 8, 9 or 2015, 2019, 2020).

        Returns:
            Path: Path object pointing to the cached Parquet file for the given version.
        """
        return self._get_global_dataset_cache_path() / f"{version}.parquet"

    def _make_url(self, year: int) -> str:
        """
        Dataset download URL for a given year.

        Args:
            year (int): Year of the dataset (e.g., 2015, 2019, 2020, 2023).

        Returns:
            str: Appropriate URL-encoded API endpoint to download the dataset in CSV format.
        """
        next_year = year + 1

        base = (
            "https://phl.carto.com/api/v2/sql?filename=incidents_part1_part2"
            "&format=csv&skipfields=cartodb_id,the_geom,the_geom_webmercator&q="
        )

        sql = (
            f"SELECT * , ST_Y(the_geom) AS lat, ST_X(the_geom) AS lng "
            f"FROM incidents_part1_part2 "
            f"WHERE dispatch_date_time >= '{year}-01-01' "
            f"AND dispatch_date_time < '{next_year}-01-01'"
        )

        return base + quote(sql)

    def download_data(self, version: Optional[Union[int, str]] = 2023) -> None:
        """
        Download and cache the Philadelphia crime dataset for a given year.

        - If the Parquet cache already exists, the download step is skipped.
        - Otherwise, the CSV is streamed from the API, converted in-memory to Parquet,
        and cached for future use.

        Args:
            version (int): Dataset year to download (e.g., 2013-2023).
                If given as a short H3 resolution code ('8', '9', '10'),
                defaults to benchmark splits of the year 2023.
        """
        if version is None or len(str(version)) <= 3:
            version = 2023

        cache_file = self._get_cache_file(str(version))
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if not cache_file.exists():
            url = self._make_url(int(version))

            print(f"Downloading crime data for {version}...")
            duckdb.read_csv(url).to_parquet(str(cache_file), compression="zstd")

    def _preprocessing(
        self, data: gpd.GeoDataFrame, version: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocess the dataset from HuggingFace.

        Args:
            data (pd.DataFrame): a dataset to preprocess
            version (str, optional): version of dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        if self.version is None or len(self.version) <= 3:
            version = "2023"
        else:
            version = self.version

        cache_file = self._get_cache_file(str(version))
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Loading cached Parquet file for {version}...")
        df = pd.read_parquet(cache_file)

        if len(str(self.version)) <= 3:
            print("Splitting into train-test subsets ...")
            valid_ids = set(data["objectid"].unique())
            df = df[df["objectid"].isin(valid_ids)]
            df = df.drop_duplicates()

        # df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["lng", "lat"], axis=1),
            geometry=gpd.points_from_xy(df["lng"], df["lat"]),
            crs=WGS84_CRS,
        )

        gdf = gdf[~gdf.geometry.is_empty]
        # TODO: Add numerical and categorical columns
        # if version in years_previous:
        #     self.numerical_columns = None
        #     self.categorical_columns = None
        # else:
        #     self.numerical_columns = None
        #     self.categorical_columns = None

        return gdf

    def load(
        self, version: Optional[Union[int, str]] = 8, hf_token: Optional[str] = None
    ) -> dict[str, gpd.GeoDataFrame]:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str or int, optional): version of a dataset.
                Available: Official spatial train-test split from year 2023 in chosen h3 resolution:
                '8', '9, '10'. Defaults to '8'. Raw data from other years available
                as: '2013', '2014', '2015', '2016', '2017', '2018','2019', '2020', '2021',
                '2022', '2023'.

        Returns:
            dict[str, gpd.GeoDataFrame]: Dictionary with all splits loaded from the dataset. Will
                contain keys "train" and "test" if available.
        """
        self.resolution = None
        self.download_data(version=version)

        from datasets import load_dataset

        result = {}

        self.train_gdf, self.val_gdf, self.test_gdf = None, None, None
        dataset_name = self.path
        self.version = str(version)

        if self.resolution is None and self.version in ("8", "9", "10"):
            with suppress(ValueError):
                # Try to parse version as int (e.g. "8" or "9")
                self.resolution = int(self.version)

        if len(str(version)) <= 3:
            data = load_dataset(dataset_name, str(version), token=hf_token, trust_remote_code=True)
        else:
            empty_dataset = Dataset.from_pandas(pd.DataFrame())
            data = {"train": empty_dataset}
        train = data["train"].to_pandas()
        processed_train = self._preprocessing(train)
        self.train_gdf = processed_train
        result["train"] = processed_train
        if "test" in data:
            test = data["test"].to_pandas()
            processed_test = self._preprocessing(test)
            self.test_gdf = processed_test
            result["test"] = processed_test

        return result
        # return super().load(hf_token=hf_token, version=version)
