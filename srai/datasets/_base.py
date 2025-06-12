"""Base classes for Datasets."""

import abc
from typing import Literal, Optional, Union

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from datasets import load_dataset
from shapely.geometry import LineString, Polygon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from srai.regionalizers import H3Regionalizer


class HuggingFaceDataset(abc.ABC):
    """Abstract class for HuggingFace datasets."""

    def __init__(
        self,
        path: str,
        version: Optional[str] = None,
        type: Optional[str] = None,
        numerical_columns: Optional[list[str]] = None,
        categorical_columns: Optional[list[str]] = None,
        target: Optional[str] = None,
        resolution: Optional[int] = None,
    ) -> None:
        self.path = path
        self.version = version
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target = target
        self.type = type
        self.train_gdf = None
        self.test_gdf = None
        self.resolution = resolution

    @abc.abstractmethod
    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocess the dataset from HuggingFace.

        Args:
            data (pd.DataFrame): a dataset to preprocess
            version (str, optional): version of dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        raise NotImplementedError

    def load(
        self, version: Optional[Union[int, str]] = None, hf_token: Optional[str] = None
    ) -> dict[str, gpd.GeoDataFrame]:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str or int, optional): version of a dataset

        Returns:
            dict[str, gpd.GeoDataFrame]: Dictionary with all splits loaded from the dataset. Will
                 contain keys "train" and "test" if available.
        """
        result = {}
        dataset_name = self.path
        self.version = str(version)
        if self.resolution is None and version is not None:
            try:
                # Try to parse version as int (e.g. "8" or "9")
                self.resolution = int(version)
            except ValueError:
                pass
        data = load_dataset(dataset_name, str(version), token=hf_token, trust_remote_code=True)
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

    @abc.abstractmethod
    def get_h3_with_labels(
        self,
        resolution: Optional[int] = None,
        target_column: Optional[str] = None,
    ) -> tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
        """
        Returns indexes with target labels from the dataset depending on dataset and task type.

        Args:
            resolution (int): h3 resolution to regionalize data.
            train_gdf (gpd.GeoDataFrame): GeoDataFrame with training data.
            test_gdf (Optional[gpd.GeoDataFrame]): GeoDataFrame with testing data.
            target_column (Optional[str], optional): Target column name.Defaults to None.

        Returns:
            tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]: Train, Test indexes with target \
                labels in GeoDataFrames
        """
        raise NotImplementedError


class PointDataset(HuggingFaceDataset):
    """Abstract class for HuggingFace datasets with Point Data."""

    def __init__(
        self,
        path: str,
        version: Optional[str] = None,
        type: Optional[str] = None,
        numerical_columns: Optional[list[str]] = None,
        categorical_columns: Optional[list[str]] = None,
        target: Optional[str] = None,
        resolution: Optional[int] = None,
    ) -> None:
        self.path = path
        self.version = version
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target = target
        self.type = type
        self.train_gdf = None
        self.test_gdf = None
        self.dev_gdf = None
        self.resolution = resolution

    def train_test_split_bucket_regression(
        self,
        target_column: Optional[str] = None,
        resolution: int = 9,
        test_size: float = 0.2,
        bucket_number: int = 7,
        random_state: Optional[int] = None,
        dev: bool = False,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Method to generate train and test split from GeoDataFrame, based on the target_column values - its statistic.

        Args:
            target_column (Optional[str], optional): Target column name. If None, split generated on basis of number \
                of points within a hex ov given resolution. In this case values are normalized to [0,1] scale. \
                      Defaults to preset dataset target column.
            resolution (int, optional): h3 resolution to regionalize data. Defaults to 9.
            test_size (float, optional): Percentage of test set. Defaults to 0.2.
            bucket_number (int, optional): Bucket number used to stratify target data. Defaults to 7.
            random_state (int, optional):  Controls the shuffling applied to the data before applying the split. \
                Pass an int for reproducible output across multiple function. Defaults to None.
            dev (bool): If True, creates a dev split from existing train split and assigns it to self.dev_gdf.

        Returns:
            tuple(gpd.GeoDataFrame, gpd.GeoDataFrame): Train-test split made on previous train subset.
        """  # noqa: E501, W505
        resolution = resolution if resolution is not None else self.resolution

        if resolution is None:
            raise ValueError(
                "No preset resolution for the dataset in self.resolution. Please \
                             provide a resolution."
            )
        elif self.resolution is not None and resolution != self.resolution:
            raise ValueError(
                "Resolution provided is different from the preset resolution for the \
                             dataset. This may result in a data leak between splits."
            )

        target_column = target_column if target_column is not None else self.target
        if target_column is None:
            # target_column = self.target
            target_column = "count"

        if self.train_gdf is None:
            raise ValueError("Train GeoDataFrame is not loaded! Load the dataset first.")
        gdf = self.train_gdf
        gdf_ = gdf.copy()

        if target_column == "count":
            regionalizer = H3Regionalizer(resolution=resolution)
            regions = regionalizer.transform(gdf)

            joined_gdf = gpd.sjoin(gdf, regions, how="left", predicate="within")  # noqa: E501
            joined_gdf.rename(columns={"index_right": "h3_index"}, inplace=True)

            averages_hex = joined_gdf.groupby("h3_index").size().reset_index(name=target_column)
            gdf_ = regions.merge(
                averages_hex, how="inner", left_on="region_id", right_on="h3_index"
            )
            gdf_.rename(columns={"h3_index": "region_id"}, inplace=True)
            gdf_.index = gdf_["region_id"]

        splits = np.linspace(
            0, 1, num=bucket_number + 1
        )  # generate splits to bucket classification
        # quantiles = gdf_[target_column].quantile(splits)  # compute quantiles
        quantiles = gdf_[target_column].quantile(splits).drop_duplicates()
        bins = quantiles.values
        # bins = [quantiles[i] for i in splits]
        gdf_["bucket"] = pd.cut(gdf_[target_column], bins=bins, include_lowest=True).apply(
            lambda x: x.mid
        )  # noqa: E501

        train_indices, test_indices = train_test_split(
            range(len(gdf_)),
            test_size=test_size,
            stratify=gdf_.bucket,  # stratify by bucket value
            random_state=random_state,
        )

        # dev_indices, test_indices = train_test_split(
        #     range(len(test_indices)),
        #     test_size=0.5,
        #     stratify=gdf_.iloc[test_indices].bucket,
        # )
        train_gdf = gdf_.iloc[train_indices]
        test_gdf = gdf_.iloc[test_indices]
        if target_column == "count":
            train_hex_indexes = train_gdf["region_id"].unique()
            test_hex_indexes = test_gdf["region_id"].unique()
            train = joined_gdf[joined_gdf["h3_index"].isin(train_hex_indexes)]
            test = joined_gdf[joined_gdf["h3_index"].isin(test_hex_indexes)]
            train = train.drop(columns=["h3_index"])
            test = test.drop(columns=["h3_index"])

        if not dev:
            self.train_gdf = train if target_column == "count" else train_gdf
            self.test_gdf = test if target_column == "count" else test_gdf
            print(f"Created new train_gdf and test_gdf. Train len: {len(self.train_gdf)}, \
                test len: {len(self.test_gdf)}")
        else:
            self.train_gdf = train if target_column == "count" else train_gdf
            self.dev_gdf = test if target_column == "count" else test_gdf
            print(f"Created new train_gdf and dev_gdf. Test split remains unchanged. \
                   Train len: {len(self.train_gdf)}, dev len: {len(self.dev_gdf)}, \
                    test len: {len(self.test_gdf)}")

        # self.train_gdf = train
        # self.test_gdf = test
        self.resolution = resolution
        if not dev:
            return self.train_gdf, self.test_gdf
        else:
            return self.train_gdf, self.dev_gdf

    def train_test_split_spatial_points(
        self,
        test_size: float = 0.2,
        resolution: int = 8,  # TODO: dodać pole per dataset z h3_train_resolution
        resolution_subsampling: int = 1,
        random_state: Optional[int] = None,
        dev: bool = False,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Method to generate train and test split from GeoDataFrame, based on the spatial h3
        resolution.

        Args:
            test_size (float, optional): Percentage of test set.. Defaults to 0.2.
            resolution (int, optional): h3 resolution to regionalize data. Defaults to 8.
            resolution_subsampling (int, optional): h3 resolution difference to subsample \
                data for stratification. Defaults to 1.
            random_state (int, optional):  Controls the shuffling applied to the data before applying the split. \
                Pass an int for reproducible output across multiple function. Defaults to None.
            dev (bool): If True, creates a dev split from existing train split and assigns it to self.dev_gdf.

        Raises:
            ValueError: If type of data is not Points.

        Returns:
            tuple(gpd.GeoDataFrame, gpd.GeoDataFrame): Train-test split made on previous train subset.
        """  # noqa: W505, E501, D205
        if self.train_gdf is None:
            raise ValueError("Train GeoDataFrame is not loaded! Load the dataset first.")
        gdf = self.train_gdf
        gdf_ = gdf.copy()

        resolution = resolution if resolution is not None else self.resolution

        if resolution is None:
            raise ValueError(
                "No preset resolution for the dataset in self.resolution. Please \
                             provide a resolution."
            )
        elif self.resolution is not None and resolution != self.resolution:
            raise ValueError(
                "Resolution provided is different from the preset resolution for the \
                             dataset. This may result in a data leak between splits."
            )

        regionalizer = H3Regionalizer(resolution=resolution)
        regions = regionalizer.transform(gdf_)

        regions.index = regions.index.map(
            lambda idx: h3.cell_to_parent(idx, resolution - resolution_subsampling)
        )  # get parent h3 region
        regions["geometry"] = regions.index.map(
            lambda idx: Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(idx)])
        )  # get localization of h3 region

        joined_gdf = gpd.sjoin(gdf_, regions, how="left", predicate="within")
        joined_gdf.rename(columns={"index_right": "h3_index"}, inplace=True)
        joined_gdf.drop_duplicates(inplace=True)

        if joined_gdf["h3_index"].isnull().sum() != 0:  # handle outliers
            joined_gdf.loc[joined_gdf["h3_index"].isnull(), "h3_index"] = "fffffffffffffff"
        # set outlier index fffffffffffffff
        outlier_indices = joined_gdf["h3_index"].value_counts()
        outlier_indices = outlier_indices[
            outlier_indices <= 4
        ].index  # if only 4 points are in hex, they're outliers
        joined_gdf.loc[joined_gdf["h3_index"].isin(outlier_indices), "h3_index"] = "fffffffffffffff"

        train_indices, test_indices = train_test_split(
            range(len(joined_gdf)),
            test_size=test_size,  # * 2,  # multiply for dev set also
            stratify=joined_gdf.h3_index,  # stratify by spatial h3
            random_state=random_state,
        )

        # dev_indices, test_indices = train_test_split(
        #     range(len(test_indices)),
        #     test_size=0.5,
        #     stratify=joined_gdf.iloc[
        #         test_indices
        #     ].h3_index,  # perform spatial stratify (by h3 index)
        # )

        # return (
        #    gdf_.iloc[train_indices],
        #   gdf_.iloc[test_indices],
        # )
        train_gdf = gdf_.iloc[train_indices]
        test_gdf = gdf_.iloc[test_indices]
        # self.train_gdf = gdf_.iloc[train_indices]
        # self.test_gdf = gdf_.iloc[test_indices]
        self.resolution = resolution
        if not dev:
            self.train_gdf = train_gdf
            self.test_gdf = test_gdf
            print(f"Created new train_gdf and test_gdf. Train len: {len(self.train_gdf)}, \
                   test len: {len(self.test_gdf)}")
        else:
            self.train_gdf = train_gdf
            self.dev_gdf = test_gdf
            print(f"Created new train_gdf and dev_gdf. Test split remains unchanged. \
                   Train len: {len(self.train_gdf)}, dev len: {len(self.dev_gdf)}, \
                    test len: {len(self.test_gdf)}")

        if not dev:
            return self.train_gdf, self.test_gdf
        else:
            return self.train_gdf, self.dev_gdf
        # , gdf_.iloc[dev_indices],

    def get_h3_with_labels(
        self,
        resolution: Optional[int] = None,
        target_column: Optional[str] = None,
    ) -> tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
        """
        Returns h3 indexes with target labels from the dataset.

        Points are aggregated to hexes and target column values are averaged or if target column \
        is None, then the number of points is calculted within a hex and scaled to [0,1].

        Args:
            resolution (int): h3 resolution to regionalize data.
            train_gdf (gpd.GeoDataFrame): GeoDataFrame with training data.
            test_gdf (Optional[gpd.GeoDataFrame]): GeoDataFrame with testing data.
            target_column (Optional[str], optional): Target column name. If None, aggregates h3 \
                on basis of number of points within a hex of given resolution. In this case values \
                     are normalized to [0,1] scale. Defaults to None.

        Returns:
            tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]: Train, Test hexes with target \
                labels in GeoDataFrames
        """
        # if target_column is None:
        #     target_column = "count"

        resolution = resolution if resolution is not None else self.resolution

        assert self.train_gdf is not None
        # If resolution is still None, raise an error
        if resolution is None:
            raise ValueError(
                "No preset resolution for the dataset in self.resolution. Please \
                             provide a resolution."
            )
        elif self.resolution is not None and resolution != self.resolution:
            raise ValueError(
                "Resolution provided is different from the preset resolution for the \
                             dataset. This may result in a data leak between splits."
            )

        if target_column is None:
            target_column = getattr(self, "target", None) or "count"

        _train_gdf = self._aggregate_hexes(self.train_gdf, resolution, target_column)

        if self.test_gdf is not None:
            _test_gdf = self._aggregate_hexes(self.test_gdf, resolution, target_column)
        else:
            _test_gdf = None

        # Scale the "count" column to [0, 1] if it is the target column
        if target_column == "count":
            scaler = MinMaxScaler()
            # Fit the scaler on the train dataset and transform
            _train_gdf["count"] = scaler.fit_transform(_train_gdf[["count"]])
            if _test_gdf is not None:
                _test_gdf["count"] = scaler.transform(_test_gdf[["count"]])
                _test_gdf["count"] = np.clip(_test_gdf["count"], 0, 1)

        return _train_gdf, _test_gdf

    def _aggregate_hexes(
        self,
        gdf: gpd.GeoDataFrame,
        resolution: int,
        target_column: str,
    ) -> gpd.GeoDataFrame:
        """
        Aggregates points and calculates them or the mean of their target column within each hex.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame with data.
            resolution (int): h3 resolution to regionalize data.
            target_column (str): Target column name. If None, aggregates h3 on \
                basis of number of points within a hex ov given resolution. Defaults to None.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with aggregated data.
        """
        gdf_ = gdf.copy()
        regionalizer = H3Regionalizer(resolution=resolution)
        regions = regionalizer.transform(gdf)
        joined_gdf = gpd.sjoin(gdf, regions, how="left", predicate="within")  # noqa: E501
        joined_gdf.rename(columns={"index_right": "h3_index"}, inplace=True)
        if target_column == "count":
            aggregated = joined_gdf.groupby("h3_index").size().reset_index(name=target_column)
        else:
            aggregated = (
                joined_gdf.groupby("h3_index")[target_column].mean().reset_index(name=target_column)
            )

        gdf_ = regions.merge(aggregated, how="inner", left_on="region_id", right_on="h3_index")
        gdf_.rename(columns={"h3_index": "region_id"}, inplace=True)

        # gdf_.index = gdf_["region_id"]

        gdf_.drop(columns=["geometry"], inplace=True)
        return gdf_


class TrajectoryDataset(HuggingFaceDataset):
    """Abstract class for HuggingFace datasets with Trajectory data."""

    def __init__(
        self,
        path: str,
        version: Optional[str] = None,
        type: Optional[str] = None,
        numerical_columns: Optional[list[str]] = None,
        categorical_columns: Optional[list[str]] = None,
        target: Optional[str] = None,
        resolution: Optional[int] = None,
    ) -> None:
        self.path = path
        self.version = version
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target = target
        self.type = type
        self.train_gdf = None
        self.test_gdf = None
        self.resolution = resolution

    def train_test_split_bucket_trajectory(
        self,
        trajectory_id_column: str = "trip_id",
        task: Literal["TTE", "HMP"] = "TTE",
        test_size: float = 0.2,
        bucket_number: int = 4,
        dev: bool = False,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Generate train/test split from trajectory GeoDataFrame stratified by task.

        Split is generated by splitting train_gdf.

        Args:
            trajectory_id_column (str): Column identifying each trajectory.
            task (Literal["TTE", "HMP"]): Task type. Stratifies by duration
                (TTE) or hex length (HMP).
            test_size (float): Fraction of data to be used as test set.
            bucket_number (int): Number of stratification bins.
            dev (bool): If True, creates a dev split from existing train split and assigns \
                it to self.dev_gdf.


        Returns:
            Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Train and test GeoDataFrames.
        """
        assert self.train_gdf is not None
        trajectory_id_column = trajectory_id_column or self.target
        gdf_copy = self.train_gdf.copy()

        if task == "TTE":
            self.version = "TTE"
            # Calculate duration in seconds from timestamps list
            # gdf_copy["stratify_col"] = gdf_copy["timestamp"].apply(
            #     lambda ts: 0.0 if len(ts) < 2 else (ts[-1] - ts[0]).total_seconds()
            # )
            if "duration" in gdf_copy.columns:
                gdf_copy["stratify_col"] = gdf_copy["duration"]
            else:
                raise ValueError("Duration column does not exist. Can't stratify it.")

        elif task == "HMP":
            self.version = "HMP"
            # Calculate trajectory length in unique hexagons
            gdf_copy["x_len"] = gdf_copy["h3_sequence_x"].apply(lambda seq: len(set(seq)))
            gdf_copy["y_len"] = gdf_copy["h3_sequence_y"].apply(lambda seq: len(set(seq)))
            gdf_copy["stratify_col"] = gdf_copy.apply(
                lambda row: row["x_len"] + row["y_len"], axis=1
            )
        else:
            raise ValueError(f"Unsupported task type: {task}")

        gdf_copy["stratification_bin"] = pd.cut(
            gdf_copy["stratify_col"], bins=bucket_number, labels=False
        )

        trajectory_indices = gdf_copy[trajectory_id_column].unique()
        duration_bins = (
            gdf_copy[[trajectory_id_column, "stratification_bin"]]
            .drop_duplicates()
            .set_index(trajectory_id_column)["stratification_bin"]
        )

        train_indices, test_indices = train_test_split(
            trajectory_indices,
            test_size=test_size,
            stratify=duration_bins.loc[trajectory_indices],
        )

        train_gdf = gdf_copy[gdf_copy[trajectory_id_column].isin(train_indices)]
        test_gdf = gdf_copy[gdf_copy[trajectory_id_column].isin(test_indices)]

        test_gdf = test_gdf.drop(
            columns=[
                col
                for col in ["x_len", "y_len", "stratification_bin", "stratify_col"]
                if col in test_gdf.columns
            ],
        )
        train_gdf = train_gdf.drop(
            columns=[
                col
                for col in ["x_len", "y_len", "stratification_bin", "stratify_col"]
                if col in test_gdf.columns
            ],
        )

        if not dev:
            self.train_gdf = train_gdf
            self.test_gdf = test_gdf
            print(f"Created new train_gdf and test_gdf. Train len: {len(self.train_gdf)}, \
                   test len: {len(self.test_gdf)}")
        else:
            self.train_gdf = train_gdf
            self.dev_gdf = test_gdf
            print(f"Created new train_gdf and dev_gdf. Test split remains unchanged. \
                   Train len: {len(self.train_gdf)}, dev len: {len(self.dev_gdf)}, \
                    test len: {len(self.test_gdf)}")
        return train_gdf, test_gdf

    def get_h3_with_labels(
        self,
        resolution: Optional[int] = None,
        target_column: Optional[str] = None,
    ) -> tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
        """
        Returns ids, h3 indexes sequences, with target labels from the dataset.

        Points are aggregated to hex trajectories and target column values are calculated \
            for each trajectory (time duration for TTE task, future movement sequence for HMP task).

        Args:
            resolution (int): h3 resolution to regionalize data.
            train_gdf (gpd.GeoDataFrame): GeoDataFrame with training data.
            test_gdf (Optional[gpd.GeoDataFrame]): GeoDataFrame with testing data.
            target_column (Optional[str], optional): Target column name. In trajectories it is\
                 usually an id of trajectory/trip.

        Returns:
            tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]: Train, Test hexes sequences with \
                target labels in GeoDataFrames
        """
        # if target_column is None:
        #     target_column = "count"

        resolution = resolution if resolution is not None else self.resolution

        assert self.train_gdf is not None
        # If resolution is still None, raise an error
        if resolution is None:
            raise ValueError(
                "No preset resolution for the dataset in self.resolution. Please \
                             provide a resolution."
            )
        elif self.resolution is not None and resolution != self.resolution:
            raise ValueError(
                "Resolution provided is different from the preset resolution for the \
                             dataset. This may result in a data leak between splits."
            )

        if self.version == "TTE":
            _train_gdf = self.train_gdf[[self.target, "h3_sequence", "duration"]]

            if self.test_gdf is not None:
                _test_gdf = self.test_gdf[[self.target, "h3_sequence", "duration"]]
            else:
                _test_gdf = None
        elif self.version == "HMP":
            _train_gdf = self.train_gdf[[self.target, "h3_sequence_x", "h3_sequence_y"]]

            if self.test_gdf is not None:
                _test_gdf = self.test_gdf[[self.target, "h3_sequence_x", "h3_sequence_y"]]
            else:
                _test_gdf = None
        elif self.version == "all":
            raise TypeError(
                "Could not provide target labels, as version 'all'\
            of dataset does not provide one."
            )

        return _train_gdf, _test_gdf

    def _agg_points_to_trajectories(
        self, gdf: gpd.GeoDataFrame, target_column: str
    ) -> gpd.GeoDataFrame:
        """
        Preprocess the dataset from HuggingFace to trajectories.

        Args:
            gdf (pd.DataFrame): a dataset to preprocess
            target_column (str): a column to aggregate trajectories (trip_id)

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        _gdf = gdf.copy()
        tqdm.pandas(desc="Building linestring trajectories")

        _gdf = gdf.sort_values(by=[target_column, "timestamp"]).copy()
        geometry_col = _gdf.geometry.name

        # Group and aggregate all columns as lists

        aggregated = _gdf.groupby(target_column).agg(lambda x: x.tolist())
        aggregated = aggregated[aggregated[geometry_col].apply(lambda x: len(x) > 1)]
        aggregated[geometry_col] = aggregated[geometry_col].progress_apply(LineString)

        traj_gdf = gpd.GeoDataFrame(aggregated.reset_index(), geometry=geometry_col, crs=_gdf.crs)

        return traj_gdf

    @abc.abstractmethod
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
        raise NotImplementedError
