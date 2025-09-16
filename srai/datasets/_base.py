"""Base classes for Datasets."""

import abc
import operator
from contextlib import suppress
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import platformdirs
from shapely.geometry import LineString
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from srai._optional import import_optional_dependencies
from srai.constants import REGIONS_INDEX
from srai.regionalizers import H3Regionalizer
from srai.spatial_split import train_test_spatial_split


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
        self.val_gdf = None
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

    @abc.abstractmethod
    def train_test_split(
        self,
        target_column: Optional[str] = None,
        resolution: Optional[int] = None,
        test_size: float = 0.2,
        n_bins: int = 7,
        random_state: Optional[int] = None,
        validation_split: bool = False,
        force_split: bool = False,
        task: Optional[str] = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Method to generate train/test or train/val split from GeoDataFrame.

        Args:
            target_column (Optional[str], optional): Target column name for Points, trajectories id\
                column fortrajectory datasets. Defaults to preset dataset target column.
            resolution (int, optional): H3 resolution, subclasses mayb use this argument to\
                regionalize data. Defaults to default value from the dataset.
            test_size (float, optional): Percentage of test set. Defaults to 0.2.
            n_bins (int, optional): Bucket number used to stratify target data.
            random_state (int, optional):  Controls the shuffling applied to the data before \
                applying the split.
                Pass an int for reproducible output across multiple function. Defaults to None.
            validation_split (bool): If True, creates a validation split from existing train split\
                and assigns it to self.val_gdf.
            force_split: If True, forces a new split to be created, even if an existing train/test\
                or validation split is already present.
                - With `validation_split=False`, regenerates and overwrites the test split.
                - With `validation_split=True`, regenerates and overwrites the validation split.
            task (Optional[str], optional): Task identifier. Subclasses may use this argument to
                determine stratification logic (e.g., by duration or spatial pattern).\
                    Defaults to None.

        Returns:
            tuple(gpd.GeoDataFrame, gpd.GeoDataFrame): Train-test or Train-val split made on\
                previous train subset.
        """
        raise NotImplementedError

    def load(
        self,
        version: Optional[Union[int, str]] = None,
        hf_token: Optional[str] = None,
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
        from datasets import load_dataset

        result = {}

        self.train_gdf, self.val_gdf, self.test_gdf = None, None, None
        dataset_name = self.path
        self.version = str(version)

        if (
            self.resolution is None
            and self.version in ("8", "9", "10")
            or (self.version in ("8", "9", "10") and str(self.resolution) != self.version)
        ):
            with suppress(ValueError):
                # Try to parse version as int (e.g. "8" or "9")
                self.resolution = int(self.version)

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
        # resolution: Optional[int] = None,
        # target_column: Optional[str] = None,
    ) -> tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
        """
        Returns indexes with target labels from the dataset depending on dataset and task type.

        Returns:
            tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]: \
                Train, Val, Test indexes with target labels in GeoDataFrames
        """
        raise NotImplementedError

    def _get_global_dataset_cache_path(self) -> Path:
        """
        Get the root cache directory for the dataset.

        Returns:
            Path: Path object pointing to the user-specific cache directory where
                the dataset should be stored.
        """
        return Path(platformdirs.user_cache_dir("srai")) / "datasets" / self.__class__.__name__


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
        import_optional_dependencies(dependency_group="datasets", modules=["datasets"])
        self.path = path
        self.version = version
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target = target
        self.type = type
        self.train_gdf = None
        self.test_gdf = None
        self.val_gdf = None
        self.resolution = resolution

    def train_test_split(
        self,
        target_column: Optional[str] = None,
        resolution: Optional[int] = None,
        test_size: float = 0.2,
        n_bins: int = 7,
        random_state: Optional[int] = None,
        validation_split: bool = False,
        force_split: bool = False,
        task: Optional[str] = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Method to generate splits from GeoDataFrame, based on the target_column values.

        Args:
            target_column (Optional[str], optional): Target column name. If None, split is\
                generated based on number of points within a hex of a given resolution.\
                Defaults to preset dataset target column.
            resolution (int, optional): h3 resolution to regionalize data. Defaults to default\
                value from the dataset.
            test_size (float, optional): Percentage of test set. Defaults to 0.2.
            n_bins (int, optional): Bucket number used to stratify target data.\
                Defaults to 7.
            random_state (int, optional):  Controls the shuffling applied to the data before\
                applying the split. \
                Pass an int for reproducible output across multiple function. Defaults to None.
            validation_split (bool): If True, creates a validation split from existing train split\
                and assigns it to self.val_gdf.
            force_split: If True, forces a new split to be created, even if an existing train/test\
                or validation split is already present.
                - With `validation_split=False`, regenerates and overwrites the test split.
                - With `validation_split=True`, regenerates and overwrites the validation split.
            task (Optional[str], optional): Currently not supported. Ignored in this subclass.

        Returns:
            tuple(gpd.GeoDataFrame, gpd.GeoDataFrame): Train-test or train-val split made on\
                previous train subset.
        """
        assert self.train_gdf is not None

        if (self.val_gdf is not None and validation_split and not force_split) or (
            self.test_gdf is not None and not validation_split and not force_split
        ):
            raise ValueError(
                "A split already exists. Use `force_split=True` to overwrite the existing "
                f"{'validation' if validation_split else 'test'} split."
            )

        resolution = resolution or self.resolution

        if resolution is None:
            raise ValueError(
                "No preset resolution for the dataset in self.resolution. Please "
                "provide a resolution."
            )
        elif self.resolution is not None and resolution != self.resolution:
            raise ValueError(
                "Resolution provided is different from the preset resolution for the "
                "dataset. This may result in a data leak between splits."
            )

        if self.resolution is None:
            self.resolution = resolution
        target_column = target_column if target_column is not None else self.target
        if target_column is None:
            target_column = "count"

        gdf = self.train_gdf
        gdf_ = gdf.copy()

        train, test = train_test_spatial_split(
            gdf_,
            parent_h3_resolution=resolution,
            target_column=target_column,
            test_size=test_size,
            n_bins=n_bins,
            random_state=random_state,
        )

        self.train_gdf = train
        if not validation_split:
            self.test_gdf = test
            test_len = len(self.test_gdf) if self.test_gdf is not None else 0
            print(
                f"Created new train_gdf and test_gdf. Train len: {len(self.train_gdf)},"
                f"test len: {test_len}"
            )
        else:
            self.val_gdf = test
            val_len = len(self.val_gdf) if self.val_gdf is not None else 0
            test_len = len(self.test_gdf) if self.test_gdf is not None else 0
            print(
                f"Created new train_gdf and val_gdf. Test split remains unchanged."
                f"Train len: {len(self.train_gdf)}, val len: {val_len},"
                f"test len: {test_len}"
            )
        return train, test

    def get_h3_with_labels(
        self,
        # resolution: Optional[int] = None,
        # target_column: Optional[str] = None,
    ) -> tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
        """
        Returns h3 indexes with target labels from the dataset.

        Points are aggregated to hexes and target column values are averaged or if target column \
        is None, then the number of points is calculted within a hex and scaled to [0,1].

        Returns:
            tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:\
                Train, Val, Test hexes with target labels in GeoDataFrames
        """
        # if target_column is None:
        #     target_column = "count"

        # resolution = resolution if resolution is not None else self.resolution

        assert self.train_gdf is not None
        # If resolution is still None, raise an error
        if self.resolution is None:
            raise ValueError(
                "No preset resolution for the dataset in self.resolution. Please"
                "provide a resolution."
            )
        # elif self.resolution is not None and resolution != self.resolution:
        #     raise ValueError(
        #         "Resolution provided is different from the preset resolution for the"
        #         "dataset. This may result in a data leak between splits."
        #     )

        _train_gdf = self._aggregate_hexes(self.train_gdf, self.resolution, self.target)

        if self.test_gdf is not None:
            _test_gdf = self._aggregate_hexes(self.test_gdf, self.resolution, self.target)
        else:
            _test_gdf = None

        if self.val_gdf is not None:
            _val_gdf = self._aggregate_hexes(self.val_gdf, self.resolution, self.target)
        else:
            _val_gdf = None

        # Scale the "count" column to [0, 1] if it is the target column
        if self.target == "count":
            scaler = MinMaxScaler()
            # Fit the scaler on the train dataset and transform
            _train_gdf["count"] = scaler.fit_transform(_train_gdf[["count"]])
            if _test_gdf is not None:
                _test_gdf["count"] = scaler.transform(_test_gdf[["count"]])
                _test_gdf["count"] = np.clip(_test_gdf["count"], 0, 1)
            if _val_gdf is not None:
                _val_gdf["count"] = scaler.transform(_val_gdf[["count"]])
                _val_gdf["count"] = np.clip(_val_gdf["count"], 0, 1)

        return _train_gdf, _val_gdf, _test_gdf

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

        if target_column == "count":
            aggregated = joined_gdf.groupby(REGIONS_INDEX).size().reset_index(name=target_column)
        else:
            aggregated = (
                joined_gdf.groupby(REGIONS_INDEX)[target_column]
                .mean()
                .reset_index(name=target_column)
            )

        gdf_ = regions.merge(
            aggregated,
            how="inner",
            left_on=REGIONS_INDEX,
            right_on=REGIONS_INDEX,
        )

        # gdf_.index = gdf_["region_id"]

        # gdf_.drop(columns=[GEOMETRY_COLUMN], inplace=True) # disabling dropping geometry
        return gdf_.set_index(REGIONS_INDEX)


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
        import_optional_dependencies(dependency_group="datasets", modules=["datasets"])
        self.path = path
        self.version = version
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target = target
        self.type = type
        self.train_gdf = None
        self.test_gdf = None
        self.val_gdf = None
        self.resolution = resolution

    def train_test_split(
        self,
        target_column: Optional[str] = None,
        resolution: Optional[int] = None,
        test_size: float = 0.2,
        n_bins: int = 4,
        random_state: Optional[int] = None,
        validation_split: bool = False,
        force_split: bool = False,
        task: Optional[str] = "TTE",
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Generate train/test split or train/val split from trajectory GeoDataFrame.

        Train-test/train-val split is generated by splitting train_gdf.

        Args:
            target_column (str): Column identifying each trajectory (contains trajectory ids).
            test_size (float): Fraction of data to be used as test set.
            n_bins (int): Number of stratification bins.
            random_state (int, optional):  Controls the shuffling applied to the data before\
                applying the split. Pass an int for reproducible output across multiple function.\
                    Defaults to None.
            validation_split (bool): If True, creates a validation split from existing train split\
                and assigns it to self.val_gdf.
            force_split: If True, forces a new split to be created, even if an existing train/test\
                or validation split is already present.
                - With `validation_split=False`, regenerates and overwrites the test split.
                - With `validation_split=True`, regenerates and overwrites the validation split.
            resolution (int, optional): H3 resolution to regionalize data. Currently ignored in\
                this subclass, different resolutions splits not supported yet.\
                    Defaults to default value from the dataset.
            task (Literal["TTE", "HMP"]): Task type. Stratifies by duration
                (TTE) or hex length (HMP).


        Returns:
            Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Train/test or train/val GeoDataFrames.
        """
        if (self.val_gdf is not None and validation_split and not force_split) or (
            self.test_gdf is not None and not validation_split and not force_split
        ):
            raise ValueError(
                "A split already exists. Use `force_split=True` to overwrite the existing "
                f"{'validation' if validation_split else 'test'} split."
            )
        assert self.train_gdf is not None
        trajectory_id_column = target_column or self.target
        gdf_copy = self.train_gdf.copy()

        if task not in {"TTE", "HMP"}:
            raise ValueError(f"Unsupported task: {task}")

        if task == "TTE":
            self.version = "TTE"
            # Calculate duration in seconds from timestamps list

            if "duration" in gdf_copy.columns:
                gdf_copy["stratify_col"] = gdf_copy["duration"]
            elif "duration" not in gdf_copy.columns and "timestamp" in gdf_copy.columns:
                gdf_copy["stratify_col"] = gdf_copy["timestamp"].apply(
                    #     lambda ts: (0.0 if len(ts) < 2 else (ts[-1] - ts[0]).total_seconds())
                    # )
                    lambda ts: (
                        0.0 if len(ts) < 2 else pd.Timedelta(ts[-1] - ts[0]).total_seconds()
                    )
                )
            else:
                raise ValueError(
                    "Duration column and timestamp column does not exist.\
                                  Can't stratify it."
                )

        elif task == "HMP":
            self.version = "HMP"

            def split_sequence(seq):
                split_idx = int(len(seq) * 0.85)
                if split_idx == len(seq):
                    split_idx = len(seq) - 1
                return seq[:split_idx], seq[split_idx:]

            if "h3_sequence_x" not in gdf_copy.columns:
                split_result = gdf_copy["h3_sequence"].apply(split_sequence)
                gdf_copy["h3_sequence_x"] = split_result.apply(operator.itemgetter(0))
                gdf_copy["h3_sequence_y"] = split_result.apply(operator.itemgetter(1))

            # Calculate trajectory length in unique hexagons
            gdf_copy["x_len"] = gdf_copy["h3_sequence_x"].apply(lambda seq: len(set(seq)))
            gdf_copy["y_len"] = gdf_copy["h3_sequence_y"].apply(lambda seq: len(set(seq)))
            gdf_copy["stratify_col"] = gdf_copy.apply(
                lambda row: row["x_len"] + row["y_len"], axis=1
            )
        else:
            raise ValueError(f"Unsupported task type: {task}")

        gdf_copy["stratification_bin"] = pd.cut(gdf_copy["stratify_col"], bins=n_bins, labels=False)

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
            random_state=random_state,
        )

        train_gdf = gdf_copy[gdf_copy[trajectory_id_column].isin(train_indices)]
        test_gdf = gdf_copy[gdf_copy[trajectory_id_column].isin(test_indices)]

        test_gdf = test_gdf.drop(
            columns=[
                col
                for col in (
                    "x_len",
                    "y_len",
                    "stratification_bin",
                    "stratify_col",
                )
                if col in test_gdf.columns
            ],
        )
        train_gdf = train_gdf.drop(
            columns=[
                col
                for col in (
                    "x_len",
                    "y_len",
                    "stratification_bin",
                    "stratify_col",
                )
                if col in test_gdf.columns
            ],
        )

        self.train_gdf = train_gdf
        if not validation_split:
            self.test_gdf = test_gdf
            test_len = len(self.test_gdf) if self.test_gdf is not None else 0
            print(
                f"Created new train_gdf and test_gdf. Train len: {len(self.train_gdf)}, "
                f"test len: {test_len}"
            )
        else:
            self.val_gdf = test_gdf
            val_len = len(self.val_gdf) if self.val_gdf is not None else 0
            test_len = len(self.test_gdf) if self.test_gdf is not None else 0
            print(
                f"Created new train_gdf and val_gdf. Test split remains unchanged. "
                f"Train len: {len(self.train_gdf)}, val len: {val_len}, "
                f"test len: {test_len}"
            )
        return train_gdf, test_gdf

    def get_h3_with_labels(
        self,
        # resolution: Optional[int] = None,
        # target_column: Optional[str] = None,
    ) -> tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
        """
        Returns ids, h3 indexes sequences, with target labels from the dataset.

        Points are aggregated to hex trajectories and target column values are calculated \
            for each trajectory (time duration for TTE task, future movement sequence for HMP task).

        Returns:
            tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]: Train,\
                Val, Test hexes sequences with target labels in GeoDataFrames
        """
        # resolution = resolution if resolution is not None else self.resolution

        assert self.train_gdf is not None
        # If resolution is still None, raise an error
        if self.resolution is None:
            raise ValueError(
                "No preset resolution for the dataset in self.resolution. Please"
                "provide a resolution."
            )
        # elif self.resolution is not None and resolution != self.resolution:
        #     raise ValueError(
        #         "Resolution provided is different from the preset resolution for the"
        #         "dataset. This may result in a data leak between splits."
        #     )

        if self.version == "TTE":
            _train_gdf = self.train_gdf[[self.target, "h3_sequence", "duration"]]

            if self.test_gdf is not None:
                _test_gdf = self.test_gdf[[self.target, "h3_sequence", "duration"]]
            else:
                _test_gdf = None

            if self.val_gdf is not None:
                _val_gdf = self.val_gdf[[self.target, "h3_sequence", "duration"]]
            else:
                _val_gdf = None

        elif self.version == "HMP":
            _train_gdf = self.train_gdf[[self.target, "h3_sequence_x", "h3_sequence_y"]]

            if self.test_gdf is not None:
                _test_gdf = self.test_gdf[[self.target, "h3_sequence_x", "h3_sequence_y"]]
            else:
                _test_gdf = None

            if self.val_gdf is not None:
                _val_gdf = self.val_gdf[[self.target, "h3_sequence_x", "h3_sequence_y"]]
            else:
                _val_gdf = None

        elif self.version == "all":
            raise TypeError(
                "Could not provide target labels, as version 'all'\
            of dataset does not provide one."
            )

        return _train_gdf, _val_gdf, _test_gdf

    def _agg_points_to_trajectories(
        self, gdf: gpd.GeoDataFrame, target_column: str, progress_bar: bool = True
    ) -> gpd.GeoDataFrame:
        """
        Preprocess the dataset from HuggingFace to trajectories.

        Args:
            gdf (pd.DataFrame): a dataset to preprocess
            target_column (str): a column to aggregate trajectories (trip_id)
            progress_bar (bool, optional): whether to show tqdm progress bar or not

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        _gdf = gdf.copy()
        if progress_bar:
            tqdm.pandas(desc="Building linestring trajectories")

        _gdf = gdf.sort_values(by=[target_column, "timestamp"]).copy()
        geometry_col = _gdf.geometry.name

        # Group and aggregate all columns as lists

        aggregated = _gdf.groupby(target_column).agg(lambda x: x.tolist())
        aggregated = aggregated[aggregated[geometry_col].apply(lambda x: len(x) > 1)]
        if progress_bar:
            aggregated[geometry_col] = aggregated[geometry_col].progress_apply(LineString)
        else:
            aggregated[geometry_col] = aggregated[geometry_col].apply(LineString)

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
