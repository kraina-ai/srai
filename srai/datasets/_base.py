"""Base classes for Datasets."""

import abc
from typing import Optional

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from datasets import load_dataset
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

    def load(self, hf_token: Optional[str] = None, version: Optional[str] = None) -> None:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str, optional): version of a dataset

        Returns:
            None
        """
        dataset_name = self.path
        version = version or self.version
        if version is not None and len(version) == 1:
            self.resolution = int(version)
        data = load_dataset(dataset_name, version, token=hf_token, trust_remote_code=True)
        train = data["train"].to_pandas()
        processed_train = self._preprocessing(train)
        self.train_gdf = processed_train
        if "test" in data:
            test = data["test"].to_pandas()
            processed_test = self._preprocessing(test)
            self.test_gdf = processed_test

    def train_test_split_bucket_regression(
        self,
        target_column: Optional[str] = None,
        resolution: int = 9,
        test_size: float = 0.2,
        bucket_number: int = 7,
        random_state: Optional[int] = None,
    ) -> None:
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

        Returns:
            None
        """  # noqa: E501, W505
        if self.type != "point":
            raise ValueError("This split can be performed only on point data type!")

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
        quantiles = gdf_[target_column].quantile(splits)  # compute quantiles
        bins = [quantiles[i] for i in splits]
        gdf_["bucket"] = pd.cut(gdf_[target_column], bins=bins, include_lowest=True).apply(
            lambda x: x.mid
        )  # noqa: E501

        train_indices, test_indices = train_test_split(
            range(len(gdf_)),
            test_size=test_size,  # * 2 multiply for dev set also
            stratify=gdf_.bucket,  # stratify by bucket value
            random_state=random_state,
        )

        # dev_indices, test_indices = train_test_split(
        #     range(len(test_indices)),
        #     test_size=0.5,
        #     stratify=gdf_.iloc[test_indices].bucket,
        # )
        train = gdf_.iloc[train_indices]
        test = gdf_.iloc[test_indices]
        if target_column == "count":
            train_hex_indexes = train["region_id"].unique()
            test_hex_indexes = test["region_id"].unique()
            train = joined_gdf[joined_gdf["h3_index"].isin(train_hex_indexes)]
            test = joined_gdf[joined_gdf["h3_index"].isin(test_hex_indexes)]
            train = train.drop(columns=["h3_index"])
            test = test.drop(columns=["h3_index"])

        # return train, test  # , gdf_.iloc[dev_indices]
        self.train_gdf = train
        self.test_gdf = test
        self.resolution = resolution

    def train_test_split_spatial_points(
        self,
        test_size: float = 0.2,
        resolution: int = 8,  # TODO: dodać pole per dataset z h3_train_resolution
        resolution_subsampling: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
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

        Raises:
            ValueError: If type of data is not Points.

        Returns:
            None
        """  # noqa: W505, E501, D205
        if self.type != "point":
            raise ValueError("This split can be performed only on Points data type!")

        if self.train_gdf is None:
            raise ValueError("Train GeoDataFrame is not loaded! Load the dataset first.")
        gdf = self.train_gdf
        gdf_ = gdf.copy()

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
        self.train_gdf = gdf_.iloc[train_indices]
        self.test_gdf = gdf_.iloc[test_indices]
        self.resolution = resolution
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
            # Calculate mean of the target column within each hex
            aggregated = (
                joined_gdf.groupby("h3_index")[target_column].mean().reset_index(name=target_column)
            )

        gdf_ = regions.merge(aggregated, how="inner", left_on="region_id", right_on="h3_index")
        gdf_.rename(columns={"h3_index": "region_id"}, inplace=True)
        # gdf_.index = gdf_["region_id"]

        gdf_.drop(columns=["geometry"], inplace=True)
        return gdf_
