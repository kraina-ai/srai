"""Base classes for Datasets."""

import abc
from typing import Optional

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split

from srai.loaders import HuggingFaceLoader
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
    ) -> None:
        self.path = path
        self.version = version
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target = target
        self.type = type

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
        self, hf_token: Optional[str] = None, version: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str, optional): version of a dataset

        Returns:
            gpd.GeoDataFrame: Loaded data.
        """
        dataset_name = self.path
        version = version or self.version
        data = HuggingFaceLoader(hf_token=hf_token).load(dataset_name=dataset_name, name=version)
        processed_data = self._preprocessing(data)

        return processed_data

    def train_test_split_bucket_regression(
        self,
        gdf: gpd.GeoDataFrame,
        target_column: Optional[str] = None,
        resolution: int = 9,
        test_size: float = 0.2,
        bucket_number: int = 7,
        random_state: Optional[int] = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Method to generate train and test split from GeoDataFrame, based on the target_column values - its statistic.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame on which train, dev, test split will be performed.
            target_column (Optional[str], optional): Target column name. If None, split generated on basis of number \
                of points within a hex ov given resolution.
            resolution (int, optional): h3 resolution to regionalize data. Defaults to 9.
            test_size (float, optional): Percentage of test set. Defaults to 0.2.
            bucket_number (int, optional): Bucket number used to stratify target data. Defaults to 7.
            random_state (int, optional):  Controls the shuffling applied to the data before applying the split. \
                Pass an int for reproducible output across multiple function. Defaults to None.

        Returns:
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Train, Test splits in GeoDataFrames
        """  # noqa: E501, W505
        if self.type != "point":
            raise ValueError("This split can be performed only on point data type!")
        if target_column is None:
            # target_column = self.target
            target_column = "count"

        gdf_ = gdf.copy()
        splits = np.linspace(
            0, 1, num=bucket_number + 1
        )  # generate splits to bucket classification
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

        return train, test  # , gdf_.iloc[dev_indices]

    def train_test_split_spatial_points(
        self,
        gdf: gpd.GeoDataFrame,
        test_size: float = 0.2,
        resolution: int = 8,  # TODO: dodać pole per dataset z h3_train_resolution
        resolution_subsampling: int = 1,
        random_state: Optional[int] = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Method to generate train and test split from GeoDataFrame, based on the spatial h3
        resolution.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame on which train, dev, test split will be performed.
            test_size (float, optional): Percentage of test set.. Defaults to 0.2.
            resolution (int, optional): h3 resolution to regionalize data. Defaults to 8.
            resolution_subsampling (int, optional): h3 resolution difference to subsample \
                data for stratification. Defaults to 1.
            random_state (int, optional):  Controls the shuffling applied to the data before applying the split. \
                Pass an int for reproducible output across multiple function. Defaults to None.

        Raises:
            ValueError: If type of data is not Points.

        Returns:
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Train, Test splits in GeoDataFrames
        """  # noqa: W505, E501, D205
        if self.type != "point":
            raise ValueError("This split can be performed only on Points data type!")
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

        return gdf_.iloc[train_indices], gdf_.iloc[test_indices]  # , gdf_.iloc[dev_indices],
