"""Base classes for Datasets."""

import abc
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from srai.loaders import HuggingFaceLoader


class HuggingFaceDataset(abc.ABC):
    """Abstract class for HuggingFace datasets."""

    def __init__(
        self,
        path: str,
        version: Optional[str] = None,
        numerical_columns: Optional[list[str]] = None,
        categorical_columns: Optional[list[str]] = None,
        target: Optional[str] = None,
    ) -> None:
        self.path = path
        self.version = version
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target = target

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

    def train_dev_test_split_bucket(
        self,
        gdf: gpd.GeoDataFrame,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        bucket_number: int = 7,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Method to generate train, dev and test split from GeoDataFrame, based on the target_column values - its statistic.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame on which train, dev, test split will be performed.
            target_column (Optional[str], optional): Target column name. Defaults to "price".
            test_size (float, optional): Percentage of test set. Defaults to 0.2.
            bucket_number (int, optional): Bucket number used to stratify target data. Defaults to 7.

        Returns:
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]: Train, Dev, Test splits in GeoDataFrames
        """  # noqa: E501, W505
        if target_column is None:
            target_column = self.target
        gdf_ = gdf.copy()
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
            test_size=test_size * 2,  # multiply for dev set also
            stratify=gdf_.bucket,  # stratify by bucket value
        )

        dev_indices, test_indices = train_test_split(
            range(len(test_indices)), test_size=0.5, stratify=gdf_.iloc[test_indices].bucket
        )

        return gdf_.iloc[train_indices], gdf_.iloc[dev_indices], gdf_.iloc[test_indices]
