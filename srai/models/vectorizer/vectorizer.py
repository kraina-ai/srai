"""
Vectorizer module.

This module contains implementation of vectorizer. Various embeddings of geolocation can be selected
"""

import warnings
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.preprocessing import StandardScaler

from srai.datasets._base import HuggingFaceDataset
from srai.embedders import GeoVexEmbedder, Hex2VecEmbedder  # noqa: F401
from srai.joiners import IntersectionJoiner
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from srai.regionalizers import H3Regionalizer

# TODO: zdefiniowane w klasie jako metadane, żeby w metadanych datastu
# TODO: testy napisać
# TODO: można wczytać plik z load z pliku
# TODO: add numerical columns to self in dataset base class. maybe categorical columns ?


class Vectorizer:
    """
    Vectorizer module.

    H3 regionalizer is used.
    Should return datasets.Dataset() generated from DF with vectors -> \
        (geo embedding + additional features) and target labels

    Hence, needed:
     - resolution
     - embedder type
     - not geo numerical column names -> to standarfize
     - not geo non-numerical column names -> to encoding
    """

    def __init__(
        self,
        gdf_dataset: gpd.GeoDataFrame,
        HF_dataset_object: Optional[HuggingFaceDataset] = None,
        target_column_name: str = "price",
        numerical_columns: Optional[list[str]] = None,
        categorical_columns: Optional[list[str]] = None,
        embedder_type: str = "Hex2VecEmbedder",
        h3_resolution: int = 8,
        embedder_hidden_sizes: Optional[list[int]] = None,
    ):
        """
        Initialization of Vectorizer.

        Args:
            gdf_dataset (gpd.GeoDataFrame): GeoDataFrame which should be prepared to training.
            HF_dataset_object (Optional[HuggingFaceDataset]): Dataset object from which gdf_dataset was loaded.
            target_column_name (str, optional): Target column in regression. Defaults to "price".
            numerical_columns (Optional[list[str]], optional): Columns from gdf_dataset with \
                numerical values that will be used to train model. If default, columns are inherited from HuggingFaceDataset.
            categorical_columns (Optional[list[str]], optional): Columns from gdf_dataset with \
                categorical values that will be used to train model. If default, columns are inherited from HuggingFaceDataset.
            embedder_type (str, optional): Type of embedder. Available embedders: Hex2VecEmbedder,\
                  GeoVecEmbedder. Defaults to "Hex2VecEmbedder".
            h3_resolution (int, optional): Resolution in h3 regionalizer. Defaults to 8.
            embedder_hidden_sizes (Optional[list[int]], optional): Hidden sizes of embedder. \
                Defaults to None.

        Raises:
            NotImplementedError: GeoVecEmbedder usage is not implemented.
            ValueError: If embedder type is not available.
        """  # noqa: W505, E501
        if HF_dataset_object is None:
            raise ValueError("Dataset object not passed!")
        if embedder_hidden_sizes is None:
            self.embedder_hidden_sizes = [150, 75, 50]
        else:
            self.embedder_hidden_sizes = embedder_hidden_sizes
        if numerical_columns is None:
            numerical_columns = HF_dataset_object.numerical_columns
        if categorical_columns is None:
            categorical_columns = HF_dataset_object.numerical_columns

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.gdf = gdf_dataset
        self.target_column_name = target_column_name

        self.regions = H3Regionalizer(resolution=h3_resolution).transform(self.gdf)
        self.osm_features = OSMPbfLoader().load(self.regions, HEX2VEC_FILTER)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if embedder_type == "Hex2VecEmbedder":
            self.embedder = Hex2VecEmbedder(
                embedder_hidden_sizes
            )  # TODO: można wczytać plik z load z pliku
        elif embedder_type == "GeoVexEmbedder":
            raise NotImplementedError
        else:
            raise ValueError(
                "Incorrect embedder type. \
                             Avaliable embedder types: Hex2VecEmbedder, GeoVexEmbedder"
            )

    def _get_embeddings(self) -> pd.DataFrame:
        """
        Method to get embeddings from geolocation.

        Returns:
            pd.DataFrame: geolocation embeddings
        """
        neighbourhood = H3Neighbourhood(self.regions)
        joint = IntersectionJoiner().transform(self.regions, self.osm_features)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embeddings = self.embedder.fit_transform(
                self.regions,
                self.osm_features,
                joint,
                neighbourhood,
                batch_size=100,
                trainer_kwargs={"max_epochs": 10, "accelerator": self.device},
            )
        return embeddings

    def _concat_columns(self, row: gpd.GeoSeries) -> np.ndarray:
        """
        Concatenate embedding values together.

        Args:
            row (gpd.GeoSeries): row of embeddings

        Returns:
            np.ndarray: concatenated embedding
        """
        return np.concatenate([np.atleast_1d(val) for val in row.values])

    def _standardize(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Method to standardize numerical columns.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame in which standarization will be performed.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with standardized numerical features.
        """
        gdf_standardized = gdf.copy()
        if self.numerical_columns is not None:
            scaler = (
                StandardScaler()
            )  # TODO: watch out for data leakage -> train test split must be before standarization!
            # TODO: add scaler to self
            gdf_standardized[self.numerical_columns] = scaler.fit_transform(
                gdf_standardized[self.numerical_columns]
            )

            return gpd.GeoDataFrame(gdf_standardized)

    def _categorical_encode(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        # TODO: Method to encode categorical columns. Columns which should be encoded \ are stored in  # noqa: W505, E501
        # HuggingFaceDataset object.

        # Args:
        #     gdf (gpd.GeoDataFrame): GeoDataFrame in which encoding of categorical values will be performed.  # noqa: W505, E501

        # Returns:
        #     gpd.GeoDataFrame: GeoDataFrame with encoded categorical features.

        gdf_encoded = gdf.copy()  # noqa: F841
        if self.categorical_columns is not None:
            pass
        raise NotImplementedError

    def get_dataset(self) -> Dataset:
        r"""
        Method to retur Hugging Face Dataset with input values to model \ (embeddings and numericalb
        features).

        Returns:
            Dataset: Hugging Face Dataset with X - input vectors, X_h3_idx - h3 indices, \
                y - target value
        """  # noqa: D205
        # TODO: load and save

        joined_gdf = gpd.sjoin(self.gdf, self.regions, how="left", op="within")
        joined_gdf.rename(columns={"index_right": "h3_index"}, inplace=True)
        if self.numerical_columns is not None:
            columns_to_add = self.numerical_columns + [
                self.target_column_name
            ]  # get features which should be added
        else:
            columns_to_add = [self.target_column_name]

        joined_gdf = self._standardize(joined_gdf)
        averages_hex = joined_gdf.groupby("h3_index")[
            columns_to_add
        ].mean()  # compute mean value per hex for all numerical values

        embeddings = self._get_embeddings()
        embeddings["h3"] = embeddings.index
        merged_gdf = embeddings.merge(
            averages_hex, how="inner", left_on="region_id", right_on="h3_index"
        )
        merge_columns = [
            col for col in merged_gdf.columns if col not in (["h3"] + [self.target_column_name])
        ]

        dataset_dict = Dataset.from_dict(
            {
                "X": merged_gdf[merge_columns].apply(self._concat_columns, axis=1).values,
                "X_h3_idx": merged_gdf["h3"].values,
                "y": merged_gdf[self.target_column_name].values,
            }
        )  # set it to be compatibile with torch  # noqa: E501
        dataset_dict.set_format(type="torch", columns=["X", "X_h3_idx", "y"])  #
        return dataset_dict
