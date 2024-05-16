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
from srai.h3 import ring_buffer_h3_regions_gdf
from srai.joiners import IntersectionJoiner
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import GEOFABRIK_LAYERS, HEX2VEC_FILTER
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
        gdf_train: gpd.GeoDataFrame,
        HF_dataset_object: Optional[HuggingFaceDataset] = None,
        target_column_name: Optional[str] = None,
        numerical_columns: Optional[list[str]] = None,
        categorical_columns: Optional[list[str]] = None,
        embedder_type: str = "Hex2VecEmbedder",
        h3_resolution: int = 8,
        embedder_hidden_sizes: Optional[list[int]] = None,
    ):
        """
        Initialization of Vectorizer.

        Args:
            gdf_train (gpd.GeoDataFrame): GeoDataFrame with training data.
            HF_dataset_object (Optional[HuggingFaceDataset], optional): Dataset object from which gdf_dataset was loaded.
            target_column_name (Optional[str], optional): Target column in regression. Defaults to None. \
                If default, columns are inherited from HuggingFaceDataset.
            numerical_columns (Optional[list[str]], optional): Columns from gdf_dataset with \
                numerical values that will be used to train model. If default, columns are inherited from HuggingFaceDataset.
            categorical_columns (Optional[list[str]], optional): Columns from gdf_dataset with \
                categorical values that will be used to train model. If default, columns are inherited from HuggingFaceDataset.
            embedder_type (str, optional): Type of embedder. Available embedders: Hex2VecEmbedder,\
                  GeoVecEmbedder. Defaults to "Hex2VecEmbedder".
            h3_resolution (int, optional): Resolution in h3 regionalizer. Defaults to 8.
            embedder_hidden_sizes (Optional[list[int]], int, optional): Hidden sizes of embedder. \
                Defaults to None. If GeoVexEmbedder type, list consisting of 1 element should be provided.

        Raises:
            NotImplementedError: GeoVecEmbedder usage is not implemented.
            ValueError: If embedder type is not available.
        """  # noqa: W505, E501
        if embedder_hidden_sizes is None:
            self.embedder_hidden_sizes = [150, 75, 50]
        else:
            self.embedder_hidden_sizes = embedder_hidden_sizes
        if (
            numerical_columns is None and HF_dataset_object is not None
        ):  # get column names from dataset object  # noqa: E501
            numerical_columns = HF_dataset_object.numerical_columns
        if categorical_columns is None and HF_dataset_object is not None:
            categorical_columns = HF_dataset_object.numerical_columns
        if target_column_name is None and HF_dataset_object is not None:
            target_column_name = HF_dataset_object.target

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.gdf = gdf_train
        self.target_column_name = target_column_name

        self.regionalizer = H3Regionalizer(resolution=h3_resolution)

        # self.regions = self.regionalizer.transform(self.gdf)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.scaler = StandardScaler().fit(self.gdf[self.numerical_columns])  # fit to trained data
        regions = self.regionalizer.transform(self.gdf)
        if embedder_type == "Hex2VecEmbedder":
            self.embedder = Hex2VecEmbedder(embedder_hidden_sizes)
            self.osm_features = OSMPbfLoader().load(regions, HEX2VEC_FILTER)

        elif embedder_type == "GeoVexEmbedder":
            k_ring_buffer_radius = 1
            buffered_h3_regions = ring_buffer_h3_regions_gdf(regions, distance=k_ring_buffer_radius)
            self.regions = buffered_h3_regions  # overwrite h3 regions
            buffered_h3_geometry = self.regions.unary_union
            self.osm_features = OSMPbfLoader().load(buffered_h3_geometry, GEOFABRIK_LAYERS)

            self.embedder = GeoVexEmbedder(
                target_features=GEOFABRIK_LAYERS,
                batch_size=10,
                neighbourhood_radius=k_ring_buffer_radius,
                convolutional_layers=2,
                embedding_size=self.embedder_hidden_sizes[-1],
            )  # type: ignore
        else:
            raise ValueError(
                "Incorrect embedder type. \
                             Avaliable embedder types: Hex2VecEmbedder, GeoVexEmbedder"
            )

        neighbourhood = H3Neighbourhood(regions)
        joint = IntersectionJoiner().transform(regions, self.osm_features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.embedder.fit(
                regions_gdf=regions,
                features_gdf=self.osm_features,
                joint_gdf=joint,
                neighbourhood=neighbourhood,
                trainer_kwargs={"max_epochs": 10, "accelerator": self.device},
            )

    def _get_embeddings(self, gdf: Optional[gpd.GeoDataFrame] = None) -> pd.DataFrame:
        """
        Method to get embeddings from geolocation.

        Args:
            gdf (Optional[gpd.GeoDataFrame], optional): GeoDataFrame from which embeddings should be extracted. Defaults to None.

        Returns:
            pd.DataFrame: geolocation embeddings
        """  # noqa: W505, E501
        if gdf is None:  # get training data
            gdf_ = self.gdf
        else:  # perform on trained embedders
            gdf_ = gdf

        regions = self.regionalizer.transform(gdf_)  # transform gdf to regions
        joint = IntersectionJoiner().transform(regions, self.osm_features)
        embeddings = self.embedder.transform(
            regions_gdf=regions, features_gdf=self.osm_features, joint_gdf=joint
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
            gdf_standardized[self.numerical_columns] = self.scaler.transform(
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

    def get_dataset(self, gdf: Optional[gpd.GeoDataFrame] = None) -> Dataset:
        r"""
        Method to return Hugging Face Dataset with input values to model \ (embeddings and numerical
        features).

        Args:
            gdf: GeoDataFrame from which dataset should be created. If None, gdf from initialization is taken.

        Returns:
            Dataset: Hugging Face Dataset with X - input vectors, X_h3_idx - h3 indices, \
                y - target value
        """  # noqa: D205, E501, W505
        if gdf is None:
            gdf_ = self.gdf
        else:
            gdf_ = gdf
        regions = self.regionalizer.transform(gdf_)  # get regions

        joined_gdf = gpd.sjoin(gdf_, regions, how="left", op="within")  # noqa: E501
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

        embeddings = self._get_embeddings(gdf_)
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
