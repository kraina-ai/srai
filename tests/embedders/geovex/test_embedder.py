"""GeoVexEmbedder tests."""

import os
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any, Union, cast

import geopandas as gpd
import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal
from pytorch_lightning import seed_everything

from srai.embedders.geovex.embedder import GeoVexEmbedder
from srai.embedders.geovex.model import GeoVexModel
from srai.exceptions import ModelNotFitException
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
from srai.neighbourhoods import H3Neighbourhood
from tests.embedders.geovex.constants import EMBEDDING_SIZE, PREDEFINED_TEST_CASES, TRAINER_KWARGS


@pytest.mark.parametrize(  # type: ignore
    "target_features, conv_layer_size, expectation",
    [
        (None, 256, does_not_raise()),
        (["building", "amenity"], 32, does_not_raise()),
        (["building", "amenity"], 256, pytest.raises(ValueError)),
    ],
)
def test_checking_encoder_sizes(
    target_features: Union[str, list[str]], conv_layer_size, expectation: Any
) -> None:
    """Test that incorrect encoder sizes raise ValueError."""
    target_tags: dict[str, list[str]] = target_features or HEX2VEC_FILTER  # type: ignore
    target_features = [f"{t}_{st}" for t in target_tags for st in HEX2VEC_FILTER[t]]  # type: ignore

    with expectation:
        GeoVexEmbedder(
            target_features=target_features,
            convolutional_layer_size=conv_layer_size,
        )


def test_embedder_not_fitted() -> None:
    """Test that GeoVexEmbedder raises ModelNotFitException if not fitted."""
    embedder = GeoVexEmbedder(
        [f"{t}_{st}" for t in HEX2VEC_FILTER for st in HEX2VEC_FILTER[t]],  # type: ignore
    )
    with pytest.raises(ModelNotFitException):
        embedder.transform(gpd.GeoDataFrame(geometry=[]), gpd.GeoDataFrame(), gpd.GeoDataFrame())


def test_embedder() -> None:
    """Test GeoVexEmbedder on predefined test cases."""
    test_files_path = Path(__file__).parent / "test_files"
    for test_case in PREDEFINED_TEST_CASES:
        name = test_case["test_case_name"]
        seed = test_case["seed"]
        radius: int = test_case["model_radius"]  # type: ignore
        print(name, seed)

        expected = pd.read_parquet(test_files_path / f"{name}_result.parquet")
        regions_gdf = gpd.read_parquet(test_files_path / f"{name}_regions.parquet")
        features_gdf = gpd.read_parquet(test_files_path / f"{name}_features.parquet")
        joint_gdf = pd.read_parquet(test_files_path / f"{name}_joint.parquet")
        seed_everything(seed, workers=True)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.use_deterministic_algorithms(True)

        neighbourhood = H3Neighbourhood(regions_gdf)
        target_features = [
            f"{st}_{t}"
            for st in test_case["tags"]  # type: ignore
            for t in test_case["tags"][st]  # type: ignore
        ]
        embedder = GeoVexEmbedder(
            target_features=target_features,
            batch_size=10,
            neighbourhood_radius=radius,
            embedding_size=EMBEDDING_SIZE,
            convolutional_layers=test_case["num_layers"],  # type: ignore
            convolutional_layer_size=test_case["convolutional_layer_size"],  # type: ignore
        )

        counts_df, _, _ = embedder._prepare_dataset(
            regions_gdf, features_gdf, joint_gdf, neighbourhood, embedder._batch_size, shuffle=True
        )

        embedder._prepare_model(counts_df, 0.001)

        for _, param in cast(GeoVexModel, embedder._model).named_parameters():
            param.data.fill_(0.01)

        result_df = embedder.fit_transform(
            regions_gdf,
            features_gdf,
            joint_gdf,
            neighbourhood,
            trainer_kwargs=TRAINER_KWARGS,
            learning_rate=0.001,
        )
        result_df.columns = result_df.columns.astype(str)
        print(result_df.head())
        print(expected.head())
        assert_frame_equal(result_df, expected, atol=1e-1)


def test_embedder_save_load() -> None:
    """Test GeoVexEmbedder model saving and loading."""
    test_files_path = Path(__file__).parent / "test_files"
    for test_case in PREDEFINED_TEST_CASES:
        name = test_case["test_case_name"]
        seed = test_case["seed"]
        radius: int = test_case["model_radius"]  # type: ignore

        # Load  data from parquet files
        regions_gdf = gpd.read_parquet(test_files_path / f"{name}_regions.parquet")
        features_gdf = gpd.read_parquet(test_files_path / f"{name}_features.parquet")
        joint_gdf = pd.read_parquet(test_files_path / f"{name}_joint.parquet")

        # Set seed for reproducibility
        seed_everything(seed, workers=True)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.use_deterministic_algorithms(True)

        # Initialize neighbourhood and target features for the embedder
        neighbourhood = H3Neighbourhood(regions_gdf)
        target_features = [
            f"{st}_{t}"
            for st in test_case["tags"]  # type: ignore
            for t in test_case["tags"][st]  # type: ignore
        ]

        # Initialize GeoVexEmbedder with the given parameters
        embedder = GeoVexEmbedder(
            target_features=target_features,
            batch_size=10,
            neighbourhood_radius=radius,
            embedding_size=EMBEDDING_SIZE,
            convolutional_layers=test_case["num_layers"],  # type: ignore
            convolutional_layer_size=test_case["convolutional_layer_size"],  # type: ignore
        )

        # Prepare dataset for the embedder
        counts_df, _, _ = embedder._prepare_dataset(
            regions_gdf, features_gdf, joint_gdf, neighbourhood, embedder._batch_size, shuffle=True
        )

        embedder._prepare_model(counts_df, 0.001)

        # Initialize model parameters to a constant value for reproducibility
        for _, param in cast(GeoVexModel, embedder._model).named_parameters():
            param.data.fill_(0.01)

        result_df = embedder.fit_transform(
            regions_gdf,
            features_gdf,
            joint_gdf,
            neighbourhood,
            trainer_kwargs=TRAINER_KWARGS,
            learning_rate=0.001,
        )

        tmp_models_dir = Path(__file__).parent / "test_files" / "tmp_models"

        # test model saving functionality
        embedder.save(tmp_models_dir / "test_model")

        # load the saved model
        loaded_embedder = GeoVexEmbedder.load(tmp_models_dir / "test_model")

        # get embeddings from the loaded model
        loaded_result_df = loaded_embedder.fit_transform(
            regions_gdf,
            features_gdf,
            joint_gdf,
            neighbourhood,
            trainer_kwargs=TRAINER_KWARGS,
            learning_rate=0.001,
        )

        # verify that the model was loaded correctly
        assert_frame_equal(result_df, loaded_result_df, atol=1e-1)

        # check type of model
        assert isinstance(loaded_embedder._model, GeoVexModel)

        # safely clean up tmp_models directory
        (tmp_models_dir / "test_model" / "model.pt").unlink()
        (tmp_models_dir / "test_model" / "config.json").unlink()
        os.rmdir(tmp_models_dir / "test_model")
        os.rmdir(tmp_models_dir)
