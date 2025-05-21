"""Hex2VecEmbedder tests."""

import os
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest_snapshot.plugin import Snapshot
from pytorch_lightning import seed_everything

from srai.embedders.hex2vec.embedder import Hex2VecEmbedder
from srai.embedders.hex2vec.model import Hex2VecModel
from srai.exceptions import ModelNotFitException
from srai.neighbourhoods import H3Neighbourhood
from tests.embedders.conftest import TRAINER_KWARGS
from tests.embedders.hex2vec.constants import (
    ENCODER_SIZES,
    PREDEFINED_TEST_CASES,
)


@pytest.mark.parametrize(  # type: ignore
    "encoder_sizes,expectation",
    [
        ([150, 75, 50], does_not_raise()),
        ([10, 5], does_not_raise()),
        ([5], does_not_raise()),
        ([], pytest.raises(ValueError)),
        ([-1, 0], pytest.raises(ValueError)),
        ([0], pytest.raises(ValueError)),
    ],
)
def test_checking_encoder_sizes(encoder_sizes: list[int], expectation: Any) -> None:
    """Test that incorrect encoder sizes raise ValueError."""
    with expectation:
        Hex2VecEmbedder(encoder_sizes)


def test_embedder_not_fitted() -> None:
    """Test that Hex2VecEmbedder raises ModelNotFitException if not fitted."""
    embedder = Hex2VecEmbedder()
    with pytest.raises(ModelNotFitException):
        embedder.transform(gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame())


def test_embedder_default_encoder_sizes() -> None:
    """Test that Hex2VecEmbedder uses default encoder sizes if not specified."""
    embedder = Hex2VecEmbedder()
    assert embedder._encoder_sizes == Hex2VecEmbedder.DEFAULT_ENCODER_SIZES


def test_embedder(snapshot: Snapshot) -> None:
    """Test Hex2VecEmbedder on predefined test cases."""
    test_files_path = Path(__file__).parent / "test_files"
    snapshot.snapshot_dir = test_files_path.as_posix()

    for test_case in PREDEFINED_TEST_CASES:
        name = test_case["test_case_name"]
        seed = test_case["seed"]
        print(name, seed)

        regions_gdf = gpd.read_parquet(test_files_path / f"{name}_regions.parquet")
        features_gdf = gpd.read_parquet(test_files_path / f"{name}_features.parquet")
        joint_gdf = gpd.read_parquet(test_files_path / f"{name}_joint.parquet")
        seed_everything(seed)

        neighbourhood = H3Neighbourhood(regions_gdf)
        embedder = Hex2VecEmbedder(encoder_sizes=ENCODER_SIZES)
        result_df = embedder.fit_transform(
            regions_gdf,
            features_gdf,
            joint_gdf,
            neighbourhood,
            trainer_kwargs=TRAINER_KWARGS,
        )
        result_df.columns = result_df.columns.astype(str)
        print(result_df.head())

        # FIXME(Calychas): readd after making neighbourhoods deterministic.
        # See [#441](https://github.com/kraina-ai/srai/pull/441)
        # snapshot.assert_match(result_df.to_json(
        #     orient="index", indent=True), f"{name}_result.json"
        # )
        assert not result_df.empty
        assert result_df.shape[0] == regions_gdf.shape[0]
        assert result_df.shape[1] == ENCODER_SIZES[-1]


def test_embedder_save_load() -> None:
    """Test GeoVexEmbedder model saving and loading."""
    test_files_path = Path(__file__).parent / "test_files"
    for test_case in PREDEFINED_TEST_CASES:
        name = test_case["test_case_name"]
        seed = test_case["seed"]

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
        embedder = Hex2VecEmbedder(
            encoder_sizes=ENCODER_SIZES,
            expected_output_features=test_case["tags"],  # type: ignore[arg-type]
        )

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
        loaded_embedder = Hex2VecEmbedder.load(tmp_models_dir / "test_model")

        # get embeddings from the loaded model
        loaded_result_df = loaded_embedder.transform(
            regions_gdf,
            features_gdf,
            joint_gdf,
        )

        # verify that the model was loaded correctly
        assert_series_equal(
            embedder.expected_output_features, loaded_embedder.expected_output_features
        )
        assert_frame_equal(result_df, loaded_result_df, atol=1e-5)

        # check type of model
        assert isinstance(loaded_embedder._model, Hex2VecModel)

        # safely clean up tmp_models directory
        (tmp_models_dir / "test_model" / "model.pt").unlink()
        (tmp_models_dir / "test_model" / "config.json").unlink()
        os.rmdir(tmp_models_dir / "test_model")
        os.rmdir(tmp_models_dir)
