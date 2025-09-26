"""S2VecEmbedder tests."""

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal, assert_series_equal
from pytorch_lightning import seed_everything

from srai.embedders.s2vec.embedder import S2VecEmbedder
from srai.embedders.s2vec.model import S2VecModel
from srai.exceptions import ModelNotFitException
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
from tests.embedders.conftest import TRAINER_KWARGS
from tests.embedders.s2vec.constants import PREDEFINED_TEST_CASES


def test_embedder_not_fitted() -> None:
    """Test that S2VecEmbedder raises ModelNotFitException if not fitted."""
    embedder = S2VecEmbedder(
        target_features=[f"{t}_{st}" for t in HEX2VEC_FILTER for st in HEX2VEC_FILTER[t]]  # type: ignore
    )
    with pytest.raises(ModelNotFitException):
        embedder.transform(gpd.GeoDataFrame(geometry=[]), gpd.GeoDataFrame())


def test_embedder() -> None:
    """Test S2VecEmbedder on predefined test cases."""
    test_files_path = Path(__file__).parent / "test_files"
    for test_case in PREDEFINED_TEST_CASES:
        name = test_case["test_case_name"]
        seed = test_case["seed"]
        print(name, seed)

        expected = pd.read_parquet(test_files_path / f"{name}_result.parquet")
        regions_gdf = gpd.read_parquet(test_files_path / f"{name}_regions.parquet")
        features_gdf = gpd.read_parquet(test_files_path / f"{name}_features.parquet")

        seed_everything(seed, workers=True)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.use_deterministic_algorithms(True)

        target_features = [
            f"{st}_{t}"
            for st in test_case["tags"]  # type: ignore
            for t in test_case["tags"][st]  # type: ignore
        ]

        embedder = S2VecEmbedder(
            target_features=target_features,
            batch_size=10,
            img_res=test_case["img_res"],  # type: ignore
            patch_res=test_case["patch_res"],  # type: ignore
            num_heads=test_case["num_heads"],  # type: ignore
            encoder_layers=test_case["encoder_layers"],  # type: ignore
            decoder_layers=test_case["decoder_layers"],  # type: ignore
            embedding_dim=test_case["embedding_dim"],  # type: ignore
            decoder_dim=test_case["decoder_dim"],  # type: ignore
            mask_ratio=test_case["mask_ratio"],  # type: ignore
            dropout_prob=test_case["dropout_prob"],  # type: ignore
        )

        counts_df, _, _ = embedder._prepare_dataset(
            regions_gdf,
            features_gdf,
            embedder._batch_size,
            shuffle=True,
        )

        embedder._prepare_model(counts_df, 0.001)

        result_df = embedder.fit_transform(
            regions_gdf,
            features_gdf,
            learning_rate=0.001,
            trainer_kwargs=TRAINER_KWARGS,
        )
        result_df.columns = result_df.columns.astype(str)
        print(result_df.head())
        print(expected.head())
        assert_frame_equal(result_df, expected, atol=1e-4)


def test_embedder_save_load() -> None:
    """Test S2VecEmbedder model saving and loading."""
    test_files_path = Path(__file__).parent / "test_files"
    for test_case in PREDEFINED_TEST_CASES:
        name = test_case["test_case_name"]
        seed = test_case["seed"]

        regions_gdf = gpd.read_parquet(test_files_path / f"{name}_regions.parquet")
        features_gdf = gpd.read_parquet(test_files_path / f"{name}_features.parquet")

        seed_everything(seed, workers=True)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.use_deterministic_algorithms(True)

        embedder = S2VecEmbedder(
            target_features=test_case["tags"],  # type: ignore
            batch_size=1,
            img_res=test_case["img_res"],  # type: ignore
            patch_res=test_case["patch_res"],  # type: ignore
            num_heads=test_case["num_heads"],  # type: ignore
            encoder_layers=test_case["encoder_layers"],  # type: ignore
            decoder_layers=test_case["decoder_layers"],  # type: ignore
            embedding_dim=test_case["embedding_dim"],  # type: ignore
            decoder_dim=test_case["decoder_dim"],  # type: ignore
            mask_ratio=test_case["mask_ratio"],  # type: ignore
        )

        result_df = embedder.fit_transform(
            regions_gdf,
            features_gdf,
            learning_rate=0.001,
            trainer_kwargs=TRAINER_KWARGS,
        )

        tmp_models_dir = test_files_path / "tmp_models"

        # test model saving functionality
        embedder.save(tmp_models_dir / "test_model")

        # load the saved model
        loaded_embedder = S2VecEmbedder.load(tmp_models_dir / "test_model")

        # get embeddings from the loaded model
        loaded_result_df = loaded_embedder.transform(
            regions_gdf,
            features_gdf,
        )

        # verify that the model was loaded correctly
        assert_series_equal(
            embedder.expected_output_features, loaded_embedder.expected_output_features
        )
        assert_frame_equal(result_df, loaded_result_df, atol=1e-5)

        # check type of model
        assert isinstance(loaded_embedder._model, S2VecModel)

        # safely clean up tmp_models directory
        (tmp_models_dir / "test_model" / "model.pt").unlink()
        (tmp_models_dir / "test_model" / "config.json").unlink()
        os.rmdir(tmp_models_dir / "test_model")
        os.rmdir(tmp_models_dir)
