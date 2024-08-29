"""Tests for GeoVex model."""

import os
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import pandas as pd
import pytest
import torch
from pytorch_lightning import seed_everything

from srai.embedders.geovex.embedder import GeoVexEmbedder
from srai.embedders.geovex.model import GeoVexModel
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from tests.embedders.geovex.constants import EMBEDDING_SIZE, PREDEFINED_TEST_CASES


@pytest.mark.parametrize(  # type: ignore
    "radius,expectation",
    [
        (1, pytest.raises(ValueError)),
        (2, does_not_raise()),
        (16, does_not_raise()),
    ],
)
def test_model_raises_with_incorrect_layer_sizes(radius: int, expectation: Any) -> None:
    """Test if GeoVexModel raises with incorrect layer_sizes."""
    with expectation:
        GeoVexModel(
            k_dim=256,
            radius=radius,
        )


@pytest.mark.parametrize(  # type: ignore
    "radius",
    [4, 9],
)
def test_layers_initialized_correctly(radius) -> None:
    """Test if GeoVexModel layers are initialized correctly."""
    k_dim = 512
    conv_layers = 3
    emb_size = 32
    conv_layer_size = 256
    model = GeoVexModel(
        k_dim=k_dim,
        radius=radius,
        conv_layers=conv_layers,
        emb_size=emb_size,
        conv_layer_size=conv_layer_size,
    )
    conv_sizes = [k_dim, *[conv_layer_size * 2**i for i in range(conv_layers)]]

    block_counter = 1
    for i, layer in enumerate(model.encoder):
        if 4 < i < (4 + conv_layers):
            # this is a conv block
            assert layer[0].conv.in_channels == conv_sizes[block_counter]
            block_counter += 1

    # check that they linear embedding dimension is correct
    assert model.encoder[-1].out_features == emb_size


def test_model_tensors() -> None:
    """Test if model batches are correctly parsed after seeding."""
    test_files_path = Path(__file__).parent / "test_files"
    for test_case in PREDEFINED_TEST_CASES:
        name = test_case["test_case_name"]
        seed = test_case["seed"]
        radius: int = test_case["model_radius"]  # type: ignore
        print(name, seed)

        regions_gdf = gpd.read_parquet(test_files_path / f"{name}_regions.parquet")
        features_gdf = gpd.read_parquet(test_files_path / f"{name}_features.parquet")
        joint_gdf = pd.read_parquet(test_files_path / f"{name}_joint.parquet")

        seed_everything(seed, workers=True)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.use_deterministic_algorithms(True)

        neighbourhood = H3Neighbourhood(regions_gdf)
        target_features = [f"{st}_{t}" for st in test_case["tags"] for t in test_case["tags"][st]]  # type: ignore
        embedder = GeoVexEmbedder(
            target_features=target_features,
            batch_size=10,
            neighbourhood_radius=radius,
            embedding_size=EMBEDDING_SIZE,
            convolutional_layers=test_case["num_layers"],  # type: ignore
            convolutional_layer_size=test_case["convolutional_layer_size"],  # type: ignore
        )

        counts_df, dataloader, _ = embedder._prepare_dataset(
            regions_gdf, features_gdf, joint_gdf, neighbourhood, embedder._batch_size, shuffle=True
        )

        embedder._prepare_model(counts_df, 0.001)

        for _, param in cast(GeoVexModel, embedder._model).named_parameters():
            param.data.fill_(0.01)

        for i, batch in enumerate(dataloader):
            expected_batch = torch.load(test_files_path / f"{name}_batch_{i}.pt")
            torch.testing.assert_close(batch, expected_batch, rtol=0, atol=0)

            expected_encoder_forward_tensor = torch.load(
                test_files_path / f"{name}_encoder_forward_{i}.pt"
            )
            encoder_forward_tensor = cast(GeoVexModel, embedder._model).encoder.forward(batch)
            torch.testing.assert_close(encoder_forward_tensor, expected_encoder_forward_tensor)

            expected_forward_tensor = torch.load(test_files_path / f"{name}_forward_{i}.pt")
            forward_tensor = cast(GeoVexModel, embedder._model).forward(batch)
            torch.testing.assert_close(forward_tensor, expected_forward_tensor)

            expected_loss_tensor = torch.load(test_files_path / f"{name}_loss_{i}.pt")
            loss_tensor = cast(GeoVexModel, embedder._model).training_step(batch, i)
            torch.testing.assert_close(loss_tensor, expected_loss_tensor)
