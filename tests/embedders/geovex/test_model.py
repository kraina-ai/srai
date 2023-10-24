"""Tests for Hex2Vec model."""
from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest

from srai.embedders.geovex.model import GeoVexModel


@pytest.mark.parametrize(  # type: ignore
    "radius,expectation",
    [
        (1, pytest.raises(ValueError)),
        (2, does_not_raise()),
        (16, does_not_raise()),
    ],
)
def test_model_raises_with_incorrect_layer_sizes(radius: int, expectation: Any) -> None:
    """Test if Hex2VecModel raises with incorrect layer_sizes."""
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
    """Test if Hex2VecModel layers are initialized correctly."""
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
