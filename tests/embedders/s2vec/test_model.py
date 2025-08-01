"""Tests for S2Vec model."""

from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest

from srai.embedders.s2vec.model import S2VecModel


@pytest.mark.parametrize(  # type: ignore
    "encoder_layers,decoder_layers,num_heads,embed_dim,decoder_dim,mask_ratio,expectation",
    [
        (0, 2, 2, 256, 128, 0.75, pytest.raises(ValueError)),
        (6, 0, 2, 256, 128, 0.75, pytest.raises(ValueError)),
        (6, 2, 0, 256, 128, 0.75, pytest.raises(ValueError)),
        (6, 2, 2, 0, 128, 0.75, pytest.raises(ValueError)),
        (6, 2, 2, 256, 0, 0.75, pytest.raises(ValueError)),
        (6, 2, 2, 256, 128, 0, pytest.raises(ValueError)),
        (6, 2, 2, 256, 128, 1.0, pytest.raises(ValueError)),
        (6, 2, 2, 256, 128, 0.75, does_not_raise()),
    ],
)
def test_model_raises_with_incorrect_params(
    encoder_layers: int,
    decoder_layers: int,
    num_heads: int,
    embed_dim: int,
    decoder_dim: int,
    mask_ratio: float,
    expectation: Any,
) -> None:
    """Test if S2VecModel raises with incorrect parameters."""
    with expectation:
        S2VecModel(
            img_size=16,
            patch_size=1,
            in_ch=347,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            num_heads=num_heads,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            mask_ratio=mask_ratio,
        )


def test_layers_initialized_correctly() -> None:
    """Test if S2VecModel layers are initialized correctly."""
    encoder_layers = 6
    decoder_layers = 2
    num_heads = 2
    embed_dim = 256
    decoder_dim = 128
    model = S2VecModel(
        img_size=16,
        patch_size=1,
        in_ch=347,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
    )
    assert len(model.encoder.blocks) == encoder_layers
    assert len(model.decoder.blocks) == decoder_layers
    for block in model.encoder.blocks:
        assert block.norm1.normalized_shape[0] == embed_dim
    for block in model.decoder.blocks:
        assert block.norm1.normalized_shape[0] == decoder_dim
