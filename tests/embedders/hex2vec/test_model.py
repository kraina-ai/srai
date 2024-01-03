"""Tests for Hex2Vec model."""

from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest

from srai.embedders.hex2vec.model import Hex2VecModel


@pytest.mark.parametrize(  # type: ignore
    "layer_sizes,expectation",
    [
        ([], pytest.raises(ValueError)),
        ([15], pytest.raises(ValueError)),
        ([15, 10], does_not_raise()),
        ([15, 10, 5], does_not_raise()),
    ],
)
def test_model_raises_with_incorrect_layer_sizes(layer_sizes: list[int], expectation: Any) -> None:
    """Test if Hex2VecModel raises with incorrect layer_sizes."""
    with expectation:
        Hex2VecModel(layer_sizes=layer_sizes)


def test_layers_initialized_correctly() -> None:
    """Test if Hex2VecModel layers are initialized correctly."""
    layer_sizes = [15, 10, 5]

    model = Hex2VecModel(layer_sizes=layer_sizes)
    for i, layer in enumerate(model.encoder):
        if i % 2 == 0:
            assert layer.in_features == layer_sizes[i // 2]
            assert layer.out_features == layer_sizes[i // 2 + 1]
