"""NeighbourDataset tests."""
from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from srai.embedders.hex2vec.neighbour_dataset import NeighbourDataset

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.parametrize(  # type: ignore
    "negative_sample_k_distance,expectation",
    [
        (-1, pytest.raises(ValueError)),
        (0, pytest.raises(ValueError)),
        (1, pytest.raises(ValueError)),
        (2, does_not_raise()),
        (3, does_not_raise()),
    ],
)
def test_raises_with_incorrect_sample_k_distance(
    negative_sample_k_distance: int, expectation: Any, request: Any
) -> None:
    """Test if NeighbourDataset checks negative_sample_k_distance correctness."""
    mocker: MockerFixture = request.getfixturevalue("mocker")
    data = pd.DataFrame()
    neighbourhood = mocker.Mock()
    with expectation:
        NeighbourDataset(data, neighbourhood, negative_sample_k_distance=negative_sample_k_distance)
