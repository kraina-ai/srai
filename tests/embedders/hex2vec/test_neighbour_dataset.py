"""NeighbourDataset tests."""

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from srai.embedders.hex2vec.neighbour_dataset import NeighbourDataset
from srai.neighbourhoods import H3Neighbourhood

if TYPE_CHECKING:  # pragma: no cover
    from pytest_mock import MockerFixture


@pytest.fixture  # type: ignore
def regions_data_df() -> pd.DataFrame:
    """Get example regions for testing."""
    root_region = "891e205194bffff"
    neighbourhood = H3Neighbourhood()
    regions_indices = list(neighbourhood.get_neighbours_up_to_distance(root_region, 25))
    regions_indices.append(root_region)
    data_df = pd.DataFrame(0, index=regions_indices, columns=["data"])
    return data_df


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
    negative_sample_k_distance: int, expectation: Any, mocker: "MockerFixture"
) -> None:
    """Test if NeighbourDataset checks negative_sample_k_distance correctness."""
    data = pd.DataFrame()
    neighbourhood = mocker.Mock()
    with expectation:
        NeighbourDataset(data, neighbourhood, negative_sample_k_distance=negative_sample_k_distance)


@pytest.mark.parametrize("negative_sample_k_distance", [2, 3, 4])  # type: ignore
def test_lookup_tables_construction(negative_sample_k_distance: int, regions_data_df: pd.DataFrame):
    """Test if NeighbourDataset constructs lookup tables correctly."""
    neighbourhood = H3Neighbourhood(regions_data_df)
    dataset = NeighbourDataset(
        regions_data_df, neighbourhood, negative_sample_k_distance=negative_sample_k_distance
    )

    positives_correct = [
        regions_data_df.index[positive_df_loc]
        in neighbourhood.get_neighbours(regions_data_df.index[anchor_df_loc])
        for anchor_df_loc, positive_df_loc in zip(
            dataset._anchor_df_locs_lookup, dataset._positive_df_locs_lookup
        )
    ]

    assert all(positives_correct)

    excluded_correct = []
    for i, index in enumerate(regions_data_df.index):
        excluded_df_locs = dataset._excluded_from_negatives[i]
        excluded_df_indices = set(regions_data_df.index[list(excluded_df_locs)])
        excluded_correct.append(
            neighbourhood.get_neighbours_up_to_distance(index, negative_sample_k_distance)
            == excluded_df_indices
        )

    assert all(excluded_correct)


def test_dataset_length(regions_data_df: pd.DataFrame) -> None:
    """Test if NeighbourDataset has correct length."""
    neighbourhood = H3Neighbourhood(regions_data_df)
    expected_length = sum(
        len(neighbourhood.get_neighbours(region)) for region in regions_data_df.index
    )
    dataset = NeighbourDataset(regions_data_df, neighbourhood)
    assert len(dataset) == expected_length
