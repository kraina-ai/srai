"""GeoVex HexagonalDataset tests."""
import os
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
import torch
from lightning_fabric import seed_everything

from srai.embedders.geovex.dataset import HexagonalDataset
from srai.embedders.geovex.embedder import GeoVexEmbedder
from srai.h3 import get_local_ij_index
from srai.neighbourhoods import AdjacencyNeighbourhood, H3Neighbourhood
from tests.embedders.geovex.constants import EMBEDDING_SIZE, PREDEFINED_TEST_CASES

ROOT_REGION = "891e205194bffff"
RING_DISTANCE = 25


@pytest.fixture  # type: ignore
def empty_gdf() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame."""
    return gpd.GeoDataFrame(geometry=[])


@pytest.fixture  # type: ignore
def regions_data_df() -> pd.DataFrame:
    """
    Get example regions for testing.

    Expects first region to be a ROOT_REGION.
    """
    neighbourhood = H3Neighbourhood()
    regions_indices = [ROOT_REGION]
    regions_indices.extend(
        list(neighbourhood.get_neighbours_up_to_distance(ROOT_REGION, RING_DISTANCE))
    )
    data_df = pd.DataFrame(
        list(range(len(regions_indices))), index=regions_indices, columns=["data"]
    )
    return data_df


@pytest.mark.parametrize(  # type: ignore
    "ring_distance,expectation",
    [
        (-1, pytest.raises(ValueError)),
        (0, pytest.raises(ValueError)),
        (1, pytest.raises(ValueError)),
        (2, does_not_raise()),
        (3, does_not_raise()),
    ],
)
def test_raises_with_incorrect_ring_distance(
    ring_distance: int,
    expectation: Any,
) -> None:
    """Test if HexagonalDataset checks ring_distance correctness."""
    data = pd.DataFrame()
    neighbourhood = H3Neighbourhood()
    with expectation:
        HexagonalDataset(data, neighbourhood, neighbor_k_ring=ring_distance)


@pytest.mark.parametrize(
    "ring_distance",
    [2, 3, 4],
)  # type: ignore
def test_dataset_length(ring_distance: int, regions_data_df: pd.DataFrame):
    """Test if HexagonalDataset constructs lookup tables correctly."""
    neighbourhood: H3Neighbourhood = H3Neighbourhood(regions_data_df)
    dataset = HexagonalDataset(
        regions_data_df, neighbourhood, neighbor_k_ring=ring_distance
    )  # type: ignore
    assert len(dataset) == len(
        neighbourhood.get_neighbours_up_to_distance(
            ROOT_REGION, distance=RING_DISTANCE - ring_distance, include_center=True, unchecked=True
        )
    )


@pytest.mark.parametrize(
    "neighborhood_cls,expectation",
    [(H3Neighbourhood, does_not_raise()), (AdjacencyNeighbourhood, pytest.raises(ValueError))],
)  # type: ignore
def test_neighborhood_type(
    neighborhood_cls: Any,
    expectation: Any,
    regions_data_df: pd.DataFrame,
    empty_gdf: gpd.GeoDataFrame,
) -> None:
    """Test if HexagonalDataset correctly accepts only H3Neighborhoods."""
    neighborhood = neighborhood_cls(empty_gdf)
    with expectation:
        HexagonalDataset(regions_data_df, neighborhood)


def test_dataset_item(regions_data_df: pd.DataFrame) -> None:
    """Test if HexagonalDataset constructs lookup tables correctly."""
    import numpy as np

    ring_distance = 2

    neighbourhood = H3Neighbourhood(regions_data_df)
    dataset = HexagonalDataset(
        regions_data_df, neighbourhood, neighbor_k_ring=ring_distance
    )  # type: ignore
    item = next(iter(dataset)).detach().numpy()
    # flatten it out and get the corresponding hexagons
    cells = regions_data_df.reset_index().set_index("data").loc[item.reshape(-1).tolist()].values

    # it starts with the root region
    # for each of the h3s, calculate the ij index
    ijs = np.array([get_local_ij_index(ROOT_REGION, _cell) for _cell in cells])

    ijs = ijs.reshape(ring_distance * 2 + 2, ring_distance * 2 + 2, 2)

    # commpare to the transposed image in the paper
    # specifically fig. 3
    # the bottom and right are padded by 0s for even #
    desired = np.array(
        [
            [(0, 0), (0, 0), (0, 2), (1, 2), (2, 2), (0, 0)],
            [(0, 0), (-1, 1), (0, 1), (1, 1), (2, 1), (0, 0)],
            [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (0, 0)],
            [(-2, -1), (-1, -1), (0, -1), (1, -1), (0, 0), (0, 0)],
            [(-2, -2), (-1, -2), (0, -2), (0, 0), (0, 0), (0, 0)],
            [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        ]
    )
    assert np.all(ijs.transpose(1, 0, -1) == desired)


def test_dataloader_batches() -> None:
    """Test if dataloader batches are in correct order after seeding."""
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
        target_features = [
            f"{st}_{t}" for st in test_case["tags"] for t in test_case["tags"][st]  # type: ignore
        ]
        embedder = GeoVexEmbedder(
            target_features=target_features,
            batch_size=10,
            neighbourhood_radius=radius,
            embedding_size=EMBEDDING_SIZE,
            convolutional_layers=test_case["num_layers"],  # type: ignore
            convolutional_layer_size=test_case["convolutional_layer_size"],  # type: ignore
        )

        _, dataloader, _ = embedder._prepare_dataset(
            regions_gdf, features_gdf, joint_gdf, neighbourhood, embedder._batch_size, shuffle=True
        )

        for i, batch in enumerate(dataloader):
            expected_batch = torch.load(test_files_path / f"{name}_batch_{i}.pt")
            torch.testing.assert_close(batch, expected_batch, rtol=0, atol=0)
