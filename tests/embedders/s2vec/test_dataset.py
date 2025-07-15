"""S2Vec Dataset tests."""

import geopandas as gpd
import pandas as pd
import pytest
import torch

from srai.embedders.s2vec.dataset import S2VecDataset
from srai.embedders.s2vec.s2_utils import get_patches_from_img_gdf

ROOT_REGIONS = ["470fc275", "470fc277"]
PARENT_LEVEL = 14
TARGET_LEVEL = 18


@pytest.fixture  # type: ignore
def empty_gdf() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame."""
    return gpd.GeoDataFrame(geometry=[])


@pytest.fixture  # type: ignore
def data_and_joint_dfs() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Get example regions for testing."""
    target_level = TARGET_LEVEL
    regions_tokens = ROOT_REGIONS
    img_df = pd.DataFrame(index=regions_tokens)
    patches_gdf, img_patch_joint_gdf = get_patches_from_img_gdf(
        img_gdf=img_df, target_level=target_level
    )
    data_df = pd.DataFrame(list(range(len(patches_gdf))), index=patches_gdf.index, columns=["data"])
    return data_df, img_patch_joint_gdf


def test_dataset_length(data_and_joint_dfs: tuple[pd.DataFrame, gpd.GeoDataFrame]) -> None:
    """Test if S2VecDataset constructs lookup tables correctly."""
    data_df, img_patch_joint_gdf = data_and_joint_dfs
    dataset = S2VecDataset(data_df, img_patch_joint_gdf)  # type: ignore
    assert len(dataset) == len(ROOT_REGIONS)


def test_dataset_get_item_shape(data_and_joint_dfs: tuple[pd.DataFrame, gpd.GeoDataFrame]) -> None:
    """Test if S2VecDataset items have the correct size."""
    data_df, img_patch_joint_gdf = data_and_joint_dfs
    dataset = S2VecDataset(data_df, img_patch_joint_gdf)  # type: ignore

    item_0 = dataset[0]

    assert isinstance(item_0, torch.Tensor)
    assert item_0.shape == torch.Size([4 ** (TARGET_LEVEL - PARENT_LEVEL), 1])
