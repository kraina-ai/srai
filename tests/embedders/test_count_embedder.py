"""CountEmbedder tests."""
from typing import List

import geopandas as gpd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from srai.embedders import CountEmbedder


@pytest.fixture  # type: ignore
def expected_embedding_df() -> pd.DataFrame:
    """Get expected CountEmbedder output for the default case."""
    expected_df = pd.DataFrame(
        {
            "region_id": ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "leisure_adult_gaming_centre": [0, 0, 1],
            "leisure_playground": [0, 1, 0],
            "amenity_pub": [1, 0, 1],
        },
    )
    expected_df.set_index("region_id", inplace=True)

    return expected_df


@pytest.fixture  # type: ignore
def expected_feature_names() -> List[str]:
    """Get expected feature names for CountEmbedder."""
    expected_feature_names = ["amenity_parking", "leisure_park", "amenity_pub"]
    return expected_feature_names


@pytest.fixture  # type: ignore
def specified_features_expected_embedding_df() -> pd.DataFrame:
    """Get expected CountEmbedder output for the case with specified features."""
    expected_df = pd.DataFrame(
        {
            "region_id": ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "amenity_parking": [0, 0, 0],
            "leisure_park": [0, 0, 0],
            "amenity_pub": [1, 0, 1],
        },
    )
    expected_df.set_index("region_id", inplace=True)

    return expected_df


def test_correct_embedding(
    gdf_regions: gpd.GeoDataFrame,
    gdf_features: gpd.GeoDataFrame,
    gdf_joint: gpd.GeoDataFrame,
    expected_embedding_df: pd.DataFrame,
) -> None:
    """Test if CountEmbedder returns correct result in the default case."""
    embedding_df = CountEmbedder().embed(
        regions_gdf=gdf_regions, features_gdf=gdf_features, joint_gdf=gdf_joint
    )
    assert_frame_equal(embedding_df, expected_embedding_df, check_dtype=False)


def test_correct_embedding_expected_features(
    gdf_regions: gpd.GeoDataFrame,
    gdf_features: gpd.GeoDataFrame,
    gdf_joint: gpd.GeoDataFrame,
    expected_feature_names: List[str],
    specified_features_expected_embedding_df: pd.DataFrame,
) -> None:
    """Test if CountEmbedder returns correct result in the specified features case."""
    embedding_df = CountEmbedder(expected_output_features=expected_feature_names).embed(
        regions_gdf=gdf_regions, features_gdf=gdf_features, joint_gdf=gdf_joint
    )
    assert_frame_equal(embedding_df, specified_features_expected_embedding_df, check_dtype=False)
