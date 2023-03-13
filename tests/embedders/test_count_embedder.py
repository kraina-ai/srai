"""CountEmbedder tests."""
from contextlib import nullcontext as does_not_raise
from typing import Any, List, Union

import geopandas as gpd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal

from srai.embedders import CountEmbedder
from srai.utils.constants import REGIONS_INDEX


@pytest.fixture  # type: ignore
def expected_embedding_df() -> pd.DataFrame:
    """Get expected CountEmbedder output for the default case."""
    expected_df = pd.DataFrame(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "leisure_adult_gaming_centre": [0, 0, 1],
            "leisure_playground": [0, 1, 0],
            "amenity_pub": [1, 0, 1],
        },
    )
    expected_df.set_index(REGIONS_INDEX, inplace=True)

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
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "amenity_parking": [0, 0, 0],
            "leisure_park": [0, 0, 0],
            "amenity_pub": [1, 0, 1],
        },
    )
    expected_df.set_index(REGIONS_INDEX, inplace=True)

    return expected_df


def test_correct_embedding(
    gdf_regions: gpd.GeoDataFrame,
    gdf_features: gpd.GeoDataFrame,
    gdf_joint: gpd.GeoDataFrame,
    expected_embedding_df: pd.DataFrame,
) -> None:
    """Test if CountEmbedder returns correct result in the default case."""
    embedding_df = CountEmbedder().transform(
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
    embedding_df = CountEmbedder(expected_output_features=expected_feature_names).transform(
        regions_gdf=gdf_regions, features_gdf=gdf_features, joint_gdf=gdf_joint
    )
    assert_frame_equal(embedding_df, specified_features_expected_embedding_df, check_dtype=False)


@pytest.mark.parametrize(  # type: ignore
    "regions_fixture,features_fixture,joint_fixture,expected_features_fixture,expectation",
    [
        (
            "gdf_regions_empty",
            "gdf_features",
            "gdf_joint",
            None,
            does_not_raise(),
        ),
        (
            "gdf_regions",
            "gdf_features_empty",
            "gdf_joint",
            None,
            pytest.raises(ValueError),
        ),
        (
            "gdf_regions",
            "gdf_features_empty",
            "gdf_joint",
            "expected_feature_names",
            does_not_raise(),
        ),
        (
            "gdf_regions",
            "gdf_features",
            "gdf_joint_empty",
            None,
            does_not_raise(),
        ),
    ],
)
def test_empty(
    regions_fixture: str,
    features_fixture: str,
    joint_fixture: str,
    expected_features_fixture: Union[str, None],
    expectation: Any,
    request: Any,
) -> None:
    """Test CountEmbedder handling of empty input data frames."""
    expected_output_features = (
        None
        if expected_features_fixture is None
        else request.getfixturevalue(expected_features_fixture)
    )
    embedder = CountEmbedder(expected_output_features)
    gdf_regions: gpd.GeoDataFrame = request.getfixturevalue(regions_fixture)
    gdf_features: gpd.GeoDataFrame = request.getfixturevalue(features_fixture)
    gdf_joint: gpd.GeoDataFrame = request.getfixturevalue(joint_fixture)

    with expectation:
        embedding = embedder.transform(gdf_regions, gdf_features, gdf_joint)
        assert len(embedding) == len(gdf_regions)
        assert_index_equal(embedding.index, gdf_regions.index)
        if expected_output_features:
            assert embedding.columns.tolist() == expected_output_features

        assert (embedding == 0).all().all()


@pytest.mark.parametrize(  # type: ignore
    "regions_fixture,features_fixture,joint_fixture,expectation",
    [
        (
            "gdf_unnamed_single_index",
            "gdf_features",
            "gdf_joint",
            pytest.raises(ValueError),
        ),
        (
            "gdf_regions",
            "gdf_unnamed_single_index",
            "gdf_joint",
            pytest.raises(ValueError),
        ),
        (
            "gdf_regions",
            "gdf_features",
            "gdf_unnamed_single_index",
            pytest.raises(ValueError),
        ),
        (
            "gdf_regions",
            "gdf_features",
            "gdf_three_level_multi_index",
            pytest.raises(ValueError),
        ),
        (
            "gdf_incorrectly_named_single_index",
            "gdf_features",
            "gdf_joint",
            pytest.raises(ValueError),
        ),
        (
            "gdf_regions",
            "gdf_incorrectly_named_single_index",
            "gdf_joint",
            pytest.raises(ValueError),
        ),
    ],
)
def test_incorrect_indexes(
    regions_fixture: str, features_fixture: str, joint_fixture: str, expectation: Any, request: Any
) -> None:
    """Test if cannot embed with incorrect dataframe indexes."""
    regions_gdf = request.getfixturevalue(regions_fixture)
    features_gdf = request.getfixturevalue(features_fixture)
    joint_gdf = request.getfixturevalue(joint_fixture)

    with expectation:
        CountEmbedder().transform(
            regions_gdf=regions_gdf, features_gdf=features_gdf, joint_gdf=joint_gdf
        )
