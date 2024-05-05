"""ContextualCountEmbedder tests."""

from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any, Union

import geopandas as gpd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from parametrization import Parametrization as P
from shapely.geometry import Polygon

from srai.constants import REGIONS_INDEX, WGS84_CRS
from srai.embedders import ContextualCountEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import GEOFABRIK_LAYERS, OsmTagsFilter
from srai.neighbourhoods import H3Neighbourhood
from srai.regionalizers import H3Regionalizer


def _create_features_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(data).set_index(REGIONS_INDEX)


@pytest.fixture  # type: ignore
def expected_embedding_df_squashed_distance_0() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output.

    Count without subcategories. Squashed features, distance 0.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "leisure": [0, 1, 1],
            "amenity": [1, 0, 1],
        }
    )


@pytest.fixture  # type: ignore
def expected_embedding_df_squashed_distance_1() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output.

    Count without subcategories. Squashed features, distance 1+.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "leisure": [0.25, 1.125, 1.125],
            "amenity": [1.125, 0.25, 1.125],
        },
    )


@pytest.fixture  # type: ignore
def expected_embedding_df_concatenated_distance_0() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output.

    Count without subcategories. Concatenated features, distance 0.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "leisure_0": [0, 1, 1],
            "amenity_0": [1, 0, 1],
        }
    )


@pytest.fixture  # type: ignore
def expected_embedding_df_concatenated_distance_1() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output.

    Count without subcategories. Concatenated features, distance 1.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "leisure_0": [0, 1, 1],
            "amenity_0": [1, 0, 1],
            "leisure_1": [1, 0.5, 0.5],
            "amenity_1": [0.5, 1, 0.5],
        },
    )


@pytest.fixture  # type: ignore
def expected_embedding_df_concatenated_distance_2() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output.

    Count without subcategories. Concatenated features, distance 2.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "leisure_0": [0, 1, 1],
            "amenity_0": [1, 0, 1],
            "leisure_1": [1, 0.5, 0.5],
            "amenity_1": [0.5, 1, 0.5],
            "leisure_2": [0, 0, 0],
            "amenity_2": [0, 0, 0],
        },
    )


@pytest.fixture  # type: ignore
def expected_subcategories_embedding_df_squashed_distance_0() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output.

    Count with subcategories. Squashed features, distance 0.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "leisure_adult_gaming_centre": [0, 0, 1],
            "leisure_playground": [0, 1, 0],
            "amenity_pub": [1, 0, 1],
        },
    )


@pytest.fixture  # type: ignore
def expected_subcategories_embedding_df_squashed_distance_1() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output.

    Count with subcategories. Squashed features, distance 1+.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "leisure_adult_gaming_centre": [0.125, 0.125, 1],
            "leisure_playground": [0.125, 1, 0.125],
            "amenity_pub": [1.125, 0.25, 1.125],
        },
    )


@pytest.fixture  # type: ignore
def expected_subcategories_embedding_df_concatenated_distance_0() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output.

    Count with subcategories. Concatenated features, distance 0.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "leisure_adult_gaming_centre_0": [0, 0, 1],
            "leisure_playground_0": [0, 1, 0],
            "amenity_pub_0": [1, 0, 1],
        },
    )


@pytest.fixture  # type: ignore
def expected_subcategories_embedding_df_concatenated_distance_1() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output.

    Count with subcategories. Concatenated features, distance 1.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "leisure_adult_gaming_centre_0": [0, 0, 1],
            "leisure_playground_0": [0, 1, 0],
            "amenity_pub_0": [1, 0, 1],
            "leisure_adult_gaming_centre_1": [0.5, 0.5, 0],
            "leisure_playground_1": [0.5, 0, 0.5],
            "amenity_pub_1": [0.5, 1, 0.5],
        },
    )


@pytest.fixture  # type: ignore
def expected_subcategories_embedding_df_concatenated_distance_2() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output.

    Count with subcategories. Concatenated features, distance 2.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "leisure_adult_gaming_centre_0": [0, 0, 1],
            "leisure_playground_0": [0, 1, 0],
            "amenity_pub_0": [1, 0, 1],
            "leisure_adult_gaming_centre_1": [0.5, 0.5, 0],
            "leisure_playground_1": [0.5, 0, 0.5],
            "amenity_pub_1": [0.5, 1, 0.5],
            "leisure_adult_gaming_centre_2": [0, 0, 0],
            "leisure_playground_2": [0, 0, 0],
            "amenity_pub_2": [0, 0, 0],
        },
    )


@pytest.fixture  # type: ignore
def expected_feature_names() -> list[str]:
    """Get expected feature names for ContextualCountEmbedder."""
    expected_feature_names = ["amenity_parking", "leisure_park", "amenity_pub"]
    return expected_feature_names


@pytest.fixture  # type: ignore
def osm_tags_filter() -> OsmTagsFilter:
    """Get osm tags filter for CountEmbedder."""
    tags_filter: OsmTagsFilter = {
        "amenity": ["parking", "pub"],
        "leisure": "park",
    }
    return tags_filter


@pytest.fixture  # type: ignore
def specified_features_expected_embedding_df_squashed_empty() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output for the case with specified features.

    Count without subcategories. Squashed features, distance 0+.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "amenity_parking": [0, 0, 0],
            "leisure_park": [0, 0, 0],
            "amenity_pub": [0, 0, 0],
        }
    )


@pytest.fixture  # type: ignore
def specified_features_expected_subcategories_embedding_df_squashed_distance_0() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output for the case with specified features.

    Count with subcategories. Squashed features, distance 0.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "amenity_parking": [0, 0, 0],
            "leisure_park": [0, 0, 0],
            "amenity_pub": [1, 0, 1],
        }
    )


@pytest.fixture  # type: ignore
def specified_features_expected_subcategories_embedding_df_squashed_distance_1() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output for the case with specified features.

    Count with subcategories. Squashed features, distance 1+.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "amenity_parking": [0, 0, 0],
            "leisure_park": [0, 0, 0],
            "amenity_pub": [1.125, 0.25, 1.125],
        }
    )


@pytest.fixture  # type: ignore
def specified_features_expected_embedding_df_concatenated_distance_0() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output for the case with specified features.

    Count without subcategories. Concatenated features, distance 0.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "amenity_parking_0": [0, 0, 0],
            "leisure_park_0": [0, 0, 0],
            "amenity_pub_0": [0, 0, 0],
        }
    )


@pytest.fixture  # type: ignore
def specified_features_expected_embedding_df_concatenated_distance_1() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output for the case with specified features.

    Count without subcategories. Concatenated features, distance 1.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "amenity_parking_0": [0, 0, 0],
            "leisure_park_0": [0, 0, 0],
            "amenity_pub_0": [0, 0, 0],
            "amenity_parking_1": [0, 0, 0],
            "leisure_park_1": [0, 0, 0],
            "amenity_pub_1": [0, 0, 0],
        }
    )


@pytest.fixture  # type: ignore
def specified_features_expected_embedding_df_concatenated_distance_2() -> pd.DataFrame:
    """
    Get expected ContextualCountEmbedder output for the case with specified features.

    Count without subcategories. Concatenated features, distance 2.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "amenity_parking_0": [0, 0, 0],
            "leisure_park_0": [0, 0, 0],
            "amenity_pub_0": [0, 0, 0],
            "amenity_parking_1": [0, 0, 0],
            "leisure_park_1": [0, 0, 0],
            "amenity_pub_1": [0, 0, 0],
            "amenity_parking_2": [0, 0, 0],
            "leisure_park_2": [0, 0, 0],
            "amenity_pub_2": [0, 0, 0],
        }
    )


@pytest.fixture  # type: ignore
def specified_features_expected_subcategories_embedding_df_concatenated_distance_0() -> (
    pd.DataFrame
):
    """
    Get expected ContextualCountEmbedder output for the case with specified features.

    Count with subcategories. Concatenated features, distance 0.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "amenity_parking_0": [0, 0, 0],
            "leisure_park_0": [0, 0, 0],
            "amenity_pub_0": [1, 0, 1],
        }
    )


@pytest.fixture  # type: ignore
def specified_features_expected_subcategories_embedding_df_concatenated_distance_1() -> (
    pd.DataFrame
):
    """
    Get expected ContextualCountEmbedder output for the case with specified features.

    Count with subcategories. Concatenated features, distance 1.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "amenity_parking_0": [0, 0, 0],
            "leisure_park_0": [0, 0, 0],
            "amenity_pub_0": [1, 0, 1],
            "amenity_parking_1": [0, 0, 0],
            "leisure_park_1": [0, 0, 0],
            "amenity_pub_1": [0.5, 1, 0.5],
        }
    )


@pytest.fixture  # type: ignore
def specified_features_expected_subcategories_embedding_df_concatenated_distance_2() -> (
    pd.DataFrame
):
    """
    Get expected ContextualCountEmbedder output for the case with specified features.

    Count with subcategories. Concatenated features, distance 2.
    """
    return _create_features_dataframe(
        {
            REGIONS_INDEX: ["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"],
            "amenity_parking_0": [0, 0, 0],
            "leisure_park_0": [0, 0, 0],
            "amenity_pub_0": [1, 0, 1],
            "amenity_parking_1": [0, 0, 0],
            "leisure_park_1": [0, 0, 0],
            "amenity_pub_1": [0.5, 1, 0.5],
            "amenity_parking_2": [0, 0, 0],
            "leisure_park_2": [0, 0, 0],
            "amenity_pub_2": [0, 0, 0],
        }
    )


@P.parameters(
    "expected_embedding_fixture",
    "neighbourhood_distance",
    "concatenate_features",
    "count_subcategories",
    "expected_features_fixture",
)  # type: ignore
@P.case(  # type: ignore
    "Squashed features, distance 0, without subcategories",
    "expected_embedding_df_squashed_distance_0",
    0,
    False,
    False,
    None,
)
@P.case(  # type: ignore
    "Squashed features, distance 1, without subcategories",
    "expected_embedding_df_squashed_distance_1",
    1,
    False,
    False,
    None,
)
@P.case(  # type: ignore
    "Squashed features, distance 2, without subcategories",
    "expected_embedding_df_squashed_distance_1",
    2,
    False,
    False,
    None,
)
@P.case(  # type: ignore
    "Concatenated features, distance 0, without subcategories",
    "expected_embedding_df_concatenated_distance_0",
    0,
    True,
    False,
    None,
)
@P.case(  # type: ignore
    "Concatenated features, distance 1, without subcategories",
    "expected_embedding_df_concatenated_distance_1",
    1,
    True,
    False,
    None,
)
@P.case(  # type: ignore
    "Concatenated features, distance 2, without subcategories",
    "expected_embedding_df_concatenated_distance_2",
    2,
    True,
    False,
    None,
)
@P.case(  # type: ignore
    "Squashed features, distance 0, witt subcategories",
    "expected_subcategories_embedding_df_squashed_distance_0",
    0,
    False,
    True,
    None,
)
@P.case(  # type: ignore
    "Squashed features, distance 1, with subcategories",
    "expected_subcategories_embedding_df_squashed_distance_1",
    1,
    False,
    True,
    None,
)
@P.case(  # type: ignore
    "Squashed features, distance 2, with subcategories",
    "expected_subcategories_embedding_df_squashed_distance_1",
    2,
    False,
    True,
    None,
)
@P.case(  # type: ignore
    "Concatenated features, distance 0, with subcategories",
    "expected_subcategories_embedding_df_concatenated_distance_0",
    0,
    True,
    True,
    None,
)
@P.case(  # type: ignore
    "Concatenated features, distance 1, with subcategories",
    "expected_subcategories_embedding_df_concatenated_distance_1",
    1,
    True,
    True,
    None,
)
@P.case(  # type: ignore
    "Concatenated features, distance 2, with subcategories",
    "expected_subcategories_embedding_df_concatenated_distance_2",
    2,
    True,
    True,
    None,
)
@P.case(  # type: ignore
    "Squashed features, distance 0, without subcategories, specified features",
    "specified_features_expected_embedding_df_squashed_empty",
    0,
    False,
    False,
    "expected_feature_names",
)
@P.case(  # type: ignore
    "Squashed features, distance 1, without subcategories, specified features",
    "specified_features_expected_embedding_df_squashed_empty",
    1,
    False,
    False,
    "expected_feature_names",
)
@P.case(  # type: ignore
    "Squashed features, distance 2, without subcategories, specified features",
    "specified_features_expected_embedding_df_squashed_empty",
    2,
    False,
    False,
    "expected_feature_names",
)
@P.case(  # type: ignore
    "Squashed features, distance 0, with subcategories, specified features",
    "specified_features_expected_subcategories_embedding_df_squashed_distance_0",
    0,
    False,
    True,
    "expected_feature_names",
)
@P.case(  # type: ignore
    "Squashed features, distance 1, with subcategories, specified features",
    "specified_features_expected_subcategories_embedding_df_squashed_distance_1",
    1,
    False,
    True,
    "expected_feature_names",
)
@P.case(  # type: ignore
    "Squashed features, distance 2, with subcategories, specified features",
    "specified_features_expected_subcategories_embedding_df_squashed_distance_1",
    2,
    False,
    True,
    "expected_feature_names",
)
@P.case(  # type: ignore
    "Concatenated features, distance 0, without subcategories, specified features",
    "specified_features_expected_embedding_df_concatenated_distance_0",
    0,
    True,
    False,
    "expected_feature_names",
)
@P.case(  # type: ignore
    "Concatenated features, distance 1, without subcategories, specified features",
    "specified_features_expected_embedding_df_concatenated_distance_1",
    1,
    True,
    False,
    "expected_feature_names",
)
@P.case(  # type: ignore
    "Concatenated features, distance 2, without subcategories, specified features",
    "specified_features_expected_embedding_df_concatenated_distance_2",
    2,
    True,
    False,
    "expected_feature_names",
)
@P.case(  # type: ignore
    "Concatenated features, distance 0, with subcategories, specified features",
    "specified_features_expected_subcategories_embedding_df_concatenated_distance_0",
    0,
    True,
    True,
    "expected_feature_names",
)
@P.case(  # type: ignore
    "Concatenated features, distance 1, with subcategories, specified features",
    "specified_features_expected_subcategories_embedding_df_concatenated_distance_1",
    1,
    True,
    True,
    "expected_feature_names",
)
@P.case(  # type: ignore
    "Concatenated features, distance 2, with subcategories, specified features",
    "specified_features_expected_subcategories_embedding_df_concatenated_distance_2",
    2,
    True,
    True,
    "expected_feature_names",
)
@P.case(  # type: ignore
    "Squashed features, distance 0, without subcategories, specified osm tags filter",
    "expected_embedding_df_squashed_distance_0",
    0,
    False,
    False,
    "osm_tags_filter",
)
@P.case(  # type: ignore
    "Squashed features, distance 1, without subcategories, specified osm tags filter",
    "expected_embedding_df_squashed_distance_1",
    1,
    False,
    False,
    "osm_tags_filter",
)
@P.case(  # type: ignore
    "Squashed features, distance 2, without subcategories, specified osm tags filter",
    "expected_embedding_df_squashed_distance_1",
    2,
    False,
    False,
    "osm_tags_filter",
)
@P.case(  # type: ignore
    "Squashed features, distance 0, with subcategories, specified osm tags filter",
    "specified_features_expected_subcategories_embedding_df_squashed_distance_0",
    0,
    False,
    True,
    "osm_tags_filter",
)
@P.case(  # type: ignore
    "Squashed features, distance 1, with subcategories, specified osm tags filter",
    "specified_features_expected_subcategories_embedding_df_squashed_distance_1",
    1,
    False,
    True,
    "osm_tags_filter",
)
@P.case(  # type: ignore
    "Squashed features, distance 2, with subcategories, specified osm tags filter",
    "specified_features_expected_subcategories_embedding_df_squashed_distance_1",
    2,
    False,
    True,
    "osm_tags_filter",
)
@P.case(  # type: ignore
    "Concatenated features, distance 0, without subcategories, specified osm tags filter",
    "expected_embedding_df_concatenated_distance_0",
    0,
    True,
    False,
    "osm_tags_filter",
)
@P.case(  # type: ignore
    "Concatenated features, distance 1, without subcategories, specified osm tags filter",
    "expected_embedding_df_concatenated_distance_1",
    1,
    True,
    False,
    "osm_tags_filter",
)
@P.case(  # type: ignore
    "Concatenated features, distance 2, without subcategories, specified osm tags filter",
    "expected_embedding_df_concatenated_distance_2",
    2,
    True,
    False,
    "osm_tags_filter",
)
@P.case(  # type: ignore
    "Concatenated features, distance 0, with subcategories, specified osm tags filter",
    "specified_features_expected_subcategories_embedding_df_concatenated_distance_0",
    0,
    True,
    True,
    "osm_tags_filter",
)
@P.case(  # type: ignore
    "Concatenated features, distance 1, with subcategories, specified osm tags filter",
    "specified_features_expected_subcategories_embedding_df_concatenated_distance_1",
    1,
    True,
    True,
    "osm_tags_filter",
)
@P.case(  # type: ignore
    "Concatenated features, distance 2, with subcategories, specified osm tags filter",
    "specified_features_expected_subcategories_embedding_df_concatenated_distance_2",
    2,
    True,
    True,
    "osm_tags_filter",
)
def test_correct_embedding(
    expected_embedding_fixture: str,
    neighbourhood_distance: int,
    concatenate_features: bool,
    count_subcategories: bool,
    expected_features_fixture: Union[str, None],
    request: Any,
) -> None:
    """Test if ContextualCountEmbedder returns correct result with different parameters."""
    expected_output_features = (
        None
        if expected_features_fixture is None
        else request.getfixturevalue(expected_features_fixture)
    )
    gdf_regions: gpd.GeoDataFrame = request.getfixturevalue("gdf_regions")
    gdf_features: gpd.GeoDataFrame = request.getfixturevalue("gdf_features")
    gdf_joint: gpd.GeoDataFrame = request.getfixturevalue("gdf_joint")

    embedder = ContextualCountEmbedder(
        neighbourhood=H3Neighbourhood(),
        neighbourhood_distance=neighbourhood_distance,
        expected_output_features=expected_output_features,
        count_subcategories=count_subcategories,
        concatenate_vectors=concatenate_features,
    )
    embedding_df = embedder.transform(
        regions_gdf=gdf_regions, features_gdf=gdf_features, joint_gdf=gdf_joint
    )

    expected_result_df = request.getfixturevalue(expected_embedding_fixture)
    assert_frame_equal(
        embedding_df.sort_index(axis=1),
        expected_result_df.sort_index(axis=1),
        check_dtype=False,
    )


def test_negative_nighbourhood_distance() -> None:
    """Test checks if negative neighbouthood distance is disallowed."""
    with pytest.raises(ValueError):
        ContextualCountEmbedder(neighbourhood=H3Neighbourhood(), neighbourhood_distance=-1)


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
@pytest.mark.parametrize("concatenate_features", [False, True])  # type: ignore
@pytest.mark.parametrize("count_subcategories", [False, True])  # type: ignore
@pytest.mark.parametrize("neighbourhood_distance", [0, 1, 2])  # type: ignore
def test_empty(
    regions_fixture: str,
    features_fixture: str,
    joint_fixture: str,
    concatenate_features: bool,
    count_subcategories: bool,
    neighbourhood_distance: int,
    expected_features_fixture: Union[str, None],
    expectation: Any,
    request: Any,
) -> None:
    """Test ContextualCountEmbedder handling of empty input data frames."""
    expected_output_features = (
        None
        if expected_features_fixture is None
        else request.getfixturevalue(expected_features_fixture)
    )
    embedder = ContextualCountEmbedder(
        neighbourhood=H3Neighbourhood(),
        neighbourhood_distance=neighbourhood_distance,
        expected_output_features=expected_output_features,
        count_subcategories=count_subcategories,
        concatenate_vectors=concatenate_features,
    )
    gdf_regions: gpd.GeoDataFrame = request.getfixturevalue(regions_fixture)
    gdf_features: gpd.GeoDataFrame = request.getfixturevalue(features_fixture)
    gdf_joint: gpd.GeoDataFrame = request.getfixturevalue(joint_fixture)

    with expectation:
        embedding = embedder.transform(gdf_regions, gdf_features, gdf_joint)
        assert len(embedding) == len(gdf_regions)
        assert embedding.index.name == gdf_regions.index.name
        if expected_output_features:
            assert len(embedding.columns) == len(expected_output_features) * (
                1 if not concatenate_features else 1 + neighbourhood_distance
            )

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
@pytest.mark.parametrize("concatenate_features", [False, True])  # type: ignore
@pytest.mark.parametrize("count_subcategories", [False, True])  # type: ignore
@pytest.mark.parametrize("neighbourhood_distance", [0, 1, 2])  # type: ignore
def test_incorrect_indexes(
    regions_fixture: str,
    features_fixture: str,
    joint_fixture: str,
    concatenate_features: bool,
    count_subcategories: bool,
    neighbourhood_distance: int,
    expectation: Any,
    request: Any,
) -> None:
    """Test if cannot embed with incorrect dataframe indexes."""
    regions_gdf: gpd.GeoDataFrame = request.getfixturevalue(regions_fixture)
    features_gdf: gpd.GeoDataFrame = request.getfixturevalue(features_fixture)
    joint_gdf: gpd.GeoDataFrame = request.getfixturevalue(joint_fixture)

    with expectation:
        ContextualCountEmbedder(
            neighbourhood=H3Neighbourhood(),
            count_subcategories=count_subcategories,
            concatenate_vectors=concatenate_features,
            neighbourhood_distance=neighbourhood_distance,
        ).transform(regions_gdf=regions_gdf, features_gdf=features_gdf, joint_gdf=joint_gdf)


def test_bigger_example() -> None:
    """Test bigger example to get multiprocessing in action."""
    geometry = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (7.416769421059001, 43.7346112362936),
                    (7.416769421059001, 43.730681304758946),
                    (7.4218262821731, 43.730681304758946),
                    (7.4218262821731, 43.7346112362936),
                ]
            )
        ],
        crs=WGS84_CRS,
    )

    regions = H3Regionalizer(resolution=13).transform(geometry)
    features = OSMPbfLoader(
        pbf_file=Path(__file__).parent.parent
        / "loaders"
        / "osm_loaders"
        / "test_files"
        / "monaco.osm.pbf"
    ).load(area=regions, tags=GEOFABRIK_LAYERS)
    joint = IntersectionJoiner().transform(regions=regions, features=features)
    embeddings = ContextualCountEmbedder(
        neighbourhood=H3Neighbourhood(),
        neighbourhood_distance=10,
        expected_output_features=GEOFABRIK_LAYERS,
    ).transform(regions_gdf=regions, features_gdf=features, joint_gdf=joint)

    assert len(embeddings) == len(
        regions
    ), f"Mismatched number of rows ({len(embeddings)}, {len(regions)})"
