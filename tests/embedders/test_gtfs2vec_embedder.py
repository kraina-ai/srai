"""GTFS2VecEmbedder tests."""

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture
from pytorch_lightning import seed_everything
from shapely.geometry import Polygon

from srai.constants import REGIONS_INDEX
from srai.embedders import GTFS2VecEmbedder
from srai.exceptions import ModelNotFitException


@pytest.fixture  # type: ignore
def gtfs2vec_features() -> gpd.GeoDataFrame:
    """Get GTFS2Vec features GeoDataFrame."""
    features_gdf = gpd.GeoDataFrame(
        {
            "trips_at_6": [1, 0, 0],
            "trips_at_7": [1, 1, 0],
            "trips_at_8": [0, 0, 1],
            "directions_at_6": [
                {"A", "A1"},
                {"B", "B1"},
                {"C"},
            ],
        },
        geometry=gpd.points_from_xy([1, 2, 5], [1, 2, 2]),
        index=[1, 2, 3],
    )
    features_gdf.index.name = "stop_id"
    return features_gdf


@pytest.fixture  # type: ignore
def gtfs2vec_regions() -> gpd.GeoDataFrame:
    """Get GTFS2Vec regions GeoDataFrame."""
    regions_gdf = gpd.GeoDataFrame(
        {
            REGIONS_INDEX: ["ff1", "ff2", "ff3"],
        },
        geometry=[
            Polygon([(0, 0), (0, 3), (3, 3), (3, 0)]),
            Polygon([(4, 0), (4, 3), (7, 3), (7, 0)]),
            Polygon([(8, 0), (8, 3), (11, 3), (11, 0)]),
        ],
    ).set_index(REGIONS_INDEX)
    return regions_gdf


@pytest.fixture  # type: ignore
def gtfs2vec_joint() -> gpd.GeoDataFrame:
    """Get GTFS2Vec joint GeoDataFrame."""
    joint_gdf = gpd.GeoDataFrame()
    joint_gdf.index = pd.MultiIndex.from_tuples(
        [("ff1", 1), ("ff1", 2), ("ff2", 3)],
        names=[REGIONS_INDEX, "stop_id"],
    )
    return joint_gdf


@pytest.fixture  # type: ignore
def features_not_embedded() -> pd.DataFrame:
    """Get features not embedded."""
    return pd.DataFrame(
        {
            "trips_at_6": [0.5, 0.0, 0.0],
            "trips_at_7": [1.0, 0.0, 0.0],
            "trips_at_8": [0.0, 0.5, 0.0],
            "directions_at_6": [1.0, 0.25, 0.0],
            REGIONS_INDEX: ["ff1", "ff2", "ff3"],
        },
    ).set_index(REGIONS_INDEX)


@pytest.fixture  # type: ignore
def features_embedded() -> pd.DataFrame:
    """Get features embedded."""
    embeddings = np.array(
        [
            [0.642446, 0.001230, -0.038590],
            [0.960088, 0.433973, 0.401301],
            [0.224951, -0.135296, -0.157667],
            [0.782488, 0.212641, 0.177253],
        ],
        dtype=np.float32,
    )
    features = pd.DataFrame(embeddings.T)
    features.index = pd.Index(["ff1", "ff2", "ff3"], name=REGIONS_INDEX)
    features.columns = pd.RangeIndex(0, 4, 1)
    return features


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
    regions_fixture: str,
    features_fixture: str,
    joint_fixture: str,
    expectation: Any,
    request: Any,
) -> None:
    """Test GTFS2VecEmbedder with incorrect input dataframes."""
    regions_gdf = request.getfixturevalue(regions_fixture)
    features_gdf = request.getfixturevalue(features_fixture)
    joint_gdf = request.getfixturevalue(joint_fixture)

    with expectation:
        embedder = GTFS2VecEmbedder(skip_autoencoder=True)
        embedder.transform(regions_gdf, features_gdf, joint_gdf)

    with expectation:
        embedder = GTFS2VecEmbedder(skip_autoencoder=True)
        embedder.fit(regions_gdf, features_gdf, joint_gdf)

    with expectation:
        embedder = GTFS2VecEmbedder(skip_autoencoder=True)
        embedder.fit_transform(regions_gdf, features_gdf, joint_gdf)


def test_transform_with_unfit_model(
    gtfs2vec_regions: gpd.GeoDataFrame,
    gtfs2vec_features: gpd.GeoDataFrame,
    gtfs2vec_joint: gpd.GeoDataFrame,
) -> None:
    """Test GTFS2VecEmbedder transform with unfitted model."""
    embedder = GTFS2VecEmbedder(skip_autoencoder=False)
    with pytest.raises(ModelNotFitException):
        embedder.transform(gtfs2vec_regions, gtfs2vec_features, gtfs2vec_joint)


def test_transform_with_mismatched_features_count(
    gtfs2vec_regions: gpd.GeoDataFrame,
    gtfs2vec_features: gpd.GeoDataFrame,
    gtfs2vec_joint: gpd.GeoDataFrame,
    mocker: MockerFixture,
) -> None:
    """Test GTFS2VecEmbedder transform with mismatched features count."""
    embedder = GTFS2VecEmbedder(skip_autoencoder=False)
    mock_model = mocker.MagicMock()
    mock_model.configure_mock(n_features=42)
    embedder._model = mock_model

    with pytest.raises(ValueError):
        embedder.transform(gtfs2vec_regions, gtfs2vec_features, gtfs2vec_joint)


@pytest.mark.parametrize(  # type: ignore
    "regions_fixture,features_fixture,joint_fixture,embedding",
    [
        (
            "gtfs2vec_regions",
            "gtfs2vec_features",
            "gtfs2vec_joint",
            True,
        ),
        (
            "gtfs2vec_regions",
            "gtfs2vec_features",
            "gtfs2vec_joint",
            False,
        ),
    ],
)
def test_embedder(
    regions_fixture: str,
    features_fixture: str,
    joint_fixture: str,
    embedding: bool,
    request: Any,
) -> None:
    """Test GTFS2VecEmbedder results."""
    regions_gdf = request.getfixturevalue(regions_fixture)
    features_gdf = request.getfixturevalue(features_fixture)
    joint_gdf = request.getfixturevalue(joint_fixture)

    if embedding:
        expected_features = request.getfixturevalue("features_embedded")
    else:
        expected_features = request.getfixturevalue("features_not_embedded")

    embedder = GTFS2VecEmbedder(hidden_size=2, embedding_size=4, skip_autoencoder=not embedding)

    seed_everything(42)
    embedder.fit(regions_gdf, features_gdf, joint_gdf)
    features_embedded = embedder.transform(regions_gdf, features_gdf, joint_gdf)

    pd.testing.assert_frame_equal(features_embedded, expected_features, atol=1e-3)

    seed_everything(42)
    features_embedded = embedder.fit_transform(regions_gdf, features_gdf, joint_gdf)

    pd.testing.assert_frame_equal(features_embedded, expected_features, atol=1e-3)
