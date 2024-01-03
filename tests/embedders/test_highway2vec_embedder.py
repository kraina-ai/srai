"""Highway2VecEmbedder tests."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pytorch_lightning import seed_everything
from shapely.geometry import LineString, Polygon

from srai.constants import FEATURES_INDEX, REGIONS_INDEX
from srai.embedders import Highway2VecEmbedder
from srai.exceptions import ModelNotFitException


@pytest.fixture  # type: ignore
def highway2vec_features() -> gpd.GeoDataFrame:
    """Get Highway2Vec features GeoDataFrame."""
    features_gdf = gpd.GeoDataFrame(
        {
            "oneway": [1, 0, 0],
            "lanes-1": [1, 1, 0],
            "lanes-2": [0, 0, 1],
            "bicycle-destination": [0, 0, 0],
        },
        geometry=[
            LineString([(1, 1), (2, 2), (5, 5)]),
            LineString([(1, 2), (2, 3), (3, 4)]),
            LineString([(5, 5), (6, 1), (10, 2)]),
        ],
        index=[1, 2, 3],
    )
    features_gdf.index.name = FEATURES_INDEX
    return features_gdf


@pytest.fixture  # type: ignore
def highway2vec_regions() -> gpd.GeoDataFrame:
    """Get Highway2Vec regions GeoDataFrame."""
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
def highway2vec_joint() -> gpd.GeoDataFrame:
    """Get Highway2Vec joint GeoDataFrame."""
    joint_gdf = gpd.GeoDataFrame()
    joint_gdf.index = pd.MultiIndex.from_tuples(
        [("ff1", 1), ("ff1", 2), ("ff2", 3), ("ff3", 3)],
        names=[REGIONS_INDEX, FEATURES_INDEX],
    )
    return joint_gdf


@pytest.fixture  # type: ignore
def highway2vec_embeddings() -> pd.DataFrame:
    """Get features embedded."""
    embeddings = np.array(
        [
            [-0.019932, 0.027169, -0.031977, -0.000582],
            [0.236779, -0.258233, 0.051689, 0.098861],
            [0.236779, -0.258233, 0.051689, 0.098861],
        ],
        dtype=np.float32,
    )

    features = pd.DataFrame(
        embeddings,
        index=pd.Index(["ff1", "ff2", "ff3"], name=REGIONS_INDEX),
        columns=pd.RangeIndex(0, 4, 1),
    )
    return features


def test_transform_with_unfit_model(
    highway2vec_regions: gpd.GeoDataFrame,
    highway2vec_features: gpd.GeoDataFrame,
    highway2vec_joint: gpd.GeoDataFrame,
) -> None:
    """Test Highway2VecEmbedder transform with unfitted model."""
    embedder = Highway2VecEmbedder()
    with pytest.raises(ModelNotFitException):
        embedder.transform(highway2vec_regions, highway2vec_features, highway2vec_joint)


def test_embedder_on_correct_input(
    highway2vec_regions: gpd.GeoDataFrame,
    highway2vec_features: gpd.GeoDataFrame,
    highway2vec_joint: gpd.GeoDataFrame,
    highway2vec_embeddings: gpd.GeoDataFrame,
) -> None:
    """Test Highway2VecEmbedder results."""
    embedder = Highway2VecEmbedder(embedding_size=4)

    seed_everything(42)
    embedder.fit(highway2vec_regions, highway2vec_features, highway2vec_joint)
    features_embedded = embedder.transform(
        highway2vec_regions, highway2vec_features, highway2vec_joint
    )
    pd.testing.assert_frame_equal(features_embedded, highway2vec_embeddings, atol=1e-3)

    seed_everything(42)
    features_embedded = embedder.fit_transform(
        highway2vec_regions, highway2vec_features, highway2vec_joint
    )
    pd.testing.assert_frame_equal(features_embedded, highway2vec_embeddings, atol=1e-3)
