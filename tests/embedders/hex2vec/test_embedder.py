"""Hex2VecEmbedder tests."""
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pandas.testing import assert_frame_equal
from pytorch_lightning import seed_everything

from srai.embedders.hex2vec.embedder import Hex2VecEmbedder
from srai.neighbourhoods import H3Neighbourhood
from tests.embedders.hex2vec.constants import ENCODER_SIZES, PREDEFINED_TEST_CASES, TRAINER_KWARGS


def test_embedder() -> None:
    """Test Hex2VecEmbedder on predefined test cases."""
    test_files_path = Path(__file__).parent / "test_files"
    for test_case in PREDEFINED_TEST_CASES:
        name = test_case["test_case_name"]
        seed = test_case["seed"]
        print(name, seed)

        expected = pd.read_parquet(test_files_path / f"{name}_result.parquet")
        regions_gdf = gpd.read_parquet(test_files_path / f"{name}_regions.parquet")
        features_gdf = gpd.read_parquet(test_files_path / f"{name}_features.parquet")
        joint_gdf = gpd.read_parquet(test_files_path / f"{name}_joint.parquet")
        seed_everything(seed)

        neighbourhood = H3Neighbourhood(regions_gdf)
        embedder = Hex2VecEmbedder(encoder_sizes=ENCODER_SIZES)
        result_df = embedder.fit_transform(
            regions_gdf, features_gdf, joint_gdf, neighbourhood, trainer_kwargs=TRAINER_KWARGS
        )
        result_df.columns = result_df.columns.astype(str)
        print(result_df.head())
        print(expected.head())
        assert_frame_equal(result_df, expected, atol=1e-1)
