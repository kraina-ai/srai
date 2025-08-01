"""Test case generation for S2VecEmbedder."""

import os
from pathlib import Path
from typing import Optional

import geopandas as gpd
import torch
from pytorch_lightning import seed_everything
from s2.s2 import s2_to_geo_boundary
from shapely.geometry import Polygon

from srai.constants import REGIONS_INDEX, WGS84_CRS
from srai.embedders.s2vec.embedder import S2VecEmbedder
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import OsmTagsFilter
from tests.embedders.conftest import TRAINER_KWARGS


def generate_test_case(
    test_case_name: str,
    root_regions_tokens: list[str],
    seed: int,
    img_res: int,
    patch_res: int,
    num_heads: int,
    encoder_layers: int,
    decoder_layers: int,
    embedding_dim: int,
    decoder_dim: int,
    mask_ratio: float,
    tags: Optional[OsmTagsFilter] = None,
) -> None:
    """Generate test case for S2VecEmbedder."""
    seed_everything(seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)

    if tags is None:
        tags = {"leisure": "park", "amenity": "restaurant"}

    geoms = [
        Polygon(s2_to_geo_boundary(token, geo_json_conformant=True))
        for token in root_regions_tokens
    ]
    regions_gdf = gpd.GeoDataFrame(index=root_regions_tokens, geometry=geoms, crs=WGS84_CRS)
    regions_gdf.index.name = REGIONS_INDEX

    loader = OSMPbfLoader()
    features_gdf = loader.load(regions_gdf, tags)

    embedder = S2VecEmbedder(
        target_features=[f"{st}_{t}" for st in tags for t in tags[st]],  # type: ignore
        batch_size=10,
        img_res=img_res,
        patch_res=patch_res,
        num_heads=num_heads,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        embedding_dim=embedding_dim,
        decoder_dim=decoder_dim,
        mask_ratio=mask_ratio,
    )

    results_df = embedder.fit_transform(
        regions_gdf=regions_gdf,
        features_gdf=features_gdf,
        trainer_kwargs=TRAINER_KWARGS,
        learning_rate=0.001,
    )

    results_df.columns = results_df.columns.astype(str)

    files_prefix = test_case_name

    output_path = Path(__file__).parent / "test_files"
    regions_gdf.to_parquet(
        output_path / f"{files_prefix}_regions.parquet",
    )
    features_gdf.to_parquet(output_path / f"{files_prefix}_features.parquet", compression="gzip")
    results_df.to_parquet(output_path / f"{files_prefix}_result.parquet")


if __name__ == "__main__":
    from constants import PREDEFINED_TEST_CASES

    for test_case in PREDEFINED_TEST_CASES:
        generate_test_case(**test_case)
