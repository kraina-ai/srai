"""Test case generation for GeoVexEmbedder."""

import os
from pathlib import Path
from typing import Optional, cast

import geopandas as gpd
import h3
import torch
from h3ronpy.pandas.vector import cells_to_polygons
from pytorch_lightning import seed_everything

from srai.constants import REGIONS_INDEX, WGS84_CRS
from srai.embedders.geovex.embedder import GeoVexEmbedder
from srai.embedders.geovex.model import GeoVexModel
from srai.h3 import ring_buffer_h3_regions_gdf
from srai.joiners import IntersectionJoiner
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER, OsmTagsFilter
from srai.neighbourhoods import H3Neighbourhood
from tests.embedders.geovex.constants import EMBEDDING_SIZE, TRAINER_KWARGS


def generate_test_case(
    test_case_name: str,
    root_region_index: str,
    region_gen_radius: int,
    model_radius: int,
    seed: int,
    tags: Optional[OsmTagsFilter] = None,
    convolutional_layer_size: int = 256,
    num_layers: int = 1,
) -> None:
    """Generate test case for GeoVexEmbedder."""
    seed_everything(seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)

    if tags is None:
        tags = HEX2VEC_FILTER

    neighbourhood = H3Neighbourhood(include_center=True)
    regions_indexes = list(
        neighbourhood.get_neighbours_up_to_distance(root_region_index, region_gen_radius)
    )

    geoms = cells_to_polygons([h3.str_to_int(r) for r in regions_indexes]).values
    regions_gdf = gpd.GeoDataFrame(index=regions_indexes, geometry=geoms, crs=WGS84_CRS)
    regions_gdf.index.name = REGIONS_INDEX

    regions_gdf = ring_buffer_h3_regions_gdf(regions_gdf, distance=model_radius).sort_index()
    buffered_geometry = regions_gdf.unary_union

    loader = OSMPbfLoader()
    features_gdf = loader.load(buffered_geometry, tags)

    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(regions_gdf, features_gdf)

    neighbourhood = H3Neighbourhood(regions_gdf)

    embedder = GeoVexEmbedder(
        target_features=[f"{st}_{t}" for st in tags for t in tags[st]],  # type: ignore
        batch_size=10,
        neighbourhood_radius=model_radius,
        embedding_size=EMBEDDING_SIZE,
        convolutional_layers=num_layers,
        convolutional_layer_size=convolutional_layer_size,
    )

    counts_df, _, _ = embedder._prepare_dataset(
        regions_gdf, features_gdf, joint_gdf, neighbourhood, embedder._batch_size, shuffle=True
    )

    embedder._prepare_model(counts_df, 0.001)

    for _, param in cast(GeoVexModel, embedder._model).named_parameters():
        param.data.fill_(0.01)

    results_df = embedder.fit_transform(
        regions_gdf=regions_gdf,
        features_gdf=features_gdf,
        joint_gdf=joint_gdf,
        neighbourhood=neighbourhood,
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
    joint_gdf.to_parquet(output_path / f"{files_prefix}_joint.parquet")
    results_df.to_parquet(output_path / f"{files_prefix}_result.parquet")


def generate_test_case_batches(
    test_case_name: str,
    root_region_index: str,
    region_gen_radius: int,
    model_radius: int,
    seed: int,
    tags: Optional[OsmTagsFilter] = None,
    convolutional_layer_size: int = 256,
    num_layers: int = 1,
) -> None:
    """Generate test case for GeoVexEmbedder."""
    seed_everything(seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)

    if tags is None:
        tags = HEX2VEC_FILTER

    neighbourhood = H3Neighbourhood(include_center=True)
    regions_indexes = list(
        neighbourhood.get_neighbours_up_to_distance(root_region_index, region_gen_radius)
    )

    geoms = cells_to_polygons([h3.str_to_int(r) for r in regions_indexes]).values
    regions_gdf = gpd.GeoDataFrame(index=regions_indexes, geometry=geoms, crs=WGS84_CRS)
    regions_gdf.index.name = REGIONS_INDEX

    regions_gdf = ring_buffer_h3_regions_gdf(regions_gdf, distance=model_radius).sort_index()
    buffered_geometry = regions_gdf.unary_union

    loader = OSMPbfLoader()
    features_gdf = loader.load(buffered_geometry, tags)

    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(regions_gdf, features_gdf)

    neighbourhood = H3Neighbourhood(regions_gdf)

    embedder = GeoVexEmbedder(
        target_features=[f"{st}_{t}" for st in tags for t in tags[st]],  # type: ignore
        batch_size=10,
        neighbourhood_radius=model_radius,
        embedding_size=EMBEDDING_SIZE,
        convolutional_layers=num_layers,
        convolutional_layer_size=convolutional_layer_size,
    )

    counts_df, dataloader, _ = embedder._prepare_dataset(
        regions_gdf, features_gdf, joint_gdf, neighbourhood, embedder._batch_size, shuffle=True
    )

    embedder._prepare_model(counts_df, 0.001)

    for _, param in cast(GeoVexModel, embedder._model).named_parameters():
        param.data.fill_(0.01)

    output_path = Path(__file__).parent / "test_files"
    files_prefix = test_case_name

    for i, batch in enumerate(dataloader):
        torch.save(batch, output_path / f"{files_prefix}_batch_{i}.pt")

        encoder_forward_tensor = cast(GeoVexModel, embedder._model).encoder.forward(batch)
        torch.save(encoder_forward_tensor, output_path / f"{files_prefix}_encoder_forward_{i}.pt")

        forward_tensor = cast(GeoVexModel, embedder._model).forward(batch)
        torch.save(forward_tensor, output_path / f"{files_prefix}_forward_{i}.pt")

        loss_tensor = cast(GeoVexModel, embedder._model).training_step(batch, i)
        torch.save(loss_tensor, output_path / f"{files_prefix}_loss_{i}.pt")


if __name__ == "__main__":
    from constants import PREDEFINED_TEST_CASES

    for test_case in PREDEFINED_TEST_CASES:
        generate_test_case(**test_case)
        generate_test_case_batches(**test_case)
