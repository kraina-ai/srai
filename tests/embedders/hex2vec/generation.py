"""Test case generation for Hex2VecEmbedder."""

from pathlib import Path
from typing import Optional

import geopandas as gpd
import h3
from h3ronpy.pandas.vector import cells_to_polygons
from pytorch_lightning import seed_everything

from srai.constants import REGIONS_INDEX, WGS84_CRS
from srai.embedders.hex2vec.embedder import Hex2VecEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import OsmTagsFilter
from srai.neighbourhoods import H3Neighbourhood
from srai.regionalizers import geocode_to_region_gdf
from tests.embedders.hex2vec.constants import ENCODER_SIZES, TRAINER_KWARGS


def generate_test_case(
    test_case_name: str,
    geocoding_name: str,
    root_region_index: str,
    h3_res: int,
    radius: int,
    seed: int,
    tags: Optional[OsmTagsFilter] = None,
) -> None:
    """Generate test case for Hex2VecEmbedder."""
    if tags is None:
        tags = {"leisure": "park", "amenity": "restaurant"}
    neighbourhood = H3Neighbourhood()
    regions_indexes = neighbourhood.get_neighbours_up_to_distance(root_region_index, radius)
    regions_indexes.add(root_region_index)
    regions_indexes = list(regions_indexes)  # type: ignore

    geoms = cells_to_polygons([h3.str_to_int(r) for r in regions_indexes]).values
    regions_gdf = gpd.GeoDataFrame(index=regions_indexes, geometry=geoms, crs=WGS84_CRS)
    regions_gdf.index.name = REGIONS_INDEX

    area_gdf = geocode_to_region_gdf(geocoding_name)
    loader = OSMPbfLoader()
    features_all = loader.load(area_gdf, tags=tags)

    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(regions_gdf, features_all)
    features_gdf = features_all[
        features_all.index.isin(joint_gdf.index.get_level_values("feature_id"))
    ]
    joint_gdf = joiner.transform(regions_gdf, features_gdf)

    neighbourhood = H3Neighbourhood(regions_gdf)
    seed_everything(seed)
    embedder = Hex2VecEmbedder(encoder_sizes=ENCODER_SIZES)
    results_df = embedder.fit_transform(
        regions_gdf, features_gdf, joint_gdf, neighbourhood, trainer_kwargs=TRAINER_KWARGS
    )
    results_df.columns = results_df.columns.astype(str)

    files_prefix = test_case_name

    output_path = Path(__file__).parent / "test_files"
    regions_gdf.to_parquet(output_path / f"{files_prefix}_regions.parquet")
    features_gdf.to_parquet(output_path / f"{files_prefix}_features.parquet")
    joint_gdf.to_parquet(output_path / f"{files_prefix}_joint.parquet")
    results_df.to_parquet(output_path / f"{files_prefix}_result.parquet")


if __name__ == "__main__":
    path = Path(__file__)

    from constants import PREDEFINED_TEST_CASES

    for test_case in PREDEFINED_TEST_CASES:
        generate_test_case(**test_case)
