"""Tests for OSMPbfLoader."""

from pathlib import Path
from typing import Union
from unittest import TestCase

import geopandas as gpd
import pytest
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import (
    GEOFABRIK_LAYERS,
    HEX2VEC_FILTER,
    GroupedOsmTagsFilter,
    OsmTagsFilter,
)

ut = TestCase()


@pytest.mark.parametrize("explode_tags", [True, False])  # type: ignore
@pytest.mark.parametrize("keep_all_tags", [True, False])  # type: ignore
def test_pbf_to_geoparquet_parsing(explode_tags: bool, keep_all_tags: bool):
    """Test if pbf to geoparquet conversion works."""
    pbf_file = Path(__file__).parent / "test_files" / "monaco.osm.pbf"
    OSMPbfLoader(pbf_file=pbf_file).load_to_geoparquet(
        area=Polygon(
            [
                (7.416769421059001, 43.7346112362936),
                (7.416769421059001, 43.730681304758946),
                (7.4218262821731, 43.730681304758946),
                (7.4218262821731, 43.7346112362936),
            ]
        ),
        tags=GEOFABRIK_LAYERS,
        ignore_cache=True,
        explode_tags=explode_tags,
        keep_all_tags=keep_all_tags,
    )


@pytest.mark.parametrize(  # type: ignore
    "test_geometries,pbf_file,query,pbf_source,expected_result_length,expected_features_columns_length,expected_features_columns_names",
    [
        (
            [
                Polygon(
                    [
                        (7.416769421059001, 43.7346112362936),
                        (7.416769421059001, 43.730681304758946),
                        (7.4218262821731, 43.730681304758946),
                        (7.4218262821731, 43.7346112362936),
                    ]
                )
            ],
            Path(__file__).parent / "test_files" / "monaco.osm.pbf",
            HEX2VEC_FILTER,
            "geofabrik",
            403,
            12,
            [
                "amenity",
                "building",
                "healthcare",
                "historic",
                "landuse",
                "leisure",
                "natural",
                "office",
                "shop",
                "sport",
                "tourism",
                "water",
            ],
        ),
        (
            [
                Polygon(
                    [
                        (7.416769421059001, 43.7346112362936),
                        (7.416769421059001, 43.730681304758946),
                        (7.4218262821731, 43.730681304758946),
                        (7.4218262821731, 43.7346112362936),
                    ]
                )
            ],
            Path(__file__).parent / "test_files" / "monaco.osm.pbf",
            GEOFABRIK_LAYERS,
            "geofabrik",
            958,
            23,
            [
                "accommodation",
                "buildings",
                "catering",
                "education",
                "fuel_parking",
                "health",
                "highway_links",
                "landuse",
                "leisure",
                "major_roads",
                "minor_roads",
                "miscpoi",
                "money",
                "natural",
                "paths_unsuitable_for_cars",
                "public",
                "shopping",
                "tourism",
                "traffic",
                "transport",
                "very_small_roads",
                "water",
                "water_traffic",
            ],
        ),
    ],
)
def test_osm_pbf_loader(
    test_geometries: list[BaseGeometry],
    pbf_file: Path,
    query: Union[OsmTagsFilter, GroupedOsmTagsFilter],
    pbf_source: str,
    expected_result_length: int,
    expected_features_columns_length: int,
    expected_features_columns_names: list[str],
):
    """Test `OSMPbfLoader.load()`."""
    download_directory = Path(__file__).parent / "test_files"
    area = gpd.GeoDataFrame(
        geometry=test_geometries,
        index=gpd.pd.Index(name=REGIONS_INDEX, data=list(range(len(test_geometries)))),
        crs=WGS84_CRS,
    )

    loader = OSMPbfLoader(
        pbf_file=pbf_file,
        download_directory=download_directory,
        download_source=pbf_source,
    )
    result = loader.load(area, tags=query, ignore_cache=True)

    assert (
        len(result) == expected_result_length
    ), f"Mismatched result length ({len(result)}, {expected_result_length})"
    assert (
        len(result.columns) == expected_features_columns_length + 1
    ), f"Mismatched columns length ({len(result.columns)}, {expected_features_columns_length + 1})"
    ut.assertCountEqual(
        result.columns,
        expected_features_columns_names + [GEOMETRY_COLUMN],
        "Mismatched columns names.",
    )
