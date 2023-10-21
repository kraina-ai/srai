"""Tests for OSMPbfLoader."""
from pathlib import Path
from typing import List, Union
from unittest import TestCase

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import (
    BASE_OSM_GROUPS_FILTER,
    HEX2VEC_FILTER,
    GroupedOsmTagsFilter,
    OsmTagsFilter,
)
from srai.loaders.osm_loaders.pbf_file_downloader import PbfFileDownloader, PbfSourceLiteral
from srai.loaders.osm_loaders.pbf_file_handler import PbfFileHandler

ut = TestCase()


@pytest.mark.parametrize(  # type: ignore
    "test_polygon",
    [
        (Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])),
        (Polygon([(3, 5), (3, 10), (7, 10), (7, 5)])),
        (Polygon([(-30, -30), (-30, 30), (30, 30), (30, -30)])),
        (
            Polygon(
                shell=[
                    (-1, 0),
                    (0, 0.5),
                    (1, 0),
                    (1, 1),
                    (0, 1),
                ],
                holes=[
                    [
                        (0.8, 0.9),
                        (0.9, 0.55),
                        (0.8, 0.3),
                        (0.5, 0.4),
                    ]
                ],
            )
        ),
        (
            gpd.read_file(
                filename=Path(__file__).parent / "test_files" / "poland.geojson"
            ).geometry[0]
        ),
        (
            gpd.read_file(
                filename=Path(__file__).parent
                / "test_files"
                / "south_africa_without_islands.geojson"
            ).geometry[0]
        ),
    ],
)
def test_geometry_preparing(test_polygon: BaseGeometry):
    """Test proper geometry preparing in `PbfFileDownloader`."""
    downloader = PbfFileDownloader(
        download_source="protomaps",
        switch_to_geofabrik_on_error=False,
    )
    prepared_polygon = downloader._prepare_polygon_for_download(test_polygon)

    assert len(prepared_polygon.exterior.coords) <= 1000
    assert len(prepared_polygon.interiors) == 0
    assert test_polygon.covered_by(prepared_polygon)


@pytest.mark.parametrize(
    "test_geometry", [Point(1, 1), LineString([(1, 1), (0, 1)])]
)  # type: ignore
@pytest.mark.parametrize(
    "pbf_source", ["geofabrik", "openstreetmap_fr", "protomaps"]
)  # type: ignore
def test_disallow_non_polygons(test_geometry: BaseGeometry, pbf_source: PbfSourceLiteral) -> None:
    """Test if `PbfFileDownloader` disallows non polygon geometries."""
    with pytest.raises(ValueError):
        regions_gdf = gpd.GeoDataFrame(
            geometry=[test_geometry],
            index=gpd.pd.Index(name=REGIONS_INDEX, data=[1]),
            crs=WGS84_CRS,
        )
        downloader = PbfFileDownloader(
            download_source=pbf_source,
            switch_to_geofabrik_on_error=False,
        )
        downloader.download_pbf_files_for_regions_gdf(regions_gdf)


@pytest.mark.parametrize(  # type: ignore
    "test_polygon,test_file_names",
    [
        (
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            ["eb2848d259345ce7dfe8af34fd1ab24503bb0b952e04e872c87c55550fa50fbf.osm.pbf"],
        ),
        (
            Polygon(
                shell=[
                    (-1, 0),
                    (0, 0.5),
                    (1, 0),
                    (1, 1),
                    (0, 1),
                ],
                holes=[
                    [
                        (0.8, 0.9),
                        (0.9, 0.55),
                        (0.8, 0.3),
                        (0.5, 0.4),
                    ]
                ],
            ),
            ["529cdcbb7a3cc103658ef31b39bed24984e421127d319c867edf2f86ff3bb098.osm.pbf"],
        ),
        (
            MultiPolygon(
                [
                    (
                        [
                            (-13, -8),
                            (-13, -9),
                            (-12, -9),
                            (-12, -8),
                        ],
                        (),
                    ),
                    (
                        [(-0.25, 0), (0.25, 0), (0, 0.2)],
                        (),
                    ),
                ]
            ),
            [
                "7a0163cb721992d6219d486b3d29517d06aa0db19dd7be049f4f1fabf6146073.osm.pbf",
                "aa756ad3a961ba6d9da46c712b0d979d0c7d4768641ceea7409b287e2d18a48f.osm.pbf",
            ],
        ),
    ],
)
def test_pbf_downloading(test_polygon: BaseGeometry, test_file_names: List[str]):
    """Test proper files downloading in `PbfFileDownloader`."""
    regions_gdf = gpd.GeoDataFrame(
        geometry=[test_polygon],
        index=gpd.pd.Index(name=REGIONS_INDEX, data=[1]),
        crs=WGS84_CRS,
    )
    downloader = PbfFileDownloader(
        download_source="protomaps",
        download_directory=Path(__file__).parent / "test_files",
        switch_to_geofabrik_on_error=False,
    )
    files = downloader.download_pbf_files_for_regions_gdf(regions_gdf)
    file_names = [path.name for path in files[1]]
    assert set(file_names) == set(test_file_names)


@pytest.mark.parametrize(  # type: ignore
    "test_file_name,query,expected_result_length,expected_features_columns_length",
    [
        (
            "d17f922ed15e9609013a6b895e1e7af2d49158f03586f2c675d17b760af3452e.osm.pbf",
            None,
            678,
            274,
        ),
        (
            "eb2848d259345ce7dfe8af34fd1ab24503bb0b952e04e872c87c55550fa50fbf.osm.pbf",
            None,
            1,
            23,
        ),
        ("529cdcbb7a3cc103658ef31b39bed24984e421127d319c867edf2f86ff3bb098.osm.pbf", None, 0, 0),
        (
            "d17f922ed15e9609013a6b895e1e7af2d49158f03586f2c675d17b760af3452e.osm.pbf",
            HEX2VEC_FILTER,
            97,
            10,
        ),
        (
            "eb2848d259345ce7dfe8af34fd1ab24503bb0b952e04e872c87c55550fa50fbf.osm.pbf",
            HEX2VEC_FILTER,
            0,
            0,
        ),
    ],
)
def test_pbf_handler(
    test_file_name: str,
    query: OsmTagsFilter,
    expected_result_length: int,
    expected_features_columns_length: int,
):
    """Test proper files loading in `PbfFileHandler`."""
    handler = PbfFileHandler(tags=query)
    features_gdf = handler.get_features_gdf(
        file_paths=[Path(__file__).parent / "test_files" / test_file_name]
    )
    assert len(features_gdf) == expected_result_length
    assert len(features_gdf.columns) == expected_features_columns_length + 1


def test_pbf_handler_geometry_filtering():  # type: ignore
    """Test proper spatial data filtering in `PbfFileHandler`."""
    file_name = "d17f922ed15e9609013a6b895e1e7af2d49158f03586f2c675d17b760af3452e.osm.pbf"
    handler = PbfFileHandler(
        tags=HEX2VEC_FILTER, region_geometry=Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    )
    features_gdf = handler.get_features_gdf(
        file_paths=[Path(__file__).parent / "test_files" / file_name]
    )
    assert len(features_gdf) == 0


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
            None,
            HEX2VEC_FILTER,
            "geofabrik",
            397,
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
            [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
            Path(__file__).parent
            / "test_files"
            / "d17f922ed15e9609013a6b895e1e7af2d49158f03586f2c675d17b760af3452e.osm.pbf",
            HEX2VEC_FILTER,
            "protomaps",
            0,
            0,
            [],
        ),
        (
            [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
            Path(__file__).parent
            / "test_files"
            / "d17f922ed15e9609013a6b895e1e7af2d49158f03586f2c675d17b760af3452e.osm.pbf",
            BASE_OSM_GROUPS_FILTER,
            "protomaps",
            0,
            0,
            [],
        ),
    ],
)
def test_osm_pbf_loader(
    test_geometries: List[BaseGeometry],
    pbf_file: Path,
    query: Union[OsmTagsFilter, GroupedOsmTagsFilter],
    pbf_source: PbfSourceLiteral,
    expected_result_length: int,
    expected_features_columns_length: int,
    expected_features_columns_names: List[str],
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
        switch_to_geofabrik_on_error=False,
    )
    result = loader.load(area, tags=query)

    assert (
        len(result) == expected_result_length
    ), f"Mismatched result length ({len(result)}, {expected_result_length})"
    assert (
        len(result.columns) == expected_features_columns_length + 1
    ), f"Mismatched columns length ({len(result.columns)}, {expected_features_columns_length + 1})"
    ut.assertCountEqual(result.columns, expected_features_columns_names + [GEOMETRY_COLUMN])
