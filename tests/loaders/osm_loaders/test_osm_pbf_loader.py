"""Tests for OSMPbfLoader."""
from pathlib import Path
from typing import List

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters.hex2vec import HEX2VEC_FILTER
from srai.loaders.osm_loaders.filters.osm_tags_type import osm_tags_type
from srai.loaders.osm_loaders.pbf_file_downloader import PbfFileDownloader
from srai.loaders.osm_loaders.pbf_file_handler import PbfFileHandler
from srai.utils.constants import REGIONS_INDEX, WGS84_CRS


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
    loader = PbfFileDownloader()
    prepared_polygon = loader._prepare_polygon_for_download(test_polygon)

    assert len(prepared_polygon.exterior.coords) <= 1000
    assert len(prepared_polygon.interiors) == 0
    assert test_polygon.covered_by(prepared_polygon)


@pytest.mark.parametrize(  # type: ignore
    "test_polygon,test_file_names",
    [
        (
            Point([(-73.981883, 40.768081)]),
            ["c5aedb967d13547767cc51218666d0d613f43a3480320f0ce794c2f6e7a87429.osm.pbf"],
        ),
        (
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            ["2dbaa121223d0f604bd1c327a9574ddb79fab5c66558f71a0ae1c53fee8f0cde.osm.pbf"],
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
            ["d6deffc61e6ad808ef3877098ed6521dc8ff1aea0670940a7469afe1840b6fda.osm.pbf"],
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
                "53dd78e53429e265489d5af87e22cc9af281e0292bf2467e8c4b13189599ae1f.osm.pbf",
                "a459f56fc8da4e8a2cfe9379ad3198df1252f0b4c2efc071e9a81f6469332379.osm.pbf",
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
    loader = PbfFileDownloader(download_directory=Path(__file__).parent / "test_files")
    files = loader.download_pbf_files_for_regions_gdf(regions_gdf)
    file_names = [path.name for path in files[1]]
    assert set(file_names) == set(test_file_names)


@pytest.mark.parametrize(  # type: ignore
    "test_file_name,query,expected_result_length,expected_features_columns_length",
    [
        (
            "c5aedb967d13547767cc51218666d0d613f43a3480320f0ce794c2f6e7a87429.osm.pbf",
            None,
            678,
            274,
        ),
        (
            "2dbaa121223d0f604bd1c327a9574ddb79fab5c66558f71a0ae1c53fee8f0cde.osm.pbf",
            None,
            1,
            23,
        ),
        ("d6deffc61e6ad808ef3877098ed6521dc8ff1aea0670940a7469afe1840b6fda.osm.pbf", None, 0, 0),
        (
            "c5aedb967d13547767cc51218666d0d613f43a3480320f0ce794c2f6e7a87429.osm.pbf",
            HEX2VEC_FILTER,
            97,
            10,
        ),
        (
            "2dbaa121223d0f604bd1c327a9574ddb79fab5c66558f71a0ae1c53fee8f0cde.osm.pbf",
            HEX2VEC_FILTER,
            0,
            0,
        ),
    ],
)
def test_pbf_handler(
    test_file_name: str,
    query: osm_tags_type,
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
    file_name = "c5aedb967d13547767cc51218666d0d613f43a3480320f0ce794c2f6e7a87429.osm.pbf"
    handler = PbfFileHandler(
        tags=HEX2VEC_FILTER, region_geometry=Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    )
    features_gdf = handler.get_features_gdf(
        file_paths=[Path(__file__).parent / "test_files" / file_name]
    )
    assert len(features_gdf) == 0


@pytest.mark.parametrize(  # type: ignore
    "test_geometries,pbf_file,query,expected_result_length,expected_features_columns_length",
    [
        ([Point([(-73.981883, 40.768081)])], None, None, 10, 38),
        ([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])], None, None, 0, 0),
        ([Point([(-73.981883, 40.768081)])], None, HEX2VEC_FILTER, 2, 3),
        (
            [Point([(-73.981883, 40.768081)])],
            Path(__file__).parent
            / "test_files"
            / "c5aedb967d13547767cc51218666d0d613f43a3480320f0ce794c2f6e7a87429.osm.pbf",
            None,
            10,
            38,
        ),
        (
            [Point([(-73.981883, 40.768081)])],
            Path(__file__).parent
            / "test_files"
            / "c5aedb967d13547767cc51218666d0d613f43a3480320f0ce794c2f6e7a87429.osm.pbf",
            HEX2VEC_FILTER,
            2,
            3,
        ),
        (
            [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
            Path(__file__).parent
            / "test_files"
            / "c5aedb967d13547767cc51218666d0d613f43a3480320f0ce794c2f6e7a87429.osm.pbf",
            None,
            0,
            0,
        ),
    ],
)
def test_osm_pbf_loader(
    test_geometries: List[BaseGeometry],
    pbf_file: Path,
    query: osm_tags_type,
    expected_result_length: int,
    expected_features_columns_length: int,
):
    """Test `OSMPbfLoader.load()`."""
    download_directory = Path(__file__).parent / "test_files"
    area = gpd.GeoDataFrame(
        geometry=test_geometries,
        index=gpd.pd.Index(name=REGIONS_INDEX, data=list(range(len(test_geometries)))),
        crs=WGS84_CRS,
    )

    loader = OSMPbfLoader(pbf_file=pbf_file, download_directory=download_directory)
    result = loader.load(area, tags=query)

    assert len(result) == expected_result_length
    assert len(result.columns) == expected_features_columns_length + 1
