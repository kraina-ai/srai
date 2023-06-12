"""
OSM PBF Loader.

This module contains loader capable of loading OpenStreetMap features from `*.osm.pbf` files.
"""
from pathlib import Path
from typing import TYPE_CHECKING, Hashable, List, Mapping, Optional, Sequence, Union

import geopandas as gpd
from tqdm import tqdm

from srai.constants import WGS84_CRS
from srai.db import delete_table, relation_to_table
from srai.loaders.osm_loaders._base import OSMLoader
from srai.loaders.osm_loaders.filters._typing import (
    grouped_osm_tags_type,
    osm_tags_type,
)

# TOFO: change import
from srai.loaders.osm_loaders.pbf_file_downloader import PbfSourceLiteral
from srai.utils._optional import import_optional_dependencies

if TYPE_CHECKING:
    import duckdb


class OSMPbfLoader(OSMLoader):
    """
    OSMPbfLoader.

    OSM(OpenStreetMap)[1] PBF(Protocolbuffer Binary Format)[2] loader is a loader
    capable of loading OSM features from a PBF file. It filters features based on OSM tags[3]
    in form of key:value pairs, that are used by OSM users to give meaning to geometries.

    This loader uses `pyosmium`[3] library capable of parsing an `*.osm.pbf` file.

    Additionally, it can download a pbf file extract for a given area using Protomaps API.

    References:
        1. https://www.openstreetmap.org/
        2. https://wiki.openstreetmap.org/wiki/PBF_Format
        3. https://osmcode.org/pyosmium/
    """

    def __init__(
        self,
        pbf_files: Optional[Sequence[Union[str, Path]]] = None,
        download_source: PbfSourceLiteral = "protomaps",
        download_directory: Union[str, Path] = "files",
    ) -> None:
        """
        Initialize OSMPbfLoader.

        Args:
            pbf_files (Sequence[Union[str, Path]], optional): Downloaded `*.osm.pbf` files
                to be used by the loader. If not provided, it will be automatically downloaded
                for a given area. Defaults to None.
            download_source (PbfSourceLiteral, optional): Source to use when downloading PBF files.
                Can be either `protomaps` or `geofabrik`.
                Defaults to "protomaps".
            download_directory (Union[str, Path], optional): Directory where to save the downloaded
                `*.osm.pbf` files. Ignored if `pbf_file` is provided. Defaults to "files"
        """
        import_optional_dependencies(dependency_group="osm", modules=["osmium"])
        self.pbf_files = pbf_files
        self.download_source = download_source
        self.download_directory = download_directory

    def load(
        self,
        area: gpd.GeoDataFrame,
        tags: Union[osm_tags_type, grouped_osm_tags_type],
    ) -> "duckdb.DuckDBPyRelation":
        """
        Load OSM features with specified tags for a given area from an `*.osm.pbf` file.

        The loader will use provided `*.osm.pbf` file, or download extracts
        using `PbfFileDownloader`. Later it will parse and filter features from files
        using `PbfFileHandler`. It will return a GeoDataFrame containing the `geometry` column
        and columns for tag keys.

        Note: Some key/value pairs might be missing from the resulting GeoDataFrame,
            simply because there are no such objects in the given area.

        Note: If you want to extract data for a big area (like country, or more), it's encouraged
            to use existing `*.osm.pbf` extracts from GeoFabrik (https://download.geofabrik.de/)
            or BBBike (https://extract.bbbike.org/). You can provide those predownloaded files in
            the constructor of the `OSMPbfLoader`.

        Args:
            area (gpd.GeoDataFrame): Area for which to download objects.
            tags (Union[osm_tags_type, grouped_osm_tags_type]): A dictionary
                specifying which tags to download.
                The keys should be OSM tags (e.g. `building`, `amenity`).
                The values should either be `True` for retrieving all objects with the tag,
                string for retrieving a single tag-value pair
                or list of strings for retrieving all values specified in the list.
                `tags={'leisure': 'park}` would return parks from the area.
                `tags={'leisure': 'park, 'amenity': True, 'shop': ['bakery', 'bicycle']}`
                would return parks, all amenity types, bakeries and bicycle shops.

        Returns:
            duckdb.DuckDBPyRelation: Downloaded features as a DuckDB relation.
        """
        from srai.loaders.osm_loaders.pbf_file_downloader import PbfFileDownloader
        from srai.loaders.osm_loaders.pbf_file_handler import read_features_from_pbf_files

        area_wgs84 = area.to_crs(crs=WGS84_CRS)

        downloaded_pbf_files: Mapping[Hashable, Sequence[Union[str, Path]]]
        if self.pbf_files:
            downloaded_pbf_files = {Path(pbf_file).name: [pbf_file] for pbf_file in self.pbf_files}
        else:
            downloaded_pbf_files = PbfFileDownloader(
                source=self.download_source, download_directory=self.download_directory
            ).download_pbf_files_for_regions_gdf(regions_gdf=area_wgs84)

        clipping_polygon = area_wgs84.geometry.unary_union

        merged_tags = self._merge_osm_tags_filter(tags)

        results: List["duckdb.DuckDBPyRelation"] = []
        for pbf_file_name, pbf_files in downloaded_pbf_files.items():
            print(pbf_file_name)
            features_relation = read_features_from_pbf_files(
                file_paths=pbf_files,
                tags=merged_tags,
                filter_region_geometry=clipping_polygon,
            )

            grouped_features_relation = self._parse_features_relation_to_groups(
                features_relation, tags
            )
            grouped_features_relation = relation_to_table(
                relation=grouped_features_relation, prefix="osm_data"
            )
            delete_table(features_relation)
            grouped_features_relation.query(
                virtual_table_name="x",
                sql_query="SELECT * EXCLUDE (geometry), ST_AsText(geometry) wkt FROM x",
            ).to_parquet(
                (Path(self.download_directory) / f"{pbf_file_name}.parquet").resolve().as_posix()
            )
            results.append(grouped_features_relation)

        result_relation = relation_to_table(relation=results[0], prefix="osm_data")
        delete_table(results[0])
        for relation in tqdm(results[1:]):
            relation.query(
                "x",
                (
                    f"SELECT x.* FROM x LEFT JOIN ({result_relation.sql_query()}) osm_data_relation"
                    " ON x.feature_id = osm_data_relation.feature_id"
                    " WHERE osm_data_relation.feature_id IS NULL"
                ),
            ).insert_into(result_relation.alias)
            delete_table(relation)

        return result_relation
