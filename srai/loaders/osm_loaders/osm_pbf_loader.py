"""
OSM PBF Loader.

This module contains loader capable of loading OpenStreetMap features from `*.osm.pbf` files.
"""

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from srai._optional import import_optional_dependencies
from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.loaders.osm_loaders._base import OSMLoader
from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter, OsmTagsFilter
from srai.loaders.osm_loaders.pbf_file_downloader import PbfSourceLiteral

if TYPE_CHECKING:
    from quackosm import PbfFileReader


class OSMPbfLoader(OSMLoader):
    """
    OSMPbfLoader.

    OSM(OpenStreetMap)[1] PBF(Protocolbuffer Binary Format)[2] loader is a loader
    capable of loading OSM features from a PBF file. It filters features based on OSM tags[3]
    in form of key:value pairs, that are used by OSM users to give meaning to geometries.

    This loader uses `duckdb`[3] engine with `spatial`[4] extension
    capable of parsing an `*.osm.pbf` file.

    Additionally, it can download a pbf file extract for a given area using different sources.

    References:
        1. https://www.openstreetmap.org/
        2. https://wiki.openstreetmap.org/wiki/PBF_Format
        3. https://duckdb.org/
        4. https://github.com/duckdb/duckdb_spatial
    """

    def __init__(
        self,
        pbf_file: Optional[Union[str, Path]] = None,
        download_source: PbfSourceLiteral = "geofabrik",
        download_directory: Union[str, Path] = "files",
        switch_to_geofabrik_on_error: bool = True,
    ) -> None:
        """
        Initialize OSMPbfLoader.

        Args:
            pbf_file (Union[str, Path], optional): Downloaded `*.osm.pbf` file to be used by
                the loader. If not provided, it will be automatically downloaded for a given area.
                Defaults to None.
            download_source (PbfSourceLiteral, optional): Source to use when downloading PBF files.
                Can be one of: `geofabrik`, `openstreetmap_fr`, `protomaps`.
                Defaults to "geofabrik".
            download_directory (Union[str, Path], optional): Directory where to save the downloaded
                `*.osm.pbf` files. Ignored if `pbf_file` is provided. Defaults to "files".
            switch_to_geofabrik_on_error (bool, optional): Flag whether to automatically
                switch `download_source` to 'geofabrik' if error occures. Defaults to `True`.
        """
        import_optional_dependencies(dependency_group="osm", modules=["quackosm"])
        self.pbf_file = pbf_file
        self.download_source = download_source
        self.download_directory = download_directory
        self.switch_to_geofabrik_on_error = switch_to_geofabrik_on_error

    def load(
        self,
        area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
        tags: Union[OsmTagsFilter, GroupedOsmTagsFilter],
        ignore_cache: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Load OSM features with specified tags for a given area from an `*.osm.pbf` file.

        The loader will use provided `*.osm.pbf` file, or download extracts
        using `PbfFileDownloader`. Later it will parse and filter features from files
        using `PbfFileReader` from `QuackOSM` library. It will return a GeoDataFrame
        containing the `geometry` column and columns for tag keys.

        Note: Some key/value pairs might be missing from the resulting GeoDataFrame,
            simply because there are no such objects in the given area.

        Args:
            area (Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame]):
                Area for which to download objects.
            tags (Union[OsmTagsFilter, GroupedOsmTagsFilter]): A dictionary
                specifying which tags to download.
                The keys should be OSM tags (e.g. `building`, `amenity`).
                The values should either be `True` for retrieving all objects with the tag,
                string for retrieving a single tag-value pair
                or list of strings for retrieving all values specified in the list.
                `tags={'leisure': 'park}` would return parks from the area.
                `tags={'leisure': 'park, 'amenity': True, 'shop': ['bakery', 'bicycle']}`
                would return parks, all amenity types, bakeries and bicycle shops.
            ignore_cache: (bool, optional): Whether to ignore precalculated geoparquet files or not.
                Defaults to False.

        Raises:
            ValueError: If PBF file is expected to be downloaded and provided geometries
                aren't shapely.geometry.Polygons.

        Returns:
            gpd.GeoDataFrame: Downloaded features as a GeoDataFrame.
        """
        area_wgs84 = self._prepare_area_gdf(area)

        pbf_reader = self._get_pbf_file_reader(area_wgs84, tags)
        pbf_files_to_load = self._get_pbf_files_to_load(area_wgs84)

        features_gdf = pbf_reader.get_features_gdf(
            file_paths=pbf_files_to_load, explode_tags=True, ignore_cache=ignore_cache
        )
        result_gdf = features_gdf.set_crs(WGS84_CRS)

        features_columns = [
            column
            for column in result_gdf.columns
            if column != GEOMETRY_COLUMN and result_gdf[column].notnull().any()
        ]
        result_gdf = result_gdf[[GEOMETRY_COLUMN, *sorted(features_columns)]]

        return result_gdf

    def load_to_geoparquet(
        self,
        area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
        tags: Union[OsmTagsFilter, GroupedOsmTagsFilter],
        ignore_cache: bool = False,
        explode_tags: bool = True,
    ) -> list[Path]:
        """
        Load OSM features with specified tags for a given area and save it to geoparquet file.

        Args:
            area (Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame]):
                Area for which to download objects.
            tags (Union[OsmTagsFilter, GroupedOsmTagsFilter]): A dictionary
                specifying which tags to download.
                The keys should be OSM tags (e.g. `building`, `amenity`).
                The values should either be `True` for retrieving all objects with the tag,
                string for retrieving a single tag-value pair
                or list of strings for retrieving all values specified in the list.
                `tags={'leisure': 'park}` would return parks from the area.
                `tags={'leisure': 'park, 'amenity': True, 'shop': ['bakery', 'bicycle']}`
                would return parks, all amenity types, bakeries and bicycle shops.
            ignore_cache: (bool, optional): Whether to ignore precalculated geoparquet files or not.
                Defaults to False.
            explode_tags: (bool, optional): Whether to split OSM tags into multiple columns or keep
                them in a single dict. Defaults to True.

        Returns:
            list[Path]: List of saved GeoParquet files.
        """
        area_wgs84 = self._prepare_area_gdf(area)

        pbf_reader = self._get_pbf_file_reader(area_wgs84, tags)
        pbf_files_to_load = self._get_pbf_files_to_load(area_wgs84)

        converted_files = []
        for downloaded_pbf_file in pbf_files_to_load:
            geoparquet_file = pbf_reader.convert_pbf_to_gpq(
                pbf_path=downloaded_pbf_file, ignore_cache=ignore_cache, explode_tags=explode_tags
            )
            converted_files.append(geoparquet_file)

        return converted_files

    def _get_pbf_files_to_load(
        self, area_wgs84: gpd.GeoDataFrame
    ) -> Sequence[Union[str, os.PathLike[str]]]:
        from srai.loaders.osm_loaders.pbf_file_downloader import PbfFileDownloader

        pbf_files_to_load: Sequence[Union[str, os.PathLike[str]]]
        if self.pbf_file is not None:
            pbf_files_to_load = [self.pbf_file]
        else:
            pbf_files_to_load = PbfFileDownloader(
                download_source=self.download_source,
                download_directory=self.download_directory,
                switch_to_geofabrik_on_error=self.switch_to_geofabrik_on_error,
            ).download_pbf_files_for_regions_gdf(regions_gdf=area_wgs84)

        return pbf_files_to_load

    def _get_pbf_file_reader(
        self, area_wgs84: gpd.GeoDataFrame, tags: Union[OsmTagsFilter, GroupedOsmTagsFilter]
    ) -> "PbfFileReader":
        from quackosm import PbfFileReader

        pbf_reader = PbfFileReader(
            tags_filter=tags,
            geometry_filter=area_wgs84.unary_union,
            working_directory=self.download_directory,
        )
        return pbf_reader
