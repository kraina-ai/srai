"""
OSM PBF Loader.

This module contains loader capable of loading OpenStreetMap features from `*.osm.pbf` files.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from srai._optional import import_optional_dependencies
from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.loaders.osm_loaders._base import OSMLoader
from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter, OsmTagsFilter

if TYPE_CHECKING:
    from quackosm import PbfFileReader
    from quackosm.osm_extracts import OsmExtractSource


class OSMPbfLoader(OSMLoader):
    """
    OSMPbfLoader.

    OSM(OpenStreetMap)[1] PBF(Protocolbuffer Binary Format)[2] loader is a loader
    capable of loading OSM features from a PBF file. It filters features based on OSM tags[3]
    in form of key:value pairs, that are used by OSM users to give meaning to geometries.

    This loader uses `PbfFileReader` from the `QuackOSM`[3] library.
    It utilizes the `duckdb`[4] engine with `spatial`[5] extension
    capable of parsing an `*.osm.pbf` file.

    Additionally, it can download a pbf file extract for a given area using different sources.

    References:
        1. https://www.openstreetmap.org/
        2. https://wiki.openstreetmap.org/wiki/PBF_Format
        3. https://github.com/kraina-ai/quackosm
        4. https://duckdb.org/
        5. https://github.com/duckdb/duckdb_spatial
    """

    def __init__(
        self,
        pbf_file: Optional[Union[str, Path]] = None,
        download_source: "OsmExtractSource" = "geofabrik",
        download_directory: Union[str, Path] = "files",
    ) -> None:
        """
        Initialize OSMPbfLoader.

        Args:
            pbf_file (Union[str, Path], optional): Downloaded `*.osm.pbf` file to be used by
                the loader. If not provided, it will be automatically downloaded for a given area.
                Defaults to None.
            download_source (OsmExtractSource, optional): Source to use when downloading PBF files.
                Can be one of: `any`, `geofabrik`, `osmfr`, `bbbike`.
                Defaults to "any".
            download_directory (Union[str, Path], optional): Directory where to save the downloaded
                `*.osm.pbf` files. Ignored if `pbf_file` is provided. Defaults to "files".
        """
        import_optional_dependencies(dependency_group="osm", modules=["quackosm"])
        self.pbf_file = pbf_file
        self.download_source = download_source
        self.download_directory = download_directory

    def load(
        self,
        area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
        tags: Union[OsmTagsFilter, GroupedOsmTagsFilter],
        ignore_cache: bool = False,
        explode_tags: bool = True,
        keep_all_tags: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Load OSM features with specified tags for a given area from an `*.osm.pbf` file.

        The loader will use provided `*.osm.pbf` file, or download extracts
        automatically. Later it will parse and filter features from files
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
            explode_tags: (bool, optional): Whether to split OSM tags into multiple columns or keep
                them in a single dict. Defaults to True.
            keep_all_tags: (bool, optional): Whether to keep all tags related to the element,
                or return only those defined in the `tags_filter`. When True, will override
                the optional grouping defined in the `tags_filter`. Defaults to False.

        Raises:
            ValueError: If PBF file is expected to be downloaded and provided geometries
                aren't shapely.geometry.Polygons.

        Returns:
            gpd.GeoDataFrame: Downloaded features as a GeoDataFrame.
        """
        area_wgs84 = self._prepare_area_gdf(area)

        pbf_reader = self._get_pbf_file_reader(area_wgs84, tags)

        if self.pbf_file is not None:
            features_gdf = pbf_reader.get_features_gdf(
                file_paths=self.pbf_file,
                keep_all_tags=keep_all_tags,
                explode_tags=explode_tags,
                ignore_cache=ignore_cache,
            )
        else:
            features_gdf = pbf_reader.get_features_gdf_from_geometry(
                keep_all_tags=keep_all_tags, explode_tags=explode_tags, ignore_cache=ignore_cache
            )

        features_gdf = features_gdf.set_crs(WGS84_CRS)

        features_columns = [
            column
            for column in features_gdf.columns
            if column != GEOMETRY_COLUMN and features_gdf[column].notnull().any()
        ]
        features_gdf = features_gdf[[GEOMETRY_COLUMN, *sorted(features_columns)]]

        return features_gdf

    def load_to_geoparquet(
        self,
        area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
        tags: Union[OsmTagsFilter, GroupedOsmTagsFilter],
        ignore_cache: bool = False,
        explode_tags: bool = True,
        keep_all_tags: bool = False,
    ) -> Path:
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
            keep_all_tags: (bool, optional): Whether to keep all tags related to the element,
                or return only those defined in the `tags_filter`. When True, will override
                the optional grouping defined in the `tags_filter`. Defaults to False.

        Returns:
            Path: Path to the saved GeoParquet file.
        """
        area_wgs84 = self._prepare_area_gdf(area)

        pbf_reader = self._get_pbf_file_reader(area_wgs84, tags)

        geoparquet_file_path: Path

        if self.pbf_file is not None:
            geoparquet_file_path = pbf_reader.convert_pbf_to_gpq(
                pbf_path=self.pbf_file,
                keep_all_tags=keep_all_tags,
                explode_tags=explode_tags,
                ignore_cache=ignore_cache,
            )
        else:
            geoparquet_file_path = pbf_reader.convert_geometry_filter_to_gpq(
                keep_all_tags=keep_all_tags, explode_tags=explode_tags, ignore_cache=ignore_cache
            )

        return geoparquet_file_path

    def _get_pbf_file_reader(
        self, area_wgs84: gpd.GeoDataFrame, tags: Union[OsmTagsFilter, GroupedOsmTagsFilter]
    ) -> "PbfFileReader":
        from quackosm import PbfFileReader
        from quackosm.osm_extracts import OsmExtractSource

        pbf_reader = PbfFileReader(
            tags_filter=tags,
            geometry_filter=area_wgs84.unary_union,
            working_directory=self.download_directory,
            osm_extract_source=OsmExtractSource(self.download_source),
        )
        return pbf_reader
