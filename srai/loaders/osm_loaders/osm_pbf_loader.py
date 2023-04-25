"""
OSM PBF Loader.

This module contains loader capable of loading OpenStreetMap features from `*.osm.pbf` files.
"""
from pathlib import Path
from typing import Hashable, List, Mapping, Optional, Sequence, Union

import geopandas as gpd
import pandas as pd

from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, WGS84_CRS
from srai.loaders.osm_loaders._base import OSMLoader
from srai.loaders.osm_loaders.filters._typing import (
    grouped_osm_tags_type,
    osm_tags_type,
)
from srai.utils._optional import import_optional_dependencies


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
        pbf_file: Optional[Union[str, Path]] = None,
        download_directory: Union[str, Path] = "files",
    ) -> None:
        """
        Initialize OSMPbfLoader.

        Args:
            pbf_file (Union[str, Path], optional): Downloaded `*.osm.pbf` file to be used by
                the loader. If not provided, it will be automatically downloaded for a given area.
                Defaults to None.
            download_directory (Union[str, Path], optional): Directory where to save the downloaded
                `*.osm.pbf` files. Ignored if `pbf_file` is provided. Defaults to "files"
        """
        import_optional_dependencies(dependency_group="osm", modules=["osmium"])
        self.pbf_file = pbf_file
        self.download_directory = download_directory

    def load(
        self,
        area: gpd.GeoDataFrame,
        tags: Union[osm_tags_type, grouped_osm_tags_type],
    ) -> gpd.GeoDataFrame:
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
            gpd.GeoDataFrame: Downloaded features as a GeoDataFrame.
        """
        from srai.loaders.osm_loaders.pbf_file_downloader import PbfFileDownloader
        from srai.loaders.osm_loaders.pbf_file_handler import PbfFileHandler

        area_wgs84 = area.to_crs(crs=WGS84_CRS)

        downloaded_pbf_files: Mapping[Hashable, Sequence[Union[str, Path]]]
        if self.pbf_file is not None:
            downloaded_pbf_files = {Path(self.pbf_file).name: [self.pbf_file]}
        else:
            downloaded_pbf_files = PbfFileDownloader(
                download_directory=self.download_directory
            ).download_pbf_files_for_regions_gdf(regions_gdf=area_wgs84)

        clipping_polygon = area_wgs84.geometry.unary_union

        merged_tags = self._merge_osm_tags_filter(tags)

        pbf_handler = PbfFileHandler(tags=merged_tags, region_geometry=clipping_polygon)

        results = []
        for region_id, pbf_files in downloaded_pbf_files.items():
            features_gdf = pbf_handler.get_features_gdf(
                file_paths=pbf_files, region_id=str(region_id)
            )
            results.append(features_gdf)

        result_gdf = self._group_gdfs(results).set_crs(WGS84_CRS)

        features_columns = result_gdf.columns.drop(labels=[GEOMETRY_COLUMN]).sort_values()
        result_gdf = result_gdf[[GEOMETRY_COLUMN, *features_columns]]

        return self._parse_features_gdf_to_groups(result_gdf, tags)

    def _group_gdfs(self, gdfs: List[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        if not gdfs:
            return self._get_empty_result()
        elif len(gdfs) == 1:
            gdf = gdfs[0]
        else:
            gdf = pd.concat(gdfs)

        return gdf[~gdf.index.duplicated(keep="first")]

    def _get_empty_result(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(index=pd.Index(name=FEATURES_INDEX), crs=WGS84_CRS, geometry=[])
