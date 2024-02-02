"""
OSM Online Loader.

This module contains loader capable of loading OpenStreetMap features from Overpass.
"""

from collections.abc import Iterable
from itertools import product
from typing import Union

import geopandas as gpd
import pandas as pd
from functional import seq
from packaging import version
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from srai._optional import import_optional_dependencies
from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, WGS84_CRS
from srai.loaders.osm_loaders._base import OSMLoader
from srai.loaders.osm_loaders.filters import (
    GroupedOsmTagsFilter,
    OsmTagsFilter,
    merge_osm_tags_filter,
)


class OSMOnlineLoader(OSMLoader):
    """
    OSMOnlineLoader.

    OSM(OpenStreetMap)[1] online loader is a loader capable of downloading objects
    from a given area from OSM. It filters features based on OSM tags[2] in form of
    key:value pairs, that are used by OSM users to give meaning to geometries.

    This loader is a wrapper around the `osmnx` library. It uses `osmnx.geometries_from_polygon`
    to make individual queries.

    References:
        1. https://www.openstreetmap.org/
        2. https://wiki.openstreetmap.org/wiki/Tags
    """

    _PBAR_FORMAT = "Downloading {}: {}"
    _ELEMENT_TYPE_INDEX_NAME = "element_type"
    _OSMID_INDEX_NAME = "osmid"
    _RESULT_INDEX_NAMES = [_ELEMENT_TYPE_INDEX_NAME, _OSMID_INDEX_NAME]

    def __init__(self) -> None:
        """Initialize OSMOnlineLoader."""
        import_optional_dependencies(dependency_group="osm", modules=["osmnx"])

    def load(
        self,
        area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
        tags: Union[OsmTagsFilter, GroupedOsmTagsFilter],
    ) -> gpd.GeoDataFrame:
        """
        Download OSM features with specified tags for a given area.

        The loader first downloads all objects with `tags`. It returns a GeoDataFrame containing
        the `geometry` column and columns for tag keys.

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

        Returns:
            gpd.GeoDataFrame: Downloaded features as a GeoDataFrame.
        """
        import osmnx as ox

        area_wgs84 = self._prepare_area_gdf(area)

        merged_tags = merge_osm_tags_filter(tags)

        _tags = self._flatten_tags(merged_tags)

        total_tags_num = len(_tags)
        total_queries = len(area_wgs84) * total_tags_num

        key_value_name_max_len = self._get_max_key_value_name_len(_tags)
        desc_max_len = key_value_name_max_len + len(self._PBAR_FORMAT.format("", ""))

        results = []

        osmnx_new_api = version.parse(ox.__version__) >= version.parse("1.5.0")
        osmnx_download_function = (
            ox.features_from_polygon if osmnx_new_api else ox.geometries_from_polygon
        )

        pbar = tqdm(product(area_wgs84[GEOMETRY_COLUMN], _tags), total=total_queries)
        for polygon, (key, value) in pbar:
            pbar.set_description(self._get_pbar_desc(key, value, desc_max_len))
            geometries = osmnx_download_function(polygon, {key: value})
            if not geometries.empty:
                results.append(geometries[[GEOMETRY_COLUMN, key]])

        result_gdf = self._group_gdfs(results).set_crs(WGS84_CRS)
        result_gdf = self._flatten_index(result_gdf)

        return self._parse_features_gdf_to_groups(result_gdf, tags)

    def _flatten_tags(self, tags: OsmTagsFilter) -> list[tuple[str, Union[str, bool]]]:
        tags_flat: list[tuple[str, Union[str, bool]]] = (
            seq(tags.items())
            .starmap(lambda k, v: product([k], v if isinstance(v, list) else [v]))
            .flatten()
            .list()
        )
        return tags_flat

    def _get_max_key_value_name_len(self, tags: list[tuple[str, Union[str, bool]]]) -> int:
        max_key_val_name_len: int = seq(tags).starmap(lambda k, v: len(k + str(v))).max()
        return max_key_val_name_len

    def _get_pbar_desc(self, key: str, val: Union[str, bool], max_desc_len: int) -> str:
        return self._PBAR_FORMAT.format(key, val).ljust(max_desc_len)

    def _group_gdfs(self, gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        if not gdfs:
            return self._get_empty_result()
        elif len(gdfs) == 1:
            gdf = gdfs[0]
        else:
            gdf = pd.concat(gdfs)

        return gdf.groupby(self._RESULT_INDEX_NAMES).first()

    def _get_empty_result(self) -> gpd.GeoDataFrame:
        result_index = pd.MultiIndex.from_arrays(arrays=[[], []], names=self._RESULT_INDEX_NAMES)
        return gpd.GeoDataFrame(index=result_index, crs=WGS84_CRS, geometry=[])

    def _flatten_index(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf.reset_index()
        gdf[FEATURES_INDEX] = (
            gdf[self._RESULT_INDEX_NAMES]
            .apply(lambda idx: "/".join(map(str, idx)), axis=1)
            .astype(str)
        )
        return gdf.set_index(FEATURES_INDEX).drop(columns=self._RESULT_INDEX_NAMES)
