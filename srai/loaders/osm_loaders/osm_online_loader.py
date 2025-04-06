"""
OSM Online Loader.

This module contains loader capable of loading OpenStreetMap features from Overpass.
"""

from contextlib import suppress
from itertools import product
from typing import Optional, Union, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from functional import seq
from packaging import version
from shapely.geometry import Polygon
from tqdm import tqdm

from srai._optional import import_optional_dependencies
from srai._typing import is_expected_type
from srai.constants import FEATURES_INDEX, FORCE_TERMINAL, GEOMETRY_COLUMN, WGS84_CRS
from srai.geometry import flatten_geometry
from srai.loaders._base import VALID_AREA_INPUT
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

    def __init__(self) -> None:
        """Initialize OSMOnlineLoader."""
        import_optional_dependencies(dependency_group="osm", modules=["osmnx"])

        import osmnx

        osmnx_new_api = version.parse(osmnx.__version__) >= version.parse("2.0.0")

        if osmnx_new_api:
            self._ELEMENT_TYPE_INDEX_NAME = "element"
            self._OSMID_INDEX_NAME = "id"

        else:
            self._ELEMENT_TYPE_INDEX_NAME = "element_type"
            self._OSMID_INDEX_NAME = "osmid"

        self._RESULT_INDEX_NAMES = [
            self._ELEMENT_TYPE_INDEX_NAME,
            self._OSMID_INDEX_NAME,
        ]

    def load(
        self,
        area: VALID_AREA_INPUT,
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

        polygons = [
            g
            for g in flatten_geometry(self._prepare_area_gdf(area).union_all())
            if isinstance(g, Polygon)
        ]

        merged_tags = merge_osm_tags_filter(tags)

        _tags = self._flatten_tags(merged_tags)

        total_tags_num = len(_tags)
        total_queries = len(polygons) * total_tags_num

        key_value_name_max_len = self._get_max_key_value_name_len(_tags)
        desc_max_len = key_value_name_max_len + len(self._PBAR_FORMAT.format("", ""))

        results = []

        osmnx_new_api = version.parse(ox.__version__) >= version.parse("1.5.0")
        osmnx_download_function = (
            ox.features_from_polygon if osmnx_new_api else ox.geometries_from_polygon
        )

        osmnx_new_exception_api = version.parse(ox.__version__) >= version.parse("1.6.0")
        if osmnx_new_exception_api:
            from osmnx._errors import InsufficientResponseError

            response_error = InsufficientResponseError
        else:
            from osmnx._errors import EmptyOverpassResponse

            response_error = EmptyOverpassResponse

        pbar = tqdm(
            product(polygons, _tags),
            total=total_queries,
            disable=FORCE_TERMINAL,
        )
        for polygon, (key, value) in pbar:
            pbar.set_description(self._get_pbar_desc(key, value, desc_max_len))
            with suppress(response_error):
                geometries = osmnx_download_function(polygon, {key: value})
                print(f"{polygon=} {geometries=}")
                if geometries is not None and not geometries.empty:
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

    def _parse_features_gdf_to_groups(
        self,
        features_gdf: gpd.GeoDataFrame,
        tags: Union[OsmTagsFilter, GroupedOsmTagsFilter],
    ) -> gpd.GeoDataFrame:
        """
        Optionally group raw OSM features into groups defined in `GroupedOsmTagsFilter`.

        Args:
            features_gdf (gpd.GeoDataFrame): Generated features from the loader.
            tags (Union[OsmTagsFilter, GroupedOsmTagsFilter]): OSM tags filter definition.

        Returns:
            gpd.GeoDataFrame: Parsed features_gdf.
        """
        if is_expected_type(tags, GroupedOsmTagsFilter):
            features_gdf = self._group_features_gdf(features_gdf, cast(GroupedOsmTagsFilter, tags))
        return features_gdf

    def _group_features_gdf(
        self, features_gdf: gpd.GeoDataFrame, group_filter: GroupedOsmTagsFilter
    ) -> gpd.GeoDataFrame:
        """
        Group raw OSM features into groups defined in `GroupedOsmTagsFilter`.

        Creates new features based on definition from `GroupedOsmTagsFilter`.
        Returns transformed GeoDataFrame with columns based on group names from the filter.
        Values are built by concatenation of matching tag key and value with
        an equal sign (eg. amenity=parking). Since many tags can match a definition
        of a single group, a first match is used as a feature value.

        Args:
            features_gdf (gpd.GeoDataFrame): Generated features from the loader.
            group_filter (GroupedOsmTagsFilter): Grouped OSM tags filter definition.

        Returns:
            gpd.GeoDataFrame: Parsed grouped features_gdf.
        """
        if len(features_gdf) == 0:
            return features_gdf[["geometry"]]

        matching_columns = []

        for group_name, osm_filter in tqdm(
            group_filter.items(),
            desc="Grouping features",
            total=len(group_filter),
            disable=FORCE_TERMINAL,
        ):
            mask = self._get_matching_mask(features_gdf, osm_filter)
            if mask.any():
                group_name_column = f"{OSMLoader.OSM_FILTER_GROUP_COLUMN_NAME}{group_name}"
                matching_columns.append(group_name_column)
                features_gdf[group_name_column] = features_gdf[mask].apply(
                    lambda row, osm_filter=osm_filter: self._get_first_matching_osm_tag_value(
                        row=row, osm_filter=osm_filter
                    ),
                    axis=1,
                )

        return (
            features_gdf[["geometry", *matching_columns]]
            .rename(
                columns={
                    column_name: column_name.replace(OSMLoader.OSM_FILTER_GROUP_COLUMN_NAME, "")
                    for column_name in matching_columns
                }
            )
            .replace(to_replace=[None], value=np.nan)
            .dropna(how="all", axis="columns")
        )

    def _get_matching_mask(
        self, features_gdf: gpd.GeoDataFrame, osm_filter: OsmTagsFilter
    ) -> pd.Series:
        """
        Create a boolean mask to identify rows matching the OSM tags filter.

        Args:
            features_gdf (gpd.GeoDataFrame): Generated features from the loader.
            osm_filter (OsmTagsFilter): OSM tags filter definition.

        Returns:
            pd.Series: Boolean mask.
        """
        mask = pd.Series(False, index=features_gdf.index)

        for osm_tag_key, osm_tag_value in osm_filter.items():
            if osm_tag_key in features_gdf.columns:
                if isinstance(osm_tag_value, bool) and osm_tag_value:
                    mask |= features_gdf[osm_tag_key]
                elif isinstance(osm_tag_value, str):
                    mask |= features_gdf[osm_tag_key] == osm_tag_value
                elif isinstance(osm_tag_value, list):
                    mask |= features_gdf[osm_tag_key].isin(osm_tag_value)

        return mask

    def _get_first_matching_osm_tag_value(
        self, row: pd.Series, osm_filter: OsmTagsFilter
    ) -> Optional[str]:
        """
        Find first matching OSM tag key and value pair for a subgroup filter.

        Returns a first matching pair of OSM tag key and value concatenated
        with an equal sign (eg. amenity=parking). If none of the values
        in the row matches the filter, `None` value is returned.

        Args:
            row (pd.Series): Row to be analysed.
            osm_filter (osm_tags_type): OSM tags filter definition.

        Returns:
            Optional[str]: New feature value.
        """
        for osm_tag_key, osm_tag_value in osm_filter.items():
            if osm_tag_key not in row or pd.isna(row[osm_tag_key]):
                continue

            is_matching_bool_filter = isinstance(osm_tag_value, bool) and osm_tag_value
            is_matching_string_filter = (
                isinstance(osm_tag_value, str) and row[osm_tag_key] == osm_tag_value
            )
            is_matching_list_filter = (
                isinstance(osm_tag_value, list) and row[osm_tag_key] in osm_tag_value
            )

            if is_matching_bool_filter or is_matching_string_filter or is_matching_list_filter:
                return f"{osm_tag_key}={row[osm_tag_key]}"

        return None
