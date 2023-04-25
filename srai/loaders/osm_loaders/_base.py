"""Base class for OSM loaders."""

import abc
from typing import Dict, Optional, Union, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from srai.loaders import Loader
from srai.loaders.osm_loaders.filters._typing import (
    grouped_osm_tags_type,
    merge_grouped_osm_tags_type,
    osm_tags_type,
)
from srai.utils.typing import is_expected_type


class OSMLoader(Loader, abc.ABC):
    """Abstract class for loaders."""

    @abc.abstractmethod
    def load(
        self,
        area: gpd.GeoDataFrame,
        tags: Union[osm_tags_type, grouped_osm_tags_type],
    ) -> gpd.GeoDataFrame:  # pragma: no cover
        """
        Load data for a given area.

        Args:
            area (gpd.GeoDataFrame): GeoDataFrame with the area of interest.
            tags (Union[osm_tags_type, grouped_osm_tags_type]): OSM tags filter.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with the downloaded data.
        """
        raise NotImplementedError

    def _merge_osm_tags_filter(
        self, tags: Union[osm_tags_type, grouped_osm_tags_type]
    ) -> osm_tags_type:
        """
        Merge OSM tags filter into `osm_tags_type` type.

        Optionally merges `grouped_osm_tags_type` into `osm_tags_type` to allow loaders to load all
        defined groups during single operation.

        Args:
            tags (Union[osm_tags_type, grouped_osm_tags_type]): OSM tags filter definition.

        Raises:
            AttributeError: When provided tags don't match both
                `osm_tags_type` or `grouped_osm_tags_type`.

        Returns:
            osm_tags_type: Merged filters.
        """
        if is_expected_type(tags, osm_tags_type):
            return cast(osm_tags_type, tags)
        elif is_expected_type(tags, grouped_osm_tags_type):
            return merge_grouped_osm_tags_type(cast(grouped_osm_tags_type, tags))

        raise AttributeError(
            "Provided tags don't match required type definitions"
            " (osm_tags_type or grouped_osm_tags_type)."
        )

    def _parse_features_gdf_to_groups(
        self, features_gdf: gpd.GeoDataFrame, tags: Union[osm_tags_type, grouped_osm_tags_type]
    ) -> gpd.GeoDataFrame:
        """
        Optionally group raw OSM features into groups defined in `grouped_osm_tags_type`.

        Args:
            features_gdf (gpd.GeoDataFrame): Generated features from the loader.
            tags (Union[osm_tags_type, grouped_osm_tags_type]): OSM tags filter definition.

        Returns:
            gpd.GeoDataFrame: Parsed features_gdf.
        """
        if is_expected_type(tags, grouped_osm_tags_type):
            features_gdf = self._group_features_gdf(features_gdf, cast(grouped_osm_tags_type, tags))
        return features_gdf

    def _group_features_gdf(
        self, features_gdf: gpd.GeoDataFrame, group_filter: grouped_osm_tags_type
    ) -> gpd.GeoDataFrame:
        """
        Group raw OSM features into groups defined in `grouped_osm_tags_type`.

        Creates new features based on definition from `grouped_osm_tags_type`.
        Returns transformed GeoDataFrame with columns based on group names from the filter.
        Values are built by concatenation of matching tag key and value with
        an equal sign (eg. amenity=parking). Since many tags can match a definition
        of a single group, a first match is used as a feature value.

        Args:
            features_gdf (gpd.GeoDataFrame): Generated features from the loader.
            group_filter (grouped_osm_tags_type): Grouped OSM tags filter definition.

        Returns:
            gpd.GeoDataFrame: Parsed grouped features_gdf.
        """
        for index, row in tqdm(
            features_gdf.iterrows(), desc="Grouping features", total=len(features_gdf.index)
        ):
            grouped_features = self._get_osm_filter_groups(row=row, group_filter=group_filter)
            for group_name, feature_value in grouped_features.items():
                features_gdf.loc[index, group_name] = feature_value

        matching_columns = [
            column for column in group_filter.keys() if column in features_gdf.columns
        ]

        return features_gdf[["geometry", *matching_columns]].replace(
            to_replace=[None], value=np.nan
        )

    def _get_osm_filter_groups(
        self, row: pd.Series, group_filter: grouped_osm_tags_type
    ) -> Dict[str, str]:
        """
        Get new group features for a single row.

        Args:
            row (pd.Series): Row to be analysed.
            group_filter (grouped_osm_tags_type): Grouped OSM tags filter definition.

        Returns:
            Dict[str, str]: Dictionary with matching group names and values.
        """
        result = {}

        for group_name, osm_filter in group_filter.items():
            matching_osm_tag = self._get_first_matching_osm_tag_value(
                row=row, osm_filter=osm_filter
            )
            if matching_osm_tag:
                result[group_name] = matching_osm_tag

        return result

    def _get_first_matching_osm_tag_value(
        self, row: pd.Series, osm_filter: osm_tags_type
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
