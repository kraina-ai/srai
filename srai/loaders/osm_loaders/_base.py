"""Base class for OSM loaders."""

import abc
from collections.abc import Iterable
from typing import Optional, Union, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from srai._typing import is_expected_type
from srai.constants import WGS84_CRS
from srai.loaders import Loader
from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter, OsmTagsFilter


def prepare_area_gdf_for_loader(
    area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """
    Prepare an area for the loader.

    Loader expects a GeoDataFrame input, but users shouldn't be limited by this requirement.
    All Shapely geometries will by transformed into GeoDataFrame with proper CRS.

    Args:
        area (Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame]):
            Area to be parsed into GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: Sanitized GeoDataFrame.
    """
    if isinstance(area, gpd.GeoDataFrame):
        # Return a GeoDataFrame with changed CRS
        return area.to_crs(WGS84_CRS)
    elif isinstance(area, gpd.GeoSeries):
        # Create a GeoDataFrame with GeoSeries
        return gpd.GeoDataFrame(geometry=area, crs=WGS84_CRS)
    elif isinstance(area, Iterable):
        # Create a GeoSeries with a list of geometries
        return prepare_area_gdf_for_loader(gpd.GeoSeries(area, crs=WGS84_CRS))
    # Wrap a single geometry with a list
    return prepare_area_gdf_for_loader([area])


class OSMLoader(Loader, abc.ABC):
    """Abstract class for loaders."""

    OSM_FILTER_GROUP_COLUMN_NAME = "osm_group_"

    @abc.abstractmethod
    def load(
        self,
        area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
        tags: Union[OsmTagsFilter, GroupedOsmTagsFilter],
    ) -> gpd.GeoDataFrame:  # pragma: no cover
        """
        Load data for a given area.

        Args:
            area (Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame]):
                Shapely geometry with the area of interest.
            tags (Union[OsmTagsFilter, GroupedOsmTagsFilter]): OSM tags filter.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with the downloaded data.
        """
        raise NotImplementedError

    def _prepare_area_gdf(
        self, area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame]
    ) -> gpd.GeoDataFrame:
        return prepare_area_gdf_for_loader(area)

    def _parse_features_gdf_to_groups(
        self, features_gdf: gpd.GeoDataFrame, tags: Union[OsmTagsFilter, GroupedOsmTagsFilter]
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
            group_filter.items(), desc="Grouping features", total=len(group_filter)
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
