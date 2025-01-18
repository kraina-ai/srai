"""
TODO.

OSM PBF Loader.

This module contains loader capable of loading OpenStreetMap features from `*.osm.pbf` files.
"""

from collections.abc import Iterable
from typing import Union

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from srai._optional import import_optional_dependencies
from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.loaders._base import Loader


class OvertureMapsLoader(Loader):
    """
    OvertureMapsLoader.

    Overture Maps[1] loader is a loader capable of loading OvertureMaps features from dedicated
    s3 bucket. It can download multiple data types for different release versions and it can filter
    features using PyArrow[2] filters.


    This loader is a wrapper around `OvertureMaestro`[3] library.
    It utilizes the PyArrow streaming capabilities as well as `duckdb`[4] engine for transforming
    the data into the required format.

    References:
        1. https://overturemaps.org/
        2. https://arrow.apache.org/docs/python/
        3. https://github.com/kraina-ai/overturemaestro
        4. https://duckdb.org/
    """

    def __init__(self) -> None:
        """Initialize Overture Maps loader."""
        import_optional_dependencies(dependency_group="overturemaps", modules=["overturemaestro"])

    def load(
        self,
        area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
        ignore_cache: bool = False,
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
        from overturemaestro.advanced_functions import (
            convert_geometry_to_wide_form_geodataframe_for_all_types,
        )

        area_wgs84 = self._prepare_area_gdf(area)

        features_gdf = convert_geometry_to_wide_form_geodataframe_for_all_types(
            geometry_filter=area_wgs84.union_all(),
            include_all_possible_columns=True,
            hierarchy_depth=None,
            ignore_cache=ignore_cache,
            working_directory="files",
            verbosity_mode="transient",
            max_workers=None,
        )

        features_gdf = features_gdf.set_crs(WGS84_CRS)

        features_columns = [
            column
            for column in features_gdf.columns
            if column != GEOMETRY_COLUMN and features_gdf[column].notnull().any()
        ]
        features_gdf = features_gdf[[GEOMETRY_COLUMN, *sorted(features_columns)]]

        return features_gdf
