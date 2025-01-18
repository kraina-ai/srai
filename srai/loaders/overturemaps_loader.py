"""
TODO.

OSM PBF Loader.

This module contains loader capable of loading OpenStreetMap features from `*.osm.pbf` files.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Optional, Union

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

    def __init__(
        self,
        theme_type_pairs: Optional[list[tuple[str, str]]] = None,
        release: Optional[str] = None,
        include_all_possible_columns: bool = True,
        hierarchy_depth: Optional[int] = None,
        download_directory: Union[str, Path] = "files",
        verbosity_mode: Literal["silent", "transient", "verbose"] = "transient",
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Initialize Overture Maps loader.

        Args:
            theme_type_pairs (Optional[list[tuple[str, str]]], optional): List of theme type pairs
                to download. If None, will download all available datasets. Defaults to None.
            release (Optional[str], optional): Release version. If not provided, will automatically
                load newest available release version. Defaults to None.
            include_all_possible_columns (bool, optional): Whether to have always the same list of
                columns in the resulting file. This ensures that always the same set of columns is
                returned for a given release for different regions. This also means, that some
                columns might be all filled with a False value. Defaults to True.
            hierarchy_depth (Optional[int]): Depth used to calculate how many hierarchy columns
                should be used to generate the wide form of the data. If None, will use all
                available columns. Defaults to None.
            download_directory (Union[str, Path], optional): Directory where to save the downloaded
                GeoParquet files. Defaults to "files".
            verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
                verbosity mode. Can be one of: silent, transient and verbose. Silent disables
                output completely. Transient tracks progress, but removes output after finished.
                Verbose leaves all progress outputs in the stdout. Defaults to "transient".
            max_workers (Optional[int], optional): Max number of multiprocessing workers used to
                process the dataset. Defaults to None.
        """
        import_optional_dependencies(dependency_group="overturemaps", modules=["overturemaestro"])
        self.theme_type_pairs = theme_type_pairs
        self.release = release
        self.include_all_possible_columns = include_all_possible_columns
        self.hierarchy_depth = hierarchy_depth
        self.download_directory = download_directory
        self.verbosity_mode = verbosity_mode
        self.max_workers = max_workers

    def load(
        self,
        area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
        ignore_cache: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Load Overture Maps features for a given area in a wide format.

        The loader will automatically download matching GeoParquet files from
        the S3 bucket provided by the Overture Maps Foundation. Later it will filter
        features and transform them into a wide format. It will return a GeoDataFrame
        containing the `geometry` column and columns for tag keys.

        Note: Some key/value pairs might be missing from the resulting GeoDataFrame,
            simply because there are no such objects in the given area.

        Args:
            area (Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame]):
                Area for which to download objects.
            ignore_cache: (bool, optional): Whether to ignore precalculated geoparquet files or not.
                Defaults to False.

        Returns:
            gpd.GeoDataFrame: Downloaded features as a GeoDataFrame.
        """
        from overturemaestro.advanced_functions import (
            convert_geometry_to_wide_form_geodataframe_for_all_types,
            convert_geometry_to_wide_form_geodataframe_for_multiple_types,
        )

        area_wgs84 = self._prepare_area_gdf(area)

        if self.theme_type_pairs:
            features_gdf = convert_geometry_to_wide_form_geodataframe_for_multiple_types(
                theme_type_pairs=self.theme_type_pairs,
                geometry_filter=area_wgs84.union_all(),
                release=self.release,
                include_all_possible_columns=self.include_all_possible_columns,
                hierarchy_depth=self.hierarchy_depth,
                ignore_cache=ignore_cache,
                working_directory=self.download_directory,
                verbosity_mode=self.verbosity_mode,
                max_workers=self.max_workers,
            )
        else:
            features_gdf = convert_geometry_to_wide_form_geodataframe_for_all_types(
                geometry_filter=area_wgs84.union_all(),
                release=self.release,
                include_all_possible_columns=self.include_all_possible_columns,
                hierarchy_depth=self.hierarchy_depth,
                ignore_cache=ignore_cache,
                working_directory=self.download_directory,
                verbosity_mode=self.verbosity_mode,
                max_workers=self.max_workers,
            )

        features_gdf = features_gdf.set_crs(WGS84_CRS)

        features_columns = [
            column
            for column in features_gdf.columns
            if column != GEOMETRY_COLUMN and features_gdf[column].notnull().any()
        ]
        features_gdf = features_gdf[[GEOMETRY_COLUMN, *sorted(features_columns)]]

        return features_gdf
