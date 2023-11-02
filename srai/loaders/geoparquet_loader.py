"""
Geoparquet loader.

This module contains geoparquet loader implementation.
"""

from pathlib import Path
from typing import Optional, Union

import geopandas as gpd

from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.loaders import Loader


class GeoparquetLoader(Loader):
    """
    GeoparquetLoader.

    Geoparquet [1] loader is a wrapper for a `geopandas.read_parquet` function
    and allows for an automatic index setting and additional geometry clipping.

    References:
        1. https://github.com/opengeospatial/geoparquet
    """

    def load(
        self,
        file_path: Union[Path, str],
        index_column: Optional[str] = None,
        columns: Optional[list[str]] = None,
        area: Optional[gpd.GeoDataFrame] = None,
    ) -> gpd.GeoDataFrame:
        """
        Load a geoparquet file.

        Args:
            file_path (Union[Path, str]): parquet file path.
            index_column (str, optional): Column that will be used as an index.
                If not provided, automatic indexing will be applied by default. Defaults to None.
            columns (List[str], optional): List of columns to load.
                If not provided, all will be loaded. Defaults to None.
            area (gpd.GeoDataFrame, optional): Mask to clip loaded data.
                If not provided, unaltered data will be returned. Defaults to None.

        Raises:
            ValueError: If provided index column doesn't exists in list of loaded columns.

        Returns:
            gpd.GeoDataFrame: Loaded geoparquet file as a GeoDataFrame.
        """
        if columns and GEOMETRY_COLUMN not in columns:
            columns.append(GEOMETRY_COLUMN)

        gdf = gpd.read_parquet(path=file_path, columns=columns)

        if index_column:
            if index_column not in gdf.columns:
                raise ValueError(f"Column {index_column} doesn't exist in a file.")
            gdf.set_index(index_column, inplace=True)

        gdf.to_crs(crs=WGS84_CRS, inplace=True)

        if area is not None:
            area_wgs84 = area.to_crs(crs=WGS84_CRS)
            gdf = gdf.clip(mask=area_wgs84, keep_geom_type=False)

        return gdf
