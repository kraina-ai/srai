"""
H3 regionizer.

This module exposes Uber's H3 Hexagonal Hierarchical Geospatial Indexing System [1] as a regionizer.

Note:
    The default API [2] was chosen (basic_str) to ease the implementation.
    It may be beneficial to try the NumPy API for computationally-heavy work.

References:
    [1] https://uber.github.io/h3-py/
    [2] https://uber.github.io/h3-py/api_comparison

"""

from typing import List

import geopandas as gpd
import h3
import pandas as pd
from functional import seq
from shapely import geometry


class H3Regionizer:
    """
    H3 Regionizer.

    H3 Regionizer allows the given geometries to be divided
    into H3 cells - hexagons with pentagons as a very rare exception

    """

    def __init__(self, resolution: int, buffer: bool = True) -> None:
        """
        Init H3Regionizer.

        Args:
            resolution (int): Resolution of the cells. See [1] for a full comparison.
            buffer (bool, optional): Whether to fully cover the geometries with
                H3 Cells (visible on the borders).
                Turn off for large geometries, as it's computationally expensive.
                Defaults to True.

        Raises:
            ValueError: If resolution is not between 0 and 15.

        References:
            [1] https://h3geo.org/docs/core-library/restable/

        """
        if not (0 <= resolution <= 15):
            raise ValueError(f"Resolution {resolution} is not between 0 and 15.")

        self.resolution = resolution
        self.buffer = buffer

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Regionize a given GeoDataFrame.

        Transforms given geometries into H3 cells of given resolution
        and optionally applies buffering.

        Args:
            gdf (gpd.GeoDataFrame): (Multi)Polygons to be regionized.

        Returns:
            gpd.GeoDataFrame: H3 cells.

        Raises:
            ValueError: If provided GeoDataFrame has no crs defined.

        """
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        # transform multipolygons to multiple polygons
        gdf_exploded = gdf_wgs84.explode(index_parts=True).reset_index(drop=True)

        h3_indexes = (
            seq(gdf_exploded["geometry"])
            .map(self._polygon_shapely_to_h3)
            .flat_map(lambda polygon: h3.polygon_to_cells(polygon, self.resolution))
            .to_list()
        )

        gdf_h3 = self._gdf_from_h3_indexes(h3_indexes)
        if self.buffer:
            return self._buffer(gdf_exploded, gdf_h3)

        return gdf_h3.to_crs(gdf.crs)

    def _polygon_shapely_to_h3(self, polygon: geometry.Polygon) -> h3.Polygon:
        """
        Convert Shapely Polygon to H3 Polygon.

        Args:
            polygon (geometry.Polygon): Shapely polygon to be converted.

        Returns:
            h3.Polygon: Converted polygon.

        """
        exterior = [coord[::-1] for coord in list(polygon.exterior.coords)]
        interiors = [
            [coord[::-1] for coord in list(interior.coords)] for interior in polygon.interiors
        ]
        return h3.Polygon(exterior, *interiors)

    def _gdf_from_h3_indexes(self, h3_indexes: List[str]) -> gpd.GeoDataFrame:
        """
        Convert H3 Indexes to GeoDataFrame with geometries.

        Args:
            h3_indexes (List[str]): H3 Indexes.

        Returns:
            gpd.GeoDataFrame: H3 cells.

        """
        return gpd.GeoDataFrame(
            None,
            index=h3_indexes,
            geometry=[self._h3_index_to_shapely_polygon(h3_index) for h3_index in h3_indexes],
            crs="epsg:4326",
        )

    def _h3_index_to_shapely_polygon(self, h3_index: str) -> geometry.Polygon:
        """
        Convert H3 Index to Shapely polygon.

        Args:
            h3_index (str): H3 Index to be converted.

        Returns:
            geometry.Polygon: Converted polygon.

        """
        return self._polygon_h3_to_shapely(h3.cells_to_polygons([h3_index])[0])

    def _polygon_h3_to_shapely(self, polygon: h3.Polygon) -> geometry.Polygon:
        """
        Convert H3 Polygon to Shapely Polygon.

        Args:
            polygon (h3.Polygon): H3 Polygon to be converted.

        Returns:
            geometry.Polygon: Converted polygon.

        """
        return geometry.Polygon(
            shell=[coord[::-1] for coord in polygon.outer],
            holes=[[coord[::-1] for coord in hole] for hole in polygon.holes],
        )

    def _buffer(self, gdf: gpd.GeoDataFrame, gdf_h3: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Buffer H3 Cells to cover the entire geometries.

        Filling the geometries using h3.polygon_to_cells() with H3 cells does
        not cover fully the geometries around the borders. To overcome this
        the approach inspired by [1] is used with the difference, that the neighbouring
        cells are found for all of the cells, not only the borders.

        Notes:
            This implementation is quite slow. There is an optimization possible to find
            only the neighbours for the border cells. It was done that way however,
            because it is hard to identify contiguous cells and handle this case correctly.

            Other solutions are to buffer the geometries by some constant * edge length,
            but that gives imprefect solutions due to different H3 cell edge lenghts across
            the globe.

        Args:
            gdf (gpd.GeoDataFrame): Geometries.
            gdf_h3 (gpd.GeoDataFrame): H3 cells.

        Returns:
            gpd.GeoDataFrame: A superset of original H3 cells with added cells on the borders.

        References:
            [1] https://stackoverflow.com/a/62519680

        """
        h3_disk = seq(gdf_h3.index).map(h3.grid_disk).flatten().distinct().to_list()
        gdf_h3_disk = self._gdf_from_h3_indexes(h3_disk)

        gdf_h3_border = (
            gdf_h3_disk.sjoin(gdf[["geometry"]], predicate="overlaps")
            .drop(columns="index_right")
            .drop_duplicates()
        )

        return pd.concat([gdf_h3, gdf_h3_border], axis=0).drop_duplicates()
