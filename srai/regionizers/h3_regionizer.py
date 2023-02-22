"""
H3 regionizer.

This module exposes Uber's H3 Hexagonal Hierarchical Geospatial Indexing System [1] as a regionizer.

Note:
    The default API [2] was chosen (basic_str) to ease the implementation.
    It may be beneficial to try the NumPy API for computationally-heavy work.

References:
    1. https://uber.github.io/h3-py/
    2. https://uber.github.io/h3-py/api_comparison
"""

from typing import List

import geopandas as gpd
import h3
from functional import seq
from shapely import geometry

from srai.utils.constants import REGIONS_INDEX, WGS84_CRS

from .base import BaseRegionizer


class H3Regionizer(BaseRegionizer):
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
                H3 Cells (visible on the borders). Defaults to True.

        Raises:
            ValueError: If resolution is not between 0 and 15.

        References:
            1. https://h3geo.org/docs/core-library/restable/
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
        gdf_wgs84 = gdf.to_crs(crs=WGS84_CRS)

        gdf_exploded = self._explode_multipolygons(gdf_wgs84)
        gdf_buffered = self._buffer(gdf_exploded) if self.buffer else gdf_exploded

        h3_indexes = (
            seq(gdf_buffered["geometry"])
            .map(self._polygon_shapely_to_h3)
            .flat_map(lambda polygon: h3.polygon_to_cells(polygon, self.resolution))
            .distinct()
            .to_list()
        )

        gdf_h3 = self._gdf_from_h3_indexes(h3_indexes)

        # there may be too many cells because of too big buffer
        gdf_h3_clipped = (
            gdf_h3.sjoin(gdf_exploded[["geometry"]]).drop(columns="index_right").drop_duplicates()
            if self.buffer
            else gdf_h3
        )

        gdf_h3_clipped.index.name = REGIONS_INDEX

        return gdf_h3_clipped.to_crs(gdf.crs)

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
            crs=WGS84_CRS,
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

    def _buffer(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Buffer geometries to generate H3 cells that cover them entirely.

        Filling the geometries using h3.polygon_to_cells() with H3 cells does
        not cover fully the geometries around the borders, because some H3 cells' centers fall out
        of the geometries. To overcome that a buffering is needed to incorporate these cells back.
        Buffering is done in meters and is liberal, which means that performing
        h3.polygon_to_cells() on new geometries results in some H3 cells not intersecting with the
        original geometries. Spatial join with the original geometries is needed later to solve
        this issue.

        Notes:
            According to [1] the ratio between the biggest and smallest hexagons at a given
            resolution is at maximum ~2. From that we can deduce, that the maximum edge length
            ratio becomes t_max / t_min = sqrt(2). As we would like to buffer at least 1 edge length
            regardless of the size, we need to multiply by at least sqrt(2).

        Args:
            gdf (gpd.GeoDataFrame): Geometries.

        Returns:
            gpd.GeoDataFrame: Geometries buffered around the edges.

        References:
            1. https://h3geo.org/docs/core-library/restable/#hexagon-min-and-max-areas
        """
        return gpd.GeoDataFrame(
            geometry=(
                gdf.to_crs(epsg=3395)
                .buffer(2 * h3.average_hexagon_edge_length(self.resolution, unit="m"))
                .to_crs(crs=WGS84_CRS)
            ),
            index=gdf.index,
        )
