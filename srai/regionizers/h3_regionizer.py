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
import geopandas as gpd
import h3
from functional import seq

from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.regionizers import Regionizer
from srai.utils import buffer_geometry
from srai.utils.h3 import gdf_from_h3_indexes, shapely_polygon_to_h3


class H3Regionizer(Regionizer):
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
            seq(gdf_buffered[GEOMETRY_COLUMN])
            .map(shapely_polygon_to_h3)
            .flat_map(lambda polygon: h3.polygon_to_cells(polygon, self.resolution))
            .distinct()
            .to_list()
        )

        gdf_h3 = gdf_from_h3_indexes(h3_indexes)

        # there may be too many cells because of too big buffer
        if self.buffer:
            gdf_h3_clipped = gdf_h3.sjoin(gdf_exploded[[GEOMETRY_COLUMN]]).drop(
                columns="index_right"
            )
            gdf_h3_clipped = gdf_h3_clipped[~gdf_h3_clipped.index.duplicated(keep="first")]
        else:
            gdf_h3_clipped = gdf_h3

        gdf_h3_clipped.index.name = REGIONS_INDEX

        return gdf_h3_clipped.to_crs(gdf.crs)


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
        buffer_distance_meters = 2 * h3.average_hexagon_edge_length(self.resolution, unit="m")
        buffered_geometries = gdf.geometry.apply(
            lambda polygon: buffer_geometry(polygon, buffer_distance_meters)
        )

        return gpd.GeoDataFrame(
            geometry=buffered_geometries,
            index=gdf.index,
        )
