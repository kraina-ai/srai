"""
H3 regionalizer.

This module exposes Uber's H3 Hexagonal Hierarchical Geospatial Indexing System [1] as
a regionalizer.

Note:
    The default API [2] was chosen (basic_str) to ease the implementation.
    It may be beneficial to try the NumPy API for computationally-heavy work.

References:
    1. https://uber.github.io/h3-py/
    2. https://uber.github.io/h3-py/api_comparison
"""

import geopandas as gpd

from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.h3 import h3_to_geoseries, shapely_geometry_to_h3
from srai.regionalizers import Regionalizer


class H3Regionalizer(Regionalizer):
    """
    H3 Regionalizer.

    H3 Regionalizer allows the given geometries to be divided
    into H3 cells - hexagons with pentagons as a very rare exception
    """

    def __init__(self, resolution: int, buffer: bool = True) -> None:
        """
        Init H3Regionalizer.

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
        Regionalize a given GeoDataFrame.

        Transforms given geometries into H3 cells of given resolution
        and optionally applies buffering.

        Args:
            gdf (gpd.GeoDataFrame): (Multi)Polygons to be regionalized.

        Returns:
            gpd.GeoDataFrame: H3 cells.

        Raises:
            ValueError: If provided GeoDataFrame has no crs defined.
        """
        gdf_wgs84 = gdf.to_crs(crs=WGS84_CRS)

        gdf_exploded = self._explode_multipolygons(gdf_wgs84)

        h3_indexes = list(
            set(
                shapely_geometry_to_h3(
                    gdf_exploded[GEOMETRY_COLUMN],
                    h3_resolution=self.resolution,
                    buffer=self.buffer,
                )
            )
        )
        gdf_h3 = gpd.GeoDataFrame(
            data={REGIONS_INDEX: h3_indexes},
            geometry=h3_to_geoseries(h3_indexes),
            crs=WGS84_CRS,
        ).set_index(REGIONS_INDEX)

        return gdf_h3.to_crs(gdf.crs)
