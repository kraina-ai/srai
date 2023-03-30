"""
Slippy map regionizer.

This module implements Slippy map tilenames [1] as regionizer.

References:
    1. https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
"""
from collections import namedtuple
from typing import Any

import geopandas as gpd
import numpy as np
import shapely.geometry as shpg
from functional import seq

from srai.constants import REGIONS_INDEX, WGS84_CRS
from srai.regionizers import Regionizer

SlippyMapId = namedtuple("SlippyMapId", "x y")


class SlippyMapRegionizer(Regionizer):
    """Regionizer class."""

    def __init__(self, z: int) -> None:
        """
        Initialize SlippyMapRegionizer.

        Args:
            z (int): zoom level

        Raises:
            ValueError: if zoom is not in [0, 19]
        """
        if not 0 <= z <= 19:
            raise ValueError
        self.zoom = z
        super().__init__()

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Regionize a given GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be regionized.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with regionized geometries.
        """
        gdf_wgs84 = gdf.to_crs(crs=WGS84_CRS)
        gdf_exploded = self._explode_multipolygons(gdf_wgs84)

        values = (
            seq(gdf_exploded["geometry"])
            .map(self._to_cells)
            .flat_map(lambda x: x)
            .distinct()
            .map(lambda item: {"geometry": item[2], REGIONS_INDEX: SlippyMapId(item[0], item[1])})
            .to_list()
        )

        gdf = gpd.GeoDataFrame(values, geometry="geometry", crs=WGS84_CRS)
        gdf = gdf.set_index(REGIONS_INDEX)
        return gdf

    def _to_cells(self, polygon: shpg.Polygon) -> list[Any]:
        gdf_bounds = polygon.bounds
        x_start, y_start = self._coordinates_to_x_y(gdf_bounds[1], gdf_bounds[0])
        x_end, y_end = self._coordinates_to_x_y(gdf_bounds[3], gdf_bounds[2])
        tiles = []
        for y in range(y_end, y_start + 1):
            for x in range(x_start, x_end + 1):
                tile_polygon = self._polygon_from_x_y(x, y)
                if self._should_not_skip(polygon, tile_polygon):
                    tiles.append((x, y, tile_polygon))
        return tiles

    def _polygon_from_x_y(self, x: int, y: int) -> shpg.Polygon:
        latitude_start, longitude_start = self._x_y_to_coordinates(x, y)
        latitude_end, longitude_end = self._x_y_to_coordinates(x + 1, y + 1)
        tile_polygon = shpg.Polygon(
            [
                (longitude_start, latitude_start),
                (longitude_end, latitude_start),
                (longitude_end, latitude_end),
                (longitude_start, latitude_end),
            ]
        )
        return tile_polygon

    def _should_not_skip(self, area: shpg.Polygon, tile: shpg.Polygon) -> bool:
        """Checks if tile is inside area boundaries."""
        intersects: bool = tile.intersects(area)
        return intersects

    def _coordinates_to_x_y(self, latitude: float, longitude: float) -> tuple[int, int]:
        """
        Counts x and y from latitude and longitude using self.zoom.

        Based on https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Implementations.
        """
        n_rows = 2**self.zoom
        x_tile = int(n_rows * ((longitude + 180) / 360))
        lat_radian = np.radians(latitude)
        y_tile = int((1 - np.arcsinh(np.tan(lat_radian)) / np.pi) / 2 * n_rows)
        return x_tile, y_tile

    def _x_y_to_coordinates(self, x: int, y: int) -> tuple[float, float]:
        """
        Counts latitude and longitude from x, y using self.zoom.

        Based on https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Implementations.
        """
        n = 2.0**self.zoom
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
        lat_deg = np.degrees(lat_rad)
        return (lat_deg, lon_deg)
