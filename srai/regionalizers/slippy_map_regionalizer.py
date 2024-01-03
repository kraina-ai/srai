"""
Slippy map regionalizer.

This module implements Slippy map tilenames [1] as a regionalizer.

References:
    1. https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
"""

from itertools import product
from typing import Any

import geopandas as gpd
import numpy as np
import shapely.geometry as shpg
from functional import seq

from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.regionalizers import Regionalizer


class SlippyMapRegionalizer(Regionalizer):
    """SlippyMapRegionalizer class."""

    def __init__(self, zoom: int) -> None:
        """
        Initialize SlippyMapRegionalizer.

        Args:
            zoom (int): zoom level

        Raises:
            ValueError: if zoom is not in [0, 19]
        """
        if not 0 <= zoom <= 19:
            raise ValueError
        self.zoom = zoom
        super().__init__()

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Regionalize a given GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be regionalized.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with regionalized geometries.

        Raises:
            ValueError: If provided GeoDataFrame has no crs defined.
        """
        gdf_wgs84 = gdf.to_crs(crs=WGS84_CRS)
        gdf_exploded = self._explode_multipolygons(gdf_wgs84)

        values = (
            seq(gdf_exploded[GEOMETRY_COLUMN])
            .map(self._to_cells)
            .flatten()
            .map(
                lambda item: (
                    item
                    | {
                        REGIONS_INDEX: f"{item['x']}_{item['y']}_{self.zoom}",
                        "z": self.zoom,
                    }
                )
            )
            .to_list()
        )

        gdf = gpd.GeoDataFrame(values, geometry=GEOMETRY_COLUMN, crs=WGS84_CRS).set_index(
            REGIONS_INDEX
        )
        return gdf.drop_duplicates()

    def _to_cells(self, polygon: shpg.Polygon) -> list[dict[str, Any]]:
        gdf_bounds = polygon.bounds
        x_start, y_start = self._coordinates_to_x_y(gdf_bounds[1], gdf_bounds[0])
        x_end, y_end = self._coordinates_to_x_y(gdf_bounds[3], gdf_bounds[2])
        tiles = []
        for x, y in product(range(x_start, x_end + 1), range(y_end, y_start + 1)):
            tile_polygon = self._polygon_from_x_y(x, y)
            if not self._should_skip_tile(polygon, tile_polygon):
                tiles.append(dict(x=x, y=y, geometry=tile_polygon))
        return tiles

    def _polygon_from_x_y(self, x: int, y: int) -> shpg.Polygon:
        latitude_start, longitude_start = self._x_y_to_coordinates(x, y)
        latitude_end, longitude_end = self._x_y_to_coordinates(x + 1, y + 1)
        tile_polygon = shpg.box(
            minx=longitude_start, miny=latitude_start, maxx=longitude_end, maxy=latitude_end
        )
        return tile_polygon

    def _should_skip_tile(self, area: shpg.Polygon, tile: shpg.Polygon) -> bool:
        """
        Check if tile is outside area boundaries.

        If so, skip it.
        """
        intersects: bool = tile.intersects(area)
        return not intersects

    def _coordinates_to_x_y(self, latitude: float, longitude: float) -> tuple[int, int]:
        """
        Convert latitude and longitude into x and y values using `self.zoom`.

        Based on https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Implementations.
        """
        n_rows = 2**self.zoom
        x_tile = int(n_rows * ((longitude + 180) / 360))
        lat_radian = np.radians(latitude)
        y_tile = int((1 - np.arcsinh(np.tan(lat_radian)) / np.pi) / 2 * n_rows)
        return x_tile, y_tile

    def _x_y_to_coordinates(self, x: int, y: int) -> tuple[float, float]:
        """
        Convert x and y values into latitude and longitude using `self.zoom`.

        Based on https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Implementations.
        """
        n = 2.0**self.zoom
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
        lat_deg = np.degrees(lat_rad)
        return (lat_deg, lon_deg)
