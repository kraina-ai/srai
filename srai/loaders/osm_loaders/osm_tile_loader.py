from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
from urllib.parse import urljoin

import numpy as np
import requests
import shapely.geometry as shpg
from matplotlib import pyplot as plt
from PIL import Image

from srai.utils import geocode_to_region_gdf

from .osm_tile_data_collector import DataCollector, InMemoryDataCollector, SavingDataCollector


class TileLoader:
    """
    Tile Loader.

    Downloads raster tiles from user specified tile server, like listed in [1]. Loader founds x, y
    coordinates [2] for specified area and downloads tiles. Address is built with schema
    {tile_server_url}/{zoom}/{x}/{y}.{resource_type}

    References:
        1. https://wiki.openstreetmap.org/wiki/Raster_tile_providers
        2. https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    """

    def __init__(
        self,
        tile_server_url: str,
        zoom: int,
        verbose: bool = False,
        resource_type: str = "png",
        auth_token: Optional[str] = None,
        collector_factory: Optional[Union[str, Callable[[], DataCollector]]] = None,
        storage_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize TileLoader.

        Args:
            tile_server_url (str): url of tile server, without z, x, y parameters
            zoom (int): zoom level [1]
            verbose (bool, optional): should print logs. Defaults to False.
            resource_type (str, optional): file extension. Added to the end of url.
                Defaults to "png".
            auth_token (Optional[str], optional): auth token. Added as access_token parameter
                to request. Defaults to None.
            collector_factory (Optional[Union[str, Callable[[], DataCollector]]], optional):Function
                returning DataCollector. If None uses InMemoryDataCollector. Defaults to None.
                    If `return` uses  InMemoryDataCollector
                    If `save` uses  SavingDataCollector
            storage_path (Optional[Union[str, Path]], optional): path to save data,
                used with SavingDataCollector. Defaults to None.

        References:
        1. https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
        """
        self.zoom = zoom
        self.verbose = verbose
        self.resource_type = resource_type
        self.base_url = urljoin(tile_server_url, "{0}/{1}/{2}." + resource_type)
        self.auth_token = auth_token
        self.save_path = storage_path
        self.collector_factory = (
            self._get_collector_factory(collector_factory)
            if collector_factory is not None
            else lambda: InMemoryDataCollector()
        )

    # TODO cover with save
    def _get_collector_factory(
        self, storage_strategy: str | Callable[[], DataCollector]
    ) -> Callable[[], DataCollector]:
        if isinstance(storage_strategy, str):
            if storage_strategy == "save":
                assert self.save_path is not None
            return {
                "save": lambda: SavingDataCollector(self.save_path, f_extension=self.resource_type),
                "return": lambda: InMemoryDataCollector(),
            }[storage_strategy]
        else:
            return storage_strategy

    def coordinates_to_x_y(self, latitude: float, longitude: float) -> Tuple[int, int]:
        """
        Counts x and y from latitude and longitude using self.zoom.

        Based on https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Implementations.
        """
        n_rows = 2**self.zoom
        x_tile = int(n_rows * ((longitude + 180) / 360))
        lat_radian = np.radians(latitude)
        y_tile = int((1 - np.arcsinh(np.tan(lat_radian)) / np.pi) / 2 * n_rows)
        return x_tile, y_tile

    def x_y_to_coordinates(self, x: int, y: int) -> Tuple[float, float]:
        """
        Counts latitude and longitude from x, y using self.zoom.

        Based on https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Implementations.
        """
        n = 2.0**self.zoom
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
        lat_deg = np.degrees(lat_rad)
        return (lat_deg, lon_deg)

    def get_tile_by_x_y(self, x: int, y: int) -> Image.Image:
        """
        Downloads single tile from tile server.

        Args:
            x: x tile x coordinate
            y: y tile y coordinate
        """
        url = self.base_url.format(self.zoom, x, y)
        if self.verbose:
            print(f"Getting tile from url: {url}")
        content = requests.get(url, params={"access_token": f"{self.auth_token}"}).content
        return Image.open(BytesIO(content))

    def get_tile_by_region_name(
        self, name: str, return_rect: bool = False
    ) -> List[List[str]] | List[List[Image.Image]]:
        """
        Returns all tiles of region.

        Args:
            name: area name, as in geocode_to_region_gdf
            return_rect: if true returns tiles out of area to keep rectangle shape of img of joined
                tiles.
        """
        gdf = geocode_to_region_gdf(name)["geometry"].item()
        gdf_bounds = gdf.bounds
        x_start, y_start = self.coordinates_to_x_y(gdf_bounds[1], gdf_bounds[0])
        x_end, y_end = self.coordinates_to_x_y(gdf_bounds[3], gdf_bounds[2])
        tiles = self.collector_factory()
        for y in range(y_end, y_start + 1):
            for x in range(x_start, x_end + 1):
                if return_rect or self._should_not_skip(gdf, x, y):
                    tile = self.get_tile_by_x_y(x, y)
                    tiles.store(x, y, tile)
                else:
                    tiles.store(x, y, None)
                    if self.verbose:
                        print(f"Skipping {x}, {y}")
        return tiles.collect()

    def _should_not_skip(self, bounds: shpg.Polygon, x: int, y: int) -> bool:
        latitude_start, longitude_start = self.x_y_to_coordinates(x, y)
        latitude_end, longitude_end = self.x_y_to_coordinates(x + 1, y + 1)
        tile_polygon = shpg.Polygon(
            [
                (longitude_start, latitude_start),
                (longitude_end, latitude_start),
                (longitude_end, latitude_end),
                (longitude_start, latitude_end),
            ]
        )
        return tile_polygon.intersects(bounds)


# TODO should be covered?
def tiles_to_img(tiles: List[List[Image.Image]]) -> Image.Image:
    """
    Creates one Image from list of tiles generated with TileLoader and InMemoryDataCollector.

    Args:
        tiles (List[List[Image.Image]]): tiles made with TileLoader.get_tile_by_region_name

    Returns:
        Image.Image: Image of all tiles
    """
    first_img = tiles[0][0]
    tile_width = first_img.width
    tile_height = first_img.height
    merged_width = len(tiles[0]) * tile_width
    merged_height = len(tiles) * tile_height

    img = Image.new("RGB", (merged_width, merged_height))

    for r, row in enumerate(tiles):
        for t, tile in enumerate(row):
            img.paste(tile, box=(t * tile_width, r * tile_height))
    print(img.width, img.height)
    return img


if __name__ == "__main__":
    loader = TileLoader("https://tile.openstreetmap.de", zoom=6, verbose=True)
    tiles = loader.get_tile_by_region_name("Wroclaw, Poland")
    plt.imshow(tiles_to_img(tiles))
    plt.show()
