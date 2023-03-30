"""
OSM tile loader.

This module implements downloading tiles from given OSM tile server.
"""
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional, Union
from urllib.parse import urljoin

import pandas as pd
import requests
from PIL import Image

from srai.regionizers.slippy_map_regionizer import SlippyMapRegionizer
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
        self.regionizer = SlippyMapRegionizer(z=self.zoom)

    def _get_collector_factory(
        self, storage_strategy: Union[str, Callable[[], DataCollector]]
    ) -> Callable[[], DataCollector]:
        if isinstance(storage_strategy, str):
            if storage_strategy == "save" and self.save_path is None:
                raise ValueError
            elif self.save_path is not None:
                save_path: Union[Path, str] = self.save_path
            return {
                "save": lambda: SavingDataCollector(save_path, f_extension=self.resource_type),
                "return": lambda: InMemoryDataCollector(),
            }[storage_strategy]
        else:
            return storage_strategy

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

    def get_tile_by_region_name(self, name: str) -> pd.DataFrame:
        """
        Returns all tiles of region.

        Args:
            name: area name, as in geocode_to_region_gdf
            return_rect: if true returns tiles out of area to keep rectangle shape of img of joined
                tiles.
        """
        tiles_collector = self.collector_factory()
        gdf = geocode_to_region_gdf(name)
        regions = self.regionizer.transform(gdf=gdf)
        data_series = regions.apply(
            lambda row: self._get_tile_for_area(row, tiles_collector), axis=1
        )
        return pd.DataFrame(data_series, columns=["tile"])

    def _get_tile_for_area(self, row: pd.Series, tiles_collector: DataCollector) -> Any:
        x, y = row.name
        tile = self.get_tile_by_x_y(x, y)
        return tiles_collector.store(x, y, tile)
