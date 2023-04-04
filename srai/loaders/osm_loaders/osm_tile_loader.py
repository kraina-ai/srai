"""
OSM tile loader.

This module implements downloading tiles from given OSM tile server.
"""
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urljoin

import pandas as pd
import requests
from PIL import Image

from srai.regionizers.slippy_map_regionizer import SlippyMapRegionizer
from srai.utils import geocode_to_region_gdf

from .osm_tile_data_collector import (
    DataCollector,
    DataCollectorType,
    InMemoryDataCollector,
    get_collector,
)


class OSMTileLoader:
    """
    OSM Tile Loader.

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
        data_collector: Optional[Union[str, DataCollector]] = None,
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
            data_collector (Optional[Union[str, DataCollector]], optional): DataCollector object or
            enum defining default collector. If None uses InMemoryDataCollector. Defaults to None.
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
        self.data_collector = (
            self._get_collector(data_collector)
            if data_collector is not None
            else InMemoryDataCollector()
        )
        self.regionizer = SlippyMapRegionizer(z=self.zoom)

    def _get_collector(
        self, storage_strategy: Union[str, DataCollectorType, DataCollector]
    ) -> DataCollector:
        if isinstance(storage_strategy, str):
            return get_collector(
                storage_strategy, save_path=self.save_path, f_extension=self.resource_type
            )
        else:
            return storage_strategy

    def get_tile_by_x_y(self, x: int, y: int) -> Image.Image:
        """
        Downloads single tile from tile server.

        Args:
            x: x tile coordinate
            y: y tile coordinate
        """
        url = self.base_url.format(self.zoom, x, y)
        if self.verbose:
            print(f"Getting tile from url: {url}")
        content = requests.get(url, params={"access_token": f"{self.auth_token}"}).content
        tile = Image.open(BytesIO(content))
        return self.data_collector.store(x, y, tile)

    def get_tile_by_region_name(self, name: str) -> pd.DataFrame:
        """
        Returns all tiles of region.

        Args:
            name: area name, as in geocode_to_region_gdf
        """
        gdf = geocode_to_region_gdf(name)
        regions = self.regionizer.transform(gdf=gdf)
        data_series = regions.apply(lambda row: self._get_tile_for_area(row), axis=1)
        return pd.DataFrame(data_series, columns=["tile"])

    def _get_tile_for_area(self, row: pd.Series) -> Any:
        x, y = row.name
        return self.get_tile_by_x_y(x, y)
