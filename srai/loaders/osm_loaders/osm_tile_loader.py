"""
OSM tile loader.

This module implements downloading tiles from given OSM tile server.
"""

from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urljoin

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry.base import BaseGeometry

from srai._optional import import_optional_dependencies
from srai.loaders.osm_loaders._base import prepare_area_gdf_for_loader
from srai.regionalizers.slippy_map_regionalizer import SlippyMapRegionalizer

from .osm_tile_data_collector import (
    DataCollector,
    DataCollectorType,
    InMemoryDataCollector,
    get_collector,
)


class OSMTileLoader:
    """
    OSM Tile Loader.

    Download raster tiles from user specified tile server, like listed in [1]. Loader finds x, y
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
            auth_token (str, optional): auth token. Added as access_token parameter
                to request. Defaults to None.
            data_collector (Union[str, DataCollector], optional): DataCollector object or
            enum defining default collector. If None uses InMemoryDataCollector. Defaults to None.
            If `return` uses  InMemoryDataCollector
            If `save` uses  SavingDataCollector
            storage_path (Union[str, Path], optional): path to save data,
                used with SavingDataCollector. Defaults to None.

        References:
            1. https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
        """
        import_optional_dependencies(dependency_group="osm", modules=["PIL"])
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
        self.regionalizer = SlippyMapRegionalizer(zoom=self.zoom)

    def _get_collector(
        self, storage_strategy: Union[str, DataCollectorType, DataCollector]
    ) -> DataCollector:
        if isinstance(storage_strategy, str):
            return get_collector(
                storage_strategy, save_path=self.save_path, file_extension=self.resource_type
            )
        return storage_strategy

    def load(
        self,
        area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
    ) -> gpd.GeoDataFrame:
        """
        Return all tiles of region.

        Args:
            area (Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame]):
                Area for which to download objects.

        Returns:
            gpd.GeoDataFrame: Pandas of tiles for each region in area transformed by DataCollector
        """
        area_wgs84 = prepare_area_gdf_for_loader(area)
        regions = self.regionalizer.transform(gdf=area_wgs84)
        regions["tile"] = regions.apply(self._get_tile_for_area, axis=1)
        return regions

    def _get_tile_for_area(self, row: pd.Series) -> Any:
        idx = row.name
        return self.get_tile_by_x_y(row["x"], row["y"], idx=idx)

    def get_tile_by_x_y(self, x: int, y: int, idx: Any = None) -> Any:
        """
        Download single tile from tile server. Return tile processed by DataCollector.

        Args:
            x(int): x tile coordinate
            y(int): y tile coordinate
            idx (Any): id of tile, if non created as x_y_self.zoom
        """
        from PIL import Image

        if idx is None:
            idx = f"{x}_{y}_{self.zoom}"
        url = self.base_url.format(self.zoom, x, y)
        if self.verbose:
            print(f"Getting tile from url: {url}")
        content = requests.get(url, params=dict(access_token=self.auth_token)).content
        tile = Image.open(BytesIO(content))
        return self.data_collector.store(idx, tile)
