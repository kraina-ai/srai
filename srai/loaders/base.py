"""Base class for loaders."""

import abc
from typing import Any

import geopandas as gpd


class BaseLoader(abc.ABC):
    """Base abstract class for loaders."""

    @abc.abstractmethod
    def download(self, gdf: gpd.GeoDataFrame, **kwargs: Any) -> gpd.GeoDataFrame:
        """Download data for a given area.

        Args:
            gdf: GeoDataFrame with the area of interest.

        Returns:
            GeoDataFrame with the downloaded data.
        """
        raise NotImplementedError
