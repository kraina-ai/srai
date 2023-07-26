"""Base class for loaders."""

import abc
from typing import Any

import geopandas as gpd


class Loader(abc.ABC):
    """Abstract class for loaders."""

    @abc.abstractmethod
    def load(self, *args: Any, **kwargs: Any) -> gpd.GeoDataFrame:  # pragma: no cover
        """
        Load data for a given area.

        Args:
            *args: Positional arguments dependating on a specific loader.
            **kwargs: Keyword arguments dependating on a specific loader.

        Returns:
            GeoDataFrame with the downloaded data.
        """
        raise NotImplementedError
