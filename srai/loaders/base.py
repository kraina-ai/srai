"""Base class for loaders."""

import abc
from typing import Any

import geopandas as gpd


class BaseLoader(abc.ABC):
    """Base abstract class for loaders."""

    @abc.abstractmethod
    def load(self, gdf: gpd.GeoDataFrame, **kwargs: Any) -> gpd.GeoDataFrame:
        """Load data for a given area.

        Args:
            gdf: GeoDataFrame with the area of interest.

        Returns:
            GeoDataFrame with the downloaded data.
        """
        raise NotImplementedError
