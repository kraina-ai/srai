"""Base class for regionizers."""

import abc
from typing import Any

import geopandas as gpd


class BaseRegionizer(abc.ABC):
    """Base abstract class for regionizers."""

    @abc.abstractmethod
    def transform(self, gdf: gpd.GeoDataFrame, **kwargs: Any) -> gpd.GeoDataFrame:
        """Regionize a given GeoDataFrame.

        Args:
            gdf: GeoDataFrame to be regionized.

        Returns:
            GeoDataFrame with the regionized data.
        """
        raise NotImplementedError
