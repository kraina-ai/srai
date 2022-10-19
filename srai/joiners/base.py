"""Base class for joiners."""

import abc

import geopandas as gpd

from ..regnionizers.base import BaseRegionizer


class BaseJoiner(abc.ABC):
    """Base abstract class for joiners."""

    def __init__(self, regionizer: BaseRegionizer):
        """Initialize the joiner.

        Args:
            regionizer: Regionizer to be used.
        """
        self.regionizer = regionizer

    @abc.abstractmethod
    def join(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Join a given GeoDataFrame.

        Args:
            gdf: GeoDataFrame to be joined.

        Returns:
            GeoDataFrame with the joined data.
        """
        raise NotImplementedError
