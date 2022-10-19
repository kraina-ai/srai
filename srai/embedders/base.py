"""Base class for embedders."""

import abc
from typing import Any

import geopandas as gpd


class BaseEmbedder(abc.ABC):
    """Base abstract class for embedders."""

    @abc.abstractmethod
    def embed(self, gdf: gpd.GeoDataFrame, **kwargs: Any) -> gpd.GeoDataFrame:
        """Embed a given GeoDataFrame.

        Args:
            gdf: GeoDataFrame to be embedded.

        Returns:
            GeoDataFrame with the embedded data.
        """
        raise NotImplementedError
