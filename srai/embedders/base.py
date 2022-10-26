"""Base class for embedders."""

import abc

import geopandas as gpd


class BaseEmbedder(abc.ABC):
    """Base abstract class for embedders."""

    @abc.abstractmethod
    def embed(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Embed a given GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be embedded.

        Returns:
            np.ndarray with embedding for each row.

        """
        raise NotImplementedError
