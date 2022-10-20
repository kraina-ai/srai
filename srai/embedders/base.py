"""Base class for embedders."""

import abc
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt


class BaseEmbedder(abc.ABC):
    """Base abstract class for embedders."""

    @abc.abstractmethod
    def embed(self, gdf: gpd.GeoDataFrame) -> npt.NDArray[np.float64]:
        """Embed a given GeoDataFrame.

        Args:
            gdf: GeoDataFrame to be embedded.

        Returns:
            np.ndarray with embedding for each row.
        """
        raise NotImplementedError
