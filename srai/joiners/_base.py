"""Base class for joiners."""

import abc
from typing import Union

from srai.geodatatable import VALID_GEO_INPUT, GeoDataTable, ParquetDataTable


class Joiner(abc.ABC):
    """Abstract class for joiners."""

    @abc.abstractmethod
    def transform(
        self,
        regions: VALID_GEO_INPUT,
        features: VALID_GEO_INPUT,
    ) -> Union[ParquetDataTable, GeoDataTable]:  # pragma: no cover
        """
        Join features to regions.

        Args:
            regions (VALID_GEO_INPUT): regions with which features are joined
            features (VALID_GEO_INPUT): features to be joined
        Returns:
            ParquetDataTable or GeoDataTable with an intersection of regions and features,
            which contains a MultiIndex and optionally a geometry with the intersection.
        """
        raise NotImplementedError
