"""Base class for loaders."""

import abc
from collections.abc import Iterable
from typing import Any, Union

from srai.geodatatable import GeoDataTable


class Loader(abc.ABC):
    """Abstract class for loaders."""

    @abc.abstractmethod
    def load(
        self, *args: Any, **kwargs: Any
    ) -> Union[GeoDataTable, Iterable[GeoDataTable]]:  # pragma: no cover
        """
        Load data for a given area.

        Args:
            *args: Positional arguments dependating on a specific loader.
            **kwargs: Keyword arguments dependating on a specific loader.

        Returns:
            GeoDataTable or multiple GeoDataTables with the downloaded data.
        """
        raise NotImplementedError
