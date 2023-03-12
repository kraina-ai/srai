"""
GTFS Downloader.

This module contains GTFS downlaoder. GTFS downloader is a proxy to The Mobility Database[1].

References:
    [1] https://database.mobilitydata.org/
"""

from pathlib import Path

import pandas as pd

from srai.utils import download_file

CATALOG_URL = "https://bit.ly/catalogs-csv"
CACHE_DIR = Path.home() / ".cache" / "srai"


class GTFSDownloader:
    """
    GTFSDownloader.

    This class provides methods to search and download GTFS feeds from The Mobility Database[1].
    """

    def __init__(self, update_catalog: bool = False) -> None:
        """
        Initialize GTFS downloader.

        Args:
            update_catalog (bool, optional): Update catalog file if present. Defaults to False.
        """
        self.catalog = self._load_catalog(update_catalog)

    def _load_catalog(self, update_catalog: bool = False) -> pd.DataFrame:
        """
        Load catalog.

        Args:
            update_catalog (bool, optional): Update catalog file if present. Defaults to False.

        Returns:
            pd.DataFrame: Catalog.
        """
        catalog_file = CACHE_DIR / "catalog.csv"

        if not catalog_file.exists() or update_catalog:
            download_file(CATALOG_URL, catalog_file)

        return pd.read_csv(catalog_file)
