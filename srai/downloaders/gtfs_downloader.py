"""
GTFS Downloader.

This module contains GTFS downlaoder. GTFS downloader is a proxy to The Mobility Database[1].

References:
    [1] https://database.mobilitydata.org/
"""

import unicodedata
from pathlib import Path

import pandas as pd
from functional import seq

from srai.utils import download_file

CATALOG_URL = "https://bit.ly/catalogs-csv"
CACHE_DIR = Path.home() / ".cache" / "srai"
CATALOG_SEARCH_COLUMNS = [
    "name",
    "location.country_code",
    "location.subdivision_name",
    "location.municipality",
    "provider",
]


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

    def update_catalog(self) -> None:
        """Update catalog file."""
        self.catalog = self._load_catalog(update_catalog=True)

    def search(self, query: str) -> pd.DataFrame:
        """
        Search catalog using queries in form of "query1, query2, ...".

        Examples: "Wrocław, PL", "New York, US", "Amtrak".

        Args:
            query (str): Search query with elements separated by comma.

        Returns:
            pd.DataFrame: Search results.
        """
        query_processed = seq(query.split(",")).map(self._remove_accents).map(str.strip).to_list()
        catalog_processed = (
            self.catalog[CATALOG_SEARCH_COLUMNS].fillna("").applymap(self._remove_accents)
        )

        return self.catalog[
            seq(catalog_processed).map(lambda row: all(q in row for q in query_processed)).to_list()
        ]

    def _remove_accents(self, text: str) -> str:
        """
        Remove accents from text.

        Will remove all accents ("ś" -> "s", "ü" -> "u") and replace "ł" with "l".

        Args:
            text (str): Text to process.

        Returns:
            str: Text without accents.
        """
        result = "".join(
            c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
        )
        result = result.replace("ł", "l")  # required for Polish

        return result

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
