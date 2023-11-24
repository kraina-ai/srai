"""
GTFS Downloader.

This module contains GTFS downlaoder. GTFS downloader is a proxy to The Mobility Database[1].

References:
    [1] https://database.mobilitydata.org/
"""

import unicodedata
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
from functional import seq
from shapely.geometry import box

from srai.constants import WGS84_CRS
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
CATALOG_BBOX_COLUMNS = [
    "location.bounding_box.minimum_longitude",
    "location.bounding_box.minimum_latitude",
    "location.bounding_box.maximum_longitude",
    "location.bounding_box.maximum_latitude",
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

    def search(
        self, query: Optional[str] = None, area: Optional[gpd.GeoDataFrame] = None
    ) -> pd.DataFrame:
        """
        Search catalog by name, location or area.

        Examples for text queries: "Wrocław, PL", "New York, US", "Amtrak".

        Args:
            query (str): Search query with elements separated by comma.
            area (gpd.GeoDataFrame): Area to search in.

        Returns:
            pd.DataFrame: Search results.

        Raises:
            ValueError: If `area` is not a GeoDataFrame (has no geometry column).
            ValueError: If neither `query` nor `area` is provided.
        """
        if query is None and area is None:
            raise ValueError("Either query or area must be provided.")

        if query is not None:
            query_filter = self._search_by_query(query)
        else:
            query_filter = [True] * len(self.catalog)

        if area is not None:
            if "geometry" not in area.columns:
                raise ValueError("Provided area has no geometry column.")

            area_filter = self._search_by_area(area)
        else:
            area_filter = [True] * len(self.catalog)

        return self.catalog[query_filter & area_filter]

    def _search_by_query(self, query: str) -> pd.Series:
        """
        Perform search by query.

        Args:
            query (str): Search query with elements separated by comma.

        Returns:
            pd.Series: Series of booleans indicating if row matches the query.
        """
        query_processed = seq(query.split(",")).map(self._remove_accents).map(str.strip).to_list()
        catalog_processed = (
            self.catalog[CATALOG_SEARCH_COLUMNS].fillna("").applymap(self._remove_accents)
        )

        res: List[bool] = (
            seq(catalog_processed).map(lambda row: all(q in row for q in query_processed)).to_list()
        )
        return pd.Series(res, dtype=bool)

    def _search_by_area(self, area: gpd.GeoDataFrame) -> pd.Series:
        """
        Perform search by area.

        Args:
            area (gpd.GeoDataFrame): Area to search in.

        Returns:
            pd.Series: Series of booleans indicating if row matches the area.
        """
        area = area.to_crs(WGS84_CRS)
        result = self.catalog.intersects(area.geometry.unary_union)
        return result

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

    def _load_catalog(self, update_catalog: bool = False) -> gpd.GeoDataFrame:
        """
        Load catalog and add geometry column.

        Args:
            update_catalog (bool, optional): Update catalog file if present. Defaults to False.

        Returns:
            pd.DataFrame: Catalog.
        """
        catalog_file = CACHE_DIR / "catalog.csv"

        if not catalog_file.exists() or update_catalog:
            download_file(CATALOG_URL, catalog_file)

        df = pd.read_csv(catalog_file)

        df[CATALOG_BBOX_COLUMNS] = df[CATALOG_BBOX_COLUMNS].fillna(0)

        df["geometry"] = df.apply(
            lambda row: (box(*row[CATALOG_BBOX_COLUMNS].tolist())),
            axis=1,
        )

        return gpd.GeoDataFrame(df, geometry="geometry", crs=WGS84_CRS)
