"""
GTFS Loader.

This module contains GTFS loader which performs time aggregations from timetable data.
It utilizes the `gtfs_kit` library [1].

References:
    [1] https://gitlab.com/mrcagney/gtfs_kit

"""

from pathlib import Path

import geopandas as gpd

from .base import BaseLoader


class GTFSLoader(BaseLoader):
    """
    GTFSLoader.

    This loader is capable of reading GTFS feed and calculates time aggregations for a given time
    slots.

    """

    def load(self, gtfs_file: Path, time_resolution: str = "1H") -> gpd.GeoDataFrame:
        """
        Load GTFS feed and calculate time aggregations for stops.

        Args:
            gtfs_file (Path): Path to the GTFS feed.
            time_resolution (str, optional): Resolution for time aggregations. Defaults to "1H".

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with time aggregations for stops.

        """
        print(f"Loading GTFS feed from {gtfs_file} at {time_resolution} resolution.")
