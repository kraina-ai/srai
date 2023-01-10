"""
GTFS Loader.

This module contains GTFS loader which performs time aggregations from timetable data.
It utilizes the `gtfs_kit` library [1]. It was originally created for the purpose of
the gtfs2vec project [2].

References:
    [1] https://gitlab.com/mrcagney/gtfs_kit
    [2] https://doi.org/10.1145/3486640.3491392

"""

from pathlib import Path

import geopandas as gpd
import gtfs_kit as gk
import pandas as pd
from shapely.geometry import Point


class GTFSLoader:
    """
    GTFSLoader.

    This loader is capable of reading GTFS feed and calculates time aggregations in 1H slots.

    """

    def __init__(self) -> None:
        """Initialize GTFS loader."""
        self.time_resolution = "1H"

    def load(self, gtfs_file: Path) -> gpd.GeoDataFrame:
        """
        Load GTFS feed and calculate time aggregations for stops.

        Args:
            gtfs_file (Path): Path to the GTFS feed.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with trip counts and list of directions for stops.

        """
        feed = gk.read_feed(gtfs_file, dist_units="km")

        trips_df = self._load_trips(feed)
        trips_df.columns = [f"trip_count_{c}" for c in trips_df.columns]

        stops_df = feed.stops[["stop_id", "stop_lat", "stop_lon"]].set_index("stop_id")
        stops_df["geometry"] = stops_df.apply(
            lambda row: Point([row["stop_lon"], row["stop_lat"]]), axis=1
        )

        result_gdf = gpd.GeoDataFrame(
            trips_df.merge(stops_df["geometry"], how="inner", on="stop_id"),
            geometry="geometry",
            crs="EPSG:4326",
        )

        result_gdf.index.name = None

        return result_gdf

    def _load_trips(self, feed: gk.Feed) -> pd.DataFrame:
        """
        Load trips from GTFS feed.

        Calculate sum of trips from stop in each time slot.

        Args:
            feed (gk.Feed): GTFS feed.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with trips.

        """
        # FIXME: this takes first wednesday from the feed, may not be the best,
        # but that is what I did in gtfs2vec
        date = feed.get_first_week()[2]
        ts = feed.compute_stop_time_series([date], freq=self.time_resolution)

        records = []

        for idx, row in ts.iterrows():
            h = idx.hour
            for s, n in row["num_trips"].items():
                records.append((s, h, n))

        df = pd.DataFrame(records, columns=["stop_id", "hour", "num_trips"])
        df = df.pivot_table(index="stop_id", columns="hour", values="num_trips", fill_value=0)

        return df

    def _load_directions(self, feed: gk.Feed) -> gpd.GeoDataFrame:
        """
        Load directions from GTFS feed.

        Create a list of unique directions for each stop and time slot.

        Args:
            feed (gk.Feed): GTFS feed.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with directions.

        """
        # TODO: implement
        raise NotImplementedError
