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
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from srai.utils._optional import import_optional_dependencies
from srai.utils.constants import WGS84_CRS

if TYPE_CHECKING:
    from gtfs_kit import Feed


class GTFSLoader:
    """
    GTFSLoader.

    This loader is capable of reading GTFS feed and calculates time aggregations in 1H slots.

    """

    def __init__(self) -> None:
        """Initialize GTFS loader."""
        import_optional_dependencies(dependency_group="gtfs", modules=["gtfs_kit"])

        self.time_resolution = "1H"

    def load(
        self,
        gtfs_file: Path,
        fail_on_validation_errors: bool = True,
        skip_validation: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Load GTFS feed and calculate time aggregations for stops.

        Args:
            gtfs_file (Path): Path to the GTFS feed.
            fail_on_validation_errors (bool): Fail if GTFS feed is invalid.
            skip_validation (bool): Skip GTFS feed validation.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with trip counts and list of directions for stops.

        """
        import gtfs_kit as gk

        feed = gk.read_feed(gtfs_file, dist_units="km")

        if not skip_validation:
            self._validate_feed(feed, fail=fail_on_validation_errors)

        trips_df = self._load_trips(feed)
        directions_df = self._load_directions(feed)

        stops_df = feed.stops[["stop_id", "stop_lat", "stop_lon"]].set_index("stop_id")
        stops_df["geometry"] = stops_df.apply(
            lambda row: Point([row["stop_lon"], row["stop_lat"]]), axis=1
        )

        result_gdf = gpd.GeoDataFrame(
            trips_df.merge(stops_df["geometry"], how="inner", on="stop_id"),
            geometry="geometry",
            crs=WGS84_CRS,
        )

        result_gdf = result_gdf.merge(directions_df, how="left", on="stop_id")

        result_gdf.index.name = None

        return result_gdf

    def _load_trips(self, feed: "Feed") -> pd.DataFrame:
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
        df = df.add_prefix("trip_count_at_")

        return df

    def _load_directions(self, feed: "Feed") -> gpd.GeoDataFrame:
        """
        Load directions from GTFS feed.

        Create a list of unique directions for each stop and time slot.

        Args:
            feed (gk.Feed): GTFS feed.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with directions.

        """
        df = feed.stop_times.merge(feed.trips, on="trip_id")
        df = df.merge(feed.stops, on="stop_id")

        df = df[df["departure_time"].notna()]

        df["hour"] = df["departure_time"].apply(lambda x: int(x[:2].replace(":", "")) % 24)

        pivoted = df.pivot_table(
            values="trip_headsign", index="stop_id", columns="hour", aggfunc=set
        )
        pivoted = pivoted.add_prefix("directions_at_")

        return pivoted

    def _validate_feed(self, feed: "Feed", fail: bool = True) -> None:
        """
        Validate GTFS feed.

        Args:
            feed (gk.Feed): GTFS feed.
            fail (bool): Fail if feed is invalid.

        """
        validation_result = feed.validate()

        if (validation_result["type"] == "error").sum() > 0:
            import warnings

            warnings.warn(f"Invalid GTFS feed: \n{validation_result}", RuntimeWarning)
            if fail:
                raise ValueError("Invalid GTFS feed.")
