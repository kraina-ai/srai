"""TODO."""
import duckdb
from geopandas import GeoDataFrame


class DuckDbGeoDataFrame(GeoDataFrame):  # type: ignore
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        self.conn = duckdb.connect()
