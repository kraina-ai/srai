"""DuckDB extensions helper."""

from typing import Optional

import duckdb


def prepare_duckdb_extensions(conn: Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """Install and load required DuckDB extensions."""
    _conn = conn or duckdb

    _conn.install_extension("spatial")
    _conn.install_extension("h3", repository="community")
    _conn.install_extension("geography", repository="community")

    _conn.load_extension("spatial")
    _conn.load_extension("h3")
    _conn.load_extension("geography")
