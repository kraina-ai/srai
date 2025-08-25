"""DuckDB extensions helper."""

from collections.abc import Iterable
from pathlib import Path
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


def relation_from_parquet_paths(
    parquet_paths: Iterable[Path],
    connection: Optional[duckdb.DuckDBPyConnection] = None,
    with_row_number: bool = False,
) -> duckdb.DuckDBPyRelation:
    """Get DuckDB relation from a list of parquet paths."""
    paths = list(map(lambda x: f"'{x}'", parquet_paths))
    if with_row_number:
        sql_query = f"""
        SELECT *, row_number() OVER () as row_number FROM read_parquet([{",".join(paths)}])
        """
    else:
        sql_query = f"SELECT * FROM read_parquet([{','.join(paths)}])"

    if connection is not None:
        return connection.sql(sql_query)

    return duckdb.sql(sql_query)
