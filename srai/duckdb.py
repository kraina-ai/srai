"""DuckDB extensions helper."""

from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import duckdb
from rq_geo_toolkit.constants import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
)
from rq_geo_toolkit.duckdb import run_query_with_memory_monitoring


def prepare_duckdb_extensions(connection: Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """Install and load required DuckDB extensions."""
    _conn = connection or duckdb

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


def relation_to_parquet(
    relation: Union[str, duckdb.DuckDBPyRelation],
    result_parquet_path: Path,
    connection: Optional[duckdb.DuckDBPyConnection] = None,
    tmp_dir_path: Optional[Path] = None,
) -> None:
    """
    Save SQL query to parquet file.

    Will run in external process or use existing DuckDB connection.

    Args:
        relation (Union[str, duckdb.DuckDBPyRelation]): Relation or SQL query to be saved.
        result_parquet_path (Path): Location where to save parquet file.
        connection (duckdb.DuckDBPyConnection, optional): Existing connection to reuse.
            Defaults to None.
        tmp_dir_path (Path, optional): Directory where to create a new DuckDB connection.
            Defaults to None.
    """
    result_parquet_path.parent.mkdir(exist_ok=True, parents=True)

    sql_query = relation if isinstance(relation, str) else relation.sql_query()

    save_query = f"""
    COPY ({sql_query}) TO '{result_parquet_path}' (
        FORMAT parquet,
        COMPRESSION {PARQUET_COMPRESSION},
        COMPRESSION_LEVEL {PARQUET_COMPRESSION_LEVEL},
        ROW_GROUP_SIZE {PARQUET_ROW_GROUP_SIZE}
    );
    """

    try:
        run_query_with_memory_monitoring(
            sql_query=save_query,
            connection=connection,
            tmp_dir_path=tmp_dir_path,
            preserve_insertion_order=True,
            limit_memory=False,
        )
    except:
        # Remove file if wasn't properly created
        result_parquet_path.unlink(missing_ok=True)
        raise
    finally:
        # Remove DuckDB tmp file if still exists
        tmp_path = result_parquet_path.with_stem(f"tmp_{result_parquet_path.stem}")
        tmp_path.unlink(missing_ok=True)
