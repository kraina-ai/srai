"""DuckDB extensions helper."""

import duckdb


def prepare_duckdb_extensions() -> None:
    """Install and load required DuckDB extensions."""
    duckdb.install_extension("spatial")
    duckdb.install_extension("h3", repository="community")
    duckdb.install_extension("geography", repository="community")

    duckdb.load_extension("spatial")
    duckdb.load_extension("h3")
    duckdb.load_extension("geography")
