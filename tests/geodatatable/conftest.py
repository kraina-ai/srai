"""GeoDataTable fixtures."""

from pathlib import Path

import geopandas as gpd
import pytest

from srai.geodatatable import GeoDataTable

EXAMPLE_GEOPARQUET_PATH = Path(__file__).parent / "test_files" / "example.parquet"


@pytest.fixture  # type: ignore
def example_geoparquet_path() -> Path:
    """Get example geoparquet path."""
    return EXAMPLE_GEOPARQUET_PATH


@pytest.fixture  # type: ignore
def example_geodatatable(example_geoparquet_path: Path) -> GeoDataTable:
    """Get example geodatatable object."""
    return GeoDataTable.from_parquet(example_geoparquet_path, persist_files=True)


@pytest.fixture  # type: ignore
def example_geodataframe(example_geoparquet_path: Path) -> gpd.GeoDataFrame:
    """Get example geodataframe object."""
    return gpd.read_parquet(example_geoparquet_path)
