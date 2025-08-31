"""Tests for loading geodatatable."""

from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import pytest
from parametrization import Parametrization as P
from shapely.geometry.base import BaseGeometry

from srai.constants import GEOMETRY_COLUMN
from srai.geodatatable import VALID_GEO_INPUT, GeoDataTable, prepare_geo_input
from tests.geodatatable.conftest import EXAMPLE_GEOPARQUET_PATH


def test_geodatatable_from_parquet_path(example_geoparquet_path: Path) -> None:
    """Test if geodatatable can be created from parquet path."""
    gdt = GeoDataTable.from_parquet(
        example_geoparquet_path, persist_files=True, index_column_names="iso_a3"
    )

    assert len(gdt) == 5, "Loaded file rows length mismatch."
    assert len(gdt.columns) == 5, "Loaded file columns length mismatch."


@pytest.mark.parametrize(
    "load_function", [GeoDataTable.from_geodataframe, GeoDataTable.from_dataframe]
)  # type: ignore
def test_geodatatable_from_geodataframe(
    example_geodataframe: gpd.GeoDataFrame, load_function: Callable[..., GeoDataTable]
) -> None:
    """Test if geodatatable can be created from geodataframe."""
    gdt = load_function(example_geodataframe)

    assert len(gdt) == len(example_geodataframe), "Loaded geodataframe rows length mismatch."
    assert len(gdt.columns) == len(example_geodataframe.columns), (
        "Loaded geodataframe columns length mismatch."
    )


@P.parameters("geo_input")  # type: ignore
@P.case("Base path", EXAMPLE_GEOPARQUET_PATH)  # type: ignore
@P.case("Full path", EXAMPLE_GEOPARQUET_PATH.resolve())  # type: ignore
@P.case("Posix path", EXAMPLE_GEOPARQUET_PATH.as_posix())  # type: ignore
@P.case("String path", str(EXAMPLE_GEOPARQUET_PATH))  # type: ignore
@P.case("List of paths", [EXAMPLE_GEOPARQUET_PATH])  # type: ignore
@P.case("GeoDataTable", GeoDataTable.from_parquet(EXAMPLE_GEOPARQUET_PATH, persist_files=True))  # type: ignore
@P.case("GeoDataFrame", gpd.read_parquet(EXAMPLE_GEOPARQUET_PATH))  # type: ignore
@P.case("GeoSeries", gpd.read_parquet(EXAMPLE_GEOPARQUET_PATH)[GEOMETRY_COLUMN])  # type: ignore
@P.case("List of geometries", gpd.read_parquet(EXAMPLE_GEOPARQUET_PATH)[GEOMETRY_COLUMN].to_list())  # type: ignore
@P.case(
    "NumPy array of geometries",
    gpd.read_parquet(EXAMPLE_GEOPARQUET_PATH)[GEOMETRY_COLUMN].to_numpy(),
)  # type: ignore
@P.case("Single geometry", gpd.read_parquet(EXAMPLE_GEOPARQUET_PATH)[GEOMETRY_COLUMN].union_all())  # type: ignore
def test_prepare_geo_input(example_geodatatable: GeoDataTable, geo_input: VALID_GEO_INPUT) -> None:
    """Test if prepare_geo_input function works."""
    gdt = prepare_geo_input(geo_input)
    if gdt.parquet_paths[0].name == "example.parquet":
        gdt.persist()

    print(gdt)

    if not isinstance(geo_input, BaseGeometry):
        assert len(gdt) == len(example_geodatatable), "Loaded file rows length mismatch."

    assert gdt.union_all().equals(example_geodatatable.union_all()), "Union geometry is not equal."
