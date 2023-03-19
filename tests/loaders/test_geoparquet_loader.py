"""Geoparquet loader tests."""
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from srai.loaders import GeoparquetLoader
from srai.utils.constants import WGS84_CRS

bbox = box(minx=-180, maxx=180, miny=-90, maxy=90)
bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]})


def test_wrong_path_error() -> None:
    """Test checks if cannot load nonexistent file."""
    with pytest.raises(FileNotFoundError):
        GeoparquetLoader().load(
            file_path=Path(__file__).parent / "test_files" / "non_existent.parquet"
        )


def test_correct_path() -> None:
    """Test checks if can load proper file."""
    gdf = GeoparquetLoader().load(
        file_path=Path(__file__).parent / "test_files" / "example.parquet"
    )
    assert len(gdf.index) == 5
    assert gdf.crs.to_epsg() == 4326


def test_wrong_index_value_error() -> None:
    """Test checks if cannot set nonexistent index."""
    with pytest.raises(ValueError):
        GeoparquetLoader().load(
            file_path=Path(__file__).parent / "test_files" / "example.parquet",
            index_column="test_column",
        )


def test_clipped_columns() -> None:
    """Test if returned a subset of columns."""
    gdf = GeoparquetLoader().load(
        file_path=Path(__file__).parent / "test_files" / "example.parquet",
        columns=["continent", "name"],
    )
    assert len(gdf.columns == 3)


def test_setting_index() -> None:
    """Test if properly set the index."""
    gdf = GeoparquetLoader().load(
        file_path=Path(__file__).parent / "test_files" / "example.parquet", index_column="name"
    )
    expected_index = {"Canada", "Fiji", "Tanzania", "United States of America", "W. Sahara"}
    assert expected_index == set(gdf.index)


def test_clipping() -> None:
    """Test if properly clips the data."""
    bbox = box(minx=-106.645646, maxx=-93.508292, miny=25.837377, maxy=36.500704)
    bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs=WGS84_CRS)
    gdf = GeoparquetLoader().load(
        file_path=Path(__file__).parent / "test_files" / "example.parquet", area=bbox_gdf
    )
    assert len(gdf.index) == 1
