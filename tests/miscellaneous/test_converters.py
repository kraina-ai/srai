"""Test if converters properly transform input data."""

from collections.abc import Iterable
from typing import Union
from unittest import TestCase

import geopandas as gpd
import pandas as pd
import pytest
from shapely import Point
from shapely.geometry.base import BaseGeometry

from srai.constants import FEATURES_INDEX, REGIONS_INDEX, WGS84_CRS
from srai.geometry import _convert_to_internal_format
from srai.loaders import convert_to_features_gdf
from srai.regionalizers import convert_to_regions_gdf

ut = TestCase()
TEST_COLUMN_NAME = "test"


@pytest.mark.parametrize(  # type: ignore
    "input",
    [
        Point(0, 0),
        [Point(0, 0), Point(1, 1)],
        gpd.GeoSeries([Point(0, 0), Point(1, 1)]),
        gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)]),
    ],
)
def test_region_gdf_converter(
    input: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
):
    """Test that region gdf converter works."""
    regions_gdf = convert_to_regions_gdf(input)
    assert regions_gdf.index.name == REGIONS_INDEX
    assert regions_gdf.crs == WGS84_CRS


@pytest.mark.parametrize(  # type: ignore
    "input",
    [
        Point(0, 0),
        [Point(0, 0), Point(1, 1)],
        gpd.GeoSeries([Point(0, 0), Point(1, 1)]),
        gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)]),
    ],
)
def test_features_gdf_converter(
    input: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame],
) -> None:
    """Test that features gdf converter works."""
    features_gdf = convert_to_features_gdf(input)
    assert features_gdf.index.name == FEATURES_INDEX
    assert features_gdf.crs == WGS84_CRS


def test_column_index_converter() -> None:
    """Test that index based on column works."""
    gdf = gpd.GeoDataFrame(
        data={TEST_COLUMN_NAME: ["a", "b"]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs=WGS84_CRS,  # cover crs conversion
    )
    features_gdf = _convert_to_internal_format(
        geometry=gdf,
        destination_index_name=FEATURES_INDEX,
        index_column=TEST_COLUMN_NAME,
    )

    ut.assertListEqual(gdf[TEST_COLUMN_NAME].to_list(), features_gdf.index.to_list())


def test_missing_column_index_converter() -> None:
    """Test that index based on missing column doesn't work."""
    with pytest.raises(ValueError):
        _convert_to_internal_format(
            geometry=gpd.GeoDataFrame(
                geometry=[Point(0, 0), Point(1, 1)],
            ),
            destination_index_name=FEATURES_INDEX,
            index_column=TEST_COLUMN_NAME,
        )


def test_multiindex_error() -> None:
    """Test if mutliindex transformation doesn't work."""
    with pytest.raises(ValueError):
        _convert_to_internal_format(
            geometry=gpd.GeoDataFrame(
                geometry=[Point(0, 0), Point(1, 1)],
                index=pd.MultiIndex.from_tuples([("a", "b"), (0, 1)]),
            ),
            destination_index_name=FEATURES_INDEX,
            index_column=None,
        )
