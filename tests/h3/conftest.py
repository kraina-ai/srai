"""Conftest for H3 tests."""
from typing import List

import geopandas as gpd
import pytest
from shapely.geometry import Point

from srai.constants import WGS84_CRS

pytest_plugins = ["tests.regionalizers.fixtures", "tests.regionalizers.test_h3_regionalizer"]


@pytest.fixture  # type: ignore
def gdf_single_point() -> gpd.GeoDataFrame:
    """Get the point case."""
    return gpd.GeoDataFrame(geometry=[Point(17.9261, 50.6696)], crs=WGS84_CRS)


@pytest.fixture  # type: ignore
def expected_point_h3_index() -> List[str]:
    """Get expected h3 index for the point case."""
    return [
        "8a1e23c44b5ffff",
    ]
