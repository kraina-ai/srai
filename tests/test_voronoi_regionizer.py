"""Voronoi regionizer tests."""
import geopandas as gpd
import pytest
from shapely.geometry import Point

from srai.regionizers import VoronoiRegionizer


def test_voronoi_regionizer_value_error() -> None:
    """Test checks if duplicate points are disallowed."""
    with pytest.raises(ValueError):
        seeds_gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0), Point(0, 0)]}, index=[1, 2])
        VoronoiRegionizer(seeds=seeds_gdf)
