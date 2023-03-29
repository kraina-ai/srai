from contextlib import nullcontext as does_not_raise
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from srai.constants import WGS84_CRS
from srai.regionizers import SlippyMapRegionizer


@pytest.fixture
def gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame approximating Warsaw bounds."""
    polygon = Polygon(
        [
            (16.8073393, 51.1389477),
            (17.0278673, 51.0426754),
            (17.1762192, 51.1063195),
            (16.9580276, 51.2093551),
            (16.8073393, 51.1389477),
        ],
    )
    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs=WGS84_CRS)
    return gdf


@pytest.fixture
def regionizer() -> SlippyMapRegionizer:
    return SlippyMapRegionizer(z=11)


def test_transform(regionizer: SlippyMapRegionizer, gdf: gpd.GeoDataFrame) -> None:
    """Tests returned regions."""
    regions = regionizer.transform(gdf)

    assert regions.shape[0] == 6, f"Invalid length {regions.shape[0]}"
    for x, y in zip([1120, 1119, 1120, 1121, 1120, 1121], [683, 684, 684, 684, 685, 685]):
        assert (x, y) in regions.index, f"{(x, y)} not in index but expected"


@pytest.mark.parametrize(
    "z, expectation",
    [
        (-1, pytest.raises(ValueError)),
        (0, does_not_raise()),
        (19, does_not_raise()),
        (20, pytest.raises(ValueError)),
    ],
)
def test_zoom_check(z: int, expectation: Any, gdf: gpd.GeoDataFrame) -> None:
    """Tests value checks."""
    with expectation:
        SlippyMapRegionizer(z=z).transform(gdf)
