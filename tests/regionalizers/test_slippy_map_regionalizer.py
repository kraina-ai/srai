"""Tests for SlippyMapRegionalizer class."""

from contextlib import nullcontext as does_not_raise
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from srai.constants import WGS84_CRS
from srai.regionalizers import SlippyMapRegionalizer

ZOOM = 11


@pytest.fixture  # type: ignore
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


@pytest.fixture  # type: ignore
def regionalizer() -> SlippyMapRegionalizer:
    """Regionalizer fixture."""
    return SlippyMapRegionalizer(zoom=ZOOM)


def test_transform(regionalizer: SlippyMapRegionalizer, gdf: gpd.GeoDataFrame) -> None:
    """Test returned regions."""
    regions = regionalizer.transform(gdf)

    assert regions.shape[0] == 6, f"Invalid length {regions.shape[0]}"
    for x, y in zip([1120, 1119, 1120, 1121, 1120, 1121], [683, 684, 684, 684, 685, 685]):
        expected_id = f"{x}_{y}_{ZOOM}"
        assert expected_id in regions.index, f"{expected_id} not in index but expected"
        assert regions.loc[expected_id]["x"] == x
        assert regions.loc[expected_id]["y"] == y
        assert regions.loc[expected_id]["z"] == ZOOM


@pytest.mark.parametrize(  # type: ignore
    "z, expectation",
    [
        (-1, pytest.raises(ValueError)),
        (0, does_not_raise()),
        (19, does_not_raise()),
        (20, pytest.raises(ValueError)),
    ],
)
def test_zoom_check(z: int, expectation: Any) -> None:
    """Test value checks."""
    with expectation:
        SlippyMapRegionalizer(zoom=z)


def test_coordinates_cast(regionalizer: SlippyMapRegionalizer) -> None:
    """Test if coordinates_to_x_y gives proper x and y value."""
    # given
    latitude, longitude = 51, 16.8
    regionalizer.zoom = 10

    # when
    x, y = regionalizer._coordinates_to_x_y(latitude=latitude, longitude=longitude)

    # then
    assert x == 559
    assert y == 342


def test_x_y_to_coordinates_should_be_inverse_to_coordinates_to_x_y(
    regionalizer: SlippyMapRegionalizer,
) -> None:
    """Test if `x_y_to_coordinates` is reversible with `coordinates_to_x_y`."""
    # given
    x, y = 50, 100

    # when
    latitude, longitude = regionalizer._x_y_to_coordinates(x, y)
    x_reverse, y_reverse = regionalizer._coordinates_to_x_y(latitude, longitude)

    # then
    assert x_reverse == x
    assert y_reverse == y
