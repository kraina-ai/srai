"""Tests for H3Regionizer."""

import osmnx as ox
import pytest

from srai.regionizers import H3Regionizer


@pytest.mark.parametrize(  # type: ignore
    "geoname,resolution,buffer",
    [
        ("Wroclaw, Poland", 7, True),
        ("Tokyo, Japan", 7, False),
        ("London, England", 7, True),
    ],
)
def test_transform(geoname: str, resolution: int, buffer: bool) -> None:
    """
    Tests transform.

    Args:
        geoname (str): name of the place
        resolution (int): resolution for H3 Regionizer
        buffer (bool): whether to buffer the H3 cells to fill the borders

    """
    gdf = ox.geocode_to_gdf(geoname)
    gdf_h3 = H3Regionizer(resolution, buffer).transform(gdf)

    assert len(gdf_h3) > 0
