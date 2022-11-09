"""Voronoi regionizer tests."""
from contextlib import nullcontext as does_not_raise
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import box

from srai.regionizers import AdministrativeBoundaryRegionizer

bbox = box(minx=-180, maxx=180, miny=-90, maxy=90)
bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs="EPSG:4326")


@pytest.mark.parametrize(  # type: ignore
    "admin_level,expectation",
    [
        (-1, pytest.raises(ValueError)),
        (0, pytest.raises(ValueError)),
        (1, does_not_raise()),
        (2, does_not_raise()),
        (3, does_not_raise()),
        (4, does_not_raise()),
        (5, does_not_raise()),
        (6, does_not_raise()),
        (7, does_not_raise()),
        (8, does_not_raise()),
        (9, does_not_raise()),
        (10, does_not_raise()),
        (11, does_not_raise()),
        (12, pytest.raises(ValueError)),
    ],
)
def test_admin_level(
    admin_level: int,
    expectation: Any,
    request: Any,
) -> None:
    """Test checks if illegal admin_level is disallowed."""
    with expectation:
        AdministrativeBoundaryRegionizer(admin_level=admin_level)


# TODO: check how empty clipping mask works, check how single points work,
# ensure empty region returns full bounding box,
# ensure no empty region is generated when not needed
