"""Overture divisions regionalizer tests."""

from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Point, box
from shapely.geometry.base import BaseGeometry

from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.regionalizers import OvertureDivisionsRegionalizer

# Bounding box covers the whole range of longitudes
ANTIMERIDIAN_DIVISION = MultiPolygon([box(-180, 0, -179, 2), box(179, 0, 180, 2)])


def _mock_divisions(
    monkeypatch: pytest.MonkeyPatch, **geometries: BaseGeometry
) -> list[dict[str, Any]]:
    """Replace download with given divisions, returns list of download calls."""
    calls: list[dict[str, Any]] = []

    def download(**kwargs: Any) -> gpd.GeoDataFrame:
        calls.append(kwargs)
        return gpd.GeoDataFrame(
            {
                "division_id": list(geometries),
                "names": [{"primary": f"Name {id[0]}", "common": None} for id in geometries],
                GEOMETRY_COLUMN: list(geometries.values()),
            },
            index=[f"area_{id}" for id in geometries],
            geometry=GEOMETRY_COLUMN,
            crs=WGS84_CRS,
        )

    monkeypatch.setattr("overturemaestro.convert_geometry_to_geodataframe", download)
    return calls


def _gdf(*geometries: BaseGeometry) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({GEOMETRY_COLUMN: list(geometries)}, crs=WGS84_CRS)


def test_unknown_subtype() -> None:
    """Test if unknown subtype is disallowed."""
    with pytest.raises(ValueError, match="Unknown division subtype: city"):
        OvertureDivisionsRegionalizer("city")  # type: ignore[arg-type]


def test_empty_gdf_attribute_error(gdf_empty) -> None:  # type: ignore
    """Test checks if empty GeoDataFrames are disallowed."""
    with pytest.raises(AttributeError):
        OvertureDivisionsRegionalizer("county").transform(gdf_empty)


def test_no_crs_gdf_value_error(gdf_no_crs) -> None:  # type: ignore
    """Test checks if GeoDataFrames without crs are disallowed."""
    with pytest.raises(ValueError):
        OvertureDivisionsRegionalizer("county").transform(gdf_no_crs)


def test_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test if divisions are filtered, clipped and properly indexed."""
    calls = _mock_divisions(
        monkeypatch,
        a1=box(0, 0, 1, 1),
        a2=box(1, 0, 2, 1),
        b1=ANTIMERIDIAN_DIVISION,
    )

    regions_gdf = OvertureDivisionsRegionalizer("county", release="r").transform(
        _gdf(box(0, 0, 1.5, 1))
    )

    assert regions_gdf.index.name == REGIONS_INDEX
    # names aren't unique
    assert regions_gdf["name"].to_dict() == {"a1": "Name a", "a2": "Name a"}
    assert regions_gdf.loc["a2", GEOMETRY_COLUMN].equals(box(1, 0, 1.5, 1))
    assert regions_gdf.crs == WGS84_CRS

    assert len(calls) == 1
    assert calls[0]["release"] == "r"
    assert calls[0]["pyarrow_filter"] == [("subtype", "=", "county"), ("class", "=", "land")]


def test_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test if full divisions are returned for points."""
    _mock_divisions(monkeypatch, a=box(0, 0, 1, 1), b=box(3, 0, 4, 1), c=ANTIMERIDIAN_DIVISION)

    regions_gdf = OvertureDivisionsRegionalizer("county", clip_regions=False).transform(
        _gdf(Point(0.5, 0.5), Point(3.5, 0.5))
    )

    assert regions_gdf.index.tolist() == ["a", "b"]
    assert regions_gdf.loc["b", GEOMETRY_COLUMN].equals(box(3, 0, 4, 1))


def test_nested_localities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test if nested divisions are kept."""
    _mock_divisions(monkeypatch, city=box(0, 0, 2, 2), district=box(0, 0, 1, 1))

    regions_gdf = OvertureDivisionsRegionalizer("locality").transform(_gdf(box(0, 0, 2, 2)))

    assert regions_gdf.area.to_dict() == pytest.approx({"city": 4, "district": 1})


@pytest.mark.parametrize("return_empty_region", [True, False])  # type: ignore
def test_empty_region(monkeypatch: pytest.MonkeyPatch, return_empty_region: bool) -> None:
    """Test if space without divisions is filled only on demand."""
    _mock_divisions(monkeypatch, a=box(0, 0, 1, 1))

    regions_gdf = OvertureDivisionsRegionalizer(
        "county", return_empty_region=return_empty_region
    ).transform(_gdf(box(0, 0, 2, 1)))

    expected = {"a": 1.0, "EMPTY": 1.0} if return_empty_region else {"a": 1.0}
    assert regions_gdf.area.to_dict() == pytest.approx(expected)


@pytest.mark.parametrize("geometries", [{}, {"b": ANTIMERIDIAN_DIVISION}])  # type: ignore
def test_no_divisions(monkeypatch: pytest.MonkeyPatch, geometries: dict[str, BaseGeometry]) -> None:
    """Test if warns and returns an empty result without divisions of a given subtype."""
    _mock_divisions(monkeypatch, **geometries)

    with pytest.warns(RuntimeWarning, match="with `subtype`=county"):
        regions_gdf = OvertureDivisionsRegionalizer("county").transform(_gdf(box(0, 0, 1, 1)))

    assert regions_gdf.empty
