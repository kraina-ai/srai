"""Administrative boundary regionalizer tests."""

from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any, Optional, Union

import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry

from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.geometry import merge_disjointed_gdf_geometries
from srai.regionalizers import AdministrativeBoundaryRegionalizer
from srai.regionalizers import administrative_boundary_regionalizer as abr_module

MADAGASCAR_BOUNDARIES_PATH = (
    Path(__file__).parent / "test_files" / "layercake_madagascar_admin_level_4.parquet"
)

bbox = box(minx=-180, maxx=180, miny=-90, maxy=90)
bbox_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [bbox]}, crs=WGS84_CRS)


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
) -> None:
    """Test checks if illegal admin_level is disallowed."""
    with expectation:
        AdministrativeBoundaryRegionalizer(admin_level=admin_level)


def test_empty_gdf_attribute_error(gdf_empty) -> None:  # type: ignore
    """Test checks if empty GeoDataFrames are disallowed."""
    with pytest.raises(AttributeError):
        abr = AdministrativeBoundaryRegionalizer(admin_level=4)
        abr.transform(gdf_empty)


def test_no_crs_gdf_value_error(gdf_no_crs) -> None:  # type: ignore
    """Test checks if GeoDataFrames without crs are disallowed."""
    with pytest.raises(ValueError):
        abr = AdministrativeBoundaryRegionalizer(admin_level=4)
        abr.transform(gdf=gdf_no_crs)


def _boundary(
    id: int,
    geometry: BaseGeometry,
    admin_level: str = "4",
    name: Optional[list[str]] = None,
    names: Optional[dict[str, list[str]]] = None,
    boundary: str = "administrative",
    type: str = "relation",
) -> dict[str, Any]:
    """Create a row with the schema of the Layercake boundaries."""
    return {
        "type": type,
        "id": id,
        "boundary": boundary,
        "admin_level": admin_level,
        "name": name,
        "names": list((names or {}).items()),
        "bbox": dict(zip(("xmin", "ymin", "xmax", "ymax"), geometry.bounds)),
        "geometry": geometry.wkb,
    }


@pytest.fixture  # type: ignore
def boundaries_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use a local file with the boundaries."""
    path = tmp_path / "boundaries.parquet"
    monkeypatch.setattr(abr_module, "LAYERCAKE_BOUNDARIES_URL", str(path))
    return path


def _save_boundaries(path: Path, *boundaries: dict[str, Any]) -> None:
    schema = pa.schema(
        {
            "type": pa.string(),
            "id": pa.int64(),
            "boundary": pa.string(),
            "admin_level": pa.string(),
            "name": pa.list_(pa.string()),
            "names": pa.map_(pa.string(), pa.list_(pa.string())),
            "bbox": pa.struct({k: pa.float32() for k in ("xmin", "ymin", "xmax", "ymax")}),
            "geometry": pa.binary(),
        }
    )
    pq.write_table(pa.Table.from_pylist(list(boundaries), schema=schema), path)


@pytest.fixture  # type: ignore
def mock_boundaries(boundaries_path: Path) -> None:
    """Mock boundaries with a single region on multiple levels."""
    geometry = box(minx=0, miny=0, maxx=1, maxy=1)
    _save_boundaries(
        boundaries_path,
        _boundary(2137, geometry, admin_level="2"),
        _boundary(2138, geometry, admin_level="4"),
    )


@pytest.mark.parametrize(  # type: ignore
    "toposimplify",
    [
        (True),
        (0.0001),
        (0.001),
        (0.01),
        (False),
        (0),
    ],
)
def test_empty_region_full_bounding_box(toposimplify: Union[bool, float], request: Any) -> None:
    """Test checks if empty region fills required bounding box."""
    request.getfixturevalue("mock_boundaries")
    request_bbox = box(minx=0, miny=0, maxx=2, maxy=2)
    request_bbox_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [request_bbox]}, crs=WGS84_CRS)
    abr = AdministrativeBoundaryRegionalizer(
        admin_level=4, return_empty_region=True, toposimplify=toposimplify
    )
    result_gdf = abr.transform(gdf=request_bbox_gdf)
    assert merge_disjointed_gdf_geometries(result_gdf).difference(request_bbox).is_empty
    assert "EMPTY" in result_gdf.index


@pytest.mark.parametrize(  # type: ignore
    "toposimplify",
    [
        (True),
        (0.0001),
        (0.001),
        (0.01),
        (False),
        (0),
    ],
)
def test_no_empty_region_full_bounding_box(toposimplify: Union[bool, float], request: Any) -> None:
    """Test checks if no empty region is generated when not needed."""
    request.getfixturevalue("mock_boundaries")
    request_bbox = box(minx=0, miny=0, maxx=1, maxy=1)
    request_bbox_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [request_bbox]}, crs=WGS84_CRS)
    abr = AdministrativeBoundaryRegionalizer(
        admin_level=2, return_empty_region=True, toposimplify=toposimplify
    )
    result_gdf = abr.transform(gdf=request_bbox_gdf)
    assert merge_disjointed_gdf_geometries(result_gdf).difference(request_bbox).is_empty
    assert "EMPTY" not in result_gdf.index


@pytest.mark.parametrize(  # type: ignore
    "toposimplify",
    [
        (True),
        (0.0001),
        (0.001),
        (0.01),
        (False),
        (0),
    ],
)
def test_points_in_result(toposimplify: Union[bool, float], request: Any) -> None:
    """Test checks case when points are in a requested region."""
    request.getfixturevalue("mock_boundaries")
    request_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [Point(0.5, 0.5)]}, crs=WGS84_CRS)

    abr = AdministrativeBoundaryRegionalizer(
        admin_level=2, return_empty_region=False, clip_regions=False, toposimplify=toposimplify
    )

    result_gdf = abr.transform(gdf=request_gdf)
    assert request_gdf.geometry[0].covered_by(result_gdf.geometry[0])


@pytest.mark.parametrize(  # type: ignore
    "toposimplify",
    [
        (True),
        (0.0001),
        (0.001),
        (0.01),
        (False),
        (0),
    ],
)
def test_toposimplify_on_real_data(
    toposimplify: Union[float, bool], boundaries_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test if toposimplify usage covers an entire region."""
    monkeypatch.setattr(abr_module, "LAYERCAKE_BOUNDARIES_URL", str(MADAGASCAR_BOUNDARIES_PATH))
    madagascar_bbox = box(
        minx=43.2541870461, miny=-25.6014344215, maxx=50.4765368996, maxy=-12.0405567359
    )
    madagascar_bbox_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [madagascar_bbox]}, crs=WGS84_CRS)

    abr = AdministrativeBoundaryRegionalizer(
        admin_level=4, return_empty_region=True, toposimplify=toposimplify
    )
    madagascar_result_gdf = abr.transform(gdf=madagascar_bbox_gdf)
    assert (
        merge_disjointed_gdf_geometries(madagascar_result_gdf).difference(madagascar_bbox).is_empty
    )


@pytest.mark.parametrize(  # type: ignore
    "return_empty_region",
    [
        (True),
        (False),
    ],
)
def test_regions_not_found(
    return_empty_region: bool, boundaries_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test if warns when can't find any regions."""
    monkeypatch.setattr(abr_module, "LAYERCAKE_BOUNDARIES_URL", str(MADAGASCAR_BOUNDARIES_PATH))
    null_island_region = box(minx=0, miny=0, maxx=0.1, maxy=0.1)
    null_island_region_gdf = gpd.GeoDataFrame(
        {GEOMETRY_COLUMN: [null_island_region]}, crs=WGS84_CRS
    )

    abr = AdministrativeBoundaryRegionalizer(
        admin_level=10, return_empty_region=return_empty_region
    )

    with pytest.warns(RuntimeWarning):
        madagascar_result_gdf = abr.transform(gdf=null_island_region_gdf)

    if return_empty_region:
        assert (
            merge_disjointed_gdf_geometries(madagascar_result_gdf)
            .difference(null_island_region)
            .is_empty
        )
    else:
        assert merge_disjointed_gdf_geometries(madagascar_result_gdf).is_empty


def _transform(*geometries: BaseGeometry, **kwargs: Any) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: list(geometries)}, crs=WGS84_CRS)
    return AdministrativeBoundaryRegionalizer(
        admin_level=4, clip_regions=False, toposimplify=False, **kwargs
    ).transform(gdf)


def test_only_administrative_boundaries_on_level(boundaries_path: Path) -> None:
    """Test if other levels and other boundaries are skipped."""
    geometry = box(0, 0, 1, 1)
    _save_boundaries(
        boundaries_path,
        _boundary(1, geometry, name=["Valid"]),
        _boundary(2, geometry, name=["Other level"], admin_level="5"),
        _boundary(3, geometry, name=["Place"], boundary="place"),
        _boundary(4, box(5, 5, 6, 6), name=["Far away"]),
    )
    assert _transform(box(0.2, 0.2, 0.8, 0.8)).index.tolist() == ["Valid"]


def test_ways(boundaries_path: Path) -> None:
    """Test if only named ways not covered by a relation on the same level are used."""
    relation = box(0, 0, 1, 1).union(box(2, 0, 3, 1))
    _save_boundaries(
        boundaries_path,
        # ids are unique only for a given type
        _boundary(1, relation, name=["Relation"]),
        _boundary(1, box(1, 0, 2, 1), name=["Way"], type="way"),
        _boundary(2, box(1, 1, 2, 2), type="way"),
        _boundary(3, box(2, 0, 3, 1), name=["Part of the relation"], type="way"),
        _boundary(4, box(0, 1, 1, 2), name=["Other level"], type="way", admin_level="5"),
    )
    assert sorted(_transform(box(0, 0, 3, 2)).index) == ["Relation", "Way"]


def test_only_ways(boundaries_path: Path) -> None:
    """Test if ways are used without any relations."""
    _save_boundaries(boundaries_path, _boundary(1, box(0, 0, 1, 1), name=["Way"], type="way"))
    assert _transform(box(0.2, 0.2, 0.8, 0.8)).index.tolist() == ["Way"]


@pytest.mark.parametrize(  # type: ignore
    "name,names,prioritize_english_name,expected_region_id",
    [
        (["Polska"], {"en": ["Poland"]}, True, "Poland"),
        (["Polska"], {"en": ["Poland"]}, False, "Polska"),
        (["Polska"], {"de": ["Polen"]}, True, "Polska"),
        (["A", "B"], None, True, "A;B"),
        (None, None, True, "1"),
    ],
)
def test_region_id(
    name: Optional[list[str]],
    names: Optional[dict[str, list[str]]],
    prioritize_english_name: bool,
    expected_region_id: str,
    boundaries_path: Path,
) -> None:
    """Test if proper name is selected."""
    _save_boundaries(boundaries_path, _boundary(1, box(0, 0, 1, 1), name=name, names=names))
    result_gdf = _transform(
        box(0.2, 0.2, 0.8, 0.8), prioritize_english_name=prioritize_english_name
    )
    assert result_gdf.index.tolist() == [expected_region_id]


def test_duplicated_names(boundaries_path: Path) -> None:
    """Test if regions with the same name are distinguishable."""
    _save_boundaries(
        boundaries_path,
        _boundary(1, box(0, 0, 1, 1), name=["Same"]),
        _boundary(1, box(1, 0, 2, 1), name=["Same"], type="way"),
        _boundary(2, box(2, 0, 3, 1), name=["Other"]),
    )
    assert sorted(_transform(box(0, 0, 3, 1)).index) == [
        "Other",
        "Same (relation/1)",
        "Same (way/1)",
    ]


def test_bounding_box_intersection_is_not_enough(boundaries_path: Path) -> None:
    """Test if a region crossing the antimeridian isn't returned for the whole latitude."""
    antimeridian = box(-180, 0, -179, 1).union(box(179, 0, 180, 1))
    _save_boundaries(
        boundaries_path,
        _boundary(1, antimeridian, name=["Antimeridian"]),
        _boundary(2, box(0, 0, 1, 1), name=["Valid"]),
    )
    assert _transform(box(0.2, 0.2, 0.8, 0.8), remove_artefact_regions=False).index.tolist() == [
        "Valid"
    ]


def test_distant_points(boundaries_path: Path) -> None:
    """Test if only regions with points are returned."""
    _save_boundaries(
        boundaries_path,
        _boundary(1, box(0, 0, 1, 1), name=["West"]),
        _boundary(2, box(50, 0, 51, 1), name=["Between"]),
        _boundary(3, box(100, 0, 101, 1), name=["East"]),
    )
    result_gdf = _transform(Point(0.5, 0.5), Point(100.5, 0.5))
    assert sorted(result_gdf.index) == ["East", "West"]


def test_real_data() -> None:
    """Test if the published file has an expected schema."""
    null_island_region_gdf = gpd.GeoDataFrame(
        {GEOMETRY_COLUMN: [box(minx=0, miny=0, maxx=0.1, maxy=0.1)]}, crs=WGS84_CRS
    )
    with pytest.warns(RuntimeWarning, match="Couldn't find any administrative boundaries"):
        AdministrativeBoundaryRegionalizer(admin_level=11).transform(null_island_region_gdf)
