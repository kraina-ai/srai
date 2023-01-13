"""Optional dependencies tests."""
import sys
from contextlib import nullcontext as does_not_raise
from typing import Any, List

import pytest

from srai.utils._optional import ImportErrorHandle, import_optional_dependency


@pytest.fixture(autouse=True)  # type: ignore
def cleanup_imports():
    """Clean imports."""
    yield
    sys.modules.pop("srai", None)


class PackageDiscarder:
    """Mock class for discarding list of packages."""

    def __init__(self) -> None:
        """Init mock class."""
        self.pkgnames: List[str] = []

    def find_spec(self, fullname, path, target=None) -> None:  # type: ignore
        """Throws ImportError if matching module."""
        if fullname in self.pkgnames:
            raise ImportError()


@pytest.fixture  # type: ignore
def no_optional_dependencies(monkeypatch):
    """Mock environment without optional dependencies."""
    d = PackageDiscarder()

    optional_packages = [
        "osmium",
        "osmnx",
        "OSMPythonTools",
        "pymap3d",
        "haversine",
        "spherical_geometry",
    ]
    for package in optional_packages:
        sys.modules.pop(package, None)
        d.pkgnames.append(package)
    sys.meta_path.insert(0, d)
    yield
    sys.meta_path.remove(d)


def _test_voronoi_regionizer() -> None:
    import geopandas as gpd
    from shapely.geometry import Point, box

    from srai.regionizers import VoronoiRegionizer

    seeds_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                Point(17.014997869227177, 51.09919872601259),
                Point(16.935542631959215, 51.09380600286582),
                Point(16.900425, 51.1162552343),
                Point(16.932700, 51.166251),
            ]
        },
        index=[1, 2, 3, 4],
        crs="EPSG:4326",
    )
    vr = VoronoiRegionizer(seeds=seeds_gdf)
    vr.transform(
        gdf=gpd.GeoDataFrame(
            {"geometry": [box(minx=-180, maxx=180, miny=-90, maxy=90)]}, crs="EPSG:4326"
        )
    )


def _test_administrative_boundary_regionizer() -> None:
    import geopandas as gpd
    from shapely.geometry import box

    from srai.regionizers.administrative_boundary_regionizer import (
        AdministrativeBoundaryRegionizer,
    )

    asia_bbox = box(
        minx=69.73278412113555,
        miny=24.988848422533074,
        maxx=88.50230949587835,
        maxy=34.846427760404225,
    )
    asia_bbox_gdf = gpd.GeoDataFrame({"geometry": [asia_bbox]}, crs="EPSG:4326")
    abr = AdministrativeBoundaryRegionizer(
        admin_level=2, return_empty_region=True, toposimplify=0.001
    )
    abr.transform(gdf=asia_bbox_gdf)


@pytest.mark.parametrize(  # type: ignore
    "test_fn",
    [
        (_test_voronoi_regionizer),
        (_test_administrative_boundary_regionizer),
    ],
)
def test_optional_available(test_fn):
    """Test if defined functions are working with optional packages."""
    test_fn()


@pytest.mark.usefixtures("no_optional_dependencies")
@pytest.mark.parametrize(  # type: ignore
    "test_fn",
    [
        (_test_voronoi_regionizer),
        (_test_administrative_boundary_regionizer),
    ],
)
def test_optional_missing(test_fn):
    """Test if defined functions are failing without optional packages."""
    with pytest.raises(ImportError):
        test_fn()


@pytest.mark.usefixtures("no_optional_dependencies")  # type: ignore
@pytest.mark.parametrize(  # type: ignore
    "import_error,expectation",
    [
        (ImportErrorHandle.RAISE, pytest.raises(ImportError)),
        (ImportErrorHandle.WARN, pytest.warns(ImportWarning)),
        (ImportErrorHandle.IGNORE, does_not_raise()),
    ],
)
def test_optional_missing_error_handle(import_error: ImportErrorHandle, expectation: Any) -> None:
    """Test checks if import error handles are working."""
    with expectation:
        import_optional_dependency(dependency_group="test", module="_srai_test", error=import_error)
