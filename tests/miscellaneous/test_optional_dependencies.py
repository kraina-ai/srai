"""Optional dependencies tests."""

import sys
from contextlib import nullcontext as does_not_raise
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import box

from srai._optional import ImportErrorHandle, import_optional_dependency
from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS


@pytest.fixture  # type: ignore
def optional_packages() -> list[str]:
    """Get a list with optional packages."""
    return [
        "quackosm",
        "osmnx",
        "overpass",
        "beautifulsoup4",
        "pymap3d",
        "haversine",
        "scipy",
        "spherical_geometry",
        "gtfs_kit",
        "folium",
        "mapclassify",
        "plotly",
        "kaleido",
        "pytorch-lightning",
        "torch",
    ]


@pytest.fixture(autouse=True)  # type: ignore
def cleanup_imports():
    """Clean imports."""
    yield
    sys.modules.pop("srai", None)


class PackageDiscarder:
    """Mock class for discarding list of packages."""

    def __init__(self) -> None:
        """Init mock class."""
        self.pkgnames: list[str] = []

    def find_spec(self, fullname, path, target=None) -> None:  # type: ignore
        """Throws ImportError if matching module."""
        if fullname in self.pkgnames:
            raise ImportError()


@pytest.fixture  # type: ignore
def no_optional_dependencies(monkeypatch, optional_packages):
    """Mock environment without optional dependencies."""
    d = PackageDiscarder()

    for package in optional_packages:
        sys.modules.pop(package, None)
        d.pkgnames.append(package)
    sys.meta_path.insert(0, d)
    yield
    sys.meta_path.remove(d)


def _test_voronoi() -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    from srai.regionalizers import VoronoiRegionalizer

    seeds_gdf = gpd.GeoDataFrame(
        {
            GEOMETRY_COLUMN: [
                Point(17.014997869227177, 51.09919872601259),
                Point(16.935542631959215, 51.09380600286582),
                Point(16.900425, 51.1162552343),
                Point(16.932700, 51.166251),
            ]
        },
        index=[1, 2, 3, 4],
        crs=WGS84_CRS,
    )
    VoronoiRegionalizer(seeds=seeds_gdf)


def _test_plotting() -> None:
    from srai.plotting import folium_wrapper, plotly_wrapper

    folium_wrapper.plot_regions(_get_regions_gdf())
    plotly_wrapper.plot_regions(_get_regions_gdf(), return_plot=True)


def _test_torch() -> None:
    from srai.embedders import (
        GeoVexEmbedder,
        GTFS2VecEmbedder,
        Hex2VecEmbedder,
        Highway2VecEmbedder,
    )

    Highway2VecEmbedder()
    GTFS2VecEmbedder()
    Hex2VecEmbedder()
    GeoVexEmbedder(["a"] * 256)


def _test_osm() -> None:
    from srai.loaders import OSMOnlineLoader, OSMPbfLoader, OSMTileLoader, OSMWayLoader
    from srai.regionalizers import AdministrativeBoundaryRegionalizer

    AdministrativeBoundaryRegionalizer(2)
    OSMPbfLoader()
    OSMOnlineLoader()
    OSMWayLoader("drive")
    OSMTileLoader("https://tile.openstreetmap.de", 9)


def _test_gtfs() -> None:
    from srai.loaders import GTFSLoader

    GTFSLoader()


def _get_regions_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        data={
            GEOMETRY_COLUMN: [
                box(
                    minx=0,
                    miny=0,
                    maxx=1,
                    maxy=1,
                )
            ],
            REGIONS_INDEX: [1],
        },
        crs=WGS84_CRS,
    )


@pytest.mark.parametrize(  # type: ignore
    "test_fn",
    [
        (_test_voronoi),
        (_test_plotting),
        (_test_torch),
        (_test_osm),
        (_test_gtfs),
    ],
)
def test_optional_available(test_fn):
    """Test if defined functions are working with optional packages."""
    test_fn()


@pytest.mark.usefixtures("no_optional_dependencies")
@pytest.mark.parametrize(  # type: ignore
    "test_fn",
    [
        (_test_voronoi),
        (_test_plotting),
        (_test_torch),
        (_test_osm),
        (_test_gtfs),
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
