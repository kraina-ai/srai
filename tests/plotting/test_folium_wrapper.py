"""Tests for folium plotting wrapper."""

import re
from typing import Union

import geopandas as gpd
import pytest
from rq_geo_toolkit.folium import MapLibreLayer
from shapely.geometry import box

from srai.constants import REGIONS_INDEX, WGS84_CRS
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from srai.plotting import plot_numeric_data, plot_regions
from srai.plotting.folium_wrapper import _get_tiles, plot_neighbours
from srai.regionalizers.geocode import geocode_to_region_gdf
from srai.regionalizers.h3_regionalizer import H3Regionalizer

STYLE_URL = "https://tiles.openfreemap.org/styles/dark"


def test_wroclaw_neighbourhood_edge_case() -> None:
    """Test plotting edge case from H3Neighbourhood example error."""
    gdf_wro = geocode_to_region_gdf("Wrocław, PL")
    regions_gdf = H3Regionalizer(8).transform(gdf_wro)
    neighbourhood_with_regions = H3Neighbourhood(regions_gdf)

    edge_region_id = "881e2050bdfffff"
    neighbours_ids = neighbourhood_with_regions.get_neighbours(edge_region_id)

    plot_neighbours(regions_gdf, edge_region_id, neighbours_ids)


@pytest.fixture  # type: ignore
def regions_gdf() -> gpd.GeoDataFrame:
    """Get a single region with a value."""
    return gpd.GeoDataFrame(
        {"value": [1.0]},
        geometry=[box(17, 51, 18, 52)],
        index=gpd.pd.Index(["a"], name=REGIONS_INDEX),
        crs=WGS84_CRS,
    )


@pytest.mark.parametrize("tiles", [STYLE_URL, MapLibreLayer(STYLE_URL)])  # type: ignore
def test_maplibre_basemap(regions_gdf: gpd.GeoDataFrame, tiles: Union[str, MapLibreLayer]) -> None:
    """Test that a style is drawn instead of raster tiles."""
    folium_map = plot_regions(regions_gdf, tiles_style=tiles)
    html = folium_map.get_root().render()

    layer = re.search(rf'var (\w+) = L.maplibreGL\({{style: "{STYLE_URL}"}}\)', html)
    assert layer and f"{layer[1]}.addTo({folium_map.get_name()})" in html
    assert "leaflet-maplibre-gl.js" in html
    assert "L.tileLayer(" not in html


def test_default_basemap_of_numeric_data(regions_gdf: gpd.GeoDataFrame) -> None:
    """Test that the default basemap can be used by many maps."""
    for _ in range(2):
        html = plot_numeric_data(regions_gdf, "value").get_root().render()
        assert "https://tiles.openfreemap.org/styles/positron" in html


def test_raster_basemap(regions_gdf: gpd.GeoDataFrame) -> None:
    """Test that other basemaps are left for Folium."""
    html = plot_regions(regions_gdf, tiles_style="OpenStreetMap").get_root().render()
    assert "L.tileLayer(" in html and "maplibre" not in html

    tiles_url = "https://example.com/{z}/{x}/{y}.png"
    assert _get_tiles(tiles_url) == tiles_url
