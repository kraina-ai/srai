"""Tests for folium plotting wrapper."""

from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from srai.plotting.folium_wrapper import plot_neighbours
from srai.regionalizers.geocode import geocode_to_region_gdf
from srai.regionalizers.h3_regionalizer import H3Regionalizer


def test_wroclaw_neighbourhood_edge_case() -> None:
    """Test plotting edge case from H3Neighbourhood example error."""
    gdf_wro = geocode_to_region_gdf("Wrocław, PL")
    regions_gdf = H3Regionalizer(8).transform(gdf_wro)
    neighbourhood_with_regions = H3Neighbourhood(regions_gdf)

    edge_region_id = "881e2050bdfffff"
    neighbours_ids = neighbourhood_with_regions.get_neighbours(edge_region_id)

    plot_neighbours(regions_gdf, edge_region_id, neighbours_ids)
