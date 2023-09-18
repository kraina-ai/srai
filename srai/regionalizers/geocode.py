"""Utility function for geocoding a name to `regions_gdf`."""
from typing import Any, Dict, List, Union

import geopandas as gpd

from srai._optional import import_optional_dependencies
from srai.constants import REGIONS_INDEX


def geocode_to_region_gdf(
    query: Union[str, List[str], Dict[str, Any]], by_osmid: bool = False
) -> gpd.GeoDataFrame:
    """
    Geocode a query to the `regions_gdf` unified format.

    This functions is a wrapper around the `ox.geocode_to_gdf`[1] function from the `osmnx` library.
    For parameters description look into the source documentation.

    Args:
        query (Union[str, List[str], Dict[str, Any]]): Query string(s) or structured dict(s)
            to geocode.
        by_osmid (bool, optional): Flag to treat query as an OSM ID lookup rather than text search.
            Defaults to False.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with geocoded regions.

    References:
        1. https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.geocoder.geocode_to_gdf
    """
    import_optional_dependencies(
        dependency_group="osm",
        modules=["osmnx"],
    )

    import osmnx as ox

    geocoded_gdf = ox.geocode_to_gdf(query=query, by_osmid=by_osmid, which_result=None)
    regions_gdf = (
        geocoded_gdf[["display_name", "geometry"]]
        .rename(columns={"display_name": REGIONS_INDEX})
        .set_index(REGIONS_INDEX)
    )
    return regions_gdf
