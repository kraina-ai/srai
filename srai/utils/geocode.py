"""Utility function for geocoding a name to `regions_gdf`."""
from typing import Any, Dict, List, Union

import geopandas as gpd
import osmnx as ox

from srai.utils.constants import REGIONS_INDEX


def geocode_to_region_gdf(
    query: Union[str, List[str], Dict[str, Any]], by_osmid: bool = False
) -> gpd.GeoDataFrame:
    """
    Geocode a query to the `regions_gdf` unified format.

    This functions is a wrapper around the `ox.geocode_to_gdf`[1] function from the `osmnx` library.
    For parameters description look into the source documentation.

    References:
        1. https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.geocoder.geocode_to_gdf
    """
    geocoded_gdf = ox.geocode_to_gdf(query=query, by_osmid=by_osmid, which_result=None)
    regions_gdf = (
        geocoded_gdf[["display_name", "geometry"]]
        .rename(columns={"display_name": REGIONS_INDEX})
        .set_index(REGIONS_INDEX)
    )
    return regions_gdf
