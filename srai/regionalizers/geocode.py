"""Utility function for geocoding a name to `regions_gdf`."""

from typing import Any, Union

import geopandas as gpd

from srai._optional import import_optional_dependencies
from srai.constants import REGIONS_INDEX


def geocode_to_region_gdf(
    query: Union[str, list[str], dict[str, Any]], by_osmid: bool = False
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

    Examples:
        Download geometry for a city
        >>> from srai.regionalizers import geocode_to_region_gdf
        >>> geocode_to_region_gdf("Wrocław, PL")
                                                          geometry
        region_id
        Wrocław, Lower Silesian Voivodeship, Poland  POLYGON ((...

        Download geometries for multiple cities

        >>> geocode_to_region_gdf(["New York City", "Washington, DC"])
                                                                    geometry
        region_id
        New York, United States                          MULTIPOLYGON (((...
        Washington, District of Columbia, United States  POLYGON ((...

        Use OSM relation IDs to get geometries.

        >>> geocode_to_region_gdf(["R175342", "R5750005"], by_osmid=True)
                                                                 geometry
        region_id
        Greater London, England, United Kingdom             POLYGON ((...
        Sydney, Council of the City of Sydney, New Sout...  POLYGON ((...
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
