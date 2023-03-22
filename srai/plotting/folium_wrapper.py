"""
Folium wrapper.

This module contains functions for quick plotting of analysed gdfs using Geopandas `explore()`
function.
"""
from itertools import cycle, islice
from typing import List, Optional, Set, Union

import folium
import geopandas as gpd
import numpy as np
import plotly.express as px

from srai.constants import REGIONS_INDEX
from srai.neighbourhoods import Neighbourhood
from srai.neighbourhoods._base import IndexType


def plot_regions(
    regions_gdf: gpd.GeoDataFrame,
    tiles_style: str = "OpenStreetMap",
    height: Union[str, float] = "100%",
    width: Union[str, float] = "100%",
    map: Optional[folium.Map] = None,
) -> folium.Map:
    """
    Plot regions shapes using Folium library.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries to plot.
        tiles_style (str, optional): Map style background. Defaults to "OpenStreetMap".
        height (Union[str, float], optional): Height of the plot. Defaults to "100%".
        width (Union[str, float], optional): Width of the plot. Defaults to "100%".
        map (folium.Map, optional): Existing map instance on which to draw the plot.
            Defaults to None.

    Returns:
        folium.Map: Generated map.
    """
    regions_gdf_copy = regions_gdf.copy()
    return regions_gdf_copy.reset_index().explore(
        column=REGIONS_INDEX,
        tooltip=REGIONS_INDEX,
        tiles=tiles_style,
        height=height,
        width=width,
        legend=False,
        cmap=px.colors.qualitative.Bold,
        categorical=True,
        style_kwds=dict(color="#444", opacity=0.5, fillOpacity=0.5),
        m=map,
    )


def plot_neighbours(
    regions_gdf: gpd.GeoDataFrame,
    region_id: IndexType,
    neighbours_ids: Set[IndexType],
    tiles_style: str = "OpenStreetMap",
    height: Union[str, float] = "100%",
    width: Union[str, float] = "100%",
    map: Optional[folium.Map] = None,
) -> folium.Map:
    """
    Plot neighbours on a map using Folium library.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries to plot.
        region_id (IndexType): Center `region_id` around which the neighbourhood should be plotted.
        neighbours_ids (Set[IndexType]): List of neighbours to highlight.
        tiles_style (str, optional): Map style background. Defaults to "OpenStreetMap".
        height (Union[str, float], optional): Height of the plot. Defaults to "100%".
        width (Union[str, float], optional): Width of the plot. Defaults to "100%".
        map (folium.Map, optional): Existing map instance on which to draw the plot.
            Defaults to None.

    Returns:
        folium.Map: Generated map.
    """
    if region_id not in regions_gdf.index:
        raise AttributeError(f"{region_id!r} doesn't exist in provided regions_gdf.")

    regions_gdf_copy = regions_gdf.copy()
    regions_gdf_copy["region"] = "other"
    regions_gdf_copy.loc[region_id, "region"] = "selected"
    regions_gdf_copy.loc[neighbours_ids, "region"] = "neighbour"
    return regions_gdf_copy.reset_index().explore(
        column="region",
        tooltip=REGIONS_INDEX,
        tiles=tiles_style,
        height=height,
        width=width,
        cmap=[
            "rgb(242, 242, 242)",
            px.colors.sequential.Sunsetdark[-1],
            px.colors.sequential.Sunsetdark[2],
        ],
        categorical=True,
        categories=["selected", "neighbour", "other"],
        style_kwds=dict(color="#444", opacity=0.5, fillOpacity=0.8),
        m=map,
    )


def plot_all_neighbourhood(
    regions_gdf: gpd.GeoDataFrame,
    region_id: IndexType,
    neighbourhood: Neighbourhood[IndexType],
    tiles_style: str = "OpenStreetMap",
    height: Union[str, float] = "100%",
    width: Union[str, float] = "100%",
    colormap: List[str] = px.colors.sequential.Agsunset_r,
    map: Optional[folium.Map] = None,
) -> folium.Map:
    """
    Plot full neighbourhood on a map using Folium library.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries to plot.
        region_id (IndexType): Center `region_id` around which the neighbourhood should be plotted.
        neighbourhood (Neighbourhood[IndexType]): `Neighbourhood` class required for finding
            neighbours.
        tiles_style (str, optional): Map style background. Defaults to "OpenStreetMap".
        height (Union[str, float], optional): Height of the plot. Defaults to "100%".
        width (Union[str, float], optional): Width of the plot. Defaults to "100%".
        colormap (List[str], optional): Colormap to apply to the nieghbourhoods.
            Defaults to `px.colors.sequential.Agsunset_r` from plotly library.
        map (folium.Map, optional): Existing map instance on which to draw the plot.
            Defaults to None.

    Returns:
        folium.Map: Generated map.
    """
    if region_id not in regions_gdf.index:
        raise AttributeError(f"{region_id!r} doesn't exist in provided regions_gdf.")

    regions_gdf_copy = regions_gdf.copy()
    regions_gdf_copy["region"] = "other"
    regions_gdf_copy.loc[region_id, "region"] = "selected"

    distance = 1
    neighbours_ids = neighbourhood.get_neighbours_at_distance(region_id, distance).intersection(
        regions_gdf.index
    )
    while neighbours_ids:
        regions_gdf_copy.loc[list(neighbours_ids), "region"] = distance
        distance += 1
        neighbours_ids = neighbourhood.get_neighbours_at_distance(region_id, distance).intersection(
            regions_gdf.index
        )

    colorscale = _generate_colorscale(
        distance, colorscale=_resample_plotly_colorscale(colormap, min(distance, 10))
    )

    return regions_gdf_copy.reset_index().explore(
        column="region",
        tooltip=[REGIONS_INDEX, "region"],
        tiles=tiles_style,
        height=height,
        width=width,
        cmap=colorscale,
        categorical=True,
        categories=["selected", *list(range(distance))[1:], "other"],
        style_kwds=dict(color="#444", opacity=0.5, fillOpacity=0.8),
        legend=distance <= 11,
        m=map,
    )


def _resample_plotly_colorscale(colorscale: List[str], steps: int) -> List[str]:
    resampled_colorscale: List[str] = px.colors.sample_colorscale(
        colorscale, np.linspace(0, 1, num=steps)
    )
    return resampled_colorscale


def _generate_colorscale(
    distance: int,
    colorscale: List[str],
    selected_color: str = "rgb(242, 242, 242)",
    other_color: str = "rgb(153, 153, 153)",
) -> List[str]:
    return [selected_color, *islice(cycle(colorscale), None, distance - 1), other_color]
