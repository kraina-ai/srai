"""
Folium wrapper.

This module contains functions for quick plotting of analysed gdfs using Geopandas `explore()`
function.
"""
from typing import List, Set, Union

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
) -> folium.Map:
    """
    Plot regions shapes using Folium library.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries to plot.
        tiles_style (str, optional): Map style background. Defaults to "OpenStreetMap".
        height (Union[str, float], optional): Height of the plot. Defaults to "100%".
        width (Union[str, float], optional): Width of the plot. Defaults to "100%".

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
        style_kwds=dict(fillOpacity=0.4, weight=2),
    )


def plot_neighbours(
    regions_gdf: gpd.GeoDataFrame,
    region_id: IndexType,
    neighbours_ids: Set[IndexType],
    tiles_style: str = "OpenStreetMap",
    height: Union[str, float] = "100%",
    width: Union[str, float] = "100%",
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

    Returns:
        folium.Map: Generated map.
    """
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
            px.colors.qualitative.Plotly[1],
            px.colors.qualitative.Plotly[2],
            px.colors.qualitative.Plotly[0],
        ],
        categories=["selected", "neighbour", "other"],
    )


def plot_all_neighbourhood(
    regions_gdf: gpd.GeoDataFrame,
    region_id: IndexType,
    neighbourhood: Neighbourhood[IndexType],
    tiles_style: str = "OpenStreetMap",
    height: Union[str, float] = "100%",
    width: Union[str, float] = "100%",
    colormap: List[str] = px.colors.sequential.Sunsetdark,
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
            Defaults to `px.colors.sequential.Sunsetdark` from plotly library.

    Returns:
        folium.Map: Generated map.
    """
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

    return regions_gdf_copy.reset_index().explore(
        column="region",
        tooltip=[REGIONS_INDEX, "region"],
        tiles=tiles_style,
        height=height,
        width=width,
        cmap=_resample_plotly_colorscale(colormap, min(distance, 10)),
        categories=["selected", *list(range(distance + 1))[1:], "other"],
    )


def _resample_plotly_colorscale(colorscale: List[str], steps: int) -> List[str]:
    resampled_colorscale: List[str] = px.colors.sample_colorscale(
        colorscale, np.linspace(0, 1, num=steps)
    )
    return resampled_colorscale
