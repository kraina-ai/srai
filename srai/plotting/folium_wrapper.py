"""
Folium wrapper.

This module contains functions for quick plotting of analysed gdfs using Geopandas `explore()`
function.
"""
from itertools import cycle, islice
from typing import List, Optional, Set, Union

from srai.utils._optional import import_optional_dependencies

import_optional_dependencies(dependency_group="plotting", modules=["folium", "plotly"])

# flake8: noqa E402

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px

from srai.constants import REGIONS_INDEX
from srai.neighbourhoods import Neighbourhood
from srai.neighbourhoods._base import IndexType


def plot_regions(
    regions_gdf: gpd.GeoDataFrame,
    tiles_style: str = "OpenStreetMap",
    height: Union[str, float] = "100%",
    width: Union[str, float] = "100%",
    colormap: Union[str, List[str]] = px.colors.qualitative.Bold,
    map: Optional[folium.Map] = None,
) -> folium.Map:
    """
    Plot regions shapes using Folium library.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries to plot.
        tiles_style (str, optional): Map style background. For more styles, look at tiles param at
            https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.explore.html.
            Defaults to "OpenStreetMap".
        height (Union[str, float], optional): Height of the plot. Defaults to "100%".
        width (Union[str, float], optional): Width of the plot. Defaults to "100%".
        colormap (Union[str, List[str]], optional): Colormap to apply to the regions.
            Defaults to `px.colors.qualitative.Bold` from plotly library.
        map (folium.Map, optional): Existing map instance on which to draw the plot.
            Defaults to None.

    Returns:
        folium.Map: Generated map.
    """
    return regions_gdf.reset_index().explore(
        column=REGIONS_INDEX,
        tooltip=REGIONS_INDEX,
        tiles=tiles_style,
        height=height,
        width=width,
        legend=False,
        cmap=colormap,
        categorical=True,
        style_kwds=dict(color="#444", opacity=0.5, fillOpacity=0.5),
        m=map,
    )


def plot_numeric_data(
    regions_gdf: gpd.GeoDataFrame,
    embedding_df: Union[pd.DataFrame, gpd.GeoDataFrame],
    data_column: str,
    tiles_style: str = "OpenStreetMap",
    height: Union[str, float] = "100%",
    width: Union[str, float] = "100%",
    colormap: Union[str, List[str]] = px.colors.sequential.Sunsetdark,
    map: Optional[folium.Map] = None,
) -> folium.Map:
    """
    Plot numerical data within regions shapes using Folium library.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries to plot.
        embedding_df (Union[pd.DataFrame, gpd.GeoDataFrame]): Region indexes and numerical data
            to plot.
        data_column (str): Name of the column used to colour the regions.
        tiles_style (str, optional): Map style background. For more styles, look at tiles param at
            https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.explore.html.
            Defaults to "OpenStreetMap".
        height (Union[str, float], optional): Height of the plot. Defaults to "100%".
        width (Union[str, float], optional): Width of the plot. Defaults to "100%".
        colormap (Union[str, List[str]], optional): Colormap to apply to the regions.
            Defaults to px.colors.sequential.Sunsetdark.
        map (folium.Map, optional): Existing map instance on which to draw the plot.
            Defaults to None.

    Returns:
        folium.Map: Generated map.
    """
    regions_gdf_copy = regions_gdf.copy()
    regions_gdf_copy = regions_gdf_copy.merge(embedding_df, on=REGIONS_INDEX)

    if not isinstance(colormap, str):
        colormap = _generate_linear_colormap(
            colormap,
            min_value=regions_gdf_copy[data_column].min(),
            max_value=regions_gdf_copy[data_column].max(),
        )

    return regions_gdf_copy.reset_index().explore(
        column=data_column,
        tooltip=[REGIONS_INDEX, data_column],
        tiles=tiles_style,
        height=height,
        width=width,
        legend=True,
        cmap=colormap,
        categorical=False,
        style_kwds=dict(color="#444", opacity=0.5, fillOpacity=0.8),
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
        tiles_style (str, optional): Map style background. For more styles, look at tiles param at
            https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.explore.html.
            Defaults to "OpenStreetMap".
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
    neighbourhood_max_distance: int = 100,
    tiles_style: str = "OpenStreetMap",
    height: Union[str, float] = "100%",
    width: Union[str, float] = "100%",
    colormap: Union[str, List[str]] = px.colors.sequential.Agsunset_r,
    map: Optional[folium.Map] = None,
) -> folium.Map:
    """
    Plot full neighbourhood on a map using Folium library.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries to plot.
        region_id (IndexType): Center `region_id` around which the neighbourhood should be plotted.
        neighbourhood (Neighbourhood[IndexType]): `Neighbourhood` class required for finding
            neighbours.
        neighbourhood_max_distance (int, optional): Max distance for rendering neighbourhoods.
            Neighbours farther away won't be coloured, and will be left as "other" regions.
            Defaults to 100.
        tiles_style (str, optional): Map style background. For more styles, look at tiles param at
            https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.explore.html.
            Defaults to "OpenStreetMap".
        height (Union[str, float], optional): Height of the plot. Defaults to "100%".
        width (Union[str, float], optional): Width of the plot. Defaults to "100%".
        colormap (Union[str, List[str]], optional): Colormap to apply to the neighbourhoods.
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
    while neighbours_ids and distance <= neighbourhood_max_distance:
        regions_gdf_copy.loc[list(neighbours_ids), "region"] = distance
        distance += 1
        neighbours_ids = neighbourhood.get_neighbours_at_distance(region_id, distance).intersection(
            regions_gdf.index
        )

    if not isinstance(colormap, str):
        colormap = _generate_colormap(
            distance, colormap=_resample_plotly_colormap(colormap, min(distance, 10))
        )

    return regions_gdf_copy.reset_index().explore(
        column="region",
        tooltip=[REGIONS_INDEX, "region"],
        tiles=tiles_style,
        height=height,
        width=width,
        cmap=colormap,
        categorical=True,
        categories=["selected", *list(range(distance))[1:], "other"],
        style_kwds=dict(color="#444", opacity=0.5, fillOpacity=0.8),
        legend=distance <= 11,
        m=map,
    )


def _resample_plotly_colormap(colormap: List[str], steps: int) -> List[str]:
    resampled_colormap: List[str] = px.colors.sample_colorscale(
        colormap, np.linspace(0, 1, num=steps)
    )
    return resampled_colormap


def _generate_colormap(
    distance: int,
    colormap: List[str],
    selected_color: str = "rgb(242, 242, 242)",
    other_color: str = "rgb(153, 153, 153)",
) -> List[str]:
    return [selected_color, *islice(cycle(colormap), None, distance - 1), other_color]


def _generate_linear_colormap(
    colormap: List[str], min_value: float, max_value: float
) -> cm.LinearColormap:
    values, _ = px.colors.convert_colors_to_same_type(colormap, colortype="tuple")
    return cm.LinearColormap(values, vmin=min_value, vmax=max_value)
