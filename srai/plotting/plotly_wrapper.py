"""
Plotly wrapper.

This module contains functions for quick plotting of analysed gdfs using Plotly library.
"""

from typing import Any, Optional

import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from shapely.geometry import Point

from srai._optional import import_optional_dependencies
from srai.constants import REGIONS_INDEX, WGS84_CRS
from srai.neighbourhoods import Neighbourhood
from srai.neighbourhoods._base import IndexType

import_optional_dependencies(dependency_group="plotting", modules=["plotly"])


def plot_regions(
    regions_gdf: gpd.GeoDataFrame,
    return_plot: bool = False,
    mapbox_style: str = "open-street-map",
    mapbox_accesstoken: Optional[str] = None,
    renderer: Optional[str] = None,
    zoom: Optional[float] = None,
    height: Optional[float] = None,
    width: Optional[float] = None,
) -> Optional[go.Figure]:
    """
    Plot regions shapes using Plotly library.

    For more info about parameters, check https://plotly.com/python/mapbox-layers/.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries to plot.
        return_plot (bool, optional): Flag whether to return the Figure object or not.
            If `True`, the plot won't be displayed automatically. Defaults to False.
        mapbox_style (str, optional): Map style background. Defaults to "open-street-map".
        mapbox_accesstoken (str, optional): Access token required for mapbox based map backgrounds.
            Defaults to None.
        renderer (str, optional): Name of renderer used for displaying the figure.
            For all descriptions, look here: https://plotly.com/python/renderers/.
            Defaults to None.
        zoom (float, optional): Map zoom. If not filled, will be approximated based on
            the bounding box of regions. Defaults to None.
        height (float, optional): Height of the plot. Defaults to None.
        width (float, optional): Width of the plot. Defaults to None.

    Returns:
        Optional[go.Figure]: Figure of the plot. Will be returned if `return_plot` is set to `True`.
    """
    regions_gdf_copy = regions_gdf.copy()
    regions_gdf_copy[REGIONS_INDEX] = regions_gdf_copy.index
    return _plot_regions(
        regions_gdf=regions_gdf_copy,
        hover_column_name=REGIONS_INDEX,
        color_feature_column=None,
        hover_data=[],
        show_legend=False,
        return_plot=return_plot,
        mapbox_style=mapbox_style,
        mapbox_accesstoken=mapbox_accesstoken,
        renderer=renderer,
        zoom=zoom,
        height=height,
        width=width,
        color_discrete_sequence=px.colors.qualitative.Safe,
        opacity=0.4,
        traces_kwargs=dict(marker_line_width=2),
    )


def plot_neighbours(
    regions_gdf: gpd.GeoDataFrame,
    region_id: IndexType,
    neighbours_ids: set[IndexType],
    return_plot: bool = False,
    mapbox_style: str = "open-street-map",
    mapbox_accesstoken: Optional[str] = None,
    renderer: Optional[str] = None,
    zoom: Optional[float] = None,
    height: Optional[float] = None,
    width: Optional[float] = None,
) -> Optional[go.Figure]:
    """
    Plot neighbours on a map using Plotly library.

    For more info about parameters, check https://plotly.com/python/mapbox-layers/.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries to plot.
        region_id (IndexType): Center `region_id` around which the neighbourhood should be plotted.
        neighbours_ids (Set[IndexType]): List of neighbours to highlight.
        return_plot (bool, optional): Flag whether to return the Figure object or not.
            If `True`, the plot won't be displayed automatically. Defaults to False.
        mapbox_style (str, optional): Map style background. Defaults to "open-street-map".
        mapbox_accesstoken (str, optional): Access token required for mapbox based map backgrounds.
            Defaults to None.
        renderer (str, optional): Name of renderer used for displaying the figure.
            For all descriptions, look here: https://plotly.com/python/renderers/.
            Defaults to None.
        zoom (float, optional): Map zoom. If not filled, will be approximated based on
            the bounding box of regions. Defaults to None.
        height (float, optional): Height of the plot. Defaults to None.
        width (float, optional): Width of the plot. Defaults to None.

    Returns:
        Optional[go.Figure]: Figure of the plot. Will be returned if `return_plot` is set to `True`.
    """
    regions_gdf_copy = regions_gdf.copy()
    regions_gdf_copy[REGIONS_INDEX] = regions_gdf_copy.index
    regions_gdf_copy["region"] = "other"
    regions_gdf_copy.loc[region_id, "region"] = "selected"
    regions_gdf_copy.loc[list(neighbours_ids), "region"] = "neighbour"
    return _plot_regions(
        regions_gdf=regions_gdf_copy,
        hover_column_name=REGIONS_INDEX,
        color_feature_column="region",
        hover_data=[],
        show_legend=True,
        return_plot=return_plot,
        mapbox_style=mapbox_style,
        mapbox_accesstoken=mapbox_accesstoken,
        renderer=renderer,
        zoom=zoom,
        height=height,
        width=width,
        layout_kwargs=dict(
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, traceorder="normal"),
        ),
        category_orders={"region": ["selected", "neighbour", "other"]},
        color_discrete_sequence=[
            px.colors.qualitative.Plotly[1],
            px.colors.qualitative.Plotly[2],
            px.colors.qualitative.Plotly[0],
        ],
    )


def plot_all_neighbourhood(
    regions_gdf: gpd.GeoDataFrame,
    region_id: IndexType,
    neighbourhood: Neighbourhood[IndexType],
    neighbourhood_max_distance: int = 100,
    return_plot: bool = False,
    mapbox_style: str = "open-street-map",
    mapbox_accesstoken: Optional[str] = None,
    renderer: Optional[str] = None,
    zoom: Optional[float] = None,
    height: Optional[float] = None,
    width: Optional[float] = None,
) -> Optional[go.Figure]:
    """
    Plot full neighbourhood on a map using Plotly library.

    For more info about parameters, check https://plotly.com/python/mapbox-layers/.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries to plot.
        region_id (IndexType): Center `region_id` around which the neighbourhood should be plotted.
        neighbourhood (Neighbourhood[IndexType]): `Neighbourhood` class required for finding
            neighbours.
        neighbourhood_max_distance (int, optional): Max distance for rendering neighbourhoods.
            Neighbours farther away won't be coloured, and will be left as "other" regions.
            Defaults to 100.
        return_plot (bool, optional): Flag whether to return the Figure object or not.
            If `True`, the plot won't be displayed automatically. Defaults to False.
        mapbox_style (str, optional): Map style background. Defaults to "open-street-map".
        mapbox_accesstoken (str, optional): Access token required for mapbox based map backgrounds.
            Defaults to None.
        renderer (str, optional): Name of renderer used for displaying the figure.
            For all descriptions, look here: https://plotly.com/python/renderers/.
            Defaults to None.
        zoom (float, optional): Map zoom. If not filled, will be approximated based on
            the bounding box of regions. Defaults to None.
        height (float, optional): Height of the plot. Defaults to None.
        width (float, optional): Width of the plot. Defaults to None.

    Returns:
        Optional[go.Figure]: Figure of the plot. Will be returned if `return_plot` is set to `True`.
    """
    regions_gdf_copy = regions_gdf.copy()
    regions_gdf_copy[REGIONS_INDEX] = regions_gdf_copy.index
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

    return _plot_regions(
        regions_gdf=regions_gdf_copy,
        hover_column_name=REGIONS_INDEX,
        color_feature_column="region",
        hover_data=[],
        show_legend=True,
        return_plot=return_plot,
        mapbox_style=mapbox_style,
        mapbox_accesstoken=mapbox_accesstoken,
        renderer=renderer,
        zoom=zoom,
        height=height,
        width=width,
        layout_kwargs=dict(
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, traceorder="normal"),
        ),
        category_orders={"region": ["selected", *range(distance), "other"]},
        color_discrete_sequence=px.colors.cyclical.Edge,
    )


def _plot_regions(
    regions_gdf: gpd.GeoDataFrame,
    hover_column_name: str,
    hover_data: list[str],
    color_feature_column: Optional[str] = None,
    show_legend: bool = False,
    return_plot: bool = False,
    mapbox_style: str = "open-street-map",
    mapbox_accesstoken: Optional[str] = None,
    opacity: float = 0.6,
    renderer: Optional[str] = None,
    zoom: Optional[float] = None,
    height: Optional[float] = None,
    width: Optional[float] = None,
    layout_kwargs: Optional[dict[str, Any]] = None,
    traces_kwargs: Optional[dict[str, Any]] = None,
    **choropleth_mapbox_kwargs: Any,
) -> Optional[go.Figure]:
    """
    Plot regions shapes using Plotly library.

    Uses `choroplethmapbox` function.
    For more info about parameters, check https://plotly.com/python/mapbox-layers/.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries to plot.
        hover_column_name (str): Column name used for hover popup title.
        hover_data (List[str]): List of column names displayed additionally on hover.
        color_feature_column (str, optional): Column name used for colouring the plot.
            Can be `None` to disable colouring.
        show_legend (bool, optional): Flag whether to show the legend or not. Defaults to False.
        return_plot (bool, optional): Flag whether to return the Figure object or not.
            If `True`, the plot won't be displayed automatically. Defaults to False.
        mapbox_style (str, optional): Map style background. Defaults to "open-street-map".
        mapbox_accesstoken (str, optional): Access token required for mapbox based map backgrounds.
            Defaults to None.
        opacity (float, optional): Markers opacity. Defaults to 0.6.
        renderer (str, optional): Name of renderer used for displaying the figure.
            For all descriptions, look here: https://plotly.com/python/renderers/.
            Defaults to None.
        zoom (float, optional): Map zoom. If not filled, will be approximated based on
            the bounding box of regions. Defaults to None.
        height (float, optional): Height of the plot. Defaults to None.
        width (float, optional): Width of the plot. Defaults to None.
        layout_kwargs (Dict[str, Any], optional): Additional parameters passed to
            the `update_layout` function. Defaults to None.
        traces_kwargs (Dict[str, Any], optional): Additional parameters passed to
            the `update_traces` function. Defaults to None.
        **choropleth_mapbox_kwargs: Additional parameters that can be passed to
            the `choropleth_mapbox` constructor.

    Returns:
        Optional[go.Figure]: Figure of the plot. Will be returned if `return_plot` is set to `True`.
    """
    center_point = _calculate_map_centroid(regions_gdf)
    if not zoom:
        zoom = _calculate_mapbox_zoom(regions_gdf)

    fig = px.choropleth_mapbox(
        regions_gdf,
        geojson=regions_gdf,
        color=color_feature_column,
        hover_name=hover_column_name,
        hover_data=hover_data,
        locations=REGIONS_INDEX,
        center={"lon": center_point.x, "lat": center_point.y},
        zoom=zoom,
        **choropleth_mapbox_kwargs,
    )

    update_layout_dict = dict(
        height=height,
        width=width,
        margin=dict(r=0, t=0, l=0, b=0),
        mapbox_style=mapbox_style,
        mapbox_accesstoken=mapbox_accesstoken,
    )
    fig.update_layout(**update_layout_dict)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)

    update_traces_dict = dict(marker_opacity=opacity, showlegend=show_legend)
    fig.update_traces(**update_traces_dict)
    if traces_kwargs:
        fig.update_traces(**traces_kwargs)

    fig.update_coloraxes(showscale=show_legend)

    if return_plot:
        return fig
    else:
        fig.show(renderer=renderer)
        return None


def _calculate_map_centroid(regions_gdf: gpd.GeoDataFrame) -> Point:
    """
    Calculate regions centroid using Equal Area Cylindrical projection [1].

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.

    Returns:
        Point: Center point in WGS84 units.

    References:
        1. https://proj.org/operations/projections/cea.html
    """
    center_point = regions_gdf.to_crs("+proj=cea").dissolve().centroid.to_crs(WGS84_CRS)[0]
    return center_point


# Inspired by:
# https://stackoverflow.com/a/65043576/7766101
def _calculate_mapbox_zoom(
    regions_gdf: gpd.GeoDataFrame,
) -> float:
    """
    Calculate approximate zoom for a plotly figure.

    Currently Plotly doesn't implement auto-fit feature for mapbox plots.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.

    Returns:
        float: zoom level for a mapbox plot.
    """
    minx, miny, maxx, maxy = regions_gdf.geometry.total_bounds
    max_bound = max(abs(maxx - minx), abs(maxy - miny)) * 111
    zoom = float(12.5 - np.log(max_bound))
    return zoom
