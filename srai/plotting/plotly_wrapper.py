# noqa
"""TODO."""
from typing import List, Optional

import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from shapely.geometry import Point

from srai.utils.constants import REGIONS_INDEX, WGS84_CRS


def plot_regions_gdf(
    regions_gdf: gpd.GeoDataFrame,
    return_plot: bool = False,
    mapbox_style: str = "open-street-map",
    mapbox_accesstoken: Optional[str] = None,
    renderer: Optional[str] = "notebook_connected",
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
            Defaults to "notebook_connected".
        zoom (float, optional): Map zoom. If not filled, will be approximated based on
            the bounding box of regions. Defaults to None.
        height (float, optional): Height of the plot. Defaults to None.
        width (float, optional): Width of the plot. Defaults to None.

    Returns:
        Optional[go.Figure]: Figure of the plot. Will be returned if `return_plot` is set to `True`.
    """
    regions_gdf_copy = regions_gdf.copy()
    regions_gdf_copy[REGIONS_INDEX] = regions_gdf_copy.index
    return _plot_regions_gdf(
        regions_gdf=regions_gdf_copy,
        hover_column_name=REGIONS_INDEX,
        color_feature_column=REGIONS_INDEX,
        hover_data=[],
        return_plot=return_plot,
        mapbox_style=mapbox_style,
        mapbox_accesstoken=mapbox_accesstoken,
        renderer=renderer,
        zoom=zoom,
        height=height,
        width=width,
    )


def _plot_regions_gdf(
    regions_gdf: gpd.GeoDataFrame,
    hover_column_name: str,
    color_feature_column: str,
    hover_data: List[str],
    return_plot: bool = False,
    mapbox_style: str = "open-street-map",
    mapbox_accesstoken: Optional[str] = None,
    renderer: Optional[str] = "notebook_connected",
    zoom: Optional[float] = None,
    height: Optional[float] = None,
    width: Optional[float] = None,
) -> Optional[go.Figure]:
    """
    Plot regions shapes using Plotly library.

    For more info about parameters, check https://plotly.com/python/mapbox-layers/.

    Args:
        regions_gdf (gpd.GeoDataFrame): Region indexes and geometries to plot.
        hover_column_name (str): Column name used for hover popup title.
        color_feature_column (str): Column name used for colouring the plot.
        hover_data (List[str]): List of column names displayed additionally on hover.
        return_plot (bool, optional): Flag whether to return the Figure object or not.
            If `True`, the plot won't be displayed automatically. Defaults to False.
        mapbox_style (str, optional): Map style background. Defaults to "open-street-map".
        mapbox_accesstoken (str, optional): Access token required for mapbox based map backgrounds.
            Defaults to None.
        renderer (str, optional): Name of renderer used for displaying the figure.
            For all descriptions, look here: https://plotly.com/python/renderers/.
            Defaults to "notebook_connected".
        zoom (float, optional): Map zoom. If not filled, will be approximated based on
            the bounding box of regions. Defaults to None.
        height (float, optional): Height of the plot. Defaults to None.
        width (float, optional): Width of the plot. Defaults to None.

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
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_traces(marker={"opacity": 0.6}, selector=dict(type="choroplethmapbox"))
    fig.update_traces(showlegend=False)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=height, width=width, margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(mapbox_style=mapbox_style, mapbox_accesstoken=mapbox_accesstoken)

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
