# noqa
"""TODO."""
from typing import Any

import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go


def plot_regions_gdf(regions_gdf: gpd.GeoDataFrame) -> go.Figure:  # noqa
    """TODO."""
    center_point = regions_gdf.dissolve().centroid[0]
    minx, miny, maxx, maxy = regions_gdf.geometry.total_bounds
    # zoom = _get_zoom_level_for_bounds(lat_min=miny, lat_max=maxy, lon_min=minx, lon_max=maxx)
    # zoom = _calculate_mapbox_zoom(lat_min=miny, lat_max=maxy, lon_min=minx, lon_max=maxx)
    zoom = get_plotting_zoom_level(lat_min=miny, lat_max=maxy, lon_min=minx, lon_max=maxx)
    # zoom, center = zoom_center(maxlat=maxy, maxlon=maxx, minlat=miny, minlon=minx)
    print(minx, miny, maxx, maxy, zoom)
    # print(center_point, center)
    fig = px.choropleth_mapbox(
        regions_gdf,
        geojson=regions_gdf,
        color=regions_gdf.index,
        locations=regions_gdf.index,
        center={"lon": center_point.x, "lat": center_point.y},
        mapbox_style="carto-positron",
        zoom=zoom,
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_traces(marker={"opacity": 0.6}, selector=dict(type="choroplethmapbox"))
    fig.update_traces(showlegend=False)
    fig.update_layout(autosize=True, margin={"r": 0, "t": 0, "l": 0, "b": 0})
    # width = fig.layout.width
    # height = fig.layout.height

    print(fig.layout)
    # minx, miny, maxx, maxy = regions_gdf.geometry.total_bounds
    # fig.update_geos(
    #     projection_type="equirectangular",
    #     # lataxis_range=[miny, maxy],
    #     # lonaxis_range=[minx, maxx],
    #     fitbounds="locations",
    #     showlakes=False,
    #     showcountries=False,
    #     showframe=False,
    #     resolution=50,
    # )
    # fig.update_layout(height=600, width=800, margin={"r": 0, "t": 0, "l": 0, "b": 0})
    print("XD")

    # fig.show(renderer="png")  # replace with fig.show() to allow interactivity
    fig.show()


def _calculate_mapbox_zoom(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> Any:  # noqa
    """TODO."""
    max_bound = max(abs(lat_max - lat_min), abs(lon_max - lon_min)) * 111
    zoom = 13 - np.log(max_bound)
    return zoom


def _get_zoom_level_for_bounds(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> Any:  # noqa
    """TODO."""
    dx = abs(lat_max - lat_min)
    dy = abs(lon_max - lon_min)
    d = dy if dy > dx else dx

    zoom_delta_lookup = {
        100: 0,
        75: 0.2,
        50: 0.4,
        40: 1.2,
        30: 1.4,
        20: 1.6,
        15: 1.7,
        10: 2.1,
        5: 2.7,
        2.5: 3.2,
        1: 3.8,
        0.5: 6,
        0.25: 7.2,
        0.125: 7.6,
        0.1: 8.2,
        0.01: 8.8,
        0.001: 9,
        0: 9.6,
    }

    for delta, zoom_value in zoom_delta_lookup.items():
        print(d, delta, zoom_value)
        if d > delta:
            return zoom_value


def zoom_center(
    maxlon: float,
    minlon: float,
    maxlat: float,
    minlat: float,
    projection: str = "mercator",
    # width_to_height: float = 2.0,
) -> Any:  # noqa
    """
    Finds optimal zoom and centering for a plotly mapbox. Must be passed (lons & lats) or lonlats.
    Temporary solution awaiting official implementation, see:
    https://github.com/plotly/plotly.js/issues/3434.

    Parameters
    --------
    lons: tuple, optional, longitude component of each location
    lats: tuple, optional, latitude component of each location
    lonlats: tuple, optional, gps locations
    format: str, specifying the order of longitud and latitude dimensions,
        expected values: 'lonlat' or 'latlon', only used if passed lonlats
    projection: str, only accepting 'mercator' at the moment,
        raises `NotImplementedError` if other is passed
    width_to_height: float, expected ratio of final graph's with to height,
        used to select the constrained axis.

    Returns
    --------
    zoom: float, from 1 to 20
    center: dict, gps position with 'lon' and 'lat' keys

    >>> print(zoom_center((-109.031387, -103.385460),
    ...     (25.587101, 31.784620)))
    (5.75, {'lon': -106.208423, 'lat': 28.685861})
    """
    center = {"lon": round((maxlon + minlon) / 2, 6), "lat": round((maxlat + minlat) / 2, 6)}

    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array(
        [
            0.0007,
            0.0014,
            0.003,
            0.006,
            0.012,
            0.024,
            0.048,
            0.096,
            0.192,
            0.3712,
            0.768,
            1.536,
            3.072,
            6.144,
            11.8784,
            23.7568,
            47.5136,
            98.304,
            190.0544,
            360.0,
        ]
    )

    if projection == "mercator":
        margin = 1.2
        # height = (maxlat - minlat) * margin * width_to_height
        height = (maxlat - minlat) * margin
        width = (maxlon - minlon) * margin
        lon_zoom = np.interp(width, lon_zoom_range, range(20, 0, -1))
        lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
        zoom = round(min(lon_zoom, lat_zoom), 2)
    else:
        raise NotImplementedError(f"{projection} projection is not implemented")

    return zoom, center


def get_plotting_zoom_level(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> Any:  # noqa
    """
    Function documentation: Basic framework adopted from Krichardson under the following thread:
    https://community.plotly.com/t/dynamic-zoom-for-mapbox/32658/6.

    # NOTE:
    # THIS IS A TEMPORARY SOLUTION UNTIL THE DASH TEAM IMPLEMENTS DYNAMIC ZOOM
    # in their plotly-functions associated with mapbox, such as go.Densitymapbox() etc.

    Returns the appropriate zoom-level for these plotly-mapbox-graphics along with
    the center coordinate tuple of all provided coordinate tuples.
    """

    width = abs(lat_max - lat_min)
    height = abs(lon_max - lon_min)

    # Otherwise, get the area of the bounding box in order to calculate a zoom-level
    area = height * width

    print(width, height, area)

    # * 1D-linear interpolation with numpy:
    # - Pass the area as the only x-value and not as a list, in order to return a scalar as well
    # - The x-points "xp" should be in parts in comparable order of magnitude of the given area
    # - The zoom-levels are adapted to the areas, i.e. start with the smallest area possible of 0
    # which leads to the highest possible zoom value 20, and so forth decreasing with increasing
    # areas as these variables are antiproportional
    zoom = np.interp(
        x=area,
        xp=[0, 5**-10, 4**-10, 3**-10, 2**-10, 1**-10, 1**-5],
        # fp=[20, 17, 16, 15, 14, 7, 5],
        fp=[20, 15, 14, 13, 12, 7, 5],
    )

    # Finally, return the zoom level and the associated boundary-box center coordinates
    return zoom
