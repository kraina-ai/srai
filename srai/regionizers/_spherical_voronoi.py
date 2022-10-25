"""
Spherical voronoi utils.

This module contains spherical voronoi implementation based on
SphericalVoronoi function from scipy library.
"""

from math import ceil, sqrt
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from haversine import haversine
from pymap3d import Ellipsoid, ecef2geodetic, geodetic2ecef
from scipy.spatial import SphericalVoronoi, geometric_slerp
from shapely.geometry import MultiPolygon, Point, Polygon, box
from spherical_geometry.polygon import SphericalPolygon

# LON: 0; LAT: 0
POINT_FRONT = (1.0, 0.0, 0.0)
# LON: 180; LAT: 0
POINT_BACK = (-1.0, 0.0, 0.0)
# LON: 0; LAT: 90
POINT_TOP = (0.0, 0.0, 1.0)
# LON: 0; LAT: -90
POINT_BOTTOM = (0.0, 0.0, -1.0)
# LON: -90; LAT: 0
POINT_LEFT = (0.0, -1.0, 0.0)
# LON: 90; LAT: 0
POINT_RIGHT = (0.0, 1.0, 0.0)

SPHERE_PARTS = [
    SphericalPolygon([POINT_FRONT, POINT_TOP, POINT_RIGHT, POINT_FRONT]),
    SphericalPolygon([POINT_RIGHT, POINT_TOP, POINT_BACK, POINT_RIGHT]),
    SphericalPolygon([POINT_BACK, POINT_TOP, POINT_LEFT, POINT_BACK]),
    SphericalPolygon([POINT_LEFT, POINT_TOP, POINT_FRONT, POINT_LEFT]),
    SphericalPolygon([POINT_FRONT, POINT_RIGHT, POINT_BOTTOM, POINT_FRONT]),
    SphericalPolygon([POINT_RIGHT, POINT_BACK, POINT_BOTTOM, POINT_RIGHT]),
    SphericalPolygon([POINT_BACK, POINT_LEFT, POINT_BOTTOM, POINT_BACK]),
    SphericalPolygon([POINT_LEFT, POINT_FRONT, POINT_BOTTOM, POINT_LEFT]),
]
SPHERE_PARTS_BOUNDING_BOXES = [
    box(minx=0, miny=0, maxx=90, maxy=90),
    box(minx=90, miny=0, maxx=180, maxy=90),
    box(minx=-180, miny=0, maxx=-90, maxy=90),
    box(minx=-90, miny=0, maxx=0, maxy=90),
    box(minx=0, miny=-90, maxx=90, maxy=0),
    box(minx=90, miny=-90, maxx=180, maxy=0),
    box(minx=-180, miny=-90, maxx=-90, maxy=0),
    box(minx=-90, miny=-90, maxx=0, maxy=0),
]


class SphereEllipsoid(Ellipsoid):  # type: ignore
    """A sphere ellipsoid."""

    def __init__(self) -> None:
        """
        Sphere ellipsoid extends Ellipsoid from pymap3d [1] library.

        Class is used for mapping lat/lon coordinates
        from and to cartesian x/y/z values on a unit sphere.
        Required for spherical voronoi algorithm.

        References:
            [1] https://github.com/geospace-code/pymap3d
        """  # noqa: W505, E501
        self.semimajor_axis = 1
        self.semiminor_axis = 1
        self.flattening = (self.semimajor_axis - self.semiminor_axis) / self.semimajor_axis
        self.thirdflattening = (self.semimajor_axis - self.semiminor_axis) / (
            self.semimajor_axis + self.semiminor_axis
        )
        self.eccentricity = sqrt(2 * self.flattening - self.flattening**2)


def map_to_geocentric(lon: float, lat: float, ell: Ellipsoid) -> Tuple[float, float, float]:
    """
    Wrapper for a geodetic2ecef function from pymap3d library.

    Args:
        lon (float): longitude of a point in a wgs84 crs.
        lat (float): latitude of a point in a wgs84 crs.

    Returns:
        Tuple[float, float, float]: (x, y, z) coordinates tuple.
    """
    x, y, z = geodetic2ecef(lat, lon, 0, ell=ell)
    return x, y, z


def map_from_geocentric(x: float, y: float, z: float, ell: Ellipsoid) -> Tuple[float, float]:
    """
        Wrapper for a ecef2geodetic function from pymap3d library.

    Args:
        x (float): X cartesian coordinate
        y (float): Y cartesian coordinate
        z (float): Z cartesian coordinate

    Returns:
        Tuple[float, float]: longitude and latitude coordinates in a wgs84 crs.
    """
    lat, lon, _ = ecef2geodetic(x, y, z, ell=ell)
    return lon, lat


def _fix_lat_lon(
    lon: float,
    lat: float,
    bbox: Polygon,
) -> Tuple[float, float]:
    """
    Fix point signs and rounding.

    Rounds latitude and longitude to 8 decimal places.
    Checks if any point is on a boundary and flips its sign
    to ensure validity of a polygon.

    Args:
        lon (float): Longitude of a point.
        lat (float): Latitude of a point.
        bbox (Polygon): Current sphere octant bounding box.

    Returns:
        Tuple[float, float]: Longitude and latitude of a point.
    """
    min_lon, min_lat, max_lon, max_lat = bbox.bounds

    # round imperfections
    lon = round(lon, 8)
    lat = round(lat, 8)

    # switch signs
    if lon and abs(lon) == abs(min_lon):
        lon = min_lon
    elif lon and abs(lon) == abs(max_lon):
        lon = max_lon
    if lat and abs(lat) == abs(min_lat):
        lat = min_lat
    elif lat and abs(lat) == abs(max_lat):
        lat = max_lat

    return lon, lat


def _create_polygon(
    spherical_polygon_points: npt.NDArray,
    bbox: Polygon,
    se: SphereEllipsoid,
    max_step: int,
) -> Polygon:
    """
    Map polygon from a sphere to Shapely polygon.

    Function maps and interpolates points from a sphere
    into a wgs84 crs while keeping integrity across all coordinates.

    Args:
        spherical_polygon_points (npt.NDArray): List of spherical points.
        bbox: (Polygon): Current sphere octant bounding box.
        se (SphereEllipsoid): SphereEllipsoid object
        max_step (int): Max step between interpolated points on an arc.

    Returns:
        Polygon: Mapped polygon in wgs84 crs.
    """
    polygon_points = []
    prev_lon = None
    prev_lat = None
    n = len(spherical_polygon_points)
    for i in range(n):
        start = spherical_polygon_points[i]
        end = spherical_polygon_points[(i + 1) % n]
        start_lon, start_lat = map_from_geocentric(start[0], start[1], start[2], se)
        end_lon, end_lat = map_from_geocentric(end[0], end[1], end[2], se)
        haversine_distance = haversine((start_lat, start_lon), (end_lat, end_lon), unit="m")
        steps = ceil(haversine_distance / max_step)
        t_vals = np.linspace(0, 1, steps)
        for pt in geometric_slerp(start, end, t_vals):
            lon, lat = map_from_geocentric(pt[0], pt[1], pt[2], se)
            lon, lat = _fix_lat_lon(lon, lat, bbox)
            if prev_lon is not None and abs(prev_lon - lon) >= 90:
                sign = 1 if lat > 0 else -1
                max_lat = sign * max(abs(prev_lat), abs(lat))
                if polygon_points[-1] != (prev_lon, max_lat):
                    polygon_points.append((prev_lon, max_lat))
                if polygon_points[-1] != (lon, lat):
                    polygon_points.append((lon, max_lat))
            polygon_points.append((lon, lat))
            prev_lon = lon
            prev_lat = lat

    polygon = Polygon(polygon_points)
    polygon = polygon.intersection(bbox)
    return polygon


def generate_voronoi_regions(
    seeds: List[Point], max_meters_between_points: int
) -> List[MultiPolygon]:
    """
    Generate Thessien polygons for a given list of seeds.

    Function generates Thessien polygons on a sphere using
    SphericalVoronoi algorithm from scipy library.

    Args:
        seeds (List[Point]): List of seeds used for generating regions.
        max_meters_between_points (int): maximal distance between points
        during interpolation of two vertices on a sphere.

    Returns:
        List[MultiPolygon]: List of regions cut into up to 8 polygons based
        on 8 parts of a sphere.
    """
    if len(seeds) == 1:
        return [MultiPolygon([box(minx=-180, maxx=180, miny=-90, maxy=90)])]

    se = SphereEllipsoid()
    mapped_points = [map_to_geocentric(pt.x, pt.y, se) for pt in seeds]
    sphere_points = np.array([[pt[0], pt[1], pt[2]] for pt in mapped_points])

    radius = 1
    center = np.array([0, 0, 0])
    sv = SphericalVoronoi(sphere_points, radius, center)
    sv.sort_vertices_of_regions()

    generated_regions: List[MultiPolygon] = []
    for region in sv.regions:
        region_vertices = [v for v in sv.vertices[region]]
        sph_pol = SphericalPolygon(region_vertices)
        sphere_intersection_parts = [
            (intersection, sphere_part_bbox)
            for sphere_part, sphere_part_bbox in zip(SPHERE_PARTS, SPHERE_PARTS_BOUNDING_BOXES)
            if (intersection := sph_pol.intersection(sphere_part)).area() > 0
        ]
        multi_polygon_parts: List[Polygon] = []
        for sphere_intersection_part, bbox in sphere_intersection_parts:
            for spherical_polygon_points in sphere_intersection_part.points:
                polygon = _create_polygon(
                    spherical_polygon_points=spherical_polygon_points,
                    bbox=bbox,
                    se=se,
                    max_step=max_meters_between_points,
                )
                multi_polygon_parts.append(polygon)

        multi_polygon = MultiPolygon(multi_polygon_parts)
        generated_regions.append(multi_polygon)

    return generated_regions
