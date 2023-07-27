"""
Spherical voronoi utils.

This module contains spherical voronoi implementation based on SphericalVoronoi function from scipy
library.
"""

from functools import partial
from math import ceil
from multiprocessing import cpu_count
from typing import Hashable, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from haversine import haversine
from pymap3d import Ellipsoid, ecef2geodetic, geodetic2ecef
from scipy.spatial import SphericalVoronoi, geometric_slerp
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.validation import make_valid
from spherical_geometry.polygon import SphericalPolygon
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from srai.constants import WGS84_CRS

SPHERE_PARTS: List[SphericalPolygon] = []
SPHERE_PARTS_BOUNDING_BOXES: List[Polygon] = []

SCIPY_THRESHOLD = 1e-8


def _generate_sphere_parts() -> None:
    global SPHERE_PARTS, SPHERE_PARTS_BOUNDING_BOXES  # noqa: PLW0603

    if not SPHERE_PARTS:
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
            SphericalPolygon([POINT_FRONT, POINT_TOP, POINT_BACK, POINT_RIGHT, POINT_FRONT]),
            SphericalPolygon([POINT_FRONT, POINT_RIGHT, POINT_BACK, POINT_BOTTOM, POINT_FRONT]),
            SphericalPolygon([POINT_FRONT, POINT_BOTTOM, POINT_BACK, POINT_LEFT, POINT_FRONT]),
            SphericalPolygon([POINT_FRONT, POINT_LEFT, POINT_BACK, POINT_TOP, POINT_FRONT]),
        ]
        SPHERE_PARTS_BOUNDING_BOXES = [
            box(minx=0, miny=0, maxx=180, maxy=90),
            box(minx=0, miny=-90, maxx=180, maxy=0),
            box(minx=-180, miny=-90, maxx=0, maxy=0),
            box(minx=-180, miny=0, maxx=0, maxy=90),
        ]


def generate_voronoi_regions(
    seeds: Union[gpd.GeoDataFrame, List[Point]],
    max_meters_between_points: int = 10_000,
    num_of_multiprocessing_workers: int = -1,
    multiprocessing_activation_threshold: Optional[int] = None,
) -> List[MultiPolygon]:
    """
    Generate Thessien polygons for a given list of seeds.

    Function generates Thessien polygons on a sphere using
    SphericalVoronoi algorithm from scipy library.

    Args:
        seeds (Union[gpd.GeoDataFrame, List[Point]]): Seeds used for generating regions.
            If list, the points are expected to be in WGS84 coordinates (lat, lon).
            Otherwise, a GeoDataFrame will be transformed into WGS84.
        max_meters_between_points (int, optional): Maximal distance in meters between two points
            in the resulting polygon. Higher number results lower resolution of a polygon.
            Defaults to 10_000 (10 kilometers).
        num_of_multiprocessing_workers (int, optional): Number of workers used for multiprocessing.
            Defaults to -1 which results in a total number of available cpu threads.
            `0` and `1` values disable multiprocessing.
            Similar to `n_jobs` parameter from `scikit-learn` library.
        multiprocessing_activation_threshold (int, optional): Number of seeds required to start
            processing on multiple processes. Activating multiprocessing for a small
            amount of points might not be feasible. Defaults to 100.

    Returns:
        List[MultiPolygon]: List of MultiPolygons cut into up to 4 polygons based
            on quadrants of a sphere.

    Raises:
        ValueError: If less than 4 seeds are provided.
        ValueError: If any seed is duplicated.
        ValueError: If any seed is outside WGS84 coordinates domain.
    """
    if isinstance(seeds, gpd.GeoDataFrame):
        seeds, region_ids = _generate_seeds(seeds)
    else:
        region_ids = list(range(len(seeds)))

    if len(seeds) < 4:
        raise ValueError("Minimum 4 seeds are required.")

    duplicated_seeds_ids = _get_duplicated_seeds_ids(seeds, region_ids)
    if duplicated_seeds_ids:
        raise ValueError(f"Duplicate seeds present: {duplicated_seeds_ids}.")

    if not _check_if_in_bounds(seeds):
        raise ValueError("Seeds outside Earth WGS84 bounding box.")

    num_of_multiprocessing_workers = _parse_num_of_multiprocessing_workers(
        num_of_multiprocessing_workers
    )
    multiprocessing_activation_threshold = _parse_multiprocessing_activation_threshold(
        multiprocessing_activation_threshold
    )

    unit_sphere_ellipsoid = Ellipsoid(
        semimajor_axis=1, semiminor_axis=1, name="Unit Sphere", model="Unit"
    )
    mapped_points = [_map_to_geocentric(pt.x, pt.y, unit_sphere_ellipsoid) for pt in seeds]
    sphere_points = np.array([[pt[0], pt[1], pt[2]] for pt in mapped_points])

    radius = 1
    center = np.array([0, 0, 0])
    sv = SphericalVoronoi(sphere_points, radius, center, threshold=SCIPY_THRESHOLD)
    sv.sort_vertices_of_regions()

    create_regions_func = partial(
        _create_region,
        sv=sv,
        ell=unit_sphere_ellipsoid,
        max_meters_between_points=max_meters_between_points,
    )

    total_regions = len(sv.regions)
    region_ids = list(range(total_regions))

    generated_regions: List[MultiPolygon]

    if num_of_multiprocessing_workers > 1 and total_regions >= multiprocessing_activation_threshold:
        generated_regions = process_map(
            create_regions_func,
            region_ids,
            desc="Generating regions",
            max_workers=num_of_multiprocessing_workers,
            chunksize=ceil(total_regions / (4 * num_of_multiprocessing_workers)),
        )
    else:
        generated_regions = [
            create_regions_func(region_id=region_id)
            for region_id in tqdm(region_ids, desc="Generating regions")
        ]

    return generated_regions


def _parse_num_of_multiprocessing_workers(num_of_multiprocessing_workers: int) -> int:
    if num_of_multiprocessing_workers == 0:
        num_of_multiprocessing_workers = 1
    elif num_of_multiprocessing_workers < 0:
        num_of_multiprocessing_workers = cpu_count()

    return num_of_multiprocessing_workers


def _parse_multiprocessing_activation_threshold(
    multiprocessing_activation_threshold: Optional[int],
) -> int:
    if not multiprocessing_activation_threshold:
        multiprocessing_activation_threshold = 100

    return multiprocessing_activation_threshold


def _generate_seeds(gdf: gpd.GeoDataFrame) -> Tuple[List[Point], List[Hashable]]:
    """Transform GeoDataFrame into list of Points with index."""
    seeds_wgs84 = gdf.to_crs(crs=WGS84_CRS)
    region_ids: List[Hashable] = []
    seeds: List[Point] = []

    for index, row in seeds_wgs84.iterrows():
        candidate_point = row.geometry.centroid
        if not candidate_point.is_empty:
            region_ids.append(index)
            seeds.append(candidate_point)

    return seeds, region_ids


def _get_duplicated_seeds_ids(seeds: List[Point], region_ids: List[Hashable]) -> List[Hashable]:
    """Return all seeds ids that overlap with another using quick sjoin operation."""
    gdf = gpd.GeoDataFrame(data={"geometry": seeds}, index=region_ids, crs=WGS84_CRS)
    duplicated_seeds = gdf.sjoin(gdf).index.value_counts().loc[lambda x: x > 1]
    duplicated_seeds_ids: List[Hashable] = duplicated_seeds.index.to_list()
    return duplicated_seeds_ids


def _check_if_in_bounds(seeds: List[Point]) -> bool:
    """Check if all seeds are within bounds."""
    wgs84_earth_bbox = (box(minx=-180, miny=-90, maxx=180, maxy=90),)
    return all(point.covered_by(wgs84_earth_bbox) for point in seeds)


def _map_to_geocentric(lon: float, lat: float, ell: Ellipsoid) -> Tuple[float, float, float]:
    """
    Wrapper for a geodetic2ecef function from pymap3d library.

    Args:
        lon (float): longitude of a point in a wgs84 crs.
        lat (float): latitude of a point in a wgs84 crs.
        ell (Ellipsoid): an ellipsoid.

    Returns:
        Tuple[float, float, float]: (x, y, z) coordinates tuple.
    """
    x, y, z = geodetic2ecef(lat, lon, 0, ell=ell)
    return x, y, z


def _create_region(
    region_id: int,
    sv: SphericalVoronoi,
    ell: Ellipsoid,
    max_meters_between_points: int,
) -> MultiPolygon:
    """
    Parse spherical region into a WGS84 MultiPolygon.

    Args:
        region_id (int): Index of region in SphericalVoronoi result.
        sv (SphericalVoronoi): SphericalVoronoi object.
        ell (Ellipsoid): Ellipsoid object.
        max_meters_between_points (int): maximal distance between points
            during interpolation of two vertices on a sphere.

    Returns:
        MultiPolygon: Parsed region in WGS84 coordinates.
    """
    region = sv.regions[region_id]
    region_vertices = [v for v in sv.vertices[region]]
    sph_pol = SphericalPolygon(region_vertices)
    sphere_intersection_parts = []
    _generate_sphere_parts()
    for sphere_part, sphere_part_bbox in zip(SPHERE_PARTS, SPHERE_PARTS_BOUNDING_BOXES):
        if sph_pol.intersects_poly(sphere_part):
            intersection = sph_pol.intersection(sphere_part)
            sphere_intersection_parts.append((intersection, sphere_part_bbox))

    multi_polygon_parts: List[Polygon] = []
    for sphere_intersection_part, bbox in sphere_intersection_parts:
        for spherical_polygon_points in sphere_intersection_part.points:
            polygon = _create_polygon(
                spherical_polygon_points=spherical_polygon_points,
                bbox=bbox,
                ell=ell,
                max_step=max_meters_between_points,
            )
            if isinstance(polygon, Polygon):
                multi_polygon_parts.append(polygon)
            elif isinstance(polygon, MultiPolygon):
                multi_polygon_parts.extend(polygon.geoms)
            elif isinstance(polygon, (LineString, MultiLineString, Point)):
                pass
            else:
                raise RuntimeError(str(type(polygon)))

    multi_polygon = MultiPolygon(multi_polygon_parts)
    return multi_polygon


def _create_polygon(
    spherical_polygon_points: npt.NDArray[np.float32],
    bbox: Polygon,
    ell: Ellipsoid,
    max_step: int,
) -> Polygon:
    """
    Map polygon from a sphere to Shapely polygon.

    Function maps and interpolates points from a sphere
    into a wgs84 crs while keeping integrity across all coordinates.

    Args:
        spherical_polygon_points (npt.NDArray): List of spherical points.
        bbox (Polygon): Current sphere octant bounding box.
        ell (Ellipsoid): Ellipsoid object.
        max_step (int): Max step between interpolated points on an arc.

    Returns:
        Polygon: Mapped polygon in wgs84 crs.
    """
    polygon_points: List[Tuple[float, float]] = []

    n = len(spherical_polygon_points)
    bbox_bounds = bbox.bounds
    for i in range(n):
        start = spherical_polygon_points[i]
        end = spherical_polygon_points[(i + 1) % n]
        start_lon, start_lat = _map_from_geocentric(start[0], start[1], start[2], ell)
        end_lon, end_lat = _map_from_geocentric(end[0], end[1], end[2], ell)
        haversine_distance = haversine((start_lat, start_lon), (end_lat, end_lon), unit="m")
        steps = ceil(haversine_distance / max_step)
        t_vals = np.linspace(0, 1, steps)

        reverse_slerp = end_lat > start_lat or (start_lat == end_lat and end_lon > start_lon)
        if reverse_slerp:
            start, end = end, start

        edge_points = _interpolate_edge(
            start_point=start, end_point=end, step_ticks=t_vals, ell=ell, bbox_bounds=bbox_bounds
        )

        polygon_points.extend(edge_points if not reverse_slerp else reversed(edge_points))

    polygon = Polygon(polygon_points)
    polygon = make_valid(polygon)
    polygon = polygon.intersection(bbox)
    if isinstance(polygon, GeometryCollection):
        for geometry in polygon.geoms:
            if isinstance(geometry, (Polygon, MultiPolygon)):
                polygon = geometry
                break
        else:
            raise RuntimeError(
                f"Intersection with a quadrant did not produce any Polygon. ({polygon})"
            )

    return polygon


def _interpolate_edge(
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    step_ticks: List[float],
    ell: Ellipsoid,
    bbox_bounds: Tuple[float, float, float, float],
) -> List[Tuple[float, float]]:
    edge_points: List[Tuple[float, float]] = []

    prev_lon = None
    prev_lat = None

    for pt in geometric_slerp(start_point, end_point, step_ticks, tol=SCIPY_THRESHOLD):
        lon, lat = _map_from_geocentric(pt[0], pt[1], pt[2], ell)
        lon, lat = _fix_lat_lon(lon, lat, bbox_bounds)
        if prev_lon is not None and abs(prev_lon - lon) >= 90:
            sign = 1 if lat > 0 else -1
            max_lat = sign * max(abs(prev_lat), abs(lat))
            if edge_points[-1] != (prev_lon, max_lat):
                edge_points.append((prev_lon, max_lat))
            if edge_points[-1] != (lon, lat):
                edge_points.append((lon, max_lat))
        edge_points.append((lon, lat))
        prev_lon = lon
        prev_lat = lat

    return edge_points


def _map_from_geocentric(x: float, y: float, z: float, ell: Ellipsoid) -> Tuple[float, float]:
    """
    Wrapper for a ecef2geodetic function from pymap3d library.

    Args:
        x (float): X cartesian coordinate.
        y (float): Y cartesian coordinate.
        z (float): Z cartesian coordinate.
        ell (Ellipsoid): an ellipsoid.

    Returns:
        Tuple[float, float]: longitude and latitude coordinates in a wgs84 crs.
    """
    lat, lon, _ = ecef2geodetic(x, y, z, ell=ell)
    return lon, lat


def _fix_lat_lon(
    lon: float,
    lat: float,
    bbox: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """
    Fix point signs and rounding.

    Rounds latitude and longitude to 8 decimal places.
    Checks if any point is on a boundary and flips its sign
    to ensure validity of a polygon.

    Args:
        lon (float): Longitude of a point.
        lat (float): Latitude of a point.
        bbox (Tuple[float, float, float, float]): Current sphere octant bounding box.

    Returns:
        Tuple[float, float]: Longitude and latitude of a point.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

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
