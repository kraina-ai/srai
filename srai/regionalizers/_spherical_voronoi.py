"""
Spherical voronoi utils.

This module contains spherical voronoi implementation based on SphericalVoronoi function from scipy
library.
"""

import hashlib
import warnings
from collections.abc import Hashable
from contextlib import suppress
from functools import partial
from math import ceil
from multiprocessing import cpu_count
from typing import Optional, Union, cast

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from haversine import haversine
from pymap3d import Ellipsoid, geodetic2ecef
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

from srai.constants import GEOMETRY_COLUMN, WGS84_CRS

SPHERE_PARTS: list[SphericalPolygon] = []
SPHERE_PARTS_BOUNDING_BOXES: list[Polygon] = []

SCIPY_THRESHOLD = 1e-8


VertexHash = bytes
EdgeHash = tuple[VertexHash, VertexHash]


def _generate_sphere_parts() -> None:
    global SPHERE_PARTS, SPHERE_PARTS_BOUNDING_BOXES  # noqa: PLW0603

    if not SPHERE_PARTS:
        POINT_FRONT = (1.0, 0.0, 0.0)  # LON: 0; LAT: 0
        POINT_BACK = (-1.0, 0.0, 0.0)  # LON: 180; LAT: 0
        POINT_TOP = (0.0, 0.0, 1.0)  # LON: 0; LAT: 90
        POINT_BOTTOM = (0.0, 0.0, -1.0)  # LON: 0; LAT: -90
        POINT_LEFT = (0.0, -1.0, 0.0)  # LON: -90; LAT: 0
        POINT_RIGHT = (0.0, 1.0, 0.0)  # LON: 90; LAT: 0

        with np.errstate(invalid="ignore"):
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
    seeds: Union[gpd.GeoDataFrame, list[Point]],
    max_meters_between_points: int = 10_000,
    num_of_multiprocessing_workers: int = -1,
    multiprocessing_activation_threshold: Optional[int] = None,
) -> list[MultiPolygon]:
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
        seeds, region_ids = _parse_geodataframe_seeds(seeds)
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

    total_regions = len(sv.regions)
    region_ids = list(range(total_regions))

    activate_multiprocessing = (
        num_of_multiprocessing_workers > 1 and total_regions >= multiprocessing_activation_threshold
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # generate all spherical polygons

        generate_spherical_polygons_parts_func = partial(
            _generate_spherical_polygons_parts,
            sv=sv,
        )

        if activate_multiprocessing:
            spherical_polygons_parts = [  # noqa: FURB179
                polygon_part_tuple
                for polygon_part_tuples in process_map(
                    generate_spherical_polygons_parts_func,
                    region_ids,
                    desc="Generating spherical polygons",
                    max_workers=num_of_multiprocessing_workers,
                    chunksize=ceil(total_regions / (4 * num_of_multiprocessing_workers)),
                )
                for polygon_part_tuple in polygon_part_tuples
            ]
        else:
            spherical_polygons_parts = [
                polygon_part_tuple
                for region_id in tqdm(region_ids, desc="Generating spherical polygons")
                for polygon_part_tuple in generate_spherical_polygons_parts_func(
                    region_id=region_id
                )
            ]

        # identify all edges

        hashed_vertices: dict[VertexHash, npt.NDArray[np.float32]] = {}
        hashed_edges: set[EdgeHash] = set()

        regions_parts: dict[int, list[tuple[int, list[EdgeHash]]]] = {}

        for (
            region_id,
            sphere_part_id,
            spherical_polygon_points,
        ) in spherical_polygons_parts:
            if region_id not in regions_parts:
                regions_parts[region_id] = []

            n = len(spherical_polygon_points)
            polygon_vertices_hashes = []
            polygon_edges = []

            # Hash all vertices
            for i in range(n):
                start: npt.NDArray[np.float32] = spherical_polygon_points[i]
                start_hash = hashlib.sha256(start.data).digest()
                hashed_vertices[start_hash] = start
                polygon_vertices_hashes.append(start_hash)

            # Map all edges
            for i in range(n):
                start_hash = polygon_vertices_hashes[i]
                end_hash = polygon_vertices_hashes[(i + 1) % n]

                if start_hash == end_hash:
                    continue

                polygon_edges.append((start_hash, end_hash))

                if (start_hash, end_hash) not in hashed_edges and (
                    end_hash,
                    start_hash,
                ) not in hashed_edges:
                    hashed_edges.add(
                        (
                            start_hash,
                            end_hash,
                        )
                    )

            regions_parts[region_id].append((sphere_part_id, polygon_edges))

        # interpolate unique ones

        interpolated_edges: dict[EdgeHash, list[tuple[float, float]]]

        interpolate_polygon_edge_func = partial(
            _interpolate_polygon_edge,
            hashed_vertices=hashed_vertices,
            ell=unit_sphere_ellipsoid,
            max_meters_between_points=max_meters_between_points,
        )

        if activate_multiprocessing:
            interpolated_edges = {
                hashed_edge: interpolated_edge
                for hashed_edge, interpolated_edge in zip(
                    hashed_edges,
                    process_map(
                        interpolate_polygon_edge_func,
                        hashed_edges,
                        desc="Interpolating edges",
                        max_workers=num_of_multiprocessing_workers,
                        chunksize=ceil(len(hashed_edges) / (4 * num_of_multiprocessing_workers)),
                    ),
                )
            }
        else:
            interpolated_edges = {
                hashed_edge: interpolate_polygon_edge_func(hashed_edge)
                for hashed_edge in tqdm(hashed_edges, desc="Interpolating edges")
            }

        # use interpolated edges to map spherical polygons into regions

        generated_regions: list[MultiPolygon] = []
        _generate_sphere_parts()

        for region_id in tqdm(region_ids, desc="Generating polygons"):
            multi_polygon_parts: list[Polygon] = []

            for sphere_part_id, region_polygon_edges in regions_parts[region_id]:
                polygon_points: list[tuple[float, float]] = []

                for edge_start, edge_end in region_polygon_edges:
                    if (edge_start, edge_end) in interpolated_edges:
                        interpolated_edge = interpolated_edges[(edge_start, edge_end)]
                    else:
                        interpolated_edge = interpolated_edges[(edge_end, edge_start)][::-1]

                    interpolated_edge = _fix_edge(
                        interpolated_edge,
                        SPHERE_PARTS_BOUNDING_BOXES[sphere_part_id].bounds,
                        prev_lon=polygon_points[-1][0] if polygon_points else None,
                        prev_lat=polygon_points[-1][1] if polygon_points else None,
                    )

                    polygon_points.extend(interpolated_edge)

                polygon = Polygon(polygon_points)
                polygon = make_valid(polygon)
                polygon = polygon.intersection(SPHERE_PARTS_BOUNDING_BOXES[sphere_part_id])
                if isinstance(polygon, GeometryCollection):
                    for geometry in polygon.geoms:
                        if isinstance(geometry, (Polygon, MultiPolygon)):
                            polygon = geometry
                            break
                    else:
                        raise RuntimeError(
                            f"Intersection with a quadrant did not produce any Polygon. ({polygon})"
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
            generated_regions.append(multi_polygon)

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


def _parse_geodataframe_seeds(
    gdf: gpd.GeoDataFrame,
) -> tuple[list[Point], list[Hashable]]:
    """Transform GeoDataFrame into list of Points with index."""
    seeds_wgs84 = gdf.to_crs(crs=WGS84_CRS)
    region_ids: list[Hashable] = []
    seeds: list[Point] = []

    for index, row in seeds_wgs84.iterrows():
        candidate_point = row.geometry.centroid
        if not candidate_point.is_empty:
            region_ids.append(index)
            seeds.append(candidate_point)

    return seeds, region_ids


def _get_duplicated_seeds_ids(seeds: list[Point], region_ids: list[Hashable]) -> list[Hashable]:
    """Return all seeds ids that overlap with another using quick sjoin operation."""
    gdf = gpd.GeoDataFrame(data={GEOMETRY_COLUMN: seeds}, index=region_ids, crs=WGS84_CRS)
    duplicated_seeds = gdf.sjoin(gdf).index.value_counts().loc[lambda x: x > 1]
    duplicated_seeds_ids: list[Hashable] = duplicated_seeds.index.to_list()
    return duplicated_seeds_ids


def _check_if_in_bounds(seeds: list[Point]) -> bool:
    """Check if all seeds are within bounds."""
    wgs84_earth_bbox = (box(minx=-180, miny=-90, maxx=180, maxy=90),)
    return all(point.covered_by(wgs84_earth_bbox) for point in seeds)


def _map_to_geocentric(lon: float, lat: float, ell: Ellipsoid) -> tuple[float, float, float]:
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


def _generate_spherical_polygons_parts(
    region_id: int,
    sv: SphericalVoronoi,
) -> list[tuple[int, int, npt.NDArray[np.float32]]]:
    """
    Generate spherical polygon intersections with sphere parts.

    Args:
        region_id (int): Index of region in SphericalVoronoi result.
        sv (SphericalVoronoi): SphericalVoronoi object.

    Returns:
        List[Tuple[int, int, npt.NDArray[np.float32]]]: List of intersections containing
            region index, an index of a sphere part and a list of vertices.
    """
    region = sv.regions[region_id]
    region_vertices = np.array([v for v in sv.vertices[region]])

    sph_pol: Optional[SphericalPolygon] = None

    sphere_intersection_parts = []
    _generate_sphere_parts()

    for sphere_part_index, sphere_part in enumerate(SPHERE_PARTS):
        # check if whole region is inside
        if all(sphere_part.contains_point(point) for point in region_vertices):
            sphere_intersection_parts.append((region_id, sphere_part_index, region_vertices))
            # skip checking other sphere parts
            break
        # check if partially inside
        elif any(sphere_part.contains_point(point) for point in region_vertices):
            # delay SphericalPolygon creation since it's an expensive operation
            if not sph_pol:
                sph_pol = SphericalPolygon(region_vertices)

            intersection = sph_pol.intersection(sphere_part)
            for points in intersection.points:
                sphere_intersection_parts.append((region_id, sphere_part_index, points))

    # second check for the corner case when the region is on the intersection of 3 or 4 sphere parts
    # and sphere part only intersects a region's arc without vertex in it
    if len(sphere_intersection_parts) in (2, 3):
        for sphere_part_index, sphere_part in enumerate(SPHERE_PARTS):
            if any(
                sphere_intersection_part_tuple[1] == sphere_part_index
                for sphere_intersection_part_tuple in sphere_intersection_parts
            ):
                continue

            intersection = cast(SphericalPolygon, sph_pol).intersection(sphere_part)
            for points in intersection.points:
                sphere_intersection_parts.append((region_id, sphere_part_index, points))

    return sphere_intersection_parts


def _interpolate_polygon_edge(
    hashed_edge: EdgeHash,
    hashed_vertices: dict[VertexHash, npt.NDArray[np.float32]],
    ell: Ellipsoid,
    max_meters_between_points: int,
) -> list[tuple[float, float]]:
    """
    Interpolates spherical polygon arc edge to the latitude and longitude.

    Args:
        hashed_edge (EdgeHash): Edge hash containing start and end point.
        hashed_vertices (Dict[VertexHash, npt.NDArray[np.float32]]): Dict of hashed vertices.
        ell (Ellipsoid): Ellipsoid object.
        max_meters_between_points (int): Maximal distance between points.

    Returns:
        List[Tuple[float, float]]: _description_
    """
    start_point_hash, end_point_hash = hashed_edge
    start_point = hashed_vertices[start_point_hash]
    end_point = hashed_vertices[end_point_hash]

    longitudes, latitudes = _map_from_geocentric(
        [start_point[0], end_point[0]],
        [start_point[1], end_point[1]],
        [start_point[2], end_point[2]],
        ell,
    )
    start_lon, start_lat = longitudes[0], latitudes[0]
    end_lon, end_lat = longitudes[1], latitudes[1]
    haversine_distance = haversine((start_lat, start_lon), (end_lat, end_lon), unit="m")
    steps = max(ceil(haversine_distance / max_meters_between_points), 2)
    t_vals = np.linspace(0, 1, steps)

    edge_points = _interpolate_edge(
        start_point=start_point, end_point=end_point, step_ticks=t_vals, ell=ell
    )

    return edge_points


def _interpolate_edge(
    start_point: tuple[float, float, float],
    end_point: tuple[float, float, float],
    step_ticks: list[float],
    ell: Ellipsoid,
) -> list[tuple[float, float]]:
    """
    Generates latitude and longitude positions for an arc on the sphere.

    Args:
        start_point (Tuple[float, float, float]): Start position on an unit sphere.
        end_point (Tuple[float, float, float]): End position on an unit sphere.
        step_ticks (List[float]): Number of steps between.
        ell (Ellipsoid): Ellipsoid object.

    Returns:
        List[Tuple[float, float]]: List of latitude and longitude coordinates of the edge.
    """
    slerped_points = geometric_slerp(start_point, end_point, step_ticks, tol=SCIPY_THRESHOLD)

    xs = [pt[0] for pt in slerped_points]
    ys = [pt[1] for pt in slerped_points]
    zs = [pt[2] for pt in slerped_points]

    longitudes, latitudes = _map_from_geocentric(xs, ys, zs, ell)

    # round imperfections
    longitudes = np.round(longitudes, 8)
    latitudes = np.round(latitudes, 8)

    return [(longitude, latitude) for longitude, latitude in zip(longitudes, latitudes)]


def _fix_edge(
    edge_points: list[tuple[float, float]],
    bbox_bounds: tuple[float, float, float, float],
    prev_lon: Optional[float] = None,
    prev_lat: Optional[float] = None,
) -> list[tuple[float, float]]:
    """
    Fixes points laying on the edge between sphere parts.

    Args:
        edge_points (List[Tuple[float, float]]): Edge points to fix.
        bbox_bounds (Tuple[float, float, float, float]): Boundary box to use in the fixing.
        prev_lon (float, optional): Previous longitude of the edge. Can be passed to keep
            the context of the previous edges. Defaults to None.
        prev_lat (float, optional): Previous latitude of the edge. Can be passed to keep
            the context of the previous edges. Defaults to None.

    Returns:
        List[Tuple[float, float]]: Fixed edge points.
    """
    fixed_edge_points: list[tuple[float, float]] = []

    if prev_lon is not None and prev_lat is not None:
        fixed_edge_points.append((prev_lon, prev_lat))

    for longitude, latitude in edge_points:
        lon, lat = _fix_lat_lon(longitude, latitude, bbox_bounds)
        if prev_lon is not None and prev_lat is not None and abs(prev_lon - lon) >= 90:
            sign = 1 if lat > 0 else -1
            max_lat = sign * max(abs(prev_lat), abs(lat))
            if fixed_edge_points[-1] != (prev_lon, max_lat):
                fixed_edge_points.append((prev_lon, max_lat))
            if fixed_edge_points[-1] != (lon, lat):
                fixed_edge_points.append((lon, max_lat))
        fixed_edge_points.append((lon, lat))
        prev_lon = lon
        prev_lat = lat

    return fixed_edge_points


def _map_from_geocentric(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    z: npt.NDArray[np.float32],
    ell: Ellipsoid,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Wrapper for a ecef2geodetic function from pymap3d library.

    Args:
        x (npt.NDArray[np.float32]): X cartesian coordinate.
        y (npt.NDArray[np.float32]): Y cartesian coordinate.
        z (npt.NDArray[np.float32]): Z cartesian coordinate.
        ell (Ellipsoid): an ellipsoid.

    Returns:
        Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: longitude and latitude coordinates
            in a wgs84 crs.
    """
    lat, lon, _ = ecef2geodetic_vectorized(x, y, z, ell=ell)
    return lon, lat


# Copyright (c) 2014-2022 Michael Hirsch, Ph.D.
# Copyright (c) 2013, Felipe Geremia Nievinski
# Copyright (c) 2004-2007 Michael Kleder
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
# and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
# conditions and the following disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def ecef2geodetic_vectorized(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    z: npt.NDArray[np.float32],
    ell: Ellipsoid,
    deg: bool = True,
) -> npt.NDArray[np.float32]:
    """
    Modified function for mapping ecdf to geodetic values from ellipsoid.

    This function is a modified copy of `ecdf2geodetic` function from `pymap3d` library
    and is redistributed under BSD-2 license.

    Args:
        x (npt.NDArray[np.float32]): X coordinates.
        y (npt.NDArray[np.float32]): Y coordinates.
        z (npt.NDArray[np.float32]): Z coordinates.
        ell (Ellipsoid): Ellipsoid object.
        deg (bool, optional): Flag whether to return values in degrees. Defaults to True.

    Returns:
        npt.NDArray[np.float32]: Parsed latitudes and longitudes
    """
    with suppress(NameError):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

    r = np.sqrt(x**2 + y**2 + z**2)

    E = np.sqrt(ell.semimajor_axis**2 - ell.semiminor_axis**2)

    # eqn. 4a
    u = np.sqrt(0.5 * (r**2 - E**2) + 0.5 * np.hypot(r**2 - E**2, 2 * E * z))

    Q = np.hypot(x, y)

    huE = np.hypot(u, E)

    # eqn. 4b
    try:
        with warnings.catch_warnings(record=False):
            warnings.simplefilter("error")
            Beta = np.arctan(huE / u * z / np.hypot(x, y))
    except (ArithmeticError, RuntimeWarning):
        is_zero_dimensions = len(x.shape) == 0

        if is_zero_dimensions:
            if np.isclose(z, 0):
                Beta = 0
            elif z > 0:
                Beta = np.pi / 2
            else:
                Beta = -np.pi / 2
        else:
            _beta_arr = []

            for _x, _y, _z, _u, _huE in zip(x, y, z, u, huE):
                try:
                    with warnings.catch_warnings(record=False):
                        warnings.simplefilter("error")
                        _beta = np.arctan(_huE / _u * _z / np.hypot(_x, _y))
                except (ArithmeticError, RuntimeWarning):
                    if np.isclose(_z, 0):
                        _beta = 0
                    elif _z > 0:
                        _beta = np.pi / 2
                    else:
                        _beta = -np.pi / 2
                _beta_arr.append(_beta)

            Beta = np.asarray(_beta_arr)

    # eqn. 13
    dBeta = ((ell.semiminor_axis * u - ell.semimajor_axis * huE + E**2) * np.sin(Beta)) / (
        ell.semimajor_axis * huE * 1 / np.cos(Beta) - E**2 * np.cos(Beta)
    )

    Beta += dBeta

    # eqn. 4c
    lat = np.arctan(ell.semimajor_axis / ell.semiminor_axis * np.tan(Beta))

    with suppress(NameError):
        # patch latitude for float32 precision loss
        lim_pi2 = np.pi / 2 - np.finfo(dBeta.dtype).eps
        lat = np.where(Beta >= lim_pi2, np.pi / 2, lat)
        lat = np.where(Beta <= -lim_pi2, -np.pi / 2, lat)

    lon = np.arctan2(y, x)

    # eqn. 7
    cosBeta = np.cos(Beta)
    with suppress(NameError):
        # patch altitude for float32 precision loss
        cosBeta = np.where(Beta >= lim_pi2, 0, cosBeta)
        cosBeta = np.where(Beta <= -lim_pi2, 0, cosBeta)

    alt = np.hypot(z - ell.semiminor_axis * np.sin(Beta), Q - ell.semimajor_axis * cosBeta)

    # inside ellipsoid?
    inside = (
        x**2 / ell.semimajor_axis**2 + y**2 / ell.semimajor_axis**2 + z**2 / ell.semiminor_axis**2
        < 1
    )

    try:
        if inside.any():
            # avoid all false assignment bug
            alt[inside] = -alt[inside]
    except (TypeError, AttributeError):
        if inside:
            alt = -alt

    if deg:
        lat = np.degrees(lat)
        lon = np.degrees(lon)

    return lat, lon, alt


def _fix_lat_lon(
    lon: float,
    lat: float,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    """
    Fix point signs and rounding.

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
