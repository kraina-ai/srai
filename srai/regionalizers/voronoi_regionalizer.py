"""
Voronoi Regionalizer.

This module contains voronoi regionalizer implementation.
"""

from typing import Hashable, List, Optional, Union

import geopandas as gpd
from shapely.geometry import Point, box

from srai._optional import import_optional_dependencies
from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.regionalizers import Regionalizer


class VoronoiRegionalizer(Regionalizer):
    """
    VoronoiRegionalizer.

    Voronoi [1] regionalizer allows the given geometries to be divided
    into Thiessen polygons using geometries that are the seeds. To
    minimize distortions tessellation will be performed on a sphere
    using SphericalVoronoi [2] function from scipy library.

    References:
        1. https://en.wikipedia.org/wiki/Voronoi_diagram
        2. https://docs.scipy.org/doc/scipy-1.9.2/reference/generated/scipy.spatial.SphericalVoronoi.html
    """  # noqa: W505, E501

    def __init__(
        self,
        seeds: Union[gpd.GeoDataFrame, List[Point]],
        max_meters_between_points: int = 10_000,
        num_of_multiprocessing_workers: int = -1,
        multiprocessing_activation_threshold: Optional[int] = None,
    ) -> None:
        """
        Init VoronoiRegionalizer.

        All (multi)polygons from seeds GeoDataFrame will be transformed to their centroids,
        because scipy function requires only points as an input.

        Args:
            seeds (Union[gpd.GeoDataFrame, List[Point]]): List of points or a GeoDataFrame
                with seeds for creating a tessellation. Every non-point geometry will be mapped
                to a centroid. Minimum 4 seeds are required. Seeds cannot lie on a single arc.
                Empty seeds will be removed.
            max_meters_between_points (int): Maximal distance in meters between two points
                in a resulting polygon. Higher number results lower resolution of a polygon.
            num_of_multiprocessing_workers (int): Number of workers used for
                multiprocessing. Defaults to `-1` which results in a total number of available
                cpu threads. `0` and `1` values disable multiprocessing.
                Similar to `n_jobs` parameter from `scikit-learn` library.
            multiprocessing_activation_threshold (int, optional): Number of seeds required to start
                processing on multiple processes. Activating multiprocessing for a small
                amount of points might not be feasible. Defaults to 100.

        Raises:
            ValueError: If any seed is duplicated.
            ValueError: If less than 4 seeds are provided.
            ValueError: If provided seeds geodataframe has no crs defined.
        """
        import_optional_dependencies(
            dependency_group="voronoi",
            modules=["haversine", "pymap3d", "scipy", "spherical_geometry"],
        )
        self.region_ids: List[Hashable] = []
        self.seeds: List[Point] = []

        if isinstance(seeds, gpd.GeoDataFrame):
            self._parse_geodataframe_seeds(seeds_gdf=seeds)
        else:
            self.seeds = seeds
            self.region_ids = list(range(len(seeds)))

        self.max_meters_between_points = max_meters_between_points
        self.num_of_multiprocessing_workers = num_of_multiprocessing_workers
        self.multiprocessing_activation_threshold = multiprocessing_activation_threshold

        if len(self.seeds) < 4:
            raise ValueError("Minimum 4 seeds are required.")

        duplicated_seeds_ids = self._get_duplicated_seeds_ids()
        if duplicated_seeds_ids:
            raise ValueError(f"Duplicate seeds present: {duplicated_seeds_ids}.")

    def transform(self, gdf: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
        """
        Regionalize a given GeoDataFrame.

        Returns a list of disjointed regions consisting of Thiessen cells generated
        using a Voronoi diagram on the sphere.

        Args:
            gdf (Optional[gpd.GeoDataFrame], optional): GeoDataFrame to be regionalized.
                Will use this list of geometries to crop resulting regions. If None, a boundary box
                with bounds (-180, -90, 180, 90) is used to return regions covering whole Earth.
                Defaults to None.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with the regionalized data cropped using input
                GeoDataFrame.

        Raises:
            ValueError: If provided geodataframe has no crs defined.
            ValueError: If seeds are laying on a single arc.
        """
        from ._spherical_voronoi import generate_voronoi_regions

        if gdf is None:
            gdf = gpd.GeoDataFrame(
                {GEOMETRY_COLUMN: [box(minx=-180, maxx=180, miny=-90, maxy=90)]}, crs=WGS84_CRS
            )

        gdf_wgs84 = gdf.to_crs(crs=WGS84_CRS)
        generated_regions = generate_voronoi_regions(
            seeds=self.seeds,
            max_meters_between_points=self.max_meters_between_points,
            num_of_multiprocessing_workers=self.num_of_multiprocessing_workers,
            multiprocessing_activation_threshold=self.multiprocessing_activation_threshold,
        )
        regions_gdf = gpd.GeoDataFrame(
            data={GEOMETRY_COLUMN: generated_regions}, index=self.region_ids, crs=WGS84_CRS
        )
        regions_gdf.index.rename(REGIONS_INDEX, inplace=True)
        clipped_regions_gdf = regions_gdf.clip(mask=gdf_wgs84, keep_geom_type=False)
        return clipped_regions_gdf

    def _parse_geodataframe_seeds(self, seeds_gdf: gpd.GeoDataFrame) -> None:
        seeds_wgs84 = seeds_gdf.to_crs(crs=WGS84_CRS)
        for index, row in seeds_wgs84.iterrows():
            candidate_point = row.geometry.centroid
            if not candidate_point.is_empty:
                self.region_ids.append(index)
                self.seeds.append(candidate_point)

    def _get_duplicated_seeds_ids(self) -> List[Hashable]:
        """Return all seeds ids that overlap with another using quick sjoin operation."""
        gdf = gpd.GeoDataFrame(
            data={GEOMETRY_COLUMN: self.seeds}, index=self.region_ids, crs=WGS84_CRS
        )
        duplicated_seeds = gdf.sjoin(gdf).index.value_counts().loc[lambda x: x > 1]
        duplicated_seeds_ids: List[Hashable] = duplicated_seeds.index.to_list()
        return duplicated_seeds_ids
