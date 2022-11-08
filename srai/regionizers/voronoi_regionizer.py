"""
Voronoi Regionizer.

This module contains voronoi regionizer implementation.

"""

from typing import Optional

import geopandas as gpd
from shapely.geometry import box

from ._spherical_voronoi import generate_voronoi_regions


class VoronoiRegionizer:
    """
    VoronoiRegionizer.

    Voronoi [1] regionizer allows the given geometries to be divided
    into Thiessen polygons using geometries that are the seeds. To
    minimize distortions tessellation will be performed on a sphere
    using SphericalVoronoi [2] function from scipy library.

    References:
        [1] https://en.wikipedia.org/wiki/Voronoi_diagram
        [2] https://docs.scipy.org/doc/scipy-1.9.2/reference/generated/scipy.spatial.SphericalVoronoi.html

    """  # noqa: W505, E501

    def __init__(self, seeds: gpd.GeoDataFrame, max_meters_between_points: int = 10_000) -> None:
        """
        Init VoronoiRegionizer.

        All (multi)polygons from seeds GeoDataFrame will be transformed to their centroids,
        because scipy function requires only points as an input.

        Args:
            seeds (gpd.GeoDataFrame): GeoDataFrame with seeds for
                creating a tessellation. Minimum 4 seeds are required.
                Seeds cannot lie on a single arc. Empty seeds will be removed.
            max_meters_between_points (int): Maximal distance in meters between two points
                in a resulting polygon. Higher number results lower resolution of a polygon.

        Raises:
            ValueError: If any seed is duplicated.
            ValueError: If less than 4 seeds are provided.
            ValueError: If provided seeds geodataframe has no crs defined.

        """
        if len(seeds.index) < 4:
            raise ValueError("Minimum 4 seeds are required.")

        seeds_wgs84 = seeds.to_crs(epsg=4326)
        self.region_ids = []
        self.seeds = []
        self.max_meters_between_points = max_meters_between_points
        for index, row in seeds_wgs84.iterrows():
            candidate_point = row.geometry.centroid
            if not candidate_point.is_empty:
                self.region_ids.append(index)
                self.seeds.append(candidate_point)

        if self._check_duplicate_points():
            raise ValueError("Duplicate seeds present.")

    def transform(self, gdf: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
        """
        Regionize a given GeoDataFrame.

        Returns a list of disjointed regions consisting of Thiessen cells generated
        using a Voronoi diagram on the sphere.

        Args:
            gdf (Optional[gpd.GeoDataFrame], optional): GeoDataFrame to be regionized.
                Will use this list of geometries to crop resulting regions. If None, a boundary box
                with bounds (-180, -90, 180, 90) is used to return regions covering whole Earth.
                Defaults to None.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with the regionized data cropped using input
                GeoDataFrame.

        Raises:
            ValueError: If provided geodataframe has no crs defined.
            ValueError: If seeds are laying on a single arc.

        """
        if gdf is None:
            gdf = gpd.GeoDataFrame(
                {"geometry": [box(minx=-180, maxx=180, miny=-90, maxy=90)]}, crs="EPSG:4326"
            )

        gds_wgs84 = gdf.to_crs(epsg=4326)
        generated_regions = generate_voronoi_regions(self.seeds, self.max_meters_between_points)
        regions_gdf = gpd.GeoDataFrame(
            data={"geometry": generated_regions}, index=self.region_ids, crs=4326
        )
        regions_gdf.index.rename("region_id", inplace=True)
        clipped_regions_gdf = regions_gdf.clip(mask=gds_wgs84, keep_geom_type=False)
        return clipped_regions_gdf

    def _check_duplicate_points(self) -> bool:
        """Check if any point overlaps with another using quick sjoin operation."""
        gdf = gpd.GeoDataFrame(data=[{"geometry": s} for s in self.seeds])
        return len(gdf.sjoin(gdf).index) != len(self.seeds)
