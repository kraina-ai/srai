"""
Administrative Boundary Regionizer.

This module contains administrative boundary regionizer implementation.

"""

# from itertools import combinations
from typing import Optional

import geopandas as gpd

# from ._spherical_voronoi import generate_voronoi_regions
import osmnx as ox
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder

# from shapely.geometry import box

# import osmnx.settings as oxs


class AdministrativeBoundaryRegionizer:
    """
    AdministrativeBoundaryRegionizer.

    Administrative boundary regionizer allows the given geometries to be divided
    into boundaries from OpenStreetMap [1] on a given `admin_level`.

    Can download those boundaries online using `osmnx` library or load it from downloaded `.pbf`
    file.

    References:
        [1] https://wiki.openstreetmap.org/wiki/Tag:boundary=administrative

    """

    # https://wiki.openstreetmap.org/wiki/Key:admin_level

    def __init__(
        self,
        admin_level: int,
        pbf_file_path: Optional[str] = None,
        prioritize_english_name: bool = True,
    ) -> None:
        """
        Inits VoronoiRegionizer.

        TODO!!! All (multi)polygons from seeds GeoDataFrame will be transformed to their centroids,
        because scipy function requires only points as an input.

        Args:
            seeds (gpd.GeoDataFrame): GeoDataFrame with seeds for
                creating a tessellation. Minimum 4 seeds are required.
                Seeds cannot lie on a single arc.
            max_meters_between_points (int): Maximal distance in meters between two points
                in a resulting polygon. Higher number results lower resolution of a polygon.

        Raises:
            ValueError: If any seed is duplicated.
            ValueError: If less than 4 seeds are provided.
            ValueError: If provided seeds geodataframe has no crs defined.

        """
        self.admin_level = admin_level
        self.prioritize_english_name = prioritize_english_name
        # self.tags = {"boundary": "administrative", "admin_level": str(admin_level)}

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Regionize a given GeoDataFrame.

        TODO!!! Returns a list of disjointed regions consisting of Thiessen cells generated
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
        # oxs.default_crs = "EPSG:4326"

        gds_wgs84 = gdf.to_crs(epsg=4326)
        raw_gdf_bounds = gds_wgs84.total_bounds
        overpass_bbox = (raw_gdf_bounds[1], raw_gdf_bounds[0], raw_gdf_bounds[3], raw_gdf_bounds[2])
        query = overpassQueryBuilder(
            bbox=overpass_bbox,
            elementType="relation",
            selector=[
                '"boundary"="administrative"',
                '"type"~"boundary|multipolygon"',
                f'"admin_level"="{self.admin_level}"',
            ],
            out="body",
            includeGeometry=True,
        )
        overpass = Overpass()
        boundaries = overpass.query(query, timeout=60)
        regions_dicts = []
        # relations_names = []
        # relations = []
        total = boundaries.countRelations()
        for idx, element in enumerate(boundaries.relations()):
            region_id = None
            if self.prioritize_english_name:
                region_id = element.tag("name:en")
            if not region_id:
                region_id = element.tag("name")
            if not region_id:
                region_id = str(element.id())

            # relations_names.append(name)

            print(region_id, element.id(), idx + 1, total)
            # multipolygon = _parse_relation_to_multipolygon(
            #     element=element.to_json(), geometries=geometries
            # )
            # geometries[unique_id] = multipolygon
            regions_dicts.append(
                {"geometry": self._get_boundary_geometry(element.id()), "region_id": region_id}
            )
            # relations.append(f"R{element.areaId()}")
            # relations.append(f"R{element.id()}")

        # regions_gdf = (
        #     ox.geocode_to_gdf(query=relations, by_osmid=True)
        #     .set_crs(epsg=4326)
        #     .set_index(relations_names)
        # )

        regions_gdf = gpd.GeoDataFrame(data=regions_dicts, crs="EPSG:4326").set_index("region_id")
        # gdf_bounds = box(
        #     minx=raw_gdf_bounds[0],
        #     miny=raw_gdf_bounds[1],
        #     maxx=raw_gdf_bounds[2],
        #     maxy=raw_gdf_bounds[3],
        # )
        # boundaries_gdf = ox.geometries_from_polygon(
        #     polygon=gdf_bounds,
        #     # west=gdf_bounds[0],
        #     # south=gdf_bounds[1],
        #     # east=gdf_bounds[2],
        #     # north=gdf_bounds[3],
        #     tags=self.tags,
        # )  # N S E W
        # clipped_regions_gdf = boundaries_gdf.clip(mask=gds_wgs84, keep_geom_type=False)
        # return clipped_regions_gdf
        clipped_regions_gdf = regions_gdf.clip(mask=gds_wgs84, keep_geom_type=False)
        return clipped_regions_gdf

    def _get_boundary_geometry(self, relation_id: int) -> gpd.GeoDataFrame:
        return ox.geocode_to_gdf(query=[f"R{relation_id}"], by_osmid=True).geometry[0]
