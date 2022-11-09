"""
Administrative Boundary Regionizer.

This module contains administrative boundary regionizer implementation.

"""

from typing import Union

import geopandas as gpd
import topojson as tp
from osmnx import geocode_to_gdf
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid
from tqdm import tqdm


class AdministrativeBoundaryRegionizer:
    """
    AdministrativeBoundaryRegionizer.

    Administrative boundary regionizer allows the given geometries to be divided
    into boundaries from OpenStreetMap on a given `admin_level` [1].

    Downloads those boundaries online using `OSMPythonTools` and `osmnx` library.

    Note: offline .pbf loading will be implemented in the future.
    Note: option to download historic data will be implemented in the future.

    References:
        [1] https://wiki.openstreetmap.org/wiki/Key:admin_level

    """

    def __init__(
        self,
        admin_level: int,
        return_empty_region: bool = False,
        prioritize_english_name: bool = True,
        toposimplify: Union[bool, float] = True,
    ) -> None:
        """
        Init AdministrativeBoundaryRegionizer.

        Args:
            admin_level (int): OpenStreetMap admin_level value. See [1] for detailed description of
                available values.
            return_empty_region (bool, optional): Whether to return an empty region to fill
                remaining space or not. Defaults to False.
            prioritize_english_name (bool, optional): Whether to use english area name if available
                as a region id first. Defaults to True.
            toposimplify (Union[bool, float], optional): Whether to simplify topology to reduce
                geometries size or not. Value is passed to `topojson` library for topology-aware
                simplification. Since provided values are treated like degrees, values between
                1e-4 and 1.0 are recommended. Defaults to True (which results in value equal 1e-4).

        Raises:
            ValueError: If admin_level is outside available range (1-11). See [2] for list of
                values with `in_wiki` selected.

        References:
            [1] https://wiki.openstreetmap.org/wiki/Tag:boundary=administrative#10_admin_level_values_for_specific_countries
            [2] https://taginfo.openstreetmap.org/keys/admin_level#values

        """  # noqa: W505, E501
        if admin_level < 1 or admin_level > 11:
            raise ValueError("admin_level outside of available range.")

        self.admin_level = admin_level
        self.prioritize_english_name = prioritize_english_name
        self.return_empty_region = return_empty_region

        if isinstance(toposimplify, float):
            self.toposimplify = toposimplify
        elif toposimplify:
            self.toposimplify = 1e-4
        else:
            self.toposimplify = False

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Regionize a given GeoDataFrame.

        Will query Overpass [1] server using `OSMPythonTools` [2] library for closed administrative
        boundaries on a given admin_level and then download geometries for each relation using
        `osmnx` [3] library.

        If `prioritize_english_name` is set to `True`, method will try to extract the `name:en` tag
        first before resorting to the `name` tag. If boundary doesn't have a `name` tag, an `id`
        will be used.

        Before returning downloaded regions, a topology is built and can be simplified to reduce
        size of geometries while keeping neighbouring regions together without introducing gaps.
        `Topojson` library [4] is used for this operation.

        Additionally, an empty region with name `EMPTY` can be introduced if returned regions do
        not fully cover a clipping mask.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be regionized.
                Will use this list of geometries to crop resulting regions.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with the regionized data cropped using input
                GeoDataFrame.

        Raises:
            RuntimeError: If simplification can't preserve a topology.

        References:
            [1] https://wiki.openstreetmap.org/wiki/Overpass_API
            [2] https://github.com/mocnik-science/osm-python-tools
            [3] https://github.com/gboeing/osmnx
            [4] https://github.com/mattijn/topojson

        """
        gds_wgs84 = gdf.to_crs(epsg=4326)
        raw_gdf_bounds = (
            gds_wgs84.total_bounds
        )  # TODO: switch to loop over all geometries (simplify to polygons or base geometries)
        overpass_bbox = (raw_gdf_bounds[1], raw_gdf_bounds[0], raw_gdf_bounds[3], raw_gdf_bounds[2])
        query = overpassQueryBuilder(
            bbox=overpass_bbox,
            elementType="relation",
            selector=[
                '"boundary"="administrative"',
                '"type"~"boundary|multipolygon"',
                f'"admin_level"="{self.admin_level}"',
            ],
            out="ids tags",
            includeGeometry=False,
        )
        overpass = Overpass()
        boundaries = overpass.query(query, timeout=60, shallow=False)
        regions_dicts = []
        for element in tqdm(boundaries.relations(), desc="Loading boundaries"):
            region_id = None
            if self.prioritize_english_name:
                region_id = element.tag("name:en")
            if not region_id:
                region_id = element.tag("name")
            if not region_id:
                region_id = str(element.id())

            regions_dicts.append(
                {"geometry": self._get_boundary_geometry(element.id()), "region_id": region_id}
            )

        regions_gdf = gpd.GeoDataFrame(data=regions_dicts, crs="EPSG:4326").set_index("region_id")
        regions_gdf = self._toposimplify_gdf(regions_gdf)

        clipped_regions_gdf = regions_gdf.clip(mask=gds_wgs84, keep_geom_type=False)

        if self.return_empty_region:
            empty_region = self._generate_empty_region(
                mask=gds_wgs84, regions_gdf=clipped_regions_gdf
            )
            if not empty_region.is_empty:
                clipped_regions_gdf.loc["EMPTY", "geometry"] = empty_region
        return clipped_regions_gdf

    def _get_boundary_geometry(self, relation_id: int) -> gpd.GeoDataFrame:
        """Download a geometry of a relation using `osmnx` library."""
        return geocode_to_gdf(query=[f"R{relation_id}"], by_osmid=True).geometry[0]

    def _toposimplify_gdf(self, regions_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Create a topology to ensure proper boundaries between regions and simplify it."""
        topo = tp.Topology(
            regions_gdf,
            prequantize=False,
            presimplify=False,
            toposimplify=self.toposimplify,
            simplify_algorithm="dp",
            prevent_oversimplify=True,
        )
        regions_gdf = topo.to_gdf(winding_order="CW_CCW", crs="EPSG:4326", validate=True)
        regions_gdf.index.rename("region_id", inplace=True)
        regions_gdf.geometry = regions_gdf.geometry.apply(make_valid)
        for idx, r in regions_gdf.iterrows():
            if not isinstance(r.geometry, (Polygon, MultiPolygon)):
                raise RuntimeError(
                    f"Simplification couldn't preserve geometry for region: {idx}."
                    " Try lowering toposimplify value."
                )
        return regions_gdf

    def _generate_empty_region(
        self, mask: gpd.GeoDataFrame, regions_gdf: gpd.GeoDataFrame
    ) -> BaseGeometry:
        """Generate a region filling the space between regions and full clipping mask."""
        joined_mask = unary_union(mask.geometry)
        joined_geometry = unary_union(regions_gdf.geometry)
        return joined_mask.difference(joined_geometry)
