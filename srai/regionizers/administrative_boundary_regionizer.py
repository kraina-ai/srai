"""
Administrative Boundary Regionizer.

This module contains administrative boundary regionizer implementation.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from OSMPythonTools.overpass import Element

from typing import Any, Dict, List, Union

import geopandas as gpd
import topojson as tp
from functional import seq
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid
from tqdm import tqdm

from srai.utils._optional import import_optional_dependencies
from srai.utils.constants import CRS

from .base import BaseRegionizer


class AdministrativeBoundaryRegionizer(BaseRegionizer):
    """
    AdministrativeBoundaryRegionizer.

    Administrative boundary regionizer allows the given geometries to be divided
    into boundaries from OpenStreetMap on a given `admin_level` [1].

    Downloads those boundaries online using `OSMPythonTools` and `osmnx` library.

    Note: offline .pbf loading will be implemented in the future.
    Note: option to download historic data will be implemented in the future.

    References:
        1. https://wiki.openstreetmap.org/wiki/Key:admin_level

    """

    def __init__(
        self,
        admin_level: int,
        clip_regions: bool = True,
        return_empty_region: bool = False,
        prioritize_english_name: bool = True,
        toposimplify: Union[bool, float] = True,
    ) -> None:
        """
        Init AdministrativeBoundaryRegionizer.

        Args:
            admin_level (int): OpenStreetMap admin_level value. See [1] for detailed description of
                available values.
            clip_regions (bool, optional): Whether to to clip regions using a provided mask.
                Turning it off can an be useful when trying to load regions using list a of points.
                Defaults to True.
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
            1. https://wiki.openstreetmap.org/wiki/Tag:boundary=administrative#10_admin_level_values_for_specific_countries
            2. https://taginfo.openstreetmap.org/keys/admin_level#values

        """  # noqa: W505, E501
        import_optional_dependencies(
            dependency_group="osm",
            modules=["osmnx", "OSMPythonTools"],
        )
        from OSMPythonTools.overpass import Overpass

        if admin_level < 1 or admin_level > 11:
            raise ValueError("admin_level outside of available range.")

        self.admin_level = admin_level
        self.prioritize_english_name = prioritize_english_name
        self.clip_regions = clip_regions
        self.return_empty_region = return_empty_region

        if isinstance(toposimplify, float):
            self.toposimplify = toposimplify
        elif toposimplify:
            self.toposimplify = 1e-4
        else:
            self.toposimplify = False

        self.overpass = Overpass()

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
            1. https://wiki.openstreetmap.org/wiki/Overpass_API
            2. https://github.com/mocnik-science/osm-python-tools
            3. https://github.com/gboeing/osmnx
            4. https://github.com/mattijn/topojson

        """
        gdf_wgs84 = gdf.to_crs(crs=CRS)

        regions_dicts = self._generate_regions_from_all_geometries(gdf_wgs84)

        regions_gdf = gpd.GeoDataFrame(data=regions_dicts, crs=CRS).set_index("region_id")
        regions_gdf = self._toposimplify_gdf(regions_gdf)

        if self.clip_regions:
            regions_gdf = regions_gdf.clip(mask=gdf_wgs84, keep_geom_type=False)

        if self.return_empty_region:
            empty_region = self._generate_empty_region(mask=gdf_wgs84, regions_gdf=regions_gdf)
            if not empty_region.is_empty:
                regions_gdf.loc["EMPTY", "geometry"] = empty_region
        return regions_gdf

    def _generate_regions_from_all_geometries(
        self, gdf_wgs84: gpd.GeoDataFrame
    ) -> List[Dict[str, Any]]:
        """Query and optimize downloading data from Overpass."""
        elements_ids = set()
        generated_regions: List[Dict[str, Any]] = []

        all_geometries = (
            seq([self._flatten_geometries(g) for g in gdf_wgs84.geometry]).flatten().list()
        )

        with tqdm(desc="Loading boundaries") as pbar:
            for geometry in all_geometries:
                unary_geometry = unary_union([r["geometry"] for r in generated_regions])
                if not geometry.within(unary_geometry):
                    query = self._generate_query_for_single_geometry(geometry)
                    boundaries = self.overpass.query(query, timeout=60, shallow=False)
                    boundaries_list = list(boundaries.relations()) if boundaries.relations() else []
                    for boundary in boundaries_list:
                        if boundary.id() not in elements_ids:
                            elements_ids.add(boundary.id())
                            generated_regions.append(self._parse_overpass_element(boundary))
                            pbar.update(1)

        return generated_regions

    def _flatten_geometries(self, g: BaseGeometry) -> List[BaseGeometry]:
        """Flatten all geometries into a list of BaseGeometries."""
        if isinstance(g, BaseMultipartGeometry):
            return list(
                seq([self._flatten_geometries(sub_geom) for sub_geom in g.geoms]).flatten().list()
            )
        return [g]

    def _generate_query_for_single_geometry(self, g: BaseGeometry) -> str:
        """Generate Overpass query for a geometry."""
        from OSMPythonTools.overpass import overpassQueryBuilder

        if isinstance(g, Point):
            query = (
                f"is_in({g.y},{g.x});"
                'area._["boundary"="administrative"]'
                '["type"~"boundary|multipolygon"]'
                f'["admin_level"="{self.admin_level}"];'
                "relation(pivot); out ids tags;"
            )
        else:
            raw_gdf_bounds = g.bounds
            overpass_bbox = (
                raw_gdf_bounds[1],
                raw_gdf_bounds[0],
                raw_gdf_bounds[3],
                raw_gdf_bounds[2],
            )
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
        return query

    def _parse_overpass_element(self, element: "Element") -> Dict[str, Any]:
        """Parse single Overpass Element and return a region."""
        region_id = None
        if self.prioritize_english_name:
            region_id = element.tag("name:en")
        if not region_id:
            region_id = element.tag("name")
        if not region_id:
            region_id = str(element.id())

        return {
            "geometry": self._get_boundary_geometry(element.id()),
            "region_id": region_id,
        }

    def _get_boundary_geometry(self, relation_id: int) -> BaseGeometry:
        """Download a geometry of a relation using `osmnx` library."""
        from osmnx import geocode_to_gdf

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
        regions_gdf = topo.to_gdf(winding_order="CW_CCW", crs=CRS, validate=True)
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
