"""
Administrative Boundary Regionizer.

This module contains administrative boundary regionizer implementation.
"""
import time
from typing import Any, Dict, List, Union

import geopandas as gpd
import topojson as tp
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid
from tqdm import tqdm

from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.regionizers import Regionizer
from srai.utils import flatten_geometry_series
from srai.utils._optional import import_optional_dependencies


class AdministrativeBoundaryRegionizer(Regionizer):
    """
    AdministrativeBoundaryRegionizer.

    Administrative boundary regionizer allows the given geometries to be divided
    into boundaries from OpenStreetMap on a given `admin_level` [1].

    Downloads those boundaries online using `overpass` and `osmnx` libraries.

    Note: offline .pbf loading will be implemented in the future.
    Note: option to download historic data will be implemented in the future.

    References:
        1. https://wiki.openstreetmap.org/wiki/Key:admin_level
    """

    EMPTY_REGION_NAME = "EMPTY"

    def __init__(
        self,
        admin_level: int,
        clip_regions: bool = True,
        return_empty_region: bool = False,
        prioritize_english_name: bool = True,
        toposimplify: Union[bool, float] = True,
        remove_artefact_regions: bool = True,
    ) -> None:
        """
        Init AdministrativeBoundaryRegionizer.

        Args:
            admin_level (int): OpenStreetMap admin_level value. See [1] for detailed description of
                available values.
            clip_regions (bool, optional): Whether to clip regions using a provided mask.
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
            remove_artefact_regions (bool, optional): Whether to remove small regions barely
                intersecting queried area. Turning it off can sometimes load unnecessary boundaries
                that touch on the edge. It removes regions that intersect with an area smaller
                than 1% of total self. Defaults to True.

        Raises:
            ValueError: If admin_level is outside available range (1-11). See [2] for list of
                values with `in_wiki` selected.

        References:
            1. https://wiki.openstreetmap.org/wiki/Tag:boundary=administrative#10_admin_level_values_for_specific_countries
            2. https://taginfo.openstreetmap.org/keys/admin_level#values
        """  # noqa: W505, E501
        import_optional_dependencies(
            dependency_group="osm",
            modules=["osmnx", "overpass"],
        )
        from overpass import API

        if admin_level < 1 or admin_level > 11:
            raise ValueError("admin_level outside of available range.")

        self.admin_level = admin_level
        self.prioritize_english_name = prioritize_english_name
        self.clip_regions = clip_regions
        self.return_empty_region = return_empty_region
        self.remove_artefact_regions = remove_artefact_regions

        if isinstance(toposimplify, (int, float)) and toposimplify > 0:
            self.toposimplify = toposimplify
        elif isinstance(toposimplify, bool) and toposimplify:
            self.toposimplify = 1e-4
        else:
            self.toposimplify = False

        self.overpass_api = API(timeout=60)

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Regionize a given GeoDataFrame.

        Will query Overpass [1] server using `overpass` [2] library for closed administrative
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
            2. https://github.com/mvexel/overpass-api-python-wrapper
            3. https://github.com/gboeing/osmnx
            4. https://github.com/mattijn/topojson
        """
        gdf_wgs84 = gdf.to_crs(crs=WGS84_CRS)

        regions_dicts = self._generate_regions_from_all_geometries(gdf_wgs84)

        if not regions_dicts:
            import warnings

            warnings.warn(
                (
                    "Couldn't find any administrative boundaries with"
                    f" `admin_level`={self.admin_level}."
                ),
                RuntimeWarning,
            )
            if self.return_empty_region:
                regions_gdf = gpd.GeoDataFrame(
                    data={
                        GEOMETRY_COLUMN: [unary_union(gdf_wgs84.geometry)],
                        REGIONS_INDEX: [AdministrativeBoundaryRegionizer.EMPTY_REGION_NAME],
                    },
                    crs=WGS84_CRS,
                ).set_index(REGIONS_INDEX)
            else:
                regions_gdf = gpd.GeoDataFrame(
                    data={
                        GEOMETRY_COLUMN: [],
                        REGIONS_INDEX: [],
                    },
                    crs=WGS84_CRS,
                )

            return regions_gdf

        regions_gdf = gpd.GeoDataFrame(data=regions_dicts, crs=WGS84_CRS).set_index(REGIONS_INDEX)
        regions_gdf = self._toposimplify_gdf(regions_gdf)

        if self.remove_artefact_regions:
            clipping_area = gdf_wgs84.geometry.unary_union
            regions_to_keep = [
                region_id
                for region_id in regions_gdf.index
                if self._calculate_intersection_area_fraction(
                    regions_gdf.loc[region_id]["geometry"], clipping_area
                )
                > 0.01
            ]
            regions_gdf = regions_gdf.loc[regions_to_keep]

        if self.clip_regions:
            regions_gdf = regions_gdf.clip(mask=gdf_wgs84, keep_geom_type=False)

        if self.return_empty_region:
            empty_region = self._generate_empty_region(mask=gdf_wgs84, regions_gdf=regions_gdf)
            if not empty_region.is_empty:
                regions_gdf.loc[
                    AdministrativeBoundaryRegionizer.EMPTY_REGION_NAME, GEOMETRY_COLUMN
                ] = empty_region
        return regions_gdf

    def _generate_regions_from_all_geometries(
        self, gdf_wgs84: gpd.GeoDataFrame
    ) -> List[Dict[str, Any]]:
        """Query and optimize downloading data from Overpass."""
        elements_ids = set()
        generated_regions: List[Dict[str, Any]] = []

        all_geometries = flatten_geometry_series(gdf_wgs84.geometry)

        with tqdm(desc="Loading boundaries") as pbar:
            for geometry in all_geometries:
                unary_geometry = unary_union([r[GEOMETRY_COLUMN] for r in generated_regions])
                if not geometry.covered_by(unary_geometry):
                    query = self._generate_query_for_single_geometry(geometry)
                    boundaries_list = self._query_overpass(query)
                    for boundary in boundaries_list:
                        if boundary["id"] not in elements_ids:
                            elements_ids.add(boundary["id"])
                            generated_regions.append(self._parse_overpass_element(boundary))
                            pbar.update(1)

        return generated_regions

    def _query_overpass(self, query: str) -> List[Dict[str, Any]]:
        """
        Query Overpass and catch exceptions.

        Since `overpass` library doesn't have useful http error wrapping like `osmnx` does [1],
        this method allows for retry after waiting some time. Additionally, caching mechanism
        uses `osmnx` internal methods built for this purpose.

        Args:
            query (str): Overpass query.

        Raises:
            ex: If exception is different than urllib.request.HTTPError or
                HTTP code is different than 429 or 504.

        Returns:
            List[Dict[str, Any]]: Query elements result from Overpass.

        References:
            1. https://github.com/gboeing/osmnx/blob/main/osmnx/downloader.py#L712
        """
        from osmnx.downloader import _retrieve_from_cache, _save_to_cache
        from overpass import MultipleRequestsError, ServerLoadError

        while True:
            try:
                query_result = _retrieve_from_cache(url=query, check_remark=False)
                if query_result is None:
                    query_result = self.overpass_api.get(
                        query, verbosity="ids tags", responseformat="json"
                    )
                    _save_to_cache(url=query, response_json=query_result, sc=200)
                elements: List[Dict[str, Any]] = query_result["elements"]
                return elements
            except (MultipleRequestsError, ServerLoadError):
                time.sleep(60)

    def _generate_query_for_single_geometry(self, g: BaseGeometry) -> str:
        """Generate Overpass query for a geometry."""
        if isinstance(g, Point):
            query = (
                f"is_in({g.y},{g.x});"
                'area._["boundary"="administrative"]'
                '["type"~"boundary|multipolygon"]'
                f'["admin_level"="{self.admin_level}"];'
                "relation(pivot);"
            )
        else:
            raw_gdf_bounds = g.bounds
            overpass_bbox = (
                raw_gdf_bounds[1],
                raw_gdf_bounds[0],
                raw_gdf_bounds[3],
                raw_gdf_bounds[2],
            )
            query = (
                '(relation["boundary"="administrative"]'
                '["type"~"boundary|multipolygon"]'
                f'["admin_level"="{self.admin_level}"]'
                f"({overpass_bbox[0]},{overpass_bbox[1]},{overpass_bbox[2]},{overpass_bbox[3]}););"
            )
        return query

    def _parse_overpass_element(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Parse single Overpass Element and return a region."""
        element_tags = element.get("tags", {})
        region_id = None
        if self.prioritize_english_name:
            region_id = element_tags.get("name:en")
        if not region_id:
            region_id = element_tags.get("name")
        if not region_id:
            region_id = str(element["id"])

        return {
            GEOMETRY_COLUMN: self._get_boundary_geometry(element["id"]),
            REGIONS_INDEX: region_id,
        }

    def _get_boundary_geometry(self, relation_id: str) -> BaseGeometry:
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
        regions_gdf = topo.to_gdf(winding_order="CW_CCW", crs=WGS84_CRS, validate=True)
        regions_gdf.index.rename(REGIONS_INDEX, inplace=True)
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

    def _calculate_intersection_area_fraction(
        self, region_geometry: BaseGeometry, clipping_area: BaseGeometry
    ) -> float:
        """Calculate intersection area fraction to check if it's big enough."""
        full_area = float(region_geometry.area)
        clip_area = float(region_geometry.intersection(clipping_area).area)
        return clip_area / full_area
