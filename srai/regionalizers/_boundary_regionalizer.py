"""
Boundary Regionalizer.

This module contains a base class for regionalizers using downloaded boundaries.
"""

from abc import abstractmethod
from typing import Optional, Union

import geopandas as gpd
import topojson as tp
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.regionalizers import Regionalizer


class BoundaryRegionalizer(Regionalizer):
    """Base class for regionalizers using downloaded boundaries."""

    EMPTY_REGION_NAME = "EMPTY"

    def __init__(
        self,
        clip_regions: bool,
        return_empty_region: bool,
        toposimplify: Union[bool, float],
        remove_artefact_regions: bool,
    ) -> None:
        """Init BoundaryRegionalizer."""
        self.clip_regions = clip_regions
        self.return_empty_region = return_empty_region
        self.remove_artefact_regions = remove_artefact_regions

        if isinstance(toposimplify, (int, float)) and toposimplify > 0:
            self.toposimplify = toposimplify
        elif isinstance(toposimplify, bool) and toposimplify:
            self.toposimplify = 1e-4
        else:
            self.toposimplify = False

    @abstractmethod
    def _load_regions(self, gdf_wgs84: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Load boundaries intersecting any of the geometries."""

    @abstractmethod
    def _get_level_description(self) -> str:
        """Get description of the level used in a warning about lack of boundaries."""

    def _transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Load boundaries, simplify them and clip to the given GeoDataFrame."""
        gdf_wgs84 = gdf.to_crs(crs=WGS84_CRS)

        regions_gdf = self._load_regions(gdf_wgs84)

        if regions_gdf.empty:
            import warnings

            warnings.warn(
                "Couldn't find any administrative boundaries with"
                f" {self._get_level_description()}.",
                RuntimeWarning,
                stacklevel=3,
            )

            return self._get_empty_geodataframe(gdf_wgs84)

        regions_gdf = self._toposimplify_gdf(regions_gdf)

        if self.remove_artefact_regions:
            points_collection: Optional[BaseGeometry] = gdf_wgs84[
                gdf_wgs84.geom_type == "Point"
            ].geometry.union_all()
            clipping_polygon_area: Optional[BaseGeometry] = gdf_wgs84[
                gdf_wgs84.geom_type != "Point"
            ].geometry.union_all()

            regions_to_keep = [
                region_id
                for region_id, row in regions_gdf.iterrows()
                if self._check_intersects_with_points(row["geometry"], points_collection)
                or self._calculate_intersection_area_fraction(
                    row["geometry"], clipping_polygon_area
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
                    self.EMPTY_REGION_NAME,
                    GEOMETRY_COLUMN,
                ] = empty_region

        return regions_gdf

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
        joined_mask = mask.geometry.union_all()
        joined_geometry = regions_gdf.geometry.union_all()
        return joined_mask.difference(joined_geometry)

    def _get_empty_geodataframe(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Get empty GeoDataFrame when zero boundaries have been found."""
        if self.return_empty_region:
            regions_gdf = gpd.GeoDataFrame(
                data={
                    GEOMETRY_COLUMN: [gdf.geometry.union_all()],
                    REGIONS_INDEX: [self.EMPTY_REGION_NAME],
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

    def _check_intersects_with_points(
        self, region_geometry: BaseGeometry, points_collection: Optional[BaseGeometry]
    ) -> bool:
        """Check if region intersects with any point in query regions."""
        return (
            points_collection is not None
            and not points_collection.is_empty
            and region_geometry.intersects(points_collection)
        )

    def _calculate_intersection_area_fraction(
        self,
        region_geometry: BaseGeometry,
        clipping_polygon_area: Optional[BaseGeometry],
    ) -> float:
        """Calculate intersection area fraction to check if it's big enough."""
        if clipping_polygon_area is None or clipping_polygon_area.is_empty:
            return 0
        full_area = float(region_geometry.area)
        clip_area = float(region_geometry.intersection(clipping_polygon_area).area)
        return clip_area / full_area
