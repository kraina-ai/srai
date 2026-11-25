"""
Overture Divisions Regionalizer.

This module contains Overture Maps divisions regionalizer implementation.
"""

from pathlib import Path
from typing import Literal, Optional, Union, get_args

import geopandas as gpd

from srai._optional import import_optional_dependencies
from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.regionalizers._boundary_regionalizer import BoundaryRegionalizer

DIVISION_SUBTYPE = Literal[
    "country",
    "dependency",
    "region",
    "county",
    "localadmin",
    "locality",
    "macrohood",
    "neighborhood",
    "microhood",
]


class OvertureDivisionsRegionalizer(BoundaryRegionalizer):
    """
    OvertureDivisionsRegionalizer.

    Overture divisions regionalizer allows the given geometries to be divided into divisions [1]
    from Overture Maps on a given `subtype` level.

    Downloads land polygons using `OvertureMaestro` [2] library.

    Only divisions of a given subtype are used and the result is empty where they don't exist. Not
    every country has all of the subtypes (there are no counties in Brazil or Ireland) and the same
    subtype can mean a different thing in different countries. Localities can be nested, so the
    regions can overlap. Polygons can be missing for levels below the county.

    References:
        1. https://docs.overturemaps.org/guides/divisions/
        2. https://github.com/kraina-ai/overturemaestro
    """

    def __init__(
        self,
        subtype: DIVISION_SUBTYPE,
        clip_regions: bool = True,
        return_empty_region: bool = False,
        toposimplify: Union[bool, float] = True,
        remove_artefact_regions: bool = True,
        release: Optional[str] = None,
        download_directory: Union[str, Path] = "files",
        verbosity_mode: Literal["silent", "transient", "verbose"] = "transient",
    ) -> None:
        """
        Init OvertureDivisionsRegionalizer.

        Args:
            subtype (DIVISION_SUBTYPE): Subtype of divisions. One of: country, dependency, region,
                county, localadmin, locality, macrohood, neighborhood, microhood.
            clip_regions (bool, optional): Whether to clip regions using a provided mask.
                Turning it off can an be useful when trying to load regions using list a of points.
                Defaults to True.
            return_empty_region (bool, optional): Whether to return an empty region to fill
                remaining space or not. Defaults to False.
            toposimplify (Union[bool, float], optional): Whether to simplify topology to reduce
                geometries size or not. Value is passed to `topojson` library for topology-aware
                simplification. Since provided values are treated like degrees, values between
                1e-4 and 1.0 are recommended. Defaults to True (which results in value equal 1e-4).
            remove_artefact_regions (bool, optional): Whether to remove small regions barely
                intersecting queried area. Turning it off can sometimes load unnecessary boundaries
                that touch on the edge. It removes regions that intersect with an area smaller
                than 1% of total self. Takes into consideration if provided query GeoDataFrame
                contains points and then skips calculating area when intersects any point.
                Defaults to True.
            release (Optional[str], optional): Release version. If not provided, will automatically
                load newest available release version. Defaults to None.
            download_directory (Union[str, Path], optional): Directory where to save the downloaded
                GeoParquet files. Defaults to "files".
            verbosity_mode (Literal["silent", "transient", "verbose"], optional): Set progress
                verbosity mode. Can be one of: silent, transient and verbose. Silent disables
                output completely. Transient tracks progress, but removes output after finished.
                Verbose leaves all progress outputs in the stdout. Defaults to "transient".

        Raises:
            ValueError: If subtype is unknown.
        """
        import_optional_dependencies(dependency_group="overturemaps", modules=["overturemaestro"])

        if subtype not in get_args(DIVISION_SUBTYPE):
            raise ValueError(
                f"Unknown division subtype: {subtype}."
                f" Available subtypes: {', '.join(get_args(DIVISION_SUBTYPE))}."
            )

        super().__init__(
            clip_regions=clip_regions,
            return_empty_region=return_empty_region,
            toposimplify=toposimplify,
            remove_artefact_regions=remove_artefact_regions,
        )
        self.subtype = subtype
        self.release = release
        self.download_directory = download_directory
        self.verbosity_mode = verbosity_mode

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Regionalize a given GeoDataFrame.

        Will download divisions of a given subtype intersecting given geometries. Divisions are
        indexed by their id, because names aren't unique. Primary name is available in the `name`
        column.

        Before returning downloaded regions, a topology is built and can be simplified to reduce
        size of geometries while keeping neighbouring regions together without introducing gaps.
        `Topojson` library [1] is used for this operation.

        Additionally, an empty region with name `EMPTY` can be introduced if returned regions do
        not fully cover a clipping mask.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be regionalized.
                Will use this list of geometries to crop resulting regions.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with the regionalized data cropped using input
                GeoDataFrame.

        Raises:
            RuntimeError: If simplification can't preserve a topology.

        References:
            1. https://github.com/mattijn/topojson
        """
        return self._transform(gdf)

    def _get_level_description(self) -> str:
        return f"`subtype`={self.subtype}"

    def _load_regions(self, gdf_wgs84: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        from overturemaestro import convert_geometry_to_geodataframe

        geometry_filter = gdf_wgs84.geometry.union_all()
        divisions = convert_geometry_to_geodataframe(
            theme="divisions",
            type="division_area",
            geometry_filter=geometry_filter,
            release=self.release,
            pyarrow_filter=[("subtype", "=", self.subtype), ("class", "=", "land")],
            columns_to_download=["id", "division_id", "names", "geometry"],
            working_directory=self.download_directory,
            verbosity_mode=self.verbosity_mode,
        )
        # OvertureMaestro compares only bounding boxes for rectangles and points
        divisions = divisions[divisions.intersects(geometry_filter)]

        return gpd.GeoDataFrame(
            {
                "name": [names["primary"] for names in divisions["names"]],
                GEOMETRY_COLUMN: divisions.geometry.to_numpy(),
            },
            index=divisions["division_id"].rename(REGIONS_INDEX),
            geometry=GEOMETRY_COLUMN,
            crs=WGS84_CRS,
        )
