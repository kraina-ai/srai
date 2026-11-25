"""
Administrative Boundary Regionalizer.

This module contains administrative boundary regionalizer implementation.
"""

from collections.abc import Iterable
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely import from_wkb

from srai._optional import import_optional_dependencies
from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.regionalizers._boundary_regionalizer import BoundaryRegionalizer

LAYERCAKE_BOUNDARIES_URL = "https://data.openstreetmap.us/layercake/boundaries.parquet"
# Bounding boxes are saved as 32-bit floats
BBOX_TOLERANCE = 1e-4


class AdministrativeBoundaryRegionalizer(BoundaryRegionalizer):
    """
    AdministrativeBoundaryRegionalizer.

    Administrative boundary regionalizer allows the given geometries to be divided
    into boundaries from OpenStreetMap on a given `admin_level` [1].

    Reads those boundaries from the Layercake [2] GeoParquet file using `duckdb` library.

    Note: offline .pbf loading will be implemented in the future.
    Note: option to download historic data will be implemented in the future.

    References:
        1. https://wiki.openstreetmap.org/wiki/Key:admin_level
        2. https://layercake.openstreetmap.us/layers/boundaries/
    """

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
        Init AdministrativeBoundaryRegionalizer.

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
                than 1% of total self. Takes into consideration if provided query GeoDataFrame
                contains points and then skips calculating area when intersects any point.
                Defaults to True.

        Raises:
            ValueError: If admin_level is outside available range (1-11). See [2] for list of
                values with `in_wiki` selected.

        References:
            1. https://wiki.openstreetmap.org/wiki/Tag:boundary=administrative#10_admin_level_values_for_specific_countries
            2. https://taginfo.openstreetmap.org/keys/admin_level#values
        """  # noqa: W505, E501
        import_optional_dependencies(dependency_group="osm", modules=["duckdb"])

        if admin_level < 1 or admin_level > 11:
            raise ValueError("admin_level outside of available range.")

        super().__init__(
            clip_regions=clip_regions,
            return_empty_region=return_empty_region,
            toposimplify=toposimplify,
            remove_artefact_regions=remove_artefact_regions,
        )
        self.admin_level = admin_level
        self.prioritize_english_name = prioritize_english_name

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Regionalize a given GeoDataFrame.

        Will query Layercake [1] GeoParquet file with OpenStreetMap boundaries using `duckdb` [2]
        library for administrative boundaries on a given admin_level that intersect given
        geometries. Relations and named closed ways are used, without ways covered by a relation.

        Geometries are stored in big chunks, so the download can take a while, especially
        for a big area. Newer `duckdb` versions (2.0+) download less data. Results aren't cached.

        If `prioritize_english_name` is set to `True`, method will try to extract the `name:en` tag
        first before resorting to the `name` tag. If boundary doesn't have a `name` tag, an `id`
        will be used. Names repeated in the result have an OpenStreetMap type and id added:
        `name (relation/id)`.

        Before returning downloaded regions, a topology is built and can be simplified to reduce
        size of geometries while keeping neighbouring regions together without introducing gaps.
        `Topojson` library [3] is used for this operation.

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
            1. https://layercake.openstreetmap.us/layers/boundaries/
            2. https://duckdb.org/
            3. https://github.com/mattijn/topojson
        """
        return self._transform(gdf)

    def _get_level_description(self) -> str:
        return f"`admin_level`={self.admin_level}"

    def _load_regions(self, gdf_wgs84: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Load boundaries intersecting any of the geometries."""
        boundaries = self._query_boundaries(gdf_wgs84.total_bounds)

        boundaries = boundaries.iloc[
            boundaries.sindex.query(
                gdf_wgs84.geometry.union_all(), predicate="intersects", sort=True
            )
        ]

        # Closed ways can be a part of a relation with the same boundary
        is_way = boundaries["type"] == "way"
        ways, relations = boundaries[is_way], boundaries[~is_way]
        duplicated_ways, _ = relations.sindex.query(ways.geometry, predicate="covered_by")
        boundaries = boundaries.drop(index=ways.index[duplicated_ways])

        region_ids = boundaries["name"].fillna(boundaries["id"].astype(str))
        if self.prioritize_english_name:
            region_ids = boundaries["name_en"].fillna(region_ids)

        # Index has to be unique, otherwise it's lost during the topology simplification
        osm_ids = boundaries["type"] + "/" + boundaries["id"].astype(str)
        region_ids = region_ids.mask(
            region_ids.duplicated(keep=False), region_ids + " (" + osm_ids + ")"
        )

        return gpd.GeoDataFrame(
            {GEOMETRY_COLUMN: boundaries.geometry.to_numpy()},
            index=pd.Index(region_ids, name=REGIONS_INDEX, dtype=object),
            geometry=GEOMETRY_COLUMN,
            crs=WGS84_CRS,
        )

    def _query_boundaries(self, bounds: Iterable[float]) -> gpd.GeoDataFrame:
        """Download boundaries with bounding boxes intersecting given bounds."""
        import duckdb

        # ponytail: single bounding box, so distant areas also read everything between them.
        # Query each cluster of areas separately, if it's too slow.
        xmin, ymin, xmax, ymax = map(float, bounds)
        # Unnamed ways are mostly parts of other boundaries
        query = f"""
        SELECT
            type,
            id,
            nullif(array_to_string(name, ';'), '') AS name,
            nullif(array_to_string(map_extract(names, 'en')[1], ';'), '') AS name_en,
            geometry
        FROM read_parquet('{LAYERCAKE_BOUNDARIES_URL}')
        WHERE boundary = 'administrative'
            AND admin_level = '{self.admin_level}'
            AND (type = 'relation' OR len(name) > 0)
            AND bbox.xmin <= {xmax + BBOX_TOLERANCE}
            AND bbox.xmax >= {xmin - BBOX_TOLERANCE}
            AND bbox.ymin <= {ymax + BBOX_TOLERANCE}
            AND bbox.ymax >= {ymin - BBOX_TOLERANCE}
        ORDER BY type, id
        """
        with duckdb.connect() as connection:
            # Keep geometries as WKB
            connection.execute("SET enable_geoparquet_conversion = false")
            rows = connection.execute(query).fetchall()

        boundaries = pd.DataFrame(rows, columns=["type", "id", "name", "name_en", GEOMETRY_COLUMN])
        return gpd.GeoDataFrame(
            boundaries, geometry=from_wkb(boundaries[GEOMETRY_COLUMN]), crs=WGS84_CRS
        )
