"""
Administrative Boundary Regionalizer.

This module contains administrative boundary regionalizer implementation.
"""

import hashlib
import tempfile
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import STRtree, box, from_wkb
from shapely.geometry.base import BaseGeometry

from srai._optional import import_optional_dependencies
from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.geometry import flatten_geometry_series
from srai.regionalizers._boundary_regionalizer import BoundaryRegionalizer

LAYERCAKE_BOUNDARIES_URL = "https://data.openstreetmap.us/layercake/boundaries.parquet"
# Bounding boxes are saved as 32-bit floats
BBOX_TOLERANCE = 1e-4


def _get_boundaries_version() -> str:
    """Get version of the published file, it's replaced under the same URL."""
    response = requests.head(LAYERCAKE_BOUNDARIES_URL, timeout=60)
    response.raise_for_status()
    return response.headers["ETag"]


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
        library for administrative boundaries on a given admin_level and then download geometries
        of those intersecting given geometries. Results are saved in the `cache` directory.

        Geometries are stored in big chunks, so first download can take a while, especially
        for a big area.

        If `prioritize_english_name` is set to `True`, method will try to extract the `name:en` tag
        first before resorting to the `name` tag. If boundary doesn't have a `name` tag, an `id`
        will be used.

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
        geometries = flatten_geometry_series(gdf_wgs84.geometry)
        version = _get_boundaries_version()
        source_query = (
            f"FROM read_parquet('{LAYERCAKE_BOUNDARIES_URL}') WHERE boundary = 'administrative'"
            f" AND type = 'relation' AND admin_level = '{self.admin_level}'"
        )

        # Geometries make up almost the whole file, so they are read only for matching boundaries
        boundaries = self._query_with_cache(
            "SELECT id, array_to_string(name, ';') AS name,"
            " array_to_string(map_extract(names, 'en')[1], ';') AS name_en,"
            f" bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax {source_query} ORDER BY id",
            version,
        )
        bounding_boxes = box(
            boundaries["xmin"] - BBOX_TOLERANCE,
            boundaries["ymin"] - BBOX_TOLERANCE,
            boundaries["xmax"] + BBOX_TOLERANCE,
            boundaries["ymax"] + BBOX_TOLERANCE,
        )
        boundaries = boundaries.iloc[self._get_intersecting(bounding_boxes, geometries)]
        # Boundaries crossing the antimeridian would extend the bounds to the whole globe
        is_wide = boundaries["xmax"] - boundaries["xmin"] > 180
        boundaries_wkb = [
            self._query_with_cache(
                f"SELECT id, geometry {source_query} AND {self._get_rows_filter(group)}", version
            )
            for group in (boundaries[~is_wide], boundaries[is_wide])
            if not group.empty
        ]
        if not boundaries_wkb:
            return gpd.GeoDataFrame({GEOMETRY_COLUMN: []}, geometry=GEOMETRY_COLUMN, crs=WGS84_CRS)

        boundaries = boundaries.merge(pd.concat(boundaries_wkb), on="id")
        boundaries[GEOMETRY_COLUMN] = from_wkb(boundaries[GEOMETRY_COLUMN])
        boundaries = boundaries.iloc[
            self._get_intersecting(boundaries[GEOMETRY_COLUMN], geometries)
        ]

        region_ids = boundaries["name"].replace("", None).fillna(boundaries["id"].astype(str))
        if self.prioritize_english_name:
            region_ids = boundaries["name_en"].replace("", None).fillna(region_ids)

        return gpd.GeoDataFrame(
            {GEOMETRY_COLUMN: boundaries[GEOMETRY_COLUMN].to_numpy()},
            index=pd.Index(region_ids, name=REGIONS_INDEX, dtype=object),
            geometry=GEOMETRY_COLUMN,
            crs=WGS84_CRS,
        )

    def _get_rows_filter(self, boundaries: pd.DataFrame) -> str:
        """
        Get filter for the selected boundaries.

        DuckDB can skip reading geometries only using the simple filters on the bounds, without them
        a list of ids downloads almost the whole file.
        """
        # ponytail: single range, so distant areas also read everything between them.
        # Split boundaries into clusters, if it's too slow.
        bounds = [
            f"bbox.{c} BETWEEN {boundaries[c].min()!s}::FLOAT AND {boundaries[c].max()!s}::FLOAT"
            for c in ("xmin", "ymin", "xmax", "ymax")
        ]
        return f"{' AND '.join(bounds)} AND id IN ({', '.join(map(str, boundaries['id']))})"

    def _get_intersecting(
        self, boundaries: "np.ndarray", geometries: list[BaseGeometry]
    ) -> "np.ndarray":
        """Get sorted positions of the boundaries intersecting any of the geometries."""
        return np.unique(STRtree(boundaries).query(geometries, predicate="intersects")[1])

    def _query_with_cache(self, query: str, version: str) -> pd.DataFrame:
        """Query boundaries and save the result to not download it again."""
        import duckdb

        query_hash = hashlib.sha256(f"{version}{query}".encode()).hexdigest()
        result_path = Path("cache").resolve() / f"{query_hash}.parquet"

        if not result_path.exists():
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                tempfile.TemporaryDirectory(dir=result_path.parent) as tmp_dir,
                duckdb.connect() as connection,
            ):
                tmp_path = Path(tmp_dir) / result_path.name
                # Keep geometries as WKB
                connection.execute("SET enable_geoparquet_conversion = false")
                connection.execute(f"COPY ({query}) TO '{tmp_path}' (FORMAT parquet)")
                tmp_path.replace(result_path)

        return pd.read_parquet(result_path)
