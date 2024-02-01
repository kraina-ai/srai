"""
OSM Way loader.

This module contains osm loader implementation for ways based on OSMnx.
"""

import logging
from enum import Enum
from typing import Any, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as shpg
from functional import seq
from tqdm.auto import tqdm

from srai._optional import import_optional_dependencies
from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, WGS84_CRS
from srai.exceptions import LoadedDataIsEmptyException
from srai.loaders import Loader

from . import constants

logger = logging.getLogger(__name__)


class OSMNetworkType(str, Enum):
    """
    Type of the street network.

    See [1] for more details.

    References:
        1. https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.graph.graph_from_place
    """

    ALL_PRIVATE = "all_private"
    ALL = "all"
    BIKE = "bike"
    DRIVE = "drive"
    DRIVE_SERVICE = "drive_service"
    WALK = "walk"


class OSMWayLoader(Loader):
    """
    OSMWayLoader downloads road infrastructure from OSM.

    OSMWayLoader loader is a wrapper for the `osmnx.graph_from_polygon()`
    and `osmnx.graph_to_gdfs()` that simplifies obtaining the road infrastructure data
    from OpenStreetMap. As the OSM data is often noisy, it can also take an opinionated approach
    to preprocessing it, with standardisation in mind - e.g. unification of units,
    discarding non-wiki values and rounding them.
    """

    def __init__(
        self,
        network_type: Union[OSMNetworkType, str],
        contain_within_area: bool = False,
        preprocess: bool = True,
        wide: bool = True,
        metadata: bool = False,
        osm_way_tags: dict[str, list[str]] = constants.OSM_WAY_TAGS,
    ) -> None:
        """
        Init OSMWayLoader.

        Args:
            network_type (Union[NetworkType, str]):
                Type of the network to download.
            contain_within_area (bool): defaults to False
                Whether to remove the roads that have one of their nodes outside of the given area.
            preprocess (bool): defaults to True
                Whether to preprocess the data.
            wide (bool): defaults to True
                Whether to return the roads in wide format.
            metadata (bool): defaults to False
                Whether to return metadata for roads.
            osm_way_tags (List[str]): defaults to constants.OSM_WAY_TAGS
                Dict of tags to take into consideration during computing.
        """
        import_optional_dependencies(dependency_group="osm", modules=["osmnx"])

        self.network_type = network_type
        self.contain_within_area = contain_within_area
        self.preprocess = preprocess
        self.wide = wide
        self.metadata = metadata
        self.osm_keys = list(osm_way_tags.keys())
        self.osm_tags_flat = (
            seq(osm_way_tags.items())
            .flat_map(lambda x: [f"{x[0]}-{v}" if x[0] not in ("oneway") else x[0] for v in x[1]])
            .distinct()
            .to_list()
        )

    def load(self, area: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Load road infrastructure for a given GeoDataFrame.

        Args:
            area (gpd.GeoDataFrame): (Multi)Polygons for which to download road infrastructure data.

        Raises:
            ValueError: If provided GeoDataFrame has no crs defined.
            ValueError: If provided GeoDataFrame is empty.
            TypeError: If provided geometries are not of type Polygon or MultiPolygon.
            LoadedDataIsEmptyException: If none of the supplied area polygons contains
                any road infrastructure data.

        Returns:
            Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Road infrastructure as (intersections, roads)
        """
        import osmnx as ox

        ox.settings.useful_tags_way = constants.OSMNX_WAY_KEYS
        ox.settings.timeout = constants.OSMNX_TIMEOUT

        if area.empty:
            raise ValueError("Provided `area` GeoDataFrame is empty.")

        gdf_wgs84 = area.to_crs(crs=WGS84_CRS)

        gdf_nodes_raw, gdf_edges_raw = self._graph_from_gdf(gdf_wgs84)
        if gdf_edges_raw.empty or gdf_edges_raw.empty:
            raise LoadedDataIsEmptyException(
                "It can happen when there is no road infrastructure in the given area."
            )

        gdf_edges = self._explode_cols(gdf_edges_raw)

        if self.preprocess:
            gdf_edges = self._preprocess(gdf_edges)

        if self.wide:
            gdf_edges = self._to_wide(gdf_edges_raw, gdf_edges)

        gdf_edges = self._unify_index_and_columns_names(gdf_edges)

        return gdf_nodes_raw, gdf_edges

    def _graph_from_gdf(self, gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Obtain the raw road infrastructure data from OSM.

        Args:
            gdf (gpd.GeoDataFrame): (Multi)Polygons for which to download road infrastructure data.

        Returns:
            Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Road infrastructure as (intersections, roads)
        """
        nodes = []
        edges = []
        for polygon in tqdm(gdf["geometry"], desc="Downloading graphs", leave=False):
            gdf_n, gdf_e = self._try_graph_from_polygon(polygon)

            if not gdf_e.empty and not self.contain_within_area:
                # perform cleaning of edges outside of an area that were incorrectly added,
                # it occures when two nodes outside of an area happen to be connected by an edge
                gdf_e = gdf_e.sjoin(
                    gpd.GeoDataFrame(geometry=[polygon], crs=WGS84_CRS),
                    how="inner",
                    predicate="intersects",
                ).drop(columns="index_right")

            nodes.append(gdf_n)
            edges.append(gdf_e)

        gdf_nodes = pd.concat(nodes, axis=0)
        gdf_edges = pd.concat(edges, axis=0)

        # remove duplicates, cannot use drop_duplicates()
        # because some columns contain unhashable type `list`
        gdf_nodes = gdf_nodes[~gdf_nodes.astype(str).duplicated()]
        gdf_edges = gdf_edges[~gdf_edges.astype(str).duplicated()]

        return gdf_nodes, gdf_edges

    def _try_graph_from_polygon(
        self, polygon: Union[shpg.Polygon, shpg.MultiPolygon]
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Try obtaining the raw road infrastructure data from OSM for a single polygon using `osmnx`.

        If `osmnx` fails, then just return the empty result.

        Args:
            polygon (Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon]):
                Polygon for which to download road infrastructure data.

        Returns:
            Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Road infrastructure as (intersections, roads)
        """
        import osmnx as ox
        from packaging import version

        osmnx_new_api = version.parse(ox.__version__) >= version.parse("1.6.0")

        response_error = (
            ox._errors.InsufficientResponseError
            if osmnx_new_api
            else ox._errors.EmptyOverpassResponse
        )

        try:
            return self._graph_from_polygon(polygon)
        except (response_error, ValueError):
            return gpd.GeoDataFrame(), gpd.GeoDataFrame()

    def _graph_from_polygon(
        self, polygon: Union[shpg.Polygon, shpg.MultiPolygon]
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Obtain the raw road infrastructure data from OSM for a single polygon.

        Args:
            polygon (Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon]):
                Polygon for which to download road infrastructure data.

        Notes:
            * The road infrastructure graph is treated as an undirected graph.

        Returns:
            Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Road infrastructure as (intersections, roads)
        """
        import osmnx as ox

        G_directed = ox.graph_from_polygon(
            polygon,
            network_type=self.network_type,
            retain_all=True,
            clean_periphery=True,
            truncate_by_edge=(not self.contain_within_area),
        )

        G_undirected = ox.utils_graph.get_undirected(G_directed)
        gdf_n, gdf_e = ox.graph_to_gdfs(G_undirected)

        return gdf_n, gdf_e

    def _explode_cols(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Explode lists in feature columns.

        Args:
            gdf (gpd.GeoDataFrame): Edges with columns to explode.

        Returns:
            gpd.GeoDataFrame: Edges with all of their columns exploded.
        """
        for col in self.osm_keys:
            if col not in gdf.columns:
                gdf = gdf.assign(**{col: None})
            gdf = gdf.explode(col)

        gdf["i"] = range(len(gdf))
        gdf.set_index("i", append=True, inplace=True)

        return gdf

    def _preprocess(
        self, gdf: gpd.GeoDataFrame, inplace: bool = False
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Preprocess edges.

        Args:
            gdf (gpd.GeoDataFrame): Edges to preprocess.
            inplace (bool): defaults to False.

        Returns:
            gpd.GeoDataFrame: Edges with preprocessed features.
        """
        if not inplace:
            gdf = gdf.copy()

        max_osm_keys_str_len = max(map(len, self.osm_keys))
        for col in (pbar := tqdm(self.osm_keys, leave=False)):
            pbar.set_description(f"Preprocessing {col:{max_osm_keys_str_len}}")
            gdf[col] = gdf[col].apply(
                lambda x, c=col: self._sanitize_and_normalize(x, c)  # noqa: FURB111
            )

        return gdf if not inplace else None

    def _sanitize_and_normalize(self, x: Any, column_name: str) -> str:
        return self._normalize(self._sanitize(str(x), column_name), column_name)

    def _normalize(self, x: Any, column_name: str) -> str:
        try:
            if x is None:
                return "None"
            elif column_name == "lanes":
                x = min(x, 15)
            elif column_name == "maxspeed":
                if x <= 0:
                    x = 0
                elif x <= 5:
                    x = 5
                elif x <= 7:
                    x = 7
                elif x <= 10:
                    x = 10
                elif x <= 15:
                    x = 15
                else:
                    x = min(int(round(x / 10) * 10), 200)
            elif column_name == "width":
                x = min(round(x * 2) / 2, 30.0)
        except Exception as e:
            logger.warning(
                f"{OSMWayLoader._normalize.__qualname__} | {column_name}: {x} - {type(x)} | {e}."
                " Returning 'None'"
            )
            return "None"

        return str(x)

    def _sanitize(self, x: Any, column_name: str) -> Any:
        if x in ("", "none", "None", np.nan, "nan", "NaN", None):
            return None

        try:
            if column_name == "lanes":
                x = int(float(x))
            elif column_name == "maxspeed":
                if x in ("signals", "variable"):
                    return None

                if x in constants.OSM_IMPLICIT_MAXSPEEDS:
                    x = constants.OSM_IMPLICIT_MAXSPEEDS[x]

                x = x.replace("km/h", "")
                if "mph" in x:
                    x = float(x.split("mph")[0].strip())
                    x = x * constants.MPH_TO_KMH
                x = float(x)
            elif column_name == "width":
                if x.endswith(("m", "meter")):
                    x = x.split("m")[0].strip()
                elif "'" in x:
                    x = float(x.split("'")[0].strip())
                    x = x * constants.INCHES_TO_METERS
                elif x.endswith("ft"):
                    x = float(x.split("ft")[0].strip())
                    x = x * constants.FEET_TO_METERS
                x = float(x)

        except Exception as e:
            logger.warning(
                f"{OSMWayLoader._sanitize.__qualname__} | {column_name}: {x} - {type(x)} | {e}."
                " Returning None"
            )
            return None

        return x

    def _to_wide(self, gdf: gpd.GeoDataFrame, gdf_exploded: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Convert edges in long format to wide.

        Args:
            gdf (gpd.GeoDataFrame): original edges.
            gdf_exploded (gpd.GeoDataFrame): edges with columns after feature explosion.

        Returns:
            gpd.GeoDataFrame: Edges in wide format.
        """
        gdf_edges_wide = (
            pd.get_dummies(gdf_exploded[self.osm_keys], prefix_sep="-")
            .droplevel(3)
            .groupby(level=[0, 1, 2])
            .max()
            .astype(np.uint8)
            .reindex(columns=self.osm_tags_flat, fill_value=0)
            .astype(np.uint8)
        )

        osm_keys_to_drop = [k for k in self.osm_keys if k in gdf.columns]
        gdf_edges_wide = gpd.GeoDataFrame(
            pd.concat(
                [
                    gdf.drop(columns=osm_keys_to_drop),
                    gdf_edges_wide,
                ],
                axis=1,
            ),
            crs=WGS84_CRS,
        )

        return gdf_edges_wide

    def _unify_index_and_columns_names(self, gdf_edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Make naming of index and columns consistent.

        Args:
            gdf_edges (gpd.GeoDataFrame): Edges to unify

        Returns:
            gpd.GeoDataFrame: Edges with unified index and columns names
        """
        gdf = gdf_edges.reset_index().drop(columns=["u", "v"])
        gdf.index.rename(FEATURES_INDEX, inplace=True)

        reindex_columns = constants.METADATA_COLUMNS if self.metadata else []
        reindex_columns += self.osm_tags_flat if self.wide else self.osm_keys
        reindex_columns += [GEOMETRY_COLUMN]
        gdf = gdf.reindex(columns=reindex_columns)

        return gdf
