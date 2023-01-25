"""
OSM Way loader.

This module contains osm loader implementation for ways based on OSMnx.
"""
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from functional import seq
from tqdm.auto import tqdm

import srai.utils.constants as srai_constants
from srai.utils._optional import import_optional_dependencies

from . import constants

logger = logging.getLogger(__name__)


class NetworkType(str, Enum):
    """
    Type of the street network.

    See [1] for more details.

    References:
        [1] https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.graph.graph_from_address
    """

    ALL_PRIVATE = "all_private"
    ALL = "all"
    BIKE = "bike"
    DRIVE = "drive"
    DRIVE_SERVICE = "drive_service"
    WALK = "walk"


class OSMWayLoader:
    """
    OSMWayLoader downloads road infrastructure from OSM.

    OSMWayLoader loader is a wrapper for the `osmnx.graph_from_polygon()`
    and `osmnx.graph_to_gdfs()` that simplifies obtaining the road infrastructure data
    from OpenStreetMap. As the OSM data is often noisy, it can also take an opinionated approach
    on preprocessing it having standarisation in mind - e.g. converting to the same units,
    disgarding non-wiki values and rounding them.
    """

    def __init__(
        self,
        network_type: Union[NetworkType, str],
        preprocess: bool = True,
        wide: bool = True,
        osm_way_tags: Dict[str, List[str]] = constants.OSM_WAY_TAGS,
    ) -> None:
        """
        Init OSMWayLoader.

        Args:
            network_type (Union[NetworkType, str]):
                Type of the network to download.
            preprocess (bool): defaults to True
                Whether to preprocess the data.
            wide (bool): defaults to True
                Whether to return the edges in wide format.
            osm_way_tags (List[str]): defaults to constants.OSM_WAY_TAGS
                Dict of tags to take into consideration during computing.
        """
        import_optional_dependencies(dependency_group="osm", modules=["osmnx"])

        self.network_type = network_type
        self.preprocess = preprocess
        self.wide = wide
        self.osm_tags_flat = (
            seq(osm_way_tags.items())
            .flat_map(lambda x: [f"{x[0]}-{v}" if x[0] not in ("oneway") else x[0] for v in x[1]])
            .distinct()
            .to_list()
        )
        self.osm_keys = list(osm_way_tags.keys())

    def load(self, area: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Load road infrastructure for a given GeoDataFrame.

        Args:
            area (gpd.GeoDataFrame): (Multi)Polygons for which to download road infrastructure data.

        Raises:
            ValueError: If provided GeoDataFrame has no crs defined.

        Returns:
            Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Road infrastructure as (nodes, edges).
        """
        import osmnx as ox

        ox.settings.useful_tags_way = constants.OSMNX_WAY_KEYS
        ox.settings.timeout = constants.OSMNX_TIMEOUT

        gdf_wgs84 = area.to_crs(crs=srai_constants.WGS84_CRS)

        gdf_nodes_raw, gdf_edges_raw = self._gdfs_from_polygons(gdf_wgs84)
        gdf_edges = self._explode_cols(gdf_edges_raw)

        if self.preprocess:
            self._preprocess(gdf_edges, inplace=True)

        if self.wide:
            gdf_edges = self._to_wide(gdf_edges_raw, gdf_edges)

        return gdf_nodes_raw, gdf_edges

    def _gdfs_from_polygons(
        self, gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Obtain the raw road infrastructure data from OSM.

        Args:
            gdf (gpd.GeoDataFrame): (Multi)Polygons for which to download road infrastructure data.

        Notes:
            * The road infrastructure graph is treated as an undirected graph.
            * `FIXME` The result may contain duplicated nodes or edges,
            when nearby polygons are queried.

        Returns:
            Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Road infrastructure as (nodes, edges).
        """
        import osmnx as ox

        nodes = []
        edges = []
        for polygon in tqdm(gdf["geometry"], desc="Downloading graphs", leave=False):
            G_directed = ox.graph_from_polygon(
                polygon, network_type=self.network_type, retain_all=True, clean_periphery=True
            )

            G_undirected = ox.utils_graph.get_undirected(G_directed)
            gdf_n, gdf_e = ox.graph_to_gdfs(G_undirected)
            nodes.append(gdf_n)
            edges.append(gdf_e)

        gdf_nodes = pd.concat(nodes, axis=0)
        gdf_edges = pd.concat(edges, axis=0)

        return gdf_nodes, gdf_edges

    def _explode_cols(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Explode lists in feature columns.

        Args:
            gdf (gpd.GeoDataFrame): Edges with columns to explode.

        Returns:
            gpd.GeoDataFrame: Edges with all of their columns exploded.
        """
        for col in self.osm_keys:
            gdf = gdf.explode(col)

        gdf["i"] = range(0, len(gdf))
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
            gdf[col] = gdf[col].apply(lambda x, c=col: self._normalize(self._sanitize(x, c), c))

        return gdf if not inplace else None

    def _normalize(self, x: Any, column_name: str) -> Any:
        try:
            if x == "None":
                return x
            elif column_name == "lanes":
                x = min(x, 15)
            elif column_name == "maxspeed":
                if x <= 5:
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
                f"{OSMWayLoader._normalize.__qualname__} | {column_name}: {x} - {type(x)} | {e}"
            )
            return "None"

        return str(x)

    def _sanitize(self, x: Any, column_name: str) -> Any:
        if x in ["", "none", "None", np.nan, "nan", "NaN"]:
            return "None"

        try:
            if column_name == "lanes":
                x = int(float(x))
            elif column_name == "maxspeed":
                if x in ("signals", "variable"):
                    return "None"

                if x in constants.OSM_IMPLICIT_MAXSPEEDS:
                    x = constants.OSM_IMPLICIT_MAXSPEEDS[x]

                x = x.replace("km/h", "")
                if "mph" in x:
                    x = float(x.split(" mph")[0])
                    x = x * constants.MPH_TO_KMH
                x = float(x)
            elif column_name == "width":
                if x.endswith(" m") or x.endswith("m") or x.endswith("meter"):
                    x = x.split("m")[0].strip()
                elif "'" in x:
                    x = float(x.split("'")[0])
                    x = x * constants.INCHES_TO_METERS
                elif x.endswith("ft"):
                    x = float(x.split(" ft")[0])
                    x = x * constants.FEET_TO_METERS
                x = float(x)

        except Exception as e:
            logger.warn(
                f"{OSMWayLoader._sanitize.__qualname__} | {column_name}: {x} - {type(x)} | {e}"
            )
            return "None"

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

        gdf_edges_wide = gpd.GeoDataFrame(
            pd.concat(
                [
                    gdf.drop(columns=self.osm_keys),
                    gdf_edges_wide,
                ],
                axis=1,
            ),
            crs=srai_constants.WGS84_CRS,
        )

        return gdf_edges_wide
