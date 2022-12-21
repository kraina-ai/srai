"""
OSM Way loader.

This module contains osm loader implementation for ways based on OSMnx.
"""
import logging
from enum import Enum
from typing import Any, List, Tuple, Union

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from functional import seq

from . import constants

ox.settings.useful_tags_way = constants.OSMNX_WAY_KEYS
ox.settings.timeout = constants.OSMNX_TIMEOUT

logger = logging.getLogger(__name__)

FEATURE_NAMES = (
    seq(constants.OSM_WAY_TAGS.items())
    .flat_map(lambda x: [f"{x[0]}-{v}" if x[0] not in ("oneway") else x[0] for v in x[1]])
    .distinct()
    .to_list()
)

COLS = list(constants.OSM_WAY_TAGS.keys())


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
    OSMWayLoader.

    OSMWayLoader loader is ...
    """

    def __init__(
        self,
        network_type: Union[NetworkType, str],
        feature_names: List[str] = FEATURE_NAMES,
        cols: List[str] = COLS,
    ) -> None:
        """TODO."""
        self.network_type = network_type
        self.feature_names = feature_names
        self.cols = cols

    def load(self, area: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        ...

        Args:
            gdf (gpd.GeoDataFrame): ...

        Raises:
            ValueError: ...

        Returns:
            gpd.GeoDataFrame: ...
        """
        gdf_wgs84 = area.to_crs(epsg=4326)

        gdf_nodes, gdf_edges = self._gdfs_from_polygons(gdf_wgs84)
        gdf_edges_exploded = self._explode_cols(gdf_edges)
        gdf_edges_preprocessed = self._preprocess(gdf_edges_exploded)
        gdf_edges_wide = self._to_wide(gdf_edges, gdf_edges_preprocessed)

        return gdf_nodes, gdf_edges_wide

    def _gdfs_from_polygons(
        self, gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        nodes = []
        edges = []
        for polygon in gdf["geometry"]:
            G_directed = ox.graph_from_polygon(
                polygon, network_type=self.network_type, retain_all=True, clean_periphery=True
            )

            G_undirected = ox.utils_graph.get_undirected(G_directed)
            gdf_n, gdf_e = ox.graph_to_gdfs(G_undirected)
            nodes.append(gdf_n)
            edges.append(gdf_e)

        # FIXME: possible duplicates
        gdf_nodes = pd.concat(nodes, axis=0)
        gdf_edges = pd.concat(edges, axis=0)

        return gdf_nodes, gdf_edges

    def _explode_cols(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        for col in self.cols:
            gdf = gdf.explode(col)

        gdf["i"] = range(0, len(gdf))
        gdf.set_index("i", append=True, inplace=True)

        return gdf

    def _to_wide(self, gdf: gpd.GeoDataFrame, gdf_exploded: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf_edges_wide = (
            pd.get_dummies(gdf_exploded[self.cols], prefix_sep="-")
            .droplevel(3)
            .groupby(level=[0, 1, 2])
            .max()
            .astype(np.uint8)
            .reindex(columns=self.feature_names, fill_value=0)
            .astype(np.uint8)
        )

        gdf_edges_wide = gpd.GeoDataFrame(
            pd.concat(
                [
                    gdf.drop(columns=self.cols),
                    gdf_edges_wide,
                ],
                axis=1,
            ),
            crs="epsg:4326",
        )

        return gdf_edges_wide

    def _preprocess(self, gdf: gpd.GeoDataFrame, inplace: bool = False) -> gpd.GeoDataFrame:
        if not inplace:
            gdf = gdf.copy()

        for col in self.cols:
            gdf[col] = gdf[col].apply(lambda x, c=col: self._normalize(self._sanitize(x, c), c))

        return gdf

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
            logger.warn(
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
                    x = x * 1.6
                x = float(x)
            elif column_name == "width":
                if x.endswith(" m") or x.endswith("m") or x.endswith("meter"):
                    x = x.split("m")[0].strip()
                elif "'" in x:
                    x = float(x.split("'")[0])
                    x = x * 0.0254
                elif x.endswith("ft"):
                    x = float(x.split(" ft")[0])
                    x = x * 0.3048
                x = float(x)

        except Exception as e:
            logger.warn(
                f"{OSMWayLoader._sanitize.__qualname__} | {column_name}: {x} - {type(x)} | {e}"
            )
            return "None"

        return x
