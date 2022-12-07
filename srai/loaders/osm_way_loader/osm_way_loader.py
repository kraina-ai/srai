"""
OSM Way loader.

This module contains osm loader implementation for ways based on OSMnx.

"""
import logging
from enum import Enum
from typing import List, Tuple, Union

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


class NetworkType(str, Enum):
    """TODO."""

    ALL_PRIVATE = "all_private"
    ALL = "all"
    BIKE = "bike"
    DRIVE = "drive"
    DRIVE_SERVICE = "drive_service"
    WALK = "walk"


# TODO: Inherit from BaseLoader
class OSMWayLoader:
    """
    OSMWayLoader.

    OSMWayLoader loader is ...

    """

    def __init__(
        self,
        network_type: Union[NetworkType, str],
        feature_names: List[str] = FEATURE_NAMES,
        return_wide: bool = True,
    ) -> None:
        """TODO."""
        self.network_type = network_type
        self.feature_names = feature_names
        self.return_wide = return_wide

    def load(self, area: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        ...

        Args:
            area (gpd.GeoDataFrame, Path): ...

        Raises:
            ValueError: ...

        Returns:
            gpd.GeoDataFrame: ...

        """
        gdf_wgs84 = area.to_crs(epsg=4326)

        nodes = []
        edges = []
        for polygon in gdf_wgs84["geometry"]:
            G_directed = ox.graph_from_polygon(
                polygon, network_type=self.network_type, retain_all=True, clean_periphery=True
            )

            # FIXME: takes a really long time, which is weird.
            # Maybe try dropping 'reversed' rows instead at later stages
            G = ox.utils_graph.get_undirected(G_directed)
            gdf_n, gdf_e = ox.graph_to_gdfs(G)
            nodes.append(gdf_n)
            edges.append(gdf_e)

        # FIXME: possible duplicates
        gdf_nodes = pd.concat(nodes, axis=0)
        gdf_edges = pd.concat(edges, axis=0)

        cols = constants.OSM_WAY_TAGS.keys()
        gdf_edges_exploded: gpd.GeoDataFrame = gdf_edges
        for col in cols:
            gdf_edges_exploded = gdf_edges_exploded.explode(col)

        gdf_edges_exploded["i"] = range(0, len(gdf_edges_exploded))
        gdf_edges_exploded.set_index("i", append=True, inplace=True)

        # TODO: preprocess data (normalize)
        #

        if not self.return_wide:
            raise NotImplementedError()  # TODO

        gdf_edges_wide = (
            pd.get_dummies(gdf_edges_exploded[cols], prefix_sep="-")
            .droplevel(3)
            .groupby(level=[0, 1, 2])
            .max()
            .astype(np.uint8)
        )
        gdf_edges_wide = gdf_edges_wide.reindex(columns=self.feature_names, fill_value=0).astype(
            np.uint8
        )
        # gdf_edges_wide.astype(pd.SparseDtype(np.uint8, 0)).info()  # TODO: Sparse

        gdf = gpd.GeoDataFrame(
            pd.concat([gdf_edges.drop(columns=cols), gdf_edges_wide], axis=1), crs="epsg:4326"
        )

        return gdf_nodes, gdf

    # def _normalize(self, x: str, column_name: str) -> str:
    #     try:
    #         if x == "None":
    #             return x
    #         elif column_name == "lanes":
    #             x = min(int(x) , 15)
    #         elif column_name == "maxspeed":
    #             x = float(x)
    #             if x <= 5:
    #                 x = 5
    #             elif x <= 7:
    #                 x = 7
    #             elif x <= 10:
    #                 x = 10
    #             elif x <= 15:
    #                 x = 15
    #             else:
    #                 x = min(int(round(x / 10) * 10), 200)
    #         elif column_name == "width":
    #             x = min(round(float(x) * 2) / 2, 30.0)
    #     except Exception as e:
    #         logger.warn(f"{column_name}: {x} - {type(x)} | {e}")
    #         return "None"

    #     return str(x)

    # def _sanitize(self, x: str, column_name: str) -> str:
    #     if x in ["", "none", "None"]:
    #         return "None"

    #     try:
    #         if column_name == "lanes":
    #             x = int(float(x))
    #         elif column_name == "maxspeed":
    #             if x in ("signals", "variable"):
    #                 return "None"

    #             x = IMPLICIT_MAXSPEEDS[x] if x in IMPLICIT_MAXSPEEDS else x
    #             x = x.replace("km/h", "")
    #             if "mph" in x:
    #                 x = float(x.split(" mph")[0])
    #                 x = x * 1.6
    #             x = float(x)
    #         elif column_name == "width":
    #             if x.endswith(" m") or x.endswith("m") or x.endswith("meter"):
    #                 x = x.split("m")[0].strip()
    #             elif "'" in x:
    #                 x = float(x.split("'")[0])
    #                 x = x * 0.0254
    #             elif x.endswith("ft"):
    #                 x = float(x.split(" ft")[0])
    #                 x = x * 0.3048
    #             x = float(x)

    #     except Exception as e:
    #         logger.warn(f"{column_name}: {x} - {type(x)} | {e}")
    #         # raise Exception()
    #         return "None"

    #     return str(x)
