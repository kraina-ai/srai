"""
S2 regionizer.

This module exposes Google's S2 Geospatial Indexing System [1] as a regionizer.
It uses the Python API [2].

References:
    [1] https://s2geometry.io/
    [2] https://github.com/JoaoCarabetta/s2-py

"""

import json
from typing import Any, Dict

import geopandas as gpd
from functional import seq
from s2 import s2
from shapely.geometry import Polygon

from . import BaseRegionizer


class S2Regionizer(BaseRegionizer):
    """
    S2 Regionizer.

    S2 Regionizer gives an opportunity to divide the given geometries into square S2 cells.

    """

    def __init__(self, resolution: int, buffer: bool = True) -> None:
        """
        Init S2 Regionizer.

        Args:
            resolution (int): Resolution of the cells. See [1] for a full comparison.
            buffer (bool, optional): Whether to fully cover geometries with S2 cells
            or use just those cells that fully fit into them. Defaults to True.

        Raises:
            ValueError: If resolution is not between 0 and 30.

        References:
            [1] https://s2geometry.io/resources/s2cell_statistics.html

        """
        if not (0 <= resolution <= 30):
            raise ValueError(f"Resolution {resolution} is not between 0 and 30.")

        self.resolution = resolution
        self.buffer = buffer

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Regionize a given GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be regionized.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with regionized geometries.

        """
        gdf_wgs84 = gdf.to_crs(epsg=4326)

        # TODO: this part is in common with H3Regionizer
        # transform multipolygons to multiple polygons
        gdf_exploded = gdf_wgs84.explode(index_parts=True).reset_index(drop=True)
        geo_json = json.loads(gdf_exploded.to_json())

        cells = (
            seq(geo_json["features"])
            .map(lambda f: self._geojson_to_cells(f["geometry"], self.resolution))
            .reduce(lambda acc, curr: {**acc, **curr})
        )

        s2_gdf = gpd.GeoDataFrame(
            None,
            index=cells.keys(),
            geometry=list(cells.values()),
            crs="EPSG:4326",
        )

        res = gpd.sjoin(
            s2_gdf,
            gdf_wgs84,
            how="inner",
            predicate="intersects" if self.buffer else "within",
        )

        return res

    def _geojson_to_cells(self, geo_json: Dict[str, Any], res: int) -> Dict[str, Polygon]:
        raw_cells = s2.polyfill(geo_json, res, with_id=True, geo_json_conformant=True)
        cells: Dict[str, Polygon] = (
            seq(raw_cells).map(lambda c: (c["id"], Polygon(c["geometry"]))).to_dict()
        )

        return cells

    def _remove_cells_outside(
        self, s2_gdf: gpd.GeoDataFrame, gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        return gpd.sjoin(s2_gdf, gdf, how="inner", predicate="within")
