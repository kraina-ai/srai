"""
S2 Regionalizer.

This module exposes Google's S2 Geospatial Indexing System [1] as a regionalizer.
It uses the Python API [2].

References:
    1. https://s2geometry.io/
    2. https://github.com/JoaoCarabetta/s2-py
"""

import json
from typing import Any

import geopandas as gpd
from functional import seq
from functional.pipeline import Sequence
from s2 import s2
from shapely.geometry import Polygon

from srai.constants import GEOMETRY_COLUMN, REGIONS_INDEX, WGS84_CRS
from srai.regionalizers import Regionalizer


class S2Regionalizer(Regionalizer):
    """
    S2 Regionalizer.

    S2 Regionalizer gives an opportunity to divide the given geometries into square S2 cells.
    """

    def __init__(self, resolution: int, buffer: bool = True) -> None:
        """
        Init S2 Regionalizer.

        Args:
            resolution (int): Resolution of the cells (S2 supports 0-30). See [1] for
                a full comparison.
            buffer (bool, optional): If True then fully cover geometries with S2 cells.
                Otherwise only use those cells that fully fit into them. Defaults to True.

        Raises:
            ValueError: If resolution is not between 0 and 30.

        References:
            1. https://s2geometry.io/resources/s2cell_statistics.html
        """
        if not (0 <= resolution <= 30):
            raise ValueError(f"Resolution {resolution} is not between 0 and 30.")

        self.resolution = resolution
        self.buffer = buffer

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Regionalize a given GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to be regionalized.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with regionalized geometries.
        """
        gdf_wgs84 = gdf.to_crs(crs=WGS84_CRS)

        s2_gdf = self._fill_with_s2_cells(self._explode_multipolygons(gdf_wgs84))

        # s2 library fills also holes in Polygons, so here we remove redundant cells
        res: gpd.GeoDataFrame = gpd.sjoin(
            s2_gdf,
            gdf_wgs84,
            how="inner",
            predicate="intersects" if self.buffer else "within",
        ).drop(columns=["index_right"])

        res = res[~res.index.duplicated(keep="first")]

        res.index.name = REGIONS_INDEX

        return res

    def _fill_with_s2_cells(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        geo_json = json.loads(gdf.to_json())
        cells = (
            seq(geo_json["features"])
            .flat_map(lambda f: self._geojson_to_cells(f[GEOMETRY_COLUMN], self.resolution))
            .to_dict()
        )
        cells_gdf = gpd.GeoDataFrame(
            None,
            index=cells.keys(),
            geometry=list(cells.values()),
            crs=WGS84_CRS,
        )

        return cells_gdf

    def _geojson_to_cells(self, geo_json: dict[str, Any], res: int) -> Sequence:
        raw_cells = s2.polyfill(geo_json, res, with_id=True, geo_json_conformant=True)
        cells: Sequence = seq(raw_cells).map(lambda c: (c["id"], Polygon(c[GEOMETRY_COLUMN])))

        return cells
