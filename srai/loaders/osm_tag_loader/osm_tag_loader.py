"""#TODO module."""
from typing import Dict, List, Union

import geopandas as gpd
import osmnx as ox
import pandas as pd
from functional import seq
from tqdm import tqdm

from srai.utils.constants import WGS84_CRS


class OSMTagLoader:
    """#TODO."""

    def load(
        self, area: gpd.GeoDataFrame, tags: Dict[str, Union[List[str], bool]]
    ) -> gpd.GeoDataFrame:
        """#TODO."""
        area_wgs84 = area.to_crs(crs=WGS84_CRS)

        def _group_gdfs(gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
            if len(gdfs) > 1:
                gdf = pd.concat(gdfs)
            else:
                gdf = gdfs[0]
            return gdf.groupby(["element_type", "osmid"]).first()

        all_tags_num = seq(tags.values()).map(lambda v: 1 if isinstance(v, bool) else len(v)).sum()
        num_queries = len(area) * all_tags_num

        with tqdm(total=num_queries) as pbar:
            results = []
            for polygon in area_wgs84["geometry"]:
                polygon_results = []
                for key, values in tags.items():
                    key_results = []
                    if isinstance(values, bool):
                        values_ = [values]
                    for value in values_:
                        pbar.set_description(f"Processing {key}: {value}")
                        tags = {key: value}
                        geometries = ox.geometries_from_polygon(polygon, tags)
                        if geometries.empty:
                            pass
                            # warnings.warn(f"No results for {key}:{value}")
                        else:
                            key_results.append(geometries[["geometry", key]])
                        pbar.update(1)
                    polygon_results.extend(key_results)
                results.extend(polygon_results)
            return _group_gdfs(results)
