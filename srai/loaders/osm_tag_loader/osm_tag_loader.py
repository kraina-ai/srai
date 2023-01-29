"""#TODO module."""
from itertools import product
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import osmnx as ox
import pandas as pd
from functional import seq
from tqdm import tqdm

from srai.utils.constants import WGS84_CRS


class OSMTagLoader:
    """#TODO."""

    _PBAR_FORMAT = "Downloading {}: {}"

    def load(
        self,
        area: gpd.GeoDataFrame,
        tags: Dict[str, Union[List[str], bool]],
        return_not_found: bool = False,
    ) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, List[Tuple[str, Union[str, bool]]]]]:
        """#TODO."""
        area_wgs84 = area.to_crs(crs=WGS84_CRS)

        _tags = self._flatten_tags(tags)

        total_tags_num = len(_tags)
        total_queries = len(area) * total_tags_num

        max_key_value_name_len = self._get_max_key_value_name_len(_tags)
        max_desc_len = max_key_value_name_len + len(self._PBAR_FORMAT.format("", ""))

        key_values_not_found = []

        results = []

        pbar = tqdm(product(area_wgs84["geometry"], _tags), total=total_queries)
        for polygon, (key, value) in pbar:
            pbar.set_description(self._get_pbar_desc(key, value, max_desc_len))
            geometries = ox.geometries_from_polygon(polygon, {key: value})
            if not geometries.empty:
                results.append(geometries[["geometry", key]])
            else:
                key_values_not_found.append((key, value))

        result_gdf = self._group_gdfs(results).set_crs(WGS84_CRS)

        if return_not_found:
            return result_gdf, key_values_not_found
        return result_gdf

    def _flatten_tags(
        self, tags: Dict[str, Union[List[str], bool]]
    ) -> List[Tuple[str, Union[str, bool]]]:
        tags_flat: List[Tuple[str, Union[str, bool]]] = (
            seq(tags.items())
            .starmap(lambda k, v: product([k], [v] if isinstance(v, bool) else v))
            .flatten()
            .list()
        )
        return tags_flat

    def _get_max_key_value_name_len(self, tags: List[Tuple[str, Union[str, bool]]]) -> int:
        max_key_val_name_len: int = seq(tags).starmap(lambda k, v: len(k + str(v))).max()
        return max_key_val_name_len

    def _get_pbar_desc(self, key: str, val: Union[bool, str], max_desc_len: int) -> str:
        return self._PBAR_FORMAT.format(key, val).ljust(max_desc_len)

    def _group_gdfs(self, gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        if len(gdfs) > 1:
            gdf = pd.concat(gdfs)
        else:
            gdf = gdfs[0]
        return gdf.groupby(["element_type", "osmid"]).first()
