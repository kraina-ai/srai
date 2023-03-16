# noqa
# """
# OSM Tag Loader.

# This module contains loader capable of loading OpenStreetMap tags.
# """
"""DOCSTRING TODO."""
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import geopandas as gpd
import pandas as pd

from srai.loaders.osm_tag_loader.filters.osm_tags_type import osm_tags_type
from srai.utils._optional import import_optional_dependencies
from srai.utils.constants import WGS84_CRS


class PbfTagLoader:
    """DOCSTRING TODO."""

    # """
    # OSMTagLoader.

    # OSM(OpenStreetMap)[1] tag loader is a loader capable of downloading objects
    # from a given area from OSM. Tags in this context mean arbitrary
    # key:value pairs, that are used by OSM users to give meaning to geometries.

    # This loader is a wrapper around the `osmnx` library. It uses `osmnx.geometries_from_polygon`
    # to make individual queries.

    # References:
    #     1. https://www.openstreetmap.org/
    # """

    def __init__(self, pbf_file: Optional[Union[str, Path]] = None) -> None:
        """DOCSTRING TODO."""
        # """Initialize OSMTagLoader."""
        import_optional_dependencies(dependency_group="osm", modules=["osmium"])
        self.pbf_file = pbf_file

    def load(
        self,
        area: gpd.GeoDataFrame,
        tags: osm_tags_type,
    ) -> gpd.GeoDataFrame:
        """DOCSTRING TODO."""
        # """
        # Download OSM objects with specified tags for a given area.

        # The loader first downloads all objects with `tags`. It returns a GeoDataFrame containing
        # the `geometry` column and columns for tag keys.

        # Note: some key/value pairs might be missing from the resulting GeoDataFrame,
        #     simply because there are no such objects in the given area.

        # Args:
        #     area (gpd.GeoDataFrame): Area for which to download objects.
        #     tags (Dict[str, Union[List[str], str, bool]]): A dictionary
        #         specifying which tags to download.
        #         The keys should be OSM tags (e.g. `building`, `amenity`).
        #         The values should either be `True` for retrieving all objects with the tag,
        #         string for retrieving a single tag-value pair
        #         or list of strings for retrieving all values specified in the list.
        #         `tags={'leisure': 'park}` would return parks from the area.
        #         `tags={'leisure': 'park, 'amenity': True, 'shop': ['bakery', 'bicycle']}`
        #         would return parks, all amenity types, bakeries and bicycle shops.

        # Returns:
        #     gpd.GeoDataFrame: Downloaded objects as a GeoDataFrame.
        # """

        from srai.loaders.osm_tag_loader.pbf_file_downloader import PbfFileDownloader
        from srai.loaders.osm_tag_loader.pbf_file_handler import PbfFileHandler

        area_wgs84 = area.to_crs(crs=WGS84_CRS)

        downloaded_pbf_files: Mapping[str, Sequence[Union[str, Path]]]
        if self.pbf_file is not None:
            downloaded_pbf_files = {region_id: [self.pbf_file] for region_id in area_wgs84.index}
        else:
            downloaded_pbf_files = PbfFileDownloader().download_pbf_files_for_region_gdf(
                region_gdf=area_wgs84
            )

        pbf_handler = PbfFileHandler(tags=tags)

        results = []
        for region_id, pbf_files in downloaded_pbf_files.items():
            clipping_polygon = area_wgs84.loc[[region_id]].geometry.unary_union

            features_gdf = pbf_handler.get_features_gdf(file_paths=pbf_files, region_id=region_id)
            features_gdf = features_gdf[features_gdf.intersects(clipping_polygon)]
            results.append(features_gdf)

        result_gdf = self._group_gdfs(results).set_crs(WGS84_CRS)

        features_columns = result_gdf.columns.drop(labels=["geometry"]).sort_values()
        result_gdf = result_gdf[["geometry", *features_columns]]

        return result_gdf

    def _group_gdfs(self, gdfs: List[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        if not gdfs:
            return self._get_empty_result()
        elif len(gdfs) == 1:
            gdf = gdfs[0]
        else:
            gdf = pd.concat(gdfs)

        return gdf[~gdf.index.duplicated(keep="first")]

    def _get_empty_result(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(index=[], crs=WGS84_CRS, geometry=[])
