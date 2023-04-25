"""
Intersection Joiner.

This module contains intersection joiner implementation.
"""

import geopandas as gpd
import pandas as pd

from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, REGIONS_INDEX
from srai.joiners import Joiner


class IntersectionJoiner(Joiner):
    """
    Intersection Joiner.

    Intersection Joiner allows to join two GeoDataFrames and find all overlapping geometries. It
    does not apply any grouping or aggregation.
    """

    def transform(
        self, regions: gpd.GeoDataFrame, features: gpd.GeoDataFrame, return_geom: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Join features to regions based on an 'intersects' predicate.

        Does not apply any grouping to regions.

        Args:
            regions (gpd.GeoDataFrame): regions with which features are joined
            features (gpd.GeoDataFrame): features to be joined
            return_geom (bool): whether to return geometry of the joined features.
                Defaults to False.

        Returns:
            GeoDataFrame with an intersection of regions and features, which contains
            a MultiIndex and optionaly a geometry with the intersection
        """
        if GEOMETRY_COLUMN not in regions.columns:
            raise ValueError("Regions must have a geometry column.")
        if GEOMETRY_COLUMN not in features.columns:
            raise ValueError("Features must have a geometry column.")

        if len(regions) == 0:
            raise ValueError("Regions must not be empty.")
        if len(features) == 0:
            raise ValueError("Features must not be empty.")

        result_gdf: gpd.GeoDataFrame

        if return_geom:
            result_gdf = self._join_with_geom(regions, features)
        else:
            result_gdf = self._join_without_geom(regions, features)

        return result_gdf

    def _join_with_geom(
        self, regions: gpd.GeoDataFrame, features: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Join features to regions with returning an intersecting geometry.

        Args:
            regions (gpd.GeoDataFrame): regions with which features are joined
            features (gpd.GeoDataFrame): features to be joined

        Returns:
            GeoDataFrame with an intersection of regions and features, which contains
            a MultiIndex and a geometry with the intersection
        """
        joined_parts = [
            gpd.overlay(
                single[[GEOMETRY_COLUMN]].reset_index(names=FEATURES_INDEX),
                regions[[GEOMETRY_COLUMN]].reset_index(names=REGIONS_INDEX),
                how="intersection",
                keep_geom_type=False,
            ).set_index([REGIONS_INDEX, FEATURES_INDEX])
            for _, single in features.groupby(features[GEOMETRY_COLUMN].geom_type)
        ]

        joint = gpd.GeoDataFrame(pd.concat(joined_parts, ignore_index=False))
        return joint

    def _join_without_geom(
        self, regions: gpd.GeoDataFrame, features: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Join features to regions without intersection caclulation.

        Args:
            regions (gpd.GeoDataFrame): regions with which features are joined
            features (gpd.GeoDataFrame): features to be joined

        Returns:
            GeoDataFrame with an intersection of regions and features, which contains
            a MultiIndex
        """
        joint = (
            gpd.sjoin(
                regions.reset_index(names=REGIONS_INDEX),
                features[[GEOMETRY_COLUMN]].reset_index(names=FEATURES_INDEX),
                how="inner",
                predicate="intersects",
            )
            .set_index([REGIONS_INDEX, FEATURES_INDEX])
            .drop(columns=["index_right", GEOMETRY_COLUMN])
        )
        return joint
