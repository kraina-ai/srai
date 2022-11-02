"""
Intersection Joiner.

This module contains intersection joiner implementation.

"""

import geopandas as gpd
import pandas as pd


class IntersectionJoiner:
    """
    Intersection Joiner.

    Intersection Joiner allows to join two GeoDataFrames and find all overlapping geometries. It
    does not apply any grouping or aggregation.

    """

    def join(self, regions: gpd.GeoDataFrame, features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Join features to regions.

        Args:
            regions (gpd.GeoDataFrame): regions with which features are joined
            features (gpd.GeoDataFrame): features to be joined

        Returns:
            GeoDataFrame with an intersection of regions and features, which contains
            a MultiIndex and a geometry with the intersection

        """
        joined_parts = []

        for _, single in features.groupby(features["geometry"].geom_type):
            joined_parts.append(
                gpd.overlay(
                    single[["geometry"]].reset_index(names="feature_id"),
                    regions[["geometry"]].reset_index(names="region_id"),
                    how="intersection",
                    keep_geom_type=False,
                ).set_index(["region_id", "feature_id"])
            )

        joint = gpd.GeoDataFrame(pd.concat(joined_parts, ignore_index=False))

        return joint
