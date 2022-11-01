"""
Intersection Joiner.

This module contains intersection joiner implementation.

"""

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd


@dataclass
class JoinerResult:
    """Joiner result dataclass."""

    regions: gpd.GeoDataFrame
    features: gpd.GeoDataFrame
    joined: gpd.GeoDataFrame


class IntersectionJoiner:
    """
    Intersection Joiner.

    Intersection Joiner allows to join two GeoDataFrames and find all
    overlapping geometries. It does not apply any grouping or
    aggregation.

    """

    def join(self, regions: gpd.GeoDataFrame, features: gpd.GeoDataFrame) -> JoinerResult:
        """
        Join features to regions.

        Args:
            regions (gpd.GeoDataFrame): regions with which features are joined
            features (gpd.GeoDataFrame): features to be joined

        Returns:
            JoinerResult object, which contains 3 GeoDataFrames:
            * original regions
            * original features
            * intersection of regions and features, which contains a MultiIndex and a geometry with
              the intersection

        """
        joined_parts = []

        for _, single in features.groupby(features["geometry"].geom_type):
            joined_parts.append(
                gpd.overlay(
                    single[["geometry"]].reset_index().rename(columns={"index": "features_id"}),
                    regions[["geometry"]].reset_index().rename(columns={"index": "hex_id"}),
                    how="intersection",
                    keep_geom_type=False,
                ).set_index(["hex_id", "features_id"])
            )

        joined = gpd.GeoDataFrame(pd.concat(joined_parts, ignore_index=True))

        return JoinerResult(
            regions=regions,
            features=features,
            joined=joined,
        )
