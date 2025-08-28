"""
Intersection Joiner.

This module contains intersection joiner implementation.
"""

from typing import Literal, Union, overload

import geopandas as gpd
import pandas as pd

from srai.constants import FEATURES_INDEX, GEOMETRY_COLUMN, REGIONS_INDEX
from srai.geodatatable import (
    VALID_GEO_INPUT,
    GeoDataTable,
    ParquetDataTable,
    prepare_geo_input,
)
from srai.joiners import Joiner


class IntersectionJoiner(Joiner):
    """
    Intersection Joiner.

    Intersection Joiner allows to join two GeoDataFrames and find all overlapping geometries. It
    does not apply any grouping or aggregation.
    """

    @overload
    def transform(
        self,
        regions: VALID_GEO_INPUT,
        features: VALID_GEO_INPUT,
    ) -> ParquetDataTable: ...

    @overload
    def transform(
        self,
        regions: VALID_GEO_INPUT,
        features: VALID_GEO_INPUT,
        return_geom: Literal[False],
    ) -> ParquetDataTable: ...

    @overload
    def transform(
        self,
        regions: VALID_GEO_INPUT,
        features: VALID_GEO_INPUT,
        return_geom: Literal[True],
    ) -> GeoDataTable: ...

    def transform(
        self,
        regions: VALID_GEO_INPUT,
        features: VALID_GEO_INPUT,
        return_geom: bool = False,
    ) -> Union[ParquetDataTable, GeoDataTable]:
        """
        Join features to regions based on an 'intersects' predicate.

        Does not apply any grouping to regions.

        Args:
            regions (VALID_GEO_INPUT): regions with which features are joined
            features (VALID_GEO_INPUT): features to be joined
            return_geom (bool): whether to return geometry of the joined features.
                Defaults to False.

        Returns:
            ParquetDataTable or GeoDataTable with an intersection of regions and features,
            which contains a MultiIndex and optionaly a geometry with the intersection.
        """
        regions_pdt = prepare_geo_input(regions)
        features_pdt = prepare_geo_input(features)

        if GEOMETRY_COLUMN not in regions_pdt.columns:
            raise ValueError("Regions must have a geometry column.")
        if GEOMETRY_COLUMN not in features_pdt.columns:
            raise ValueError("Features must have a geometry column.")

        if regions_pdt.empty:
            raise ValueError("Regions must not be empty.")
        if features_pdt.empty:
            raise ValueError("Features must not be empty.")

        if return_geom:
            return self._join_with_geom(regions_pdt, features_pdt)

        return self._join_without_geom(regions_pdt, features_pdt)

    def _join_with_geom(self, regions: GeoDataTable, features: GeoDataTable) -> GeoDataTable:
        """
        Join features to regions with returning an intersecting geometry.

        Args:
            regions (GeoDataTable): regions with which features are joined
            features (GeoDataTable): features to be joined

        Returns:
            GeoDataTable with an intersection of regions and features, which contains
            a MultiIndex and a geometry with the intersection
        """
        regions_gdf = regions.to_geodataframe()
        features_gdf = features.to_geodataframe()
        joined_parts = [
            gpd.overlay(
                single[[GEOMETRY_COLUMN]].reset_index(names=FEATURES_INDEX),
                regions_gdf[[GEOMETRY_COLUMN]].reset_index(names=REGIONS_INDEX),
                how="intersection",
                keep_geom_type=False,
            ).set_index([REGIONS_INDEX, FEATURES_INDEX])
            for _, single in features_gdf.groupby(features_gdf[GEOMETRY_COLUMN].geom_type)
        ]

        joint = gpd.GeoDataFrame(pd.concat(joined_parts, ignore_index=False))
        return GeoDataTable.from_geodataframe(joint)

    def _join_without_geom(self, regions: GeoDataTable, features: GeoDataTable) -> ParquetDataTable:
        """
        Join features to regions without intersection caclulation.

        Args:
            regions (GeoDataTable): regions with which features are joined
            features (GeoDataTable): features to be joined

        Returns:
            ParquetDataTable with an intersection of regions and features, which contains
            a MultiIndex
        """
        regions_gdf = regions.to_geodataframe()
        features_gdf = features.to_geodataframe()

        features_idx, region_idx = regions_gdf.sindex.query(
            features_gdf[GEOMETRY_COLUMN], predicate="intersects"
        )
        joint = pd.DataFrame(
            {
                REGIONS_INDEX: regions_gdf.index[region_idx],
                FEATURES_INDEX: features_gdf.index[features_idx],
            }
        ).set_index([REGIONS_INDEX, FEATURES_INDEX])
        return ParquetDataTable.from_dataframe(joint)
