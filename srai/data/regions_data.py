from typing import Optional

import geopandas as gpd


class RegionsData:
    def __init__(
        self,
        regions: Optional[gpd.GeoDataFrame],
        features: Optional[gpd.GeoDataFrame],
        joint: Optional[gpd.GeoDataFrame],
    ) -> None:
        self._regions = regions
        self._features = features
        self._joint = joint

    @property
    def regions(self) -> Optional[gpd.GeoDataFrame]:
        return self._regions

    @regions.setter
    def regions(self, regions: Optional[gpd.GeoDataFrame]) -> None:
        self._regions = regions

    @property
    def features(self) -> Optional[gpd.GeoDataFrame]:
        return self._features

    @features.setter
    def features(self, features: Optional[gpd.GeoDataFrame]) -> None:
        self._features = features

    @property
    def joint(self) -> Optional[gpd.GeoDataFrame]:
        return self._joint

    @joint.setter
    def joint(self, joint: Optional[gpd.GeoDataFrame]) -> None:
        self._joint = joint
