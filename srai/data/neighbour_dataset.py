from typing import Optional

import geopandas as gpd

from srai.data.dataset import Dataset
from srai.neighbourhoods import Neighbourhood


class NeighbourhoodDataset(Dataset):
    def __init__(
        self,
        regions: Optional[gpd.GeoDataFrame],
        features: Optional[gpd.GeoDataFrame],
        joint: Optional[gpd.GeoDataFrame],
        neighbourhood: Optional[Neighbourhood],
    ) -> None:
        super().__init__(regions, features, joint)
        self._neighbourhood = neighbourhood

    @property
    def neighbourhood(self) -> Optional[Neighbourhood]:
        return self._neighbourhood

    @neighbourhood.setter
    def neighbourhood(self, neighbourhood: Optional[Neighbourhood]) -> None:
        self._neighbourhood = neighbourhood
