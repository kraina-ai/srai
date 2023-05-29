from dataclasses import dataclass
from typing import List, Optional
import geopandas as gpd
from pathlib import Path

@dataclass
class RegionizedData:
    area_gdf: Optional[gpd.GeoDataFrame] = None
    regions_gdf: Optional[gpd.GeoDataFrame] = None
    features_gdf: Optional[gpd.GeoDataFrame] = None
    joint_gdf: Optional[gpd.GeoDataFrame] = None

    def save(self, dir_path: Path):
        dir_path.mkdir(parents=True, exist_ok=True)
        for name, gdf in [
            ("area", self.area_gdf),
            ("regions", self.regions_gdf),
            ("features", self.features_gdf),
            ("joint", self.joint_gdf),
        ]:
            if gdf is not None:
                gdf.to_parquet(dir_path / f"{name}.parquet", index=True)
    
    @classmethod
    def load(cls, dir_path: Path):
        def _load_gdf(name: str) -> Optional[gpd.GeoDataFrame]:
            try:
                return gpd.read_parquet(dir_path / f"{name}.parquet")
            except:
                return None
        area_gdf = _load_gdf("area")
        regions_gdf = _load_gdf("regions")
        features_gdf = _load_gdf("features")
        joint_gdf = _load_gdf("joint")
        return cls(area_gdf, regions_gdf, features_gdf, joint_gdf)