from srai.datasets import Dataset
import geopandas as gpd
from shapely.geometry import LineString
from tqdm.contrib.concurrent import process_map


class Brightkite(Dataset):
    def _preprocessing(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of Brightkite trajectories dataset.

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        df = data.copy()
        df["geometry"] = process_map(LineString, df["geometry"], chunksize=1000, max_workers=20)
        gdf = gpd.GeoDataFrame(data=df, geometry="geometry", crs="EPSG:4326")

        return gdf
