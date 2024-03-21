from srai.datasets import Dataset
import geopandas as gpd
from shapely.geometry import LineString


class TDrive(Dataset):
    def _preprocessing(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of T-Drive dataset.

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        data = data.copy()
        gdf = gpd.GeoDataFrame(
            data.drop(["arrays_geometry"], axis=1),
            geometry=gpd.GeoSeries(data["arrays_geometry"].map(LineString)),
            crs="EPSG:4326",
        )
        gdf = gdf.set_index("taxi_id")

        return gdf
