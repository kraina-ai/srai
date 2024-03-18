from srai.datasets import Dataset
import geopandas as gpd


class HouseSalesInKingCountry(Dataset):

    def _preprocessing(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of House Sales In King Country dataset.

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        gdf = gpd.GeoDataFrame(
            data.drop(["lat", "long"], axis=1),
            geometry=gpd.points_from_xy(x=data["long"], y=data["lat"]),
            crs="EPSG:4326",
        )
        return gdf
