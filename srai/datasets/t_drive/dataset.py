"""
T-Drive dataset loader.

This module contains T-Drive Dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from srai.datasets import HFDataset


class TDrive(HFDataset):
    """
    T-Drive dataset.

    This dataset contains the GPS trajectories of 10,357 taxis during the
    period of Feb. 2 to Feb. 8, 2008 within Beijing. The total number of
    points in this dataset is about 15 million and the total distance of
    the trajectories reaches to 9 million kilometers.

    [1] Jing Yuan, Yu Zheng, Xing Xie, and Guangzhong Sun. Driving with
    knowledge from the physical world. In The 17th ACM SIGKDD international
    conference on Knowledge Discovery and Data mining,
    KDD '11, New York, NY, USA, 2011. ACM. [2] Jing Yuan, Yu Zheng,
    Chengyang Zhang, Wenlei Xie, Xing Xie, Guangzhong Sun, and Yan Huang.
    Tdrive: driving directions based on taxi trajectories. In Proceedings
    of the 18th SIGSPATIAL International Conference on Advances in
    Geographic Information Systems, GIS '10, pages 99'108, New York, NY, USA,
    2010. ACM.
    """

    def _preprocessing(
        self, data: pd.DataFrame, data_version_name: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of T-Drive dataset.
            data_version_name: version of dataset

        Returns:
            GeoDataFrame of dataset, contatins location data.
        """
        data = data.copy()
        gdf = gpd.GeoDataFrame(
            data.drop(["arrays_geometry"], axis=1).set_index("taxi_id"),
            geometry=gpd.GeoSeries(data["arrays_geometry"].map(LineString)),
            crs="EPSG:4326",
        )

        return gdf
