"""
This module contains dataset used to load dataset containing spatial information.

Datasets can be loaded using .load() method. Some of them may need name of version.
"""

from ._base import HuggingFaceDataset
from .airbnb_multicity.dataset import AirbnbMulticityDataset
from .brightkite.dataset import BrightkiteDataset
from .chicago_crime.dataset import ChicagoCrimeDataset
from .foursquare_checkins.dataset import FoursquareCheckinsDataset
from .geolife.dataset import GeolifeDataset
from .gowalla.dataset import GowallaDataset
from .house_sales_in_king_county.dataset import HouseSalesInKingCountyDataset
from .nyc_bike.dataset import NYCBikeDataset
from .philadelphia_crime.dataset import PhiladelphiaCrimeDataset
from .police_department_incidents.dataset import PoliceDepartmentIncidentsDataset
from .porto_taxi.dataset import PortoTaxiDataset
from .t_drive.dataset import TDriveDataset

__all__ = [
    "HuggingFaceDataset",
    "AirbnbMulticityDataset",
    "ChicagoCrimeDataset",
    "NYCBikeDataset",
    "FoursquareCheckinsDataset",
    "GeolifeDataset",
    "GowallaDataset",
    "HouseSalesInKingCountyDataset",
    "BrightkiteDataset",
    "PhiladelphiaCrimeDataset",
    "PoliceDepartmentIncidentsDataset",
    "PortoTaxiDataset",
    "TDriveDataset",
]
