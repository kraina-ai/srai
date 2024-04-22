"""
This module contains dataset used to load dataset containing spatial information.

Datasets can be loaded using .load() method. Some of them may need name of version.
"""

from ._base import HuggingFaceDataset
from .airbnb_multicity.dataset import AirbnbMulticity
from .brightkite.dataset import Brightkite
from .chicago_crime.dataset import ChicagoCrime
from .foursquare_checkins.dataset import FoursquareCheckins
from .geolife.dataset import Geolife
from .gowalla.dataset import Gowalla
from .house_sales_in_king_country.dataset import HouseSalesInKingCountry
from .nyc_bike.dataset import NYCBike
from .philadelphia_crime.dataset import PhiladelphiaCrime
from .police_department_incidents.dataset import PoliceDepartmentIncidents
from .porto_taxi.dataset import PortoTaxi
from .t_drive.dataset import TDrive

__all__ = [
    "HuggingFaceDataset",
    "AirbnbMulticity",
    "ChicagoCrime",
    "NYCBike",
    "FoursquareCheckins",
    "Geolife",
    "Gowalla",
    "HouseSalesInKingCountry",
    "Brightkite",
    "PhiladelphiaCrime",
    "PoliceDepartmentIncidents",
    "PortoTaxi",
    "TDrive",
]
