"""
This module contains dataset used to load dataset containing spatial information.

Datasets can be loaded using .load() method. Some of them may need name of version.
"""

from ._base import Dataset
from .airbnb_multicity import AirbnbMulticity
from .brightkite import Brightkite
from .chicago_crime import ChicagoCrime
from .foursquare_checkins import FoursquareCheckins
from .geolife import Geolife
from .gowalla import Gowalla
from .house_sales_in_king_country import HouseSalesInKingCountry
from .nyc_bike import NYCBike
from .philadelphia_crime import PhiladelphiaCrime
from .police_department_incidents import PoliceDepartmentIncidents
from .porto_taxi import PortoTaxi
from .t_drive import TDrive

__all__ = [
    "Dataset",
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
