"""
This module contains dataset used to load dataset containing spatial information.

Datasets can be loaded using .load() method. Some of them may need name of version.
"""

from ._base import HuggingFaceDataset
from .airbnb_multicity import AirbnbMulticityDataset
from .chicago_crime import ChicagoCrimeDataset
from .house_sales_in_king_county import HouseSalesInKingCountyDataset
from .philadelphia_crime import PhiladelphiaCrimeDataset
from .police_department_incidents import PoliceDepartmentIncidentsDataset

__all__ = [
    "HuggingFaceDataset",
    "AirbnbMulticityDataset",
    "HouseSalesInKingCountyDataset",
    "PhiladelphiaCrimeDataset",
    "ChicagoCrimeDataset",
    "PoliceDepartmentIncidentsDataset",
]
