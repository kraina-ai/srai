"""
This module contains dataset used to load dataset containing spatial information.

Datasets can be loaded using .load() method. Some of them may need name of version.
"""

from ._base import HuggingFaceDataset
from .airbnb_multicity.dataset import AirbnbMulticityDataset
from .chicago_crime.dataset import ChicagoCrimeDataset
from .philadelphia_crime.dataset import PhiladelphiaCrimeDataset
from .police_department_incidents.dataset import PoliceDepartmentIncidentsDataset

__all__ = [
    "HuggingFaceDataset",
    "AirbnbMulticityDataset",
    "PhiladelphiaCrimeDataset",
    "ChicagoCrimeDataset",
    "PoliceDepartmentIncidentsDataset",
]
