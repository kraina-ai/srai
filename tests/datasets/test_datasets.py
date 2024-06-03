from typing import Callable, Optional

import pandas as pd
from parametrization import Parametrization as P
from pytest_mock import MockerFixture

from srai.constants import GEOMETRY_COLUMN, WGS84_CRS
from srai.datasets import (
    AirbnbMulticityDataset,
    BrightkiteDataset,
    ChicagoCrimeDataset,
    FoursquareCheckinsDataset,
    GeolifeDataset,
    GowallaDataset,
    HouseSalesInKingCountryDataset,
    HuggingFaceDataset,
    NYCBikeDataset,
    PhiladelphiaCrimeDataset,
    PoliceDepartmentIncidentsDataset,
    PortoTaxiDataset,
    TDriveDataset,
)
from srai.loaders import HuggingFaceLoader


@P.parameters("dataset_fn", "loaded_df", "version")  # type: ignore
@P.case(  # type: ignore
    "AirbnbMulticityDataset",
    AirbnbMulticityDataset,
    pd.DataFrame({"f1": ["a", "b"], "latitude": [10, 15], "longitude": [20, 30]}),
    None,
)
@P.case(  # type: ignore
    "BrightkiteDataset",
    BrightkiteDataset,
    pd.DataFrame({"f1": ["a", "b"], "geometry": [[[1, 2], [3, 4]], [[10, 11], [23, 24]]]}),
    None,
)
@P.case(  # type: ignore
    "ChicagoCrimeDataset",
    ChicagoCrimeDataset,
    pd.DataFrame(
        {
            "f1": ["a", "b"],
            "Latitude": [10, 15],
            "Longitude": [20, 30],
            "X Coordinate": [2, 3],
            "Y Coordinate": [4, 5],
        }
    ),
    None,
)
@P.case(  # type: ignore
    "FoursquareCheckinsDataset",
    FoursquareCheckinsDataset,
    pd.DataFrame(
        {"f1": ["a", "b"], "latitude": [[10, 15], [20, 25]], "longitude": [[20, 30], [45, 50]]}
    ),
    None,
)
@P.case(  # type: ignore
    "GeolifeDataset",
    GeolifeDataset,
    pd.DataFrame({"f1": ["a", "b"], "arrays_geometry": [[[1, 2], [3, 4]], [[10, 11], [23, 24]]]}),
    None,
)
@P.case(  # type: ignore
    "GowallaDataset",
    GowallaDataset,
    pd.DataFrame({"f1": ["a", "b"], "geometry": [[[1, 2], [3, 4]], [[10, 11], [23, 24]]]}),
    None,
)
@P.case(  # type: ignore
    "HouseSalesInKingCountryDataset",
    HouseSalesInKingCountryDataset,
    pd.DataFrame({"f1": ["a", "b"], "lat": [10, 15], "long": [20, 30]}),
    None,
)
@P.case(  # type: ignore
    "NYCBikeDataset_2013",
    NYCBikeDataset,
    pd.DataFrame(
        {
            "f1": ["a", "b"],
            "start station latitude": [10, 15],
            "start station longitude": [20, 30],
            "end station latitude": [10, 15],
            "end station longitude": [20, 30],
        }
    ),
    "nyc_bike_2013",
)
@P.case(  # type: ignore
    "NYCBikeDataset_2023",
    NYCBikeDataset,
    pd.DataFrame(
        {
            "f1": ["a", "b"],
            "start_lat": [10, 15],
            "start_lng": [20, 30],
            "end_lat": [10, 15],
            "end_lng": [20, 30],
        }
    ),
    "nyc_bike_2023",
)
@P.case(  # type: ignore
    "PhiladelphiaCrimeDataset",
    PhiladelphiaCrimeDataset,
    pd.DataFrame({"f1": ["a", "b"], "lat": [10, 15], "lng": [20, 30]}),
    None,
)
@P.case(  # type: ignore
    "PoliceDepartmentIncidentsDataset",
    PoliceDepartmentIncidentsDataset,
    pd.DataFrame({"f1": ["a", "b"], "Latitude": [10, 15], "Longitude": [20, 30]}),
    None,
)
@P.case(  # type: ignore
    "PortoTaxiDataset",
    PortoTaxiDataset,
    pd.DataFrame({"f1": ["a", "b"], "geometry": [[[1, 2], [3, 4]], [[10, 11], [23, 24]]]}),
    None,
)
@P.case(  # type: ignore
    "TDriveDataset",
    TDriveDataset,
    pd.DataFrame(
        {
            "taxi_id": ["taxi_1", "taxi_2"],
            "f1": ["a", "b"],
            "arrays_geometry": [[[1, 2], [3, 4]], [[10, 11], [23, 24]]],
        }
    ),
    None,
)
def test_hugging_face_dataset(
    mocker: MockerFixture,
    dataset_fn: Callable[[], HuggingFaceDataset],
    loaded_df: pd.DataFrame,
    version: Optional[str],
) -> None:
    mocker.patch.object(HuggingFaceLoader, "load").return_value = loaded_df

    dataset = dataset_fn()
    gdf = dataset.load(version=version)
    assert not gdf.empty
    assert GEOMETRY_COLUMN in gdf
    assert gdf.crs == WGS84_CRS
    assert "f1" in gdf
