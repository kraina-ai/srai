"""Parsing of the points, target and regions to the positional values."""

from collections.abc import Iterable, Sequence
from typing import Any, Optional, Union

import geopandas as gpd
import h3
import numpy as np
import numpy.typing as npt
import pandas as pd
from shapely import Point, STRtree

from srai.constants import WGS84_CRS

PointsInput = Union[gpd.GeoDataFrame, gpd.GeoSeries, Iterable[Point]]
TargetInput = Union[str, Sequence[Any], npt.NDArray[Any], pd.Series]
RegionsInput = Union[int, gpd.GeoSeries, gpd.GeoDataFrame]


def to_geopandas(points: PointsInput) -> Union[gpd.GeoDataFrame, gpd.GeoSeries]:
    """Change an iterable of points to the GeoSeries."""
    if isinstance(points, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return points
    return gpd.GeoSeries(list(points), crs=getattr(points, "crs", None))


def parse_points(points: gpd.GeoSeries) -> gpd.GeoSeries:
    """
    Validate points and reset their index.

    Args:
        points (gpd.GeoSeries): Points to be validated.

    Returns:
        gpd.GeoSeries: Points with a positional index and defined CRS (WGS84 if it was missing).
    """
    points = points.reset_index(drop=True)

    if points.empty:
        raise ValueError("Cannot split empty input.")

    missing_points = int((points.isna() | points.is_empty).sum())
    if missing_points:
        raise ValueError(f"Found {missing_points} missing or empty geometries in the input.")

    if not (points.geom_type == "Point").all():
        raise ValueError(
            "Only point geometries can be parsed."
            " Use centroids if you want to split other types of geometries."
        )

    return _ensure_crs(points)


def parse_target(
    points: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    target: Optional[TargetInput],
    categorical: bool,
) -> Optional[pd.Series]:
    """
    Get validated target values in the order of the points.

    Args:
        points (Union[gpd.GeoDataFrame, gpd.GeoSeries]): Points being split.
        target (Optional[TargetInput]): Column name or values of the target.
        categorical (bool): Whether the target is categorical.

    Returns:
        Optional[pd.Series]: Target values with a positional index.
    """
    if target is None:
        if categorical:
            raise ValueError(
                "If categorical is True, target must be specified."
                " It should contain categorical values to stratify the data."
            )
        return None

    if isinstance(target, str):
        if not isinstance(points, gpd.GeoDataFrame):
            raise ValueError(
                "Target can be a name of the column only if points are a GeoDataFrame."
            )
        target = points[target]

    values = pd.Series(target).reset_index(drop=True)
    if not categorical:
        values = values.infer_objects()
    if pd.api.types.is_extension_array_dtype(values) and pd.api.types.is_float_dtype(values):
        # Nullable types can have NaN values that are not reported as missing
        values = values.astype("float64")

    if len(values) != len(points):
        raise ValueError(
            f"Target has a different length ({len(values)}) than the points ({len(points)})."
        )

    missing_values = int(values.isna().sum())
    if missing_values:
        raise ValueError(f"Found {missing_values} missing values in the target.")

    if not categorical and pd.api.types.infer_dtype(values) in ("string", "categorical"):
        raise ValueError(
            "Target isn't numerical. Set categorical to True to stratify by categories."
        )

    if not categorical and pd.api.types.is_float_dtype(values) and np.isinf(values).any():
        raise ValueError("Found infinite values in the target.")

    return values


def get_groups(points: gpd.GeoSeries, regions: RegionsInput) -> npt.NDArray[Any]:
    """
    Get a label of the region for each point.

    Args:
        points (gpd.GeoSeries): Parsed points.
        regions (RegionsInput): H3 resolution or geometries used to group points.

    Returns:
        npt.NDArray[Any]: H3 cells or positions of the geometries.
    """
    if isinstance(regions, (int, np.integer)):
        return _get_h3_cells(points, int(regions))

    if isinstance(regions, (gpd.GeoSeries, gpd.GeoDataFrame)):
        return _get_first_matching_regions(points, _ensure_crs(regions.geometry))

    raise TypeError(
        "Parameter regions must be an H3 resolution (int), GeoSeries or GeoDataFrame,"
        f" not {type(regions).__name__}."
    )


def _get_h3_cells(points: gpd.GeoSeries, resolution: int) -> npt.NDArray[Any]:
    points = points.to_crs(WGS84_CRS)
    # Longitudes in the 0-360 range are properly wrapped by the H3
    if not (points.y.abs().le(90) & points.x.abs().le(360)).all():
        raise ValueError(
            "Coordinates of the points are outside of the WGS84 bounds."
            " Make sure that the CRS of the points is defined."
        )
    return np.array([h3.latlng_to_cell(y, x, resolution) for x, y in zip(points.x, points.y)])


def _get_first_matching_regions(
    points: gpd.GeoSeries, regions: gpd.GeoSeries
) -> npt.NDArray[np.int_]:
    points_positions, regions_positions = STRtree(regions.to_numpy()).query(
        points.to_crs(regions.crs).to_numpy(), predicate="intersects"
    )
    matches = np.full(len(points), len(regions))
    np.minimum.at(matches, points_positions, regions_positions)

    points_outside = int((matches == len(regions)).sum())
    if points_outside:
        raise ValueError(f"Found {points_outside} points outside of the provided regions.")

    return matches


def _ensure_crs(geometries: gpd.GeoSeries) -> gpd.GeoSeries:
    return geometries.set_crs(WGS84_CRS) if geometries.crs is None else geometries
