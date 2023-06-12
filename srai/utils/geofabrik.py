"""TODO."""
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import geopandas as gpd
import requests
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from srai.utils.geometry import flatten_geometry

GEOFABRIK_INDEX = "https://download.geofabrik.de/index-v1.json"

GEOFABRIK_INDEX_GDF: Optional[gpd.GeoDataFrame] = None


@dataclass
class GeofabrikExtract:
    """TODO."""

    id: str
    url: str
    geometry: BaseGeometry


def find_smallest_containing_extracts_urls(
    polygon: BaseGeometry,
) -> List[GeofabrikExtract]:
    """
    Find smallest extracts from Geofabrik that contains given polygon.

    Iterates a geofabrik index and finds smallest polygons that covers a given polygon.

    Args:
        polygon (BaseGeometry): Polygon to be contained.

    Returns:
        List[GeofabrikExtract]: List of extracts name, URL to download it and boundary polygon.
    """
    global GEOFABRIK_INDEX_GDF  # noqa: PLW0603

    if GEOFABRIK_INDEX_GDF is None:
        GEOFABRIK_INDEX_GDF = _load_geofabrik_index()

    joined_extracts: List[GeofabrikExtract] = []

    for geometry in tqdm(flatten_geometry(polygon)):
        joined_extracts.extend(_find_smallest_containing_extracts_urls_for_single_polygon(geometry))

    unique_extracts: Dict[str, GeofabrikExtract] = {}
    for extract in joined_extracts:
        if extract.id in unique_extracts:
            continue
        unique_extracts[extract.id] = extract

    extracts_filtered = _filter_extracts(polygon, list(unique_extracts.keys()))

    return extracts_filtered


def _find_smallest_containing_extracts_urls_for_single_polygon(
    polygon: BaseGeometry,
) -> List[GeofabrikExtract]:
    if GEOFABRIK_INDEX_GDF is None:
        raise RuntimeError("Geofabrik index is empty.")

    extracts: List[GeofabrikExtract] = []
    polygon_to_cover = polygon.buffer(0)
    iterations = 100
    while not polygon_to_cover.is_empty and iterations > 0:
        matching_rows = GEOFABRIK_INDEX_GDF[
            (~GEOFABRIK_INDEX_GDF["id"].isin(extract.id for extract in extracts))
            & (GEOFABRIK_INDEX_GDF.intersects(polygon_to_cover))
        ]
        if len(matching_rows) == 0 or iterations == 0:
            raise RuntimeError("Couldn't find extracts matching given polygon.")

        smallest_extract = matching_rows.iloc[0]
        polygon_to_cover = polygon_to_cover.difference(smallest_extract.geometry)
        extracts.append(
            GeofabrikExtract(
                id=smallest_extract.id,
                url=smallest_extract["urls"]["pbf"],
                geometry=smallest_extract.geometry,
            )
        )
        iterations -= 1
    return extracts


def _filter_extracts(polygon: BaseGeometry, extracts_ids: Sequence[str]) -> List[GeofabrikExtract]:
    if GEOFABRIK_INDEX_GDF is None:
        raise RuntimeError("Geofabrik index is empty.")

    sorted_extracts_gdf = GEOFABRIK_INDEX_GDF[
        GEOFABRIK_INDEX_GDF["id"].isin(extracts_ids)
    ].sort_values(by="area", ignore_index=True, ascending=False)

    extracts_filtered: List[GeofabrikExtract] = []

    polygon_to_cover = polygon.buffer(0)
    for _, extract_row in sorted_extracts_gdf.iterrows():
        if polygon_to_cover.is_empty:
            break

        if extract_row.geometry.disjoint(polygon_to_cover):
            continue

        extract = GeofabrikExtract(
            id=extract_row.id,
            url=extract_row["urls"]["pbf"],
            geometry=extract_row.geometry,
        )

        polygon_to_cover = polygon_to_cover.difference(extract.geometry)
        extracts_filtered.append(extract)

    return extracts_filtered


def _load_geofabrik_index() -> gpd.GeoDataFrame:
    result = requests.get(GEOFABRIK_INDEX)
    parsed_data = json.loads(result.text)
    gdf = gpd.GeoDataFrame.from_features(parsed_data["features"])
    gdf["area"] = gdf.geometry.area
    gdf.sort_values(by="area", ignore_index=True, inplace=True)
    gdf[[col for col in gdf.columns if col != "geometry" and col != "urls"]].to_csv(
        "geofabrik_index.csv"
    )
    return gdf
