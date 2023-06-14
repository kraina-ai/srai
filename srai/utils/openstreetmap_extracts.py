"""TODO."""
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, List, Optional, Set

import geopandas as gpd
import requests
from bs4 import BeautifulSoup
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from srai.constants import WGS84_CRS
from srai.utils.geometry import flatten_geometry

OPENSTREETMAP_FR_POLYGONS_INDEX = "https://download.openstreetmap.fr/polygons"
OPENSTREETMAP_FR_EXTRACTS_INDEX = "https://download.openstreetmap.fr/extracts"
OPENSTREETMAP_FR_INDEX_GDF: Optional[gpd.GeoDataFrame] = None

GEOFABRIK_INDEX = "https://download.geofabrik.de/index-v1.json"
GEOFABRIK_INDEX_GDF: Optional[gpd.GeoDataFrame] = None


@dataclass
class OpenStreetMapExtract:
    """TODO."""

    id: str
    url: str
    geometry: BaseGeometry


def find_smallest_containing_geofabrik_extracts_urls(
    polygon: BaseGeometry,
) -> List[OpenStreetMapExtract]:
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

    return _find_smallest_containing_extracts_urls(polygon, GEOFABRIK_INDEX_GDF)


def find_smallest_containing_openstreetmap_fr_extracts_urls(
    polygon: BaseGeometry,
) -> List[OpenStreetMapExtract]:
    """
    Find smallest extracts from Geofabrik that contains given polygon.

    Iterates a geofabrik index and finds smallest polygons that covers a given polygon.

    Args:
        polygon (BaseGeometry): Polygon to be contained.

    Returns:
        List[GeofabrikExtract]: List of extracts name, URL to download it and boundary polygon.
    """
    global OPENSTREETMAP_FR_INDEX_GDF  # noqa: PLW0603

    if OPENSTREETMAP_FR_INDEX_GDF is None:
        OPENSTREETMAP_FR_INDEX_GDF = _load_openstreetmap_fr_index()

    return _find_smallest_containing_extracts_urls(polygon, OPENSTREETMAP_FR_INDEX_GDF)


def _find_smallest_containing_extracts_urls(
    polygon: BaseGeometry, polygons_index_gdf: gpd.GeoDataFrame
) -> List[OpenStreetMapExtract]:
    unique_extracts_ids: Set[str] = set()

    for geometry in tqdm(flatten_geometry(polygon), desc="Finding matching extracts"):
        unique_extracts_ids.update(
            _find_smallest_containing_extracts_urls_for_single_polygon(geometry, polygons_index_gdf)
        )

    extracts_filtered = _filter_extracts(polygon, unique_extracts_ids, polygons_index_gdf)

    return extracts_filtered


def _find_smallest_containing_extracts_urls_for_single_polygon(
    polygon: BaseGeometry, polygons_index_gdf: gpd.GeoDataFrame
) -> Set[str]:
    if polygons_index_gdf is None:
        raise RuntimeError("Extracts index is empty.")

    extracts_ids: Set[str] = set()
    polygon_to_cover = polygon.buffer(0)
    iterations = 100
    while not polygon_to_cover.is_empty and iterations > 0:
        matching_rows = polygons_index_gdf[
            (~polygons_index_gdf["id"].isin(extracts_ids))
            & (polygons_index_gdf.intersects(polygon_to_cover))
        ]
        if len(matching_rows) == 0 or iterations == 0:
            raise RuntimeError("Couldn't find extracts matching given polygon.")

        smallest_extract = matching_rows.iloc[0]
        polygon_to_cover = polygon_to_cover.difference(smallest_extract.geometry)
        extracts_ids.add(smallest_extract.id)
        iterations -= 1
    return extracts_ids


def _filter_extracts(
    polygon: BaseGeometry, extracts_ids: Iterable[str], polygons_index_gdf: gpd.GeoDataFrame
) -> List[OpenStreetMapExtract]:
    if polygons_index_gdf is None:
        raise RuntimeError("Geofabrik index is empty.")

    sorted_extracts_gdf = polygons_index_gdf[
        polygons_index_gdf["id"].isin(extracts_ids)
    ].sort_values(by="area", ignore_index=True, ascending=False)

    filtered_extracts: List[OpenStreetMapExtract] = []
    filtered_extracts_ids: Set[str] = set()
    filtered_extracts_geometry: Optional[BaseGeometry] = None

    for sub_polygon in tqdm(flatten_geometry(polygon), desc="Filtering extracts"):
        polygon_to_cover = sub_polygon.buffer(0)

        if filtered_extracts_geometry:
            polygon_to_cover = polygon_to_cover.difference(filtered_extracts_geometry)

        for _, extract_row in sorted_extracts_gdf.iterrows():
            if extract_row.id in filtered_extracts_ids:
                continue

            if polygon_to_cover.is_empty:
                break

            if extract_row.geometry.disjoint(polygon_to_cover):
                continue

            extract = OpenStreetMapExtract(
                id=extract_row.id,
                url=extract_row["urls"]["pbf"],
                geometry=extract_row.geometry,
            )

            polygon_to_cover = polygon_to_cover.difference(extract.geometry)
            if filtered_extracts_geometry:
                filtered_extracts_geometry = filtered_extracts_geometry.union(extract.geometry)
            else:
                filtered_extracts_geometry = extract.geometry
            filtered_extracts.append(extract)
            filtered_extracts_ids.add(extract_row.id)

    return filtered_extracts


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


def _load_openstreetmap_fr_index() -> gpd.GeoDataFrame:
    with tqdm() as pbar:
        extracts = _iterate_openstreetmap_fr_index("osm_fr", "/", True, pbar)
    gdf = gpd.GeoDataFrame(
        data=[asdict(extract) for extract in extracts], geometry="geometry"
    ).set_crs(WGS84_CRS)
    gdf["area"] = gdf.geometry.area
    gdf.sort_values(by="area", ignore_index=True, inplace=True)
    gdf[[col for col in gdf.columns if col != "geometry" and col != "urls"]].to_csv(
        "osm_fr_index.csv"
    )
    return gdf


def _iterate_openstreetmap_fr_index(
    id_prefix: str, directory_url: str, return_extracts: bool, pbar: tqdm
) -> List[OpenStreetMapExtract]:
    pbar.set_description_str(id_prefix)
    extracts = []
    result = requests.get(f"{OPENSTREETMAP_FR_EXTRACTS_INDEX}{directory_url}")
    soup = BeautifulSoup(result.text, "html.parser")
    if return_extracts:
        extracts_urls = soup.find_all(string=re.compile("-latest\\.osm\\.pbf$"))
        for extract in extracts_urls:
            link = extract.find_parent("tr").find("a")
            name = link.text.replace("-latest.osm.pbf", "")
            polygon = _parse_polygon_file(f"{directory_url}{name}.poly")
            if polygon is None:
                continue
            extracts.append(
                OpenStreetMapExtract(
                    id=f"{id_prefix}_{name}",
                    url=f"{OPENSTREETMAP_FR_EXTRACTS_INDEX}{directory_url}{link['href']}",
                    geometry=polygon,
                )
            )
            pbar.update()
    directories = soup.find_all(src="/icons/folder.gif")
    for directory in directories:
        link = directory.find_parent("tr").find("a")
        name = link.text.replace("/", "")
        extracts.extend(
            _iterate_openstreetmap_fr_index(
                id_prefix=f"{id_prefix}_{name}",
                directory_url=f"{directory_url}{link['href']}",
                return_extracts=True,
                pbar=pbar,
            )
        )

    return extracts


def _parse_polygon_file(polygon_url: str) -> Optional[MultiPolygon]:
    result = requests.get(f"{OPENSTREETMAP_FR_POLYGONS_INDEX}/{polygon_url}")
    if result.status_code == 404:
        return None
    result.raise_for_status()
    poly = parse_poly(result.text.splitlines())
    return poly


def parse_poly(lines: List[str]) -> MultiPolygon:
    """
    Parse an Osmosis polygon filter file.

    Accept a sequence of lines from a polygon file, return a shapely.geometry.MultiPolygon object.
    Based on: https://wiki.openstreetmap.org/wiki/Osmosis/Polygon_Filter_File_Python_Parsing

    http://wiki.openstreetmap.org/wiki/Osmosis/Polygon_Filter_File_Format
    """
    in_ring = False
    coords: List[Any] = []

    for index, line in enumerate(lines):
        if index == 0:
            # first line is junk.
            continue

        elif index == 1:
            # second line is the first polygon ring.
            coords.append([[], []])
            ring = coords[-1][0]
            in_ring = True

        elif in_ring and line.strip() == "END":
            # we are at the end of a ring, perhaps with more to come.
            in_ring = False

        elif in_ring:
            # we are in a ring and picking up new coordinates.
            ring.append(list(map(float, line.split())))

        elif not in_ring and line.strip() == "END":
            # we are at the end of the whole polygon.
            break

        elif not in_ring and line.startswith("!"):
            # we are at the start of a polygon part hole.
            coords[-1][1].append([])
            ring = coords[-1][1][-1]
            in_ring = True

        elif not in_ring:
            # we are at the start of a polygon part.
            coords.append([[], []])
            ring = coords[-1][0]
            in_ring = True

    return MultiPolygon(coords)
