"""
OpenStreetMap extracts.

This module contains iterators for publically available OpenStreetMap `*.osm.pbf` files
repositories.
"""
import json
import re
from dataclasses import asdict, dataclass
from functools import partial
from math import ceil
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set, Union

import geopandas as gpd
import requests
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from srai._optional import import_optional_dependencies
from srai.constants import WGS84_CRS
from srai.geometry import flatten_geometry

OPENSTREETMAP_FR_POLYGONS_INDEX = "https://download.openstreetmap.fr/polygons"
OPENSTREETMAP_FR_EXTRACTS_INDEX = "https://download.openstreetmap.fr/extracts"
OPENSTREETMAP_FR_INDEX_GDF: Optional[gpd.GeoDataFrame] = None

GEOFABRIK_INDEX = "https://download.geofabrik.de/index-v1.json"
GEOFABRIK_INDEX_GDF: Optional[gpd.GeoDataFrame] = None


@dataclass
class OpenStreetMapExtract:
    """OSM Extract metadata object."""

    id: str
    url: str
    geometry: BaseGeometry


def find_smallest_containing_geofabrik_extracts(
    geometry: Union[BaseGeometry, BaseMultipartGeometry],
) -> List[OpenStreetMapExtract]:
    """
    Find smallest extracts from Geofabrik that contains given geometry.

    Iterates a geofabrik index and finds smallest extracts that covers a given geometry.

    Args:
        geometry (Union[BaseGeometry, BaseMultipartGeometry]): Geometry to be covered.

    Returns:
        List[GeofabrikExtract]: List of extracts name, URL to download it and boundary polygon.
    """
    global GEOFABRIK_INDEX_GDF  # noqa: PLW0603

    if GEOFABRIK_INDEX_GDF is None:
        GEOFABRIK_INDEX_GDF = _load_geofabrik_index()

    return _find_smallest_containing_extracts(geometry, GEOFABRIK_INDEX_GDF)


def find_smallest_containing_openstreetmap_fr_extracts(
    geometry: Union[BaseGeometry, BaseMultipartGeometry],
) -> List[OpenStreetMapExtract]:
    """
    Find smallest extracts from OpenStreetMap.fr that contains given polygon.

    Iterates an osm.fr index and finds smallest extracts that covers a given geometry.

    Args:
        geometry (Union[BaseGeometry, BaseMultipartGeometry]): Geometry to be covered.

    Returns:
        List[GeofabrikExtract]: List of extracts name, URL to download it and boundary polygon.
    """
    global OPENSTREETMAP_FR_INDEX_GDF  # noqa: PLW0603

    import_optional_dependencies(dependency_group="osm", modules=["bs4"])

    if OPENSTREETMAP_FR_INDEX_GDF is None:
        OPENSTREETMAP_FR_INDEX_GDF = _load_openstreetmap_fr_index()

    return _find_smallest_containing_extracts(geometry, OPENSTREETMAP_FR_INDEX_GDF)


def _find_smallest_containing_extracts(
    geometry: Union[BaseGeometry, BaseMultipartGeometry],
    polygons_index_gdf: gpd.GeoDataFrame,
    num_of_multiprocessing_workers: int = -1,
    multiprocessing_activation_threshold: Optional[int] = None,
) -> List[OpenStreetMapExtract]:
    """
    Find smallest set of extracts covering a given geometry.

    Iterates a provided extracts index and searches for a smallest set that cover a given geometry.
    It's not guaranteed that this set will be the smallest and there will be no overlaps.

    Geometry will be flattened into singluar geometries if it's `BaseMultipartGeometry`.

    Args:
        geometry (Union[BaseGeometry, BaseMultipartGeometry]): Geometry to be covered.
        polygons_index_gdf (gpd.GeoDataFrame): Index of available extracts.
        num_of_multiprocessing_workers (int, optional): Number of workers used for multiprocessing.
            Defaults to -1 which results in a total number of available cpu threads.
            `0` and `1` values disable multiprocessing.
            Similar to `n_jobs` parameter from `scikit-learn` library.
        multiprocessing_activation_threshold (int, optional): Number of gometries required to start
            processing on multiple processes. Activating multiprocessing for a small
            amount of points might not be feasible. Defaults to 100.

    Returns:
        List[OpenStreetMapExtract]: List of extracts covering a given geometry.
    """
    if num_of_multiprocessing_workers == 0:
        num_of_multiprocessing_workers = 1
    elif num_of_multiprocessing_workers < 0:
        num_of_multiprocessing_workers = cpu_count()

    if not multiprocessing_activation_threshold:
        multiprocessing_activation_threshold = 100

    unique_extracts_ids: Set[str] = set()

    geometries = flatten_geometry(geometry)

    total_polygons = len(geometries)

    if (
        num_of_multiprocessing_workers > 1
        and total_polygons >= multiprocessing_activation_threshold
    ):
        find_extracts_func = partial(
            _find_smallest_containing_extracts_for_single_geometry,
            polygons_index_gdf=polygons_index_gdf,
        )

        for extract_ids_list in process_map(
            find_extracts_func,
            geometries,
            desc="Finding matching extracts",
            max_workers=num_of_multiprocessing_workers,
            chunksize=ceil(total_polygons / (4 * num_of_multiprocessing_workers)),
        ):
            unique_extracts_ids.update(extract_ids_list)
    else:
        for sub_geometry in tqdm(geometries, desc="Finding matching extracts"):
            unique_extracts_ids.update(
                _find_smallest_containing_extracts_for_single_geometry(
                    sub_geometry, polygons_index_gdf
                )
            )

    extracts_filtered = _filter_extracts(
        geometry,
        unique_extracts_ids,
        polygons_index_gdf,
        num_of_multiprocessing_workers,
        multiprocessing_activation_threshold,
    )

    return extracts_filtered


def _find_smallest_containing_extracts_for_single_geometry(
    geometry: BaseGeometry, polygons_index_gdf: gpd.GeoDataFrame
) -> Set[str]:
    """
    Find smallest set of extracts covering a given singular geometry.

    Args:
        geometry (BaseGeometry): Geometry to be covered.
        polygons_index_gdf (gpd.GeoDataFrame): Index of available extracts.

    Raises:
        RuntimeError: If provided extracts index is empty.
        RuntimeError: If there is no extracts covering a given geometry (singularly or in group).

    Returns:
        Set[str]: Selected extract index string values.
    """
    if polygons_index_gdf is None:
        raise RuntimeError("Extracts index is empty.")

    extracts_ids: Set[str] = set()
    geometry_to_cover = geometry.buffer(0)

    exactly_matching_geometry = polygons_index_gdf[
        polygons_index_gdf.geometry.geom_almost_equals(geometry)
    ]
    if len(exactly_matching_geometry) == 1:
        extracts_ids.add(exactly_matching_geometry.iloc[0].id)
        return extracts_ids

    iterations = 100
    while not geometry_to_cover.is_empty and iterations > 0:
        matching_rows = polygons_index_gdf[
            (~polygons_index_gdf["id"].isin(extracts_ids))
            & (polygons_index_gdf.intersects(geometry_to_cover))
        ]
        if 0 in (len(matching_rows), iterations):
            raise RuntimeError("Couldn't find extracts matching given geometry.")

        smallest_extract = matching_rows.iloc[0]
        geometry_to_cover = geometry_to_cover.difference(smallest_extract.geometry)
        extracts_ids.add(smallest_extract.id)
        iterations -= 1
    return extracts_ids


def _filter_extracts(
    geometry: BaseGeometry,
    extracts_ids: Iterable[str],
    polygons_index_gdf: gpd.GeoDataFrame,
    num_of_multiprocessing_workers: int,
    multiprocessing_activation_threshold: int,
) -> List[OpenStreetMapExtract]:
    """
    Filter a set of extracts to include least overlaps in it.

    Args:
        geometry (Union[BaseGeometry, BaseMultipartGeometry]): Geometry to be covered.
        extracts_ids (Iterable[str]): Group of selected extracts indexes.
        polygons_index_gdf (gpd.GeoDataFrame): Index of available extracts.
        num_of_multiprocessing_workers (int): Number of workers used for multiprocessing.
            Similar to `n_jobs` parameter from `scikit-learn` library.
        multiprocessing_activation_threshold (int): Number of gometries required to start
            processing on multiple processes.

    Raises:
        RuntimeError: If provided extracts index is empty.

    Returns:
        List[OpenStreetMapExtract]: Filtered list of extracts.
    """
    if polygons_index_gdf is None:
        raise RuntimeError("Extracts index is empty.")

    sorted_extracts_gdf = polygons_index_gdf[
        polygons_index_gdf["id"].isin(extracts_ids)
    ].sort_values(by="area", ignore_index=True, ascending=False)

    filtered_extracts: List[OpenStreetMapExtract] = []
    filtered_extracts_ids: Set[str] = set()

    geometries = flatten_geometry(geometry)

    total_geometries = len(geometries)

    if (
        num_of_multiprocessing_workers > 1
        and total_geometries >= multiprocessing_activation_threshold
    ):
        filter_extracts_func = partial(
            _filter_extracts_for_single_geometry,
            sorted_extracts_gdf=sorted_extracts_gdf,
        )

        for extract_ids_list in process_map(
            filter_extracts_func,
            geometries,
            desc="Filtering extracts",
            max_workers=num_of_multiprocessing_workers,
            chunksize=ceil(total_geometries / (4 * num_of_multiprocessing_workers)),
        ):
            filtered_extracts_ids.update(extract_ids_list)
    else:
        for sub_geometry in tqdm(geometries, desc="Filtering extracts"):
            filtered_extracts_ids.update(
                _filter_extracts_for_single_geometry(sub_geometry, sorted_extracts_gdf)
            )

    simplified_extracts_ids = _simplify_selected_extracts(
        filtered_extracts_ids, sorted_extracts_gdf
    )

    for _, extract_row in sorted_extracts_gdf[
        sorted_extracts_gdf["id"].isin(simplified_extracts_ids)
    ].iterrows():
        extract = OpenStreetMapExtract(
            id=extract_row.id,
            url=extract_row["urls"]["pbf"],
            geometry=extract_row.geometry,
        )
        filtered_extracts.append(extract)

    return filtered_extracts


def _filter_extracts_for_single_geometry(
    geometry: BaseGeometry, sorted_extracts_gdf: gpd.GeoDataFrame
) -> Set[str]:
    """
    Filter a set of extracts to include least overlaps in it for a single geometry.

    Works by selecting biggest extracts (by area) and not including smaller ones if they don't
    increase a coverage.

    Args:
        geometry (BaseGeometry): Geometry to be covered.
        sorted_extracts_gdf (gpd.GeoDataFrame): Sorted index of available extracts.

    Returns:
        Set[str]: Selected extract index string values.
    """
    polygon_to_cover = geometry.buffer(0)
    filtered_extracts_ids: Set[str] = set()

    polygon_to_cover = geometry.buffer(0)
    for _, extract_row in sorted_extracts_gdf.iterrows():
        if polygon_to_cover.is_empty:
            break

        if extract_row.geometry.disjoint(polygon_to_cover):
            continue

        polygon_to_cover = polygon_to_cover.difference(extract_row.geometry)
        filtered_extracts_ids.add(extract_row.id)

    return filtered_extracts_ids


def _simplify_selected_extracts(
    filtered_extracts_ids: Set[str], sorted_extracts_gdf: gpd.GeoDataFrame
) -> Set[str]:
    simplified_extracts_ids: Set[str] = filtered_extracts_ids.copy()

    matching_extracts = sorted_extracts_gdf[sorted_extracts_gdf["id"].isin(simplified_extracts_ids)]

    simplify_again = True
    while simplify_again:
        simplify_again = False
        extract_to_remove = None
        for extract_id in simplified_extracts_ids:
            extract_geometry = (
                matching_extracts[sorted_extracts_gdf["id"] == extract_id].iloc[0].geometry
            )
            other_geometries = matching_extracts[
                sorted_extracts_gdf["id"] != extract_id
            ].unary_union
            if extract_geometry.covered_by(other_geometries):
                extract_to_remove = extract_id
                simplify_again = True
                break

        if extract_to_remove is not None:
            simplified_extracts_ids.remove(extract_to_remove)

    return simplified_extracts_ids


def _load_geofabrik_index() -> gpd.GeoDataFrame:
    """
    Load available extracts from GeoFabrik download service.

    Returns:
        gpd.GeoDataFrame: Extracts index with metadata.
    """
    result = requests.get(
        GEOFABRIK_INDEX,
        headers={"User-Agent": "SRAI Python package (https://github.com/kraina-ai/srai)"},
    )
    parsed_data = json.loads(result.text)
    gdf = gpd.GeoDataFrame.from_features(parsed_data["features"])
    gdf["area"] = gdf.geometry.area
    gdf.sort_values(by="area", ignore_index=True, inplace=True)

    save_path = "cache/geofabrik_index.csv"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    gdf[[col for col in gdf.columns if col != "geometry" and col != "urls"]].to_csv(save_path)
    return gdf


def _load_openstreetmap_fr_index() -> gpd.GeoDataFrame:
    """
    Load available extracts from OpenStreetMap.fr download service.

    Returns:
        gpd.GeoDataFrame: Extracts index with metadata.
    """
    with tqdm() as pbar:
        extracts = _iterate_openstreetmap_fr_index("osm_fr", "/", True, pbar)
    gdf = gpd.GeoDataFrame(
        data=[asdict(extract) for extract in extracts], geometry="geometry"
    ).set_crs(WGS84_CRS)
    gdf["area"] = gdf.geometry.area
    gdf.sort_values(by="area", ignore_index=True, inplace=True)

    save_path = "cache/osm_fr_index.csv"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    gdf[[col for col in gdf.columns if col != "geometry" and col != "urls"]].to_csv(save_path)
    return gdf


def _iterate_openstreetmap_fr_index(
    id_prefix: str, directory_url: str, return_extracts: bool, pbar: tqdm
) -> List[OpenStreetMapExtract]:
    """
    Iterate OpenStreetMap.fr extracts service page.

    Works recursively, by scraping whole available directory.

    Args:
        id_prefix (str): Prefix to be applies to extracts names.
        directory_url (str): Directory URL to load.
        return_extracts (bool): Whether to return collected extracts or not.
        pbar (tqdm): Progress bar.

    Returns:
        List[OpenStreetMapExtract]: List of loaded osm.fr extracts objects.
    """
    from bs4 import BeautifulSoup

    pbar.set_description_str(id_prefix)
    extracts = []
    result = requests.get(
        f"{OPENSTREETMAP_FR_EXTRACTS_INDEX}{directory_url}",
        headers={"User-Agent": "SRAI Python package (https://github.com/kraina-ai/srai)"},
    )
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
    """
    Parse poly file from URL to geometry.

    Args:
        polygon_url (str): URL to load a poly file.

    Returns:
        Optional[MultiPolygon]: Parsed polygon.
            Empty if request returns 404 not found.
    """
    result = requests.get(
        f"{OPENSTREETMAP_FR_POLYGONS_INDEX}/{polygon_url}",
        headers={"User-Agent": "SRAI Python package (https://github.com/kraina-ai/srai)"},
    )
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
