"""DOCSTRING TODO."""
import hashlib
from pathlib import Path
from time import sleep, time
from typing import Any, Dict, Sequence

import geopandas as gpd
import requests
import shapely.wkt as wktlib
import topojson as tp
from shapely.geometry import Polygon, mapping
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid
from tqdm import tqdm

from srai.utils.constants import WGS84_CRS
from srai.utils.download import download
from srai.utils.geometry import flatten_geometry, remove_interiors


class PbfFileDownloader:
    """DOCSTRING TODO."""

    PROTOMAPS_API_START_URL = "https://app.protomaps.com/downloads/osm"
    PROTOMAPS_API_DOWNLOAD_URL = "https://app.protomaps.com/downloads/{}/download"

    _PBAR_FORMAT = "Downloading pbf file ({})"

    SIMPLIFICATION_TOLERANCE_VALUES = [
        1e-07,
        2e-07,
        5e-07,
        1e-06,
        2e-06,
        5e-06,
        1e-05,
        2e-05,
        5e-05,
        0.0001,
        0.0002,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
    ]

    def download_pbf_files_for_region_gdf(
        self, region_gdf: gpd.GeoDataFrame
    ) -> Dict[str, Sequence[Path]]:
        """DOCSTRING TODO."""
        regions_mapping: Dict[str, Sequence[Path]] = {}

        for region_id, row in region_gdf.iterrows():
            polygons = flatten_geometry(row.geometry)
            regions_mapping[region_id] = [
                self.download_pbf_file_for_polygon(polygon) for polygon in polygons
            ]

        return regions_mapping

    def download_pbf_file_for_polygon(self, polygon: Polygon) -> Path:
        """DOCSTRING TODO."""
        closed_polygon = remove_interiors(polygon)
        simplified_polygon = self._simplify_polygon(closed_polygon)
        geometry_hash = self._get_geometry_hash(simplified_polygon)
        pbf_file_path = Path().resolve() / "files" / f"{geometry_hash}.pbf"

        if not pbf_file_path.exists():
            geometry_geojson = mapping(simplified_polygon)

            s = requests.Session()

            req = s.get(url=self.PROTOMAPS_API_START_URL)

            csrf_token = req.cookies["csrftoken"]
            headers = {
                "Referer": self.PROTOMAPS_API_START_URL,
                "Cookie": f"csrftoken={csrf_token}",
                "X-CSRFToken": csrf_token,
                "Content-Type": "application/json; charset=utf-8",
            }
            request_payload = {
                "region": {"type": "geojson", "data": geometry_geojson},
                "name": geometry_hash,
            }

            start_extract_request = s.post(
                url=self.PROTOMAPS_API_START_URL,
                json=request_payload,
                headers=headers,
                cookies=dict(csrftoken=csrf_token),
            )
            start_extract_request.raise_for_status()

            start_extract_result = start_extract_request.json()
            extraction_uuid = start_extract_result["uuid"]
            status_check_url = start_extract_result["url"]

            with tqdm() as pbar:
                status_response: Dict[str, Any] = {}
                cells_total = 0
                nodes_total = 0
                elems_total = 0
                while not status_response.get("Complete", False):
                    sleep(0.5)
                    status_response = s.get(url=status_check_url).json()
                    cells_total = max(cells_total, status_response.get("CellsTotal", 0))
                    nodes_total = max(nodes_total, status_response.get("NodesTotal", 0))
                    elems_total = max(elems_total, status_response.get("ElemsTotal", 0))

                    cells_prog = status_response.get("CellsProg", None)
                    nodes_prog = status_response.get("NodesProg", None)
                    elems_prog = status_response.get("ElemsProg", None)

                    if cells_total > 0 and cells_prog is not None and cells_prog < cells_total:
                        pbar.set_description(self._PBAR_FORMAT.format("Cells"))
                        pbar.total = cells_total
                        pbar.n = cells_prog
                        pbar.last_print_n = cells_prog
                    elif nodes_total > 0 and nodes_prog is not None and nodes_prog < nodes_total:
                        pbar.set_description(self._PBAR_FORMAT.format("Nodes"))
                        pbar.total = nodes_total
                        pbar.n = nodes_prog
                        pbar.last_print_n = nodes_prog
                    elif elems_total > 0 and elems_prog is not None and elems_prog < elems_total:
                        pbar.set_description(self._PBAR_FORMAT.format("Elements"))
                        pbar.total = elems_total
                        pbar.n = elems_prog
                        pbar.last_print_n = elems_prog
                    else:
                        pbar.total = elems_total
                        pbar.n = elems_total
                        pbar.last_print_n = elems_total

                    pbar.start_t = time()
                    pbar.last_print_t = time()
                    pbar.refresh()

            download(
                url=self.PROTOMAPS_API_DOWNLOAD_URL.format(extraction_uuid),
                fname=pbf_file_path.as_posix(),
            )

        return pbf_file_path

    def _simplify_polygon(self, polygon: Polygon) -> Polygon:
        simplified_polygon = polygon

        for simplify_tolerance in self.SIMPLIFICATION_TOLERANCE_VALUES:
            simplified_polygon = (
                tp.Topology(
                    polygon,
                    toposimplify=simplify_tolerance,
                    prevent_oversimplify=True,
                )
                .to_gdf(winding_order="CW_CCW", crs=WGS84_CRS, validate=True)
                .geometry[0]
            )
            simplified_polygon = make_valid(simplified_polygon)
            if len(simplified_polygon.exterior.coords) < 1000:
                break

        if len(simplified_polygon.exterior.coords) > 1000:
            simplified_polygon = polygon.convex_hull

        if len(simplified_polygon.exterior.coords) > 1000:
            simplified_polygon = polygon.minimum_rotated_rectangle

        return simplified_polygon

    def _get_geometry_hash(self, geometry: BaseGeometry) -> str:
        wkt_string = wktlib.dumps(geometry)
        h = hashlib.new("sha256")
        h.update(wkt_string.encode())
        return h.hexdigest()
