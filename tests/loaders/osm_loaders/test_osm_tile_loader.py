"""Tests for TileLoader class."""

from io import BytesIO
from urllib.parse import urljoin

import geopandas as gpd
import pytest
import requests_mock
from functional import seq
from numpy.random import default_rng
from PIL import Image
from shapely.geometry import Polygon

from srai.constants import WGS84_CRS
from srai.loaders.osm_loaders import OSMTileLoader

TEST_DOMAIN = "http://mock_server"
RESOURCE_TYPE = "png"
LOCATION = "Wroclaw, Poland"
SAVE_TO_DISK_DIR = "123"
ZOOM = 10

rng = default_rng()


def to_bytes(img: Image.Image) -> bytes:
    """Convert image into bytes."""
    img_bytes = BytesIO()
    img.save(img_bytes, RESOURCE_TYPE)
    return img_bytes.getvalue()


@pytest.fixture  # type: ignore
def loader() -> OSMTileLoader:
    """Create default TileLoader object."""
    return OSMTileLoader(TEST_DOMAIN, zoom=ZOOM, verbose=False)


@pytest.fixture  # type: ignore
def images() -> list[bytes]:
    """Create list of images as bytes."""
    return [
        to_bytes(Image.fromarray(rng.integers(low=0, high=256, size=(4, 4, 3), dtype="uint8")))
        for _ in range(3)
    ]


@pytest.fixture  # type: ignore
def gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame approximating Wroclaw bounds."""
    polygon = Polygon(
        [
            (16.8073393, 51.1389477),
            (17.0278673, 51.0426754),
            (17.1762192, 51.1063195),
            (16.9580276, 51.2093551),
            (16.8073393, 51.1389477),
        ]
    )
    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs=WGS84_CRS)
    return gdf


def mock_requests(images: list[bytes], m: requests_mock.Mocker) -> None:
    """Make mocks for requests."""
    m.get(urljoin(TEST_DOMAIN, f"10/560/341.{RESOURCE_TYPE}"), content=images[0])
    m.get(urljoin(TEST_DOMAIN, f"10/559/342.{RESOURCE_TYPE}"), content=images[1])
    m.get(urljoin(TEST_DOMAIN, f"10/560/342.{RESOURCE_TYPE}"), content=images[2])


def test_load_images_properly(
    images: list[bytes],
    gdf: gpd.GeoDataFrame,
    loader: OSMTileLoader,
) -> None:
    """Test if load returns proper images list according to location."""
    with requests_mock.Mocker() as m:
        mock_requests(images, m)
        tiles = loader.load(gdf)

        assert len(m.request_history) == 3, f"Got {len(m.request_history)} requests."

    assert to_bytes(tiles.loc[f"560_341_{ZOOM}", "tile"]) == images[0]
    assert to_bytes(tiles.loc[f"559_342_{ZOOM}", "tile"]) == images[1]
    assert to_bytes(tiles.loc[f"560_342_{ZOOM}", "tile"]) == images[2]

    xs_by_index = seq(tiles.index).map(lambda x: int(x[:3])).to_list()
    ys_by_index = seq(tiles.index).map(lambda x: int(x[4:7])).to_list()
    zs_by_index = seq(tiles.index).map(lambda x: int(x[8:])).to_list()

    assert xs_by_index == tiles["x"].tolist()
    assert ys_by_index == tiles["y"].tolist()
    assert zs_by_index == tiles["z"].tolist()


def test_should_throw_with_save_and_not_path() -> None:
    """Test checking if throws on none path with save strategy."""
    with pytest.raises(ValueError):
        _ = OSMTileLoader(
            tile_server_url=TEST_DOMAIN, zoom=ZOOM, data_collector="save", storage_path=None
        )
