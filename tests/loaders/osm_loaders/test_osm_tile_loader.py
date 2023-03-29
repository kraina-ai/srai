from io import BytesIO
from urllib.parse import urljoin

import geopandas as gpd
import pytest
import requests_mock
from numpy.random import default_rng
from PIL import Image
from pytest_mock import MockerFixture
from shapely.geometry import Polygon

from srai.constants import WGS84_CRS
from srai.loaders.osm_loaders import TileLoader
from srai.regionizers.slippy_map_regionizer import SlippyMapId

TEST_DOMAIN = "http://mock_server"
RESOURCE_TYPE = "png"
LOCATION = "Wroclaw, Poland"
SAVE_TO_DISK_DIR = "123"
ZOOM = 10

rng = default_rng()


def to_bytes(img: Image.Image) -> bytes:
    """Converts image into bytes."""
    img_bytes = BytesIO()
    img.save(img_bytes, RESOURCE_TYPE)
    return img_bytes.getvalue()


@pytest.fixture
def loader() -> TileLoader:
    """Creates default TileLoader object."""
    return TileLoader(TEST_DOMAIN, zoom=ZOOM, verbose=False)


@pytest.fixture
def images() -> list[bytes]:
    """Creates list of images as bytes."""
    return [
        to_bytes(Image.fromarray(rng.integers(low=0, high=256, size=(4, 4, 3), dtype="uint8")))
        for _ in range(3)
    ]


@pytest.fixture
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


def test_get_tiles_returns_images_properly(
    images: list[bytes],
    mocker: MockerFixture,
    gdf: gpd.GeoDataFrame,
    loader: TileLoader,
) -> None:
    """Tests if get_tile_by_region_name returns proper images list according to location."""
    mocker.patch(
        "srai.loaders.osm_loaders.osm_tile_loader.geocode_to_region_gdf",
        return_value=gdf,
        autospec=True,
    )
    with requests_mock.Mocker() as m:
        mock_requests(images, m)
        tiles = loader.get_tile_by_region_name(LOCATION)

        assert len(m.request_history) == 3, f"Got {len(m.request_history)} requests."
    assert to_bytes(tiles[SlippyMapId(560, 341)]) == images[0]
    assert to_bytes(tiles[SlippyMapId(559, 342)]) == images[1]
    assert to_bytes(tiles[SlippyMapId(560, 342)]) == images[2]


def test_should_throw_with_save_and_not_path() -> None:
    """Test checking if throws on none path with save strategy."""
    with pytest.raises(AssertionError):
        _ = TileLoader(
            tile_server_url=TEST_DOMAIN, zoom=ZOOM, collector_factory="save", storage_path=None
        )
