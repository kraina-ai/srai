from io import BytesIO
from urllib.parse import urljoin

import geopandas as gpd
import pytest
import requests_mock
import shapely.geometry as shpg
from functional import seq
from numpy.random import default_rng
from PIL import Image
from pytest_mock import MockerFixture

from srai.loaders.osm_loaders import TileLoader

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
        for _ in range(4)
    ]


@pytest.fixture
def gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame approximating Wroclaw bounds."""
    polygon = shpg.Polygon(
        [
            (20.8516882, 52.2009766),
            (20.9915988, 52.103114),
            (21.2711512, 52.1721505),
            (21.2504688, 52.2626919),
            (21.1050091, 52.2739945),
            (21.0738087, 52.3672883),
            (20.912429, 52.3529758),
            (20.8516882, 52.2009766),
        ]
    )
    gdf = gpd.GeoDataFrame({"geometry": [polygon]})
    return gdf


def mock_requests(images: list[bytes], m: requests_mock.Mocker) -> None:
    """Make mocks for requests."""
    m.get(urljoin(TEST_DOMAIN, f"10/571/336.{RESOURCE_TYPE}"), content=images[0])
    m.get(urljoin(TEST_DOMAIN, f"10/572/336.{RESOURCE_TYPE}"), content=images[1])
    m.get(urljoin(TEST_DOMAIN, f"10/571/337.{RESOURCE_TYPE}"), content=images[2])
    m.get(urljoin(TEST_DOMAIN, f"10/572/337.{RESOURCE_TYPE}"), content=images[3])


def test_coordinates_cast(loader: TileLoader) -> None:
    """Tests if coordinates_to_x_y gives proper x and y value."""
    # given
    latitude, longitude = 51, 16.8

    # when
    x, y = loader.coordinates_to_x_y(latitude=latitude, longitude=longitude)

    # then
    assert x == 559
    assert y == 342


def test_get_tiles_returns_images_properly(
    images: list[bytes],
    mocker: MockerFixture,
    gdf: gpd.GeoDataFrame,
    loader: TileLoader,
) -> None:
    """Tests if get_tile_by_region_name returns proper images list according to location."""
    # given
    mocker.patch(
        "srai.loaders.osm_loaders.osm_tile_loader.geocode_to_region_gdf",
        return_value=gdf,
        autospec=True,
    )
    with requests_mock.Mocker() as m:
        mock_requests(images, m)
        tiles = loader.get_tile_by_region_name(LOCATION, return_rect=True)

        assert len(m.request_history) == 4
    assert to_bytes(tiles[0][0]) == images[0]
    assert to_bytes(tiles[0][1]) == images[1]
    assert to_bytes(tiles[1][0]) == images[2]
    assert to_bytes(tiles[1][1]) == images[3]


def test_should_skip_intersections(images: list[bytes], loader: TileLoader) -> None:
    """Tests if tiles out of area are skipped if specified."""
    # given
    zoom_with_skips = 11
    x_to_skip = 1119
    y_to_skip = 683
    should_skip_uri = f"{zoom_with_skips}/{x_to_skip}/{y_to_skip}.{RESOURCE_TYPE}"
    loader.zoom = zoom_with_skips
    with requests_mock.Mocker() as m:
        m.get(requests_mock.ANY, content=images[0])

        tiles = loader.get_tile_by_region_name(LOCATION, return_rect=False)

        assert len(m.request_history) == 8
        assert should_skip_uri not in seq(m.request_history).map(lambda x: x.path).to_set()
        for row in tiles:
            assert len(row) == 3


def test_x_y_to_coordinates_should_be_inverse_to_coordinates_to_x_y(
    loader: TileLoader,
) -> None:
    """Tests if `x_y_to_coordinates` is reversible with `coordinates_to_x_y`."""
    # given
    x, y = 50, 100

    # when
    latitude, longitude = loader.x_y_to_coordinates(x, y)
    x_reverse, y_reverse = loader.coordinates_to_x_y(latitude, longitude)

    # then
    assert x_reverse == x
    assert y_reverse == y


def test_should_throw_with_save_and_not_path() -> None:
    """Test checking if throws on none path with save strategy."""
    with pytest.raises(AssertionError):
        _ = TileLoader(
            tile_server_url=TEST_DOMAIN, zoom=ZOOM, collector_factory="save", storage_path=None
        )
