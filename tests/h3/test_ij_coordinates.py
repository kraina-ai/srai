"""H3 IJ coordinates tests."""

import h3
import numpy as np
import pytest

from srai.h3 import get_local_ij_index


@pytest.mark.parametrize(
    "h3_origin",
    [
        "891e2040d4bffff",
        "871e20400ffffff",
        "821f77fffffffff",
        "81743ffffffffff",
    ],
)  # type: ignore
def test_self_ok(h3_origin: str) -> None:
    """Test checks if self coordinates are in origin."""
    coordinates = get_local_ij_index(origin_index=h3_origin, h3_index=h3_origin)
    assert coordinates == (0, 0)


@pytest.mark.parametrize(
    "h3_origin, h3_cell",
    [
        ("871f53c93ffffff", "871f53c91ffffff"),
        ("861fae207ffffff", "861fae22fffffff"),
        ("81597ffffffffff", "813fbffffffffff"),
        ("84be185ffffffff", "84be181ffffffff"),
    ],
)  # type: ignore
def test_string_ok(h3_origin: str, h3_cell: str) -> None:
    """Test checks if pairs are in right orientation."""
    coordinates = get_local_ij_index(origin_index=h3_origin, h3_index=h3_cell)
    assert coordinates == (0, 1)


@pytest.mark.parametrize(
    "h3_origin, h3_cells",
    [
        (
            "892a100d6d3ffff",
            [
                "892a100896fffff",
                "892a100d6d7ffff",
                "892a100d6c3ffff",
                "892a100d6dbffff",
                "892a1008ba7ffff",
                "892a100896bffff",
            ],
        ),
        (
            "86195da4fffffff",
            [
                "86194ad37ffffff",
                "86194ad17ffffff",
                "86194ada7ffffff",
                "86195da5fffffff",
                "86195da47ffffff",
                "86195da6fffffff",
            ],
        ),
        (
            "8a1e24aa5637fff",
            [
                "8a1e24aa5627fff",
                "8a1e24aa5607fff",
                "8a1e24aa5617fff",
                "8a1e24aa578ffff",
                "8a1e24aa57affff",
                "8a1e24aa571ffff",
            ],
        ),
    ],
)  # type: ignore
@pytest.mark.parametrize("return_as_numpy", [True, False])  # type: ignore
def test_list_ok(h3_origin: str, h3_cells: list[str], return_as_numpy: bool) -> None:
    """Test checks if lists are parsed correctly."""
    coordinates = get_local_ij_index(
        origin_index=h3_origin, h3_index=h3_cells, return_as_numpy=return_as_numpy
    )

    expected_coordinates = [(0, 1), (1, 1), (1, 0), (0, -1), (-1, -1), (-1, 0)]

    if return_as_numpy:
        np.testing.assert_array_equal(coordinates, expected_coordinates)
    else:
        assert coordinates == expected_coordinates


@pytest.mark.parametrize(
    "h3_origin, h3_cell",
    [
        ("83a75dfffffffff", "83a791fffffffff"),
        ("84a605bffffffff", "84a6021ffffffff"),
        ("836200fffffffff", "837400fffffffff"),
    ],
)  # type: ignore
def test_pentagon_error(h3_origin: str, h3_cell: str) -> None:
    """Test checks if method fails over pentagon pairs."""
    with pytest.raises(h3._cy.error_system.H3FailedError):
        get_local_ij_index(origin_index=h3_origin, h3_index=h3_cell)
