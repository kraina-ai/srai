"""H3 to class to spatially static labels."""

from dataclasses import dataclass

# Based on: https://stackoverflow.com/a/49766444
from typing import ClassVar, Union

import numpy as np


def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
    """
    Pad a NumPy array with zeros along a specified axis to reach a target length.

    Args:
        array (np.ndarray): The input array to pad.
        target_length (int): Desired length along the specified axis.
        axis (int, optional): Axis along which to pad. Defaults to 0.

    Returns:
        np.ndarray: Padded array.
    """
    pad_size: int = target_length - array.shape[axis]
    if pad_size <= 0:
        return array

    npad: list[tuple[int, int]] = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode="constant", constant_values=0)


@dataclass
class CubeHexCoords:
    """Cube coordinate representation of a hexagonal grid point."""

    q: int
    r: int
    s: int

    @classmethod
    def from_axial(cls, axial_coords: tuple[int, int]) -> "CubeHexCoords":
        """
        Create CubeHexCoords from axial coordinates.

        Args:
            axial_coords (Tuple[int, int]): Axial coordinates (q, r).

        Returns:
            CubeHexCoords: Corresponding cube coordinates.
        """
        q: int = axial_coords[0]
        r: int = axial_coords[1]
        s: int = -q - r
        return cls(q, r, s)

    @classmethod
    def origin(cls) -> "CubeHexCoords":
        """
        Return the origin cube coordinates.

        Returns:
            CubeHexCoords: (0, 0, 0)
        """
        return cls(0, 0, 0)

    def __hash__(self) -> int:
        """Compute hash value for coordinate tuple."""
        return hash((self.q, self.r, self.s))

    def __eq__(self, other: object) -> bool:
        """Check for equality with another CubeHexCoords or tuple."""
        # if isinstance(other, tuple):
        # return self.q == other[0] and self.r == other[1] and self.s == other[2]
        if (
            isinstance(other, tuple)
            and len(other) == 3
            and all(isinstance(x, int) for x in other)
        ):
            return (self.q, self.r, self.s) == other
        if isinstance(other, CubeHexCoords):
            other_coords = other
            return (
                self.q == other_coords.q
                and self.r == other_coords.r
                and self.s == other_coords.s
            )
        return False

    def __add__(self, other: "CubeHexCoords") -> "CubeHexCoords":
        """Add two cube coordinates."""
        return CubeHexCoords(self.q + other.q, self.r + other.r, self.s + other.s)

    def __sub__(self, other: "CubeHexCoords") -> "CubeHexCoords":
        """Subtract one cube coordinate from another."""
        return CubeHexCoords(self.q - other.q, self.r - other.r, self.s - other.s)

    def scale(self, factor: int) -> "CubeHexCoords":
        """
        Scale coordinate by an integer factor.

        Args:
            factor (int): Scalar multiplier.

        Returns:
            CubeHexCoords: Scaled coordinates.
        """
        return CubeHexCoords(self.q * factor, self.r * factor, self.s * factor)

    def __repr__(self) -> str:
        """String representation."""
        return f"(q={self.q}, r={self.r}, s={self.s})"

    def to_tuple(self) -> tuple[int, int, int]:
        """
        Convert to tuple.

        Returns:
            Tuple[int, int, int]: (q, r, s)
        """
        return (self.q, self.r, self.s)

    def to_numpy(self) -> np.ndarray:
        """
        Convert to NumPy array.

        Returns:
            np.ndarray: 1D array [q, r, s]
        """
        return np.array([self.q, self.r, self.s])


class SpiralH3NeighbourhoodIndexer:
    """
    Indexer for generating and ordering hexagonal spiral neighborhoods in H3.

    Based on https://www.redblobgames.com/grids/hexagons/
    """

    CUBE_NEIGHBOURS: ClassVar[list[CubeHexCoords]] = [
        CubeHexCoords(1, -1, 0),
        CubeHexCoords(0, -1, 1),
        CubeHexCoords(-1, 0, 1),
        CubeHexCoords(-1, 1, 0),
        CubeHexCoords(0, 1, -1),
        CubeHexCoords(1, 0, -1),
    ]

    def __init__(
        self,
        neighbourhood_size: int,
        starting_direction: int = 0,
        reversed_order: bool = False,
    ) -> None:
        """
        Initialize the indexer.

        Args:
            neighbourhood_size (int): Radius of the neighborhood.
            starting_direction (int): Spiral starting direction (0–5).
            reversed_order (bool): Whether to reverse the spiral direction.
        """
        self.neighbourhood_size = neighbourhood_size
        self.starting_direction = starting_direction
        self.reversed_order = reversed_order
        self.cube_direction_vectors = self._generate_directions()
        self.spiral_order = self._generate_spiral_order()

    def _get_cube_coords(
        self, origin_index: Union[str, int], h3_index: Union[str, int]
    ) -> CubeHexCoords:
        """
        Convert H3 index to cube coordinates relative to a center.

        Args:
            origin_index (Union[str, int]): Center H3 index.
            h3_index (Union[str, int]): Target H3 index.

        Returns:
            CubeHexCoords: Local cube coordinates.
        """
        if isinstance(origin_index, str):
            import h3.api.basic_str as h3
        else:
            import h3.api.basic_int as h3

        a, b = h3.cell_to_local_ij(origin_index, origin_index)
        i, j = h3.cell_to_local_ij(origin_index, h3_index)
        q: int = a - i
        r: int = j - b
        s: int = -q - r
        return CubeHexCoords(q, r, s)

    def get_hexes_order(self, regions: list[str], center: str) -> list[int]:
        """
        Get the spiral order index for a list of H3 regions.

        Args:
            regions (List[str]): H3 indices of regions.
            center (str): Central H3 index.

        Returns:
            List[int]: Spiral order indices.
        """
        coords_3d = [self._get_cube_coords(center, _h) for _h in regions]
        return [self.spiral_order[(c.q, c.r, c.s)] for c in coords_3d]

    def order_neighbourhood(self, regions: list[str], center: str) -> list[str]:
        """
        Order regions in a spiral neighborhood around a center.

        Args:
            regions (List[str]): List of H3 indices.
            center (str): Central H3 index.

        Returns:
            List[str]: H3 indices in spiral order.
        """
        coords_3d = [self._get_cube_coords(center, _h) for _h in regions]
        order = [self.spiral_order[(c.q, c.r, c.s)] for c in coords_3d]
        return [regions[i] for i in order]

    def _generate_directions(self) -> list[CubeHexCoords]:
        """Generate cube direction vectors for the spiral."""
        directions: list[CubeHexCoords] = []
        step_direction = 1 if not self.reversed_order else -1

        for i in range(6):
            current_index = (self.starting_direction + i * step_direction) % 6
            next_index = (current_index + step_direction) % 6
            current_hex = SpiralH3NeighbourhoodIndexer.CUBE_NEIGHBOURS[current_index]
            next_hex = SpiralH3NeighbourhoodIndexer.CUBE_NEIGHBOURS[next_index]
            directions.append(next_hex - current_hex)

        return directions

    def _generate_spiral_order(self) -> dict[tuple[int, int, int], int]:
        """Generate mapping from cube coords to spiral order index."""
        results: dict[tuple[int, int, int], int] = {(0, 0, 0): 0}
        next_index = 1
        for radius in range(1, self.neighbourhood_size + 1):
            for coords in self._generate_ring_order(radius):
                results[(coords.q, coords.r, coords.s)] = next_index
                next_index += 1
        return results

    def _generate_ring_order(self, radius: int) -> list[CubeHexCoords]:
        """
        Generate coordinates for a ring at a given radius.

        Args:
            radius (int): Ring radius.

        Returns:
            List[CubeHexCoords]: Ordered ring of coordinates.
        """
        results: list[CubeHexCoords] = []
        hex_coords: CubeHexCoords = (
            CubeHexCoords.origin()
            + SpiralH3NeighbourhoodIndexer.CUBE_NEIGHBOURS[
                self.starting_direction
            ].scale(radius)
        )

        for direction in range(6):
            for _ in range(radius):
                results.append(hex_coords)
                hex_coords = hex_coords + self.cube_direction_vectors[direction]

        return results


SPIRAL_INDEXERS: dict[int, SpiralH3NeighbourhoodIndexer] = {}


def get_coords(
    regions: list[str],
    center: str,
    neighbourhood_size: int,
) -> np.ndarray:
    """
    Retrieve the spiral order coordinates of regions around a center.

    Args:
        regions (List[str]): List of H3 indices.
        center (str): Central H3 index.
        neighbourhood_size (int): Radius of neighborhood.

    Returns:
        np.ndarray: Spiral order indices as a NumPy array.
    """
    # global SPIRAL_INDEXERS
    if neighbourhood_size not in SPIRAL_INDEXERS:
        SPIRAL_INDEXERS[neighbourhood_size] = SpiralH3NeighbourhoodIndexer(
            neighbourhood_size
        )

    return np.asarray(
        SPIRAL_INDEXERS[neighbourhood_size].get_hexes_order(regions, center)
    )
