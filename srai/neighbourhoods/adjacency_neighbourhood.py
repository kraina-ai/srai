"""
Adjacency neighbourhood.

This module contains the AdjacencyNeighbourhood class, that allows to get the neighbours of any
region based on its borders.
"""

from collections.abc import Hashable
from typing import Optional

import duckdb
import pandas as pd

from srai.constants import GEOMETRY_COLUMN
from srai.geodatatable import VALID_GEO_INPUT, prepare_geo_input
from srai.neighbourhoods import Neighbourhood


class AdjacencyNeighbourhood(Neighbourhood[Hashable]):
    """
    Adjacency Neighbourhood.

    This class allows to get the neighbours of any region based on common border. Additionally, a
    lookup table is implemented to accelerate repeated queries.

    By default, a lookup table will be populated lazily based on queries. A dedicated function
    `generate_neighbourhoods` allows for precalculation of all the neighbourhoods at once.
    """

    def __init__(self, regions: VALID_GEO_INPUT, include_center: bool = False) -> None:
        """
        Init AdjacencyNeighbourhood.

        Args:
            regions (VALID_GEO_INPUT): regions for which a neighbourhood will be calculated.
            include_center (bool): Whether to include the region itself in the neighbours.
            This is the default value used for all the methods of the class,
            unless overridden in the function call.

        Raises:
            ValueError: If regions_gdf doesn't have geometry column.
        """
        super().__init__(include_center)
        self.regions_gdt = prepare_geo_input(regions)
        self.lookup: dict[Hashable, set[Hashable]] = {}
        self._has_generated_neighbourhood = False

    def generate_neighbourhoods(self) -> None:
        """Generate the lookup table for all regions."""
        relation = self.regions_gdt.to_duckdb()

        if self.regions_gdt.has_multiindex:
            _index_col = "joined_index"
            index_names = self.regions_gdt.index_column_names or []
            index_lookup = duckdb.sql(
                f"""
                SELECT
                    {",".join(index_names)},
                    CONCAT_WS('_', {",".join(index_names)}) as {_index_col}
                FROM ({relation.sql_query()})
                """
            ).to_df()
            index_lookup_dict = (
                pd.Series(index_lookup[index_names].values.tolist(), index=index_lookup[_index_col])
                .apply(tuple)
                .to_dict()
            )
            relation = relation.project(
                f"CONCAT_WS('_', {','.join(index_names)}) as {_index_col}, {GEOMETRY_COLUMN}"
            )
        elif self.regions_gdt.index_name:
            _index_col = self.regions_gdt.index_name
        else:
            raise ValueError("Provided GeoDataTable has not index.")

        regions_with_neighbours = (
            relation.set_alias("current")
            .join(
                relation.set_alias("other"),
                condition=f"ST_Touches(current.{GEOMETRY_COLUMN}, other.{GEOMETRY_COLUMN})",
            )
            .project(f"current.{_index_col}, other.{_index_col} as neighbour_id")
            .aggregate(
                f"{_index_col}, ARRAY_AGG(DISTINCT neighbour_id) as neighbour_ids", _index_col
            )
        )

        for batch in (
            relation.join(regions_with_neighbours, _index_col, how="left")
            .project(f"{_index_col}, COALESCE(neighbour_ids, []) as neighbour_ids")
            .fetch_arrow_reader()
        ):
            for region_id, neighbour_ids in zip(batch[_index_col], batch["neighbour_ids"]):
                lookup_region_id = region_id.as_py()
                lookup_neighbour_ids = neighbour_ids.as_py()
                if self.regions_gdt.has_multiindex:
                    lookup_region_id = index_lookup_dict[lookup_region_id]
                    lookup_neighbour_ids = [
                        index_lookup_dict[region_id] for region_id in lookup_neighbour_ids
                    ]
                self.lookup[lookup_region_id] = set(lookup_neighbour_ids)

        self._has_generated_neighbourhood = True

    def get_neighbours(
        self, index: Hashable, include_center: Optional[bool] = None
    ) -> set[Hashable]:
        """
        Get the direct neighbours of any region using its index.

        Args:
            index (Hashable): Unique identifier of the region.
            include_center (Optional[bool]): Whether to include the region itself in the neighbours.
            If None, the value set in __init__ is used. Defaults to None.

        Returns:
            Set[Hashable]: Indexes of the neighbours.
        """
        if not self._has_generated_neighbourhood:
            self.generate_neighbourhoods()

        if self._index_incorrect(index):
            return set()

        neighbours = self.lookup[index]
        neighbours = self._handle_center(
            index, 1, neighbours, at_distance=False, include_center_override=include_center
        )
        return neighbours

    def _index_incorrect(self, index: Hashable) -> bool:
        return index not in self.lookup
