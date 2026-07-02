"""Tests validating example notebooks touched by the folium/H3-index API changes.

These tests operate purely on the notebook JSON structure (without executing them), checking
that the source cells were migrated away from removed/renamed APIs and that the notebooks remain
well-formed. Full execution of notebooks is already covered by the `jupyter nbconvert` step in
the CI workflows.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"

# Notebooks touched by the folium plotting wrapper argument rename (`tiles_style` -> `tiles`,
# `map` -> `m`).
NOTEBOOKS_WITH_RENAMED_PLOTTING_KWARGS = [
    "embedders/count_embedder.ipynb",
    "embedders/geovex_embedder.ipynb",
    "embedders/hex2vec_embedder.ipynb",
    "embedders/highway2vec_embedder.ipynb",
    "embedders/load_and_save.ipynb",
    "loaders/osm_online_loader.ipynb",
    "loaders/osm_way_loader.ipynb",
    "neighbourhoods/adjacency_neighbourhood.ipynb",
    "regionalizers/administrative_boundary_regionalizer.ipynb",
    "regionalizers/h3_regionalizer.ipynb",
]

# All notebooks touched in this PR (superset of the above), used for generic structural checks.
ALL_TOUCHED_NOTEBOOKS = sorted(
    set(NOTEBOOKS_WITH_RENAMED_PLOTTING_KWARGS)
    | {
        "embedders/s2vec_embedder.ipynb",
        "loaders/osm_pbf_loader.ipynb",
        "neighbourhoods/h3_neighbourhood.ipynb",
        "neighbourhoods/overriding_include_center.ipynb",
    }
)

# Matches the removed `map=` / `tiles_style=` keyword arguments (but not e.g. `m=` or unrelated
# identifiers that merely end with `map`, such as `folium_map`).
OLD_MAP_KWARG_PATTERN = re.compile(r"(?<![\w.])map=")
OLD_TILES_STYLE_KWARG_PATTERN = re.compile(r"(?<![\w.])tiles_style=")


def _load_notebook(relative_path: str) -> dict[str, Any]:
    return json.loads((EXAMPLES_DIR / relative_path).read_text(encoding="utf-8"))


def _concat_source(notebook: dict[str, Any]) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


@pytest.mark.parametrize("notebook_path", ALL_TOUCHED_NOTEBOOKS)
def test_notebook_is_valid_json(notebook_path: str) -> None:
    """Test that the notebook file is valid, parseable JSON."""
    notebook = _load_notebook(notebook_path)
    assert isinstance(notebook, dict)


@pytest.mark.parametrize("notebook_path", ALL_TOUCHED_NOTEBOOKS)
def test_notebook_has_expected_structure(notebook_path: str) -> None:
    """Test that the notebook has the expected nbformat structure and metadata."""
    notebook = _load_notebook(notebook_path)

    assert notebook["nbformat"] == 4
    assert "cells" in notebook and len(notebook["cells"]) > 0
    assert "kernelspec" in notebook["metadata"]
    assert notebook["metadata"]["kernelspec"]["language"] == "python"


@pytest.mark.parametrize("notebook_path", NOTEBOOKS_WITH_RENAMED_PLOTTING_KWARGS)
def test_notebook_does_not_use_deprecated_tiles_style_kwarg(notebook_path: str) -> None:
    """Test that notebooks no longer call plotting functions with `tiles_style=`."""
    source = _concat_source(_load_notebook(notebook_path))
    assert not OLD_TILES_STYLE_KWARG_PATTERN.search(source), (
        f"{notebook_path} still uses the removed `tiles_style=` keyword argument"
    )


@pytest.mark.parametrize("notebook_path", NOTEBOOKS_WITH_RENAMED_PLOTTING_KWARGS)
def test_notebook_does_not_use_deprecated_map_kwarg(notebook_path: str) -> None:
    """Test that notebooks no longer call plotting functions with `map=`."""
    source = _concat_source(_load_notebook(notebook_path))
    assert not OLD_MAP_KWARG_PATTERN.search(source), (
        f"{notebook_path} still uses the removed `map=` keyword argument"
    )


def test_geoparquet_loader_notebook_was_removed() -> None:
    """Test that the notebook for the removed `GeoparquetLoader` no longer exists."""
    assert not (EXAMPLES_DIR / "loaders" / "geoparquet_loader.ipynb").exists()


class TestH3IndexIntMigration:
    """Tests for the H3 index str -> int migration in neighbourhood example notebooks."""

    @pytest.mark.parametrize(
        "notebook_path",
        ["neighbourhoods/h3_neighbourhood.ipynb", "neighbourhoods/overriding_include_center.ipynb"],
    )
    def test_region_ids_are_integers_not_strings(self, notebook_path: str) -> None:
        """Test that region id examples use int literals instead of quoted H3 hex strings."""
        source = _concat_source(_load_notebook(notebook_path))

        # Old style used quoted lowercase-hex H3 strings, e.g. "881e204089fffff".
        assert not re.search(r'["\']881e20[0-9a-f]+fffff["\']', source)

        # New style assigns a plain integer literal to `region_id`.
        assert re.search(r"region_id\s*=\s*\d{5,}\b", source)

    def test_h3_neighbourhood_uses_consistent_int_ids(self) -> None:
        """Test that both region ids used in the H3 neighbourhood notebook are integers."""
        source = _concat_source(_load_notebook("neighbourhoods/h3_neighbourhood.ipynb"))

        assert "region_id = 613019531251548159" in source
        assert "edge_region_id = 613019535601041407" in source


def test_adjacency_neighbourhood_notebook_uses_regions_keyword() -> None:
    """Test that `AdjacencyNeighbourhood` is instantiated with the `regions=` keyword.

    The constructor argument was standardized to `regions` (previously some call sites used
    positional arguments or the `regions_gdf=` keyword).
    """
    source = _concat_source(_load_notebook("neighbourhoods/adjacency_neighbourhood.ipynb"))

    constructor_calls = re.findall(r"AdjacencyNeighbourhood\(([^)]*)\)", source)
    assert constructor_calls, "Expected at least one AdjacencyNeighbourhood(...) call"

    for call_args in constructor_calls:
        assert "regions=" in call_args
        assert "regions_gdf=" not in call_args


def test_s2vec_embedder_notebook_does_not_import_unused_plot_numeric_data() -> None:
    """Test that the s2vec notebook only imports the plotting helper it actually uses."""
    notebook = _load_notebook("embedders/s2vec_embedder.ipynb")
    source = _concat_source(notebook)

    import_lines = [line for line in source.splitlines() if "from srai.plotting import" in line]
    assert import_lines, "Expected an import from srai.plotting"

    for line in import_lines:
        assert "plot_regions" in line
        assert "plot_numeric_data" not in line

    # And it shouldn't be used anywhere else in the notebook either.
    assert "plot_numeric_data" not in source


def test_osm_pbf_loader_notebook_contains_geoparquet_todo_marker() -> None:
    """Regression test for the TODO marker left before the `load_to_geoparquet` call.

    This documents a known follow-up noted in the PR; if the TODO is resolved and removed,
    this test should be updated accordingly rather than silently passing on stale content.
    """
    source = _concat_source(_load_notebook("loaders/osm_pbf_loader.ipynb"))
    assert "# TODO: change" in source
    assert "load_to_geoparquet" in source