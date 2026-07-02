"""Tests validating example notebooks match the updated plotting/neighbourhood API.

This PR renamed several keyword arguments (`tiles_style` -> `tiles`, `map` -> `m` in the
folium plotting wrapper, `regions_gdf` -> `regions` in `AdjacencyNeighbourhood`) and switched
H3 region identifiers from strings to integers. These tests make sure the example notebooks
that were updated as part of this change actually use the new API, so the notebooks keep
executing successfully (they are executed via `jupyter nbconvert` in CI).
"""

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_notebook(relative_path: str) -> dict[str, Any]:
    """Load a notebook file (valid JSON) as a dict."""
    notebook_path = EXAMPLES_DIR / relative_path
    with notebook_path.open(encoding="utf-8") as f:
        return json.load(f)


def _cell_sources(notebook: dict[str, Any]) -> list[str]:
    """Return the joined source text of every cell in the notebook."""
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def _full_source(notebook: dict[str, Any]) -> str:
    """Return the concatenated source of the entire notebook."""
    return "\n".join(_cell_sources(notebook))


UPDATED_TILES_NOTEBOOKS = [
    "embedders/count_embedder.ipynb",
    "embedders/geovex_embedder.ipynb",
    "embedders/hex2vec_embedder.ipynb",
    "embedders/highway2vec_embedder.ipynb",
    "embedders/load_and_save.ipynb",
    "loaders/osm_online_loader.ipynb",
    "loaders/osm_way_loader.ipynb",
    "regionalizers/administrative_boundary_regionalizer.ipynb",
    "neighbourhoods/adjacency_neighbourhood.ipynb",
]


@pytest.mark.parametrize("relative_path", UPDATED_TILES_NOTEBOOKS)  # type: ignore
def test_notebook_does_not_use_removed_tiles_style_argument(relative_path: str) -> None:
    """Test that updated notebooks no longer call plotting functions with `tiles_style=`."""
    notebook = _load_notebook(relative_path)
    source = _full_source(notebook)
    assert "tiles_style" not in source


@pytest.mark.parametrize(  # type: ignore
    "relative_path",
    [
        "embedders/count_embedder.ipynb",
        "embedders/geovex_embedder.ipynb",
        "embedders/hex2vec_embedder.ipynb",
        "embedders/highway2vec_embedder.ipynb",
        "embedders/load_and_save.ipynb",
        "loaders/osm_online_loader.ipynb",
        "loaders/osm_way_loader.ipynb",
        "regionalizers/administrative_boundary_regionalizer.ipynb",
        "neighbourhoods/adjacency_neighbourhood.ipynb",
    ],
)
def test_notebook_uses_new_tiles_keyword(relative_path: str) -> None:
    """Test that notebooks calling plot_regions with a tile style use the `tiles=` keyword."""
    notebook = _load_notebook(relative_path)
    source = _full_source(notebook)
    assert 'tiles="CartoDB positron"' in source


def test_h3_regionalizer_notebook_uses_m_keyword_not_map() -> None:
    """Test that h3_regionalizer.ipynb passes the folium map using `m=` instead of `map=`."""
    notebook = _load_notebook("regionalizers/h3_regionalizer.ipynb")
    source = _full_source(notebook)

    assert "map=folium_map" not in source
    assert source.count("m=folium_map") == 2


def test_s2vec_embedder_notebook_no_longer_imports_unused_plot_numeric_data() -> None:
    """Test that the unused `plot_numeric_data` import was removed from s2vec_embedder.ipynb."""
    notebook = _load_notebook("embedders/s2vec_embedder.ipynb")
    import_cell_source = _cell_sources(notebook)[0]

    assert "from srai.plotting import plot_regions" in import_cell_source
    assert "plot_numeric_data" not in import_cell_source


def test_loaders_readme_no_longer_references_geoparquet_loader() -> None:
    """Test that the loaders examples README dropped the GeoparquetLoader entry."""
    readme_path = EXAMPLES_DIR / "loaders" / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    assert "GeoparquetLoader" not in content
    assert "geoparquet_loader.ipynb" not in content


def test_geoparquet_loader_notebook_was_removed() -> None:
    """Test that the geoparquet_loader.ipynb example notebook no longer exists."""
    notebook_path = EXAMPLES_DIR / "loaders" / "geoparquet_loader.ipynb"
    assert not notebook_path.exists()


def test_osm_pbf_loader_notebook_flags_geoparquet_example_for_followup() -> None:
    """Test that the geoparquet export example in osm_pbf_loader.ipynb keeps its TODO marker."""
    notebook = _load_notebook("loaders/osm_pbf_loader.ipynb")
    source = _full_source(notebook)

    assert "# TODO: change" in source
    assert "load_to_geoparquet" in source


class TestAdjacencyNeighbourhoodNotebook:
    """Tests for examples/neighbourhoods/adjacency_neighbourhood.ipynb."""

    @pytest.fixture()  # type: ignore
    def notebook(self) -> dict[str, Any]:
        """Load the adjacency_neighbourhood notebook."""
        return _load_notebook("neighbourhoods/adjacency_neighbourhood.ipynb")

    def test_uses_regions_keyword_instead_of_regions_gdf(
        self, notebook: dict[str, Any]
    ) -> None:
        """Test that AdjacencyNeighbourhood is constructed with the new `regions=` keyword."""
        source = _full_source(notebook)

        assert "AdjacencyNeighbourhood(regions_gdf=" not in source
        assert "AdjacencyNeighbourhood(it_regions_gdf)" not in source
        assert source.count("AdjacencyNeighbourhood(regions=") == 2

    def test_kernelspec_updated_to_python_3_12(self, notebook: dict[str, Any]) -> None:
        """Test that the notebook kernel metadata reflects the Python 3.12 environment."""
        metadata = notebook["metadata"]

        assert metadata["kernelspec"]["display_name"] == "srai-3.12"
        assert metadata["language_info"]["version"] == "3.12.9"


class TestH3NeighbourhoodNotebook:
    """Tests for examples/neighbourhoods/h3_neighbourhood.ipynb."""

    @pytest.fixture()  # type: ignore
    def notebook(self) -> dict[str, Any]:
        """Load the h3_neighbourhood notebook."""
        return _load_notebook("neighbourhoods/h3_neighbourhood.ipynb")

    def test_uses_integer_region_ids_instead_of_strings(
        self, notebook: dict[str, Any]
    ) -> None:
        """Test that region ids are now plain integers rather than hex strings."""
        source = _full_source(notebook)

        assert "881e204089fffff" not in source
        assert "881e2050bdfffff" not in source
        assert "region_id = 613019531251548159" in source
        assert source.count("edge_region_id = 613019535601041407") == 2

    def test_kernelspec_updated_to_python_3_12(self, notebook: dict[str, Any]) -> None:
        """Test that the notebook kernel metadata reflects the Python 3.12 environment."""
        metadata = notebook["metadata"]

        assert metadata["kernelspec"]["display_name"] == "srai-3.12"
        assert metadata["language_info"]["version"] == "3.12.9"


def test_overriding_include_center_notebook_uses_integer_region_id() -> None:
    """Test that overriding_include_center.ipynb uses an int region id, not a hex string."""
    notebook = _load_notebook("neighbourhoods/overriding_include_center.ipynb")
    source = _full_source(notebook)

    assert "881e204089fffff" not in source
    assert "region_id = 613019531251548159" in source