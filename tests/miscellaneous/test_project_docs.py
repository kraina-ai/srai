"""Tests validating top-level project documentation content and consistency with the API."""

import inspect
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Matches the removed `map=` / `tiles_style=` keyword arguments (but not e.g. `m=` or unrelated
# identifiers that merely end with `map`, such as `folium_map`).
OLD_MAP_KWARG_PATTERN = re.compile(r"(?<![\w.])map=")
OLD_TILES_STYLE_KWARG_PATTERN = re.compile(r"(?<![\w.])tiles_style=")


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


class TestReadme:
    """Tests for `README.md`."""

    @pytest.fixture
    def content(self) -> str:
        """Read README.md content."""
        return _read("README.md")

    def test_no_deprecated_tiles_style_kwarg(self, content: str) -> None:
        """Test that README examples no longer use the renamed `tiles_style` argument."""
        assert not OLD_TILES_STYLE_KWARG_PATTERN.search(content)

    def test_no_deprecated_map_kwarg(self, content: str) -> None:
        """Test that README examples no longer use the renamed `map` argument."""
        assert not OLD_MAP_KWARG_PATTERN.search(content)

    def test_uses_new_tiles_kwarg(self, content: str) -> None:
        """Test that README examples use the new `tiles=` argument name."""
        assert 'tiles="CartoDB positron"' in content

    def test_uses_new_m_kwarg(self, content: str) -> None:
        """Test that README examples use the new `m=` argument name."""
        assert "m=folium_map" in content

    def test_plot_regions_signature_matches_readme_usage(self) -> None:
        """Test that `plot_regions` actually accepts the `tiles` and `m` keyword arguments.

        This guards against README examples silently drifting from the real API.
        """
        from srai.plotting.folium_wrapper import plot_regions

        parameters = inspect.signature(plot_regions).parameters
        assert "tiles" in parameters
        assert "m" in parameters
        assert "tiles_style" not in parameters
        assert "map" not in parameters

    def test_plot_numeric_data_signature_matches_readme_usage(self) -> None:
        """Test that `plot_numeric_data` actually accepts the `tiles` and `m` keyword arguments."""
        from srai.plotting.folium_wrapper import plot_numeric_data

        parameters = inspect.signature(plot_numeric_data).parameters
        assert "tiles" in parameters
        assert "m" in parameters
        assert "tiles_style" not in parameters
        assert "map" not in parameters


class TestContributing:
    """Tests for `CONTRIBUTING.md`."""

    @pytest.fixture
    def content(self) -> str:
        """Read CONTRIBUTING.md content."""
        return _read("CONTRIBUTING.md")

    def test_minimum_python_version_bumped_to_310(self, content: str) -> None:
        """Test that the minimum Python version instructions reference 3.10+."""
        assert "**3.10+**" in content
        assert "**3.9+**" not in content

    def test_venv_creation_instructions_use_310(self, content: str) -> None:
        """Test that the venv creation snippet uses Python 3.10."""
        assert "pdm venv create 3.10" in content
        assert "pdm venv create 3.9" not in content

    def test_python_conventions_reference_310(self, content: str) -> None:
        """Test that the coding conventions section requires Python 3.10+ compatibility."""
        assert "compatible with Python 3.10+" in content

    def test_tox_example_uses_a_supported_version(self, content: str) -> None:
        """Test that the local-testing tox example references a currently supported version."""
        match = re.search(r"tox -e python(3\.\d+)", content)
        assert match is not None
        assert match.group(1) in {"3.10", "3.11", "3.12", "3.13"}


class TestChangelog:
    """Tests for `CHANGELOG.md`."""

    @pytest.fixture
    def content(self) -> str:
        """Read CHANGELOG.md content."""
        return _read("CHANGELOG.md")

    def test_follows_keep_a_changelog_format(self, content: str) -> None:
        """Test that the changelog declares the Keep a Changelog format."""
        assert "Keep a Changelog" in content
        assert "Semantic Versioning" in content

    def test_has_unreleased_section(self, content: str) -> None:
        """Test that an `[Unreleased]` section header exists."""
        assert "## [Unreleased]" in content

    def _get_unreleased_section(self, content: str) -> str:
        start = content.index("## [Unreleased]")
        # The next top-level version header marks the end of the Unreleased section.
        next_header_match = re.search(r"\n## \[", content[start + 1 :])
        end = start + 1 + next_header_match.start() if next_header_match else len(content)
        return content[start:end]

    def test_unreleased_mentions_python_39_removal(self, content: str) -> None:
        """Test that dropping Python 3.9 support is documented as a removal."""
        section = self._get_unreleased_section(content)
        assert "### Removed" in section
        assert "Support for Python 3.9" in section

    def test_unreleased_mentions_geoparquet_loader_removal(self, content: str) -> None:
        """Test that the `GeoparquetLoader` removal is documented."""
        section = self._get_unreleased_section(content)
        assert "GeoparquetLoader" in section

    def test_unreleased_mentions_breaking_folium_rename(self, content: str) -> None:
        """Test that the breaking folium wrapper argument rename is documented."""
        section = self._get_unreleased_section(content)
        assert "BREAKING!" in section
        assert "tiles_style" in section
        assert "tiles" in section

    def test_unreleased_has_expected_section_headers(self, content: str) -> None:
        """Test that the Unreleased section only uses standard Keep a Changelog headers."""
        section = self._get_unreleased_section(content)
        allowed_headers = {"Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"}
        found_headers = re.findall(r"^### (\w+)", section, flags=re.MULTILINE)

        assert found_headers, "Expected at least one ### section header in Unreleased"
        assert set(found_headers) <= allowed_headers


class TestLoadersExamplesReadme:
    """Tests for `examples/loaders/README.md`."""

    @pytest.fixture
    def content(self) -> str:
        """Read the loaders examples README content."""
        return _read("examples/loaders/README.md")

    def test_geoparquet_loader_entry_removed(self, content: str) -> None:
        """Test that the removed GeoparquetLoader is no longer listed."""
        assert "GeoparquetLoader" not in content
        assert "geoparquet_loader.ipynb" not in content

    def test_geoparquet_loader_notebook_file_removed(self) -> None:
        """Test that the notebook file for GeoparquetLoader was actually deleted."""
        assert not (REPO_ROOT / "examples" / "loaders" / "geoparquet_loader.ipynb").exists()

    def test_all_linked_notebooks_exist(self, content: str) -> None:
        """Test that every notebook link in the README points to an existing file."""
        linked_notebooks = re.findall(r"\]\(([\w./-]+\.ipynb)\)", content)
        assert linked_notebooks, "Expected at least one notebook link in the README"

        loaders_dir = REPO_ROOT / "examples" / "loaders"
        for notebook in linked_notebooks:
            assert (loaders_dir / notebook).exists(), f"Missing notebook referenced: {notebook}"

    def test_remaining_loaders_still_referenced(self, content: str) -> None:
        """Test that unrelated loader entries were preserved after the removal."""
        for loader_name in ["GTFSLoader", "OSMOnlineLoader", "OSMPbfLoader", "OSMWayLoader"]:
            assert loader_name in content