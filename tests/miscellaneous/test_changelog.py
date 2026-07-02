"""Tests validating the CHANGELOG.md content for the Unreleased section."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"


def _read_changelog() -> str:
    """Read the full contents of CHANGELOG.md."""
    return CHANGELOG_PATH.read_text(encoding="utf-8")


def _unreleased_section() -> str:
    """Extract only the `[Unreleased]` section content from the changelog."""
    content = _read_changelog()
    match = re.search(r"## \[Unreleased\](.*?)\n## \[", content, flags=re.DOTALL)
    assert match is not None, "Could not find [Unreleased] section in CHANGELOG.md"
    return match.group(1)


def test_changelog_follows_keep_a_changelog_format() -> None:
    """Test that the changelog header still references Keep a Changelog and SemVer."""
    content = _read_changelog()
    assert "Keep a Changelog" in content
    assert "Semantic Versioning" in content


def test_changelog_has_unreleased_section() -> None:
    """Test that an [Unreleased] section exists."""
    content = _read_changelog()
    assert "## [Unreleased]" in content


def test_unreleased_added_section_lists_new_dependencies() -> None:
    """Test that the Added section documents the new dependencies."""
    unreleased = _unreleased_section()
    assert "### Added" in unreleased
    assert "rq-geo-toolkit" in unreleased
    assert "geoarrow-rust-core" in unreleased
    assert "pooch" in unreleased


def test_unreleased_changed_section_documents_breaking_changes() -> None:
    """Test that breaking changes are clearly flagged in the Changed section."""
    unreleased = _unreleased_section()
    assert "### Changed" in unreleased

    breaking_lines = [
        line for line in unreleased.splitlines() if line.strip().startswith("- BREAKING!")
    ]
    assert len(breaking_lines) >= 2

    combined_breaking_text = "\n".join(breaking_lines)
    assert "tiles_style" in combined_breaking_text
    assert "tiles" in combined_breaking_text
    assert "map" in combined_breaking_text
    assert "H3 index" in combined_breaking_text


def test_unreleased_changed_section_mentions_quackosm_bump() -> None:
    """Test that the QuackOSM version bump is documented."""
    unreleased = _unreleased_section()
    assert "QuackOSM" in unreleased
    assert "0.16.2" in unreleased


def test_unreleased_changed_section_mentions_neighbourhood_index_support() -> None:
    """Test that support for int/str H3 indexes is documented with an issue link."""
    unreleased = _unreleased_section()
    assert "int and str H3 indexes" in unreleased
    assert "https://github.com/kraina-ai/srai/issues/541" in unreleased


def test_unreleased_fixed_section_mentions_folium_colouring_fix() -> None:
    """Test that the categorical colouring fix is documented."""
    unreleased = _unreleased_section()
    assert "### Fixed" in unreleased
    assert "Categorical colouring" in unreleased


def test_unreleased_removed_section_lists_geoparquet_loader_and_py39() -> None:
    """Test that GeoparquetLoader removal and Python 3.9 support drop are documented."""
    unreleased = _unreleased_section()
    assert "### Removed" in unreleased
    assert "GeoparquetLoader" in unreleased
    assert "Python 3.9" in unreleased


def test_unreleased_section_precedes_previous_release() -> None:
    """Test that [Unreleased] appears before the most recent numbered release."""
    content = _read_changelog()
    unreleased_index = content.index("## [Unreleased]")
    previous_release_index = content.index("## [0.9.9]")
    assert unreleased_index < previous_release_index