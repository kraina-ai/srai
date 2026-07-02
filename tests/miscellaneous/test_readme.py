"""Tests validating README.md usage examples match the current plotting API."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"


def _read_readme() -> str:
    """Read the full contents of README.md."""
    return README_PATH.read_text(encoding="utf-8")


def _code_blocks() -> list[str]:
    """Extract the content of all fenced python code blocks in README.md."""
    content = _read_readme()
    return re.findall(r"```python\n(.*?)```", content, flags=re.DOTALL)


def test_readme_does_not_use_removed_tiles_style_argument() -> None:
    """Test that no code example still uses the removed `tiles_style` keyword argument."""
    content = _read_readme()
    assert "tiles_style" not in content


def test_readme_does_not_use_removed_map_keyword_argument() -> None:
    """Test that folium .explore()/plot_* calls no longer use the removed `map=` keyword."""
    content = _read_readme()
    assert "map=folium_map" not in content


def test_readme_plot_regions_examples_use_tiles_keyword() -> None:
    """Test that plot_regions examples use the new `tiles=` keyword argument."""
    code_blocks = _code_blocks()
    plot_regions_calls = [
        line
        for block in code_blocks
        for line in block.splitlines()
        if "plot_regions(" in line and "tiles=" in line
    ]
    assert len(plot_regions_calls) >= 4
    for call in plot_regions_calls:
        assert 'tiles="CartoDB positron"' in call or "tiles_style" not in call


def test_readme_folium_explore_and_plot_calls_use_m_keyword() -> None:
    """Test that map-passing calls use the new `m=` keyword instead of `map=`."""
    code_blocks = _code_blocks()
    m_kwarg_calls = [
        line for block in code_blocks for line in block.splitlines() if re.search(r"\bm=folium_map\b", line)
    ]
    assert len(m_kwarg_calls) >= 5


def test_readme_has_at_least_one_python_code_block() -> None:
    """Sanity check that README.md still contains python usage examples."""
    assert len(_code_blocks()) > 0