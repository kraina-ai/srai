"""Tests validating CONTRIBUTING.md reflects the Python 3.10+ requirement."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRIBUTING_PATH = REPO_ROOT / "CONTRIBUTING.md"


def _read_contributing() -> str:
    """Read the full contents of CONTRIBUTING.md."""
    return CONTRIBUTING_PATH.read_text(encoding="utf-8")


def test_minimum_python_version_statement_updated() -> None:
    """Test that the minimum required Python version is stated as 3.10+."""
    content = _read_contributing()
    assert "**3.10+**" in content
    assert "**3.9+**" not in content


def test_pdm_venv_create_uses_python_3_10() -> None:
    """Test that the pdm venv creation example uses Python 3.10."""
    content = _read_contributing()
    assert "pdm venv create 3.10" in content
    assert "pdm venv create 3.9" not in content


def test_tox_example_command_updated() -> None:
    """Test that the local tox testing example uses an available Python version (3.12)."""
    content = _read_contributing()
    assert "tox -e python3.12" in content
    assert "tox -e python3.9" not in content


def test_python_conventions_statement_updated() -> None:
    """Test that the Python conventions section requires 3.10+ compatibility."""
    content = _read_contributing()
    assert "compatible with Python 3.10+" in content
    assert "compatible with Python 3.9+" not in content