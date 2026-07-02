"""Tests validating GitHub Actions workflow configuration files."""

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def _load_workflow(name: str) -> dict[str, Any]:
    """Load and parse a GitHub Actions workflow YAML file."""
    workflow_path = WORKFLOWS_DIR / name
    with workflow_path.open() as f:
        return yaml.safe_load(f)


def test_tests_workflow_is_valid_yaml() -> None:
    """Test that _tests.yml is syntactically valid YAML."""
    workflow = _load_workflow("_tests.yml")
    assert workflow is not None
    assert "jobs" in workflow


def test_tests_workflow_python_version_matrix() -> None:
    """Test that the main test matrix uses supported Python versions (3.9 dropped)."""
    workflow = _load_workflow("_tests.yml")
    matrix = workflow["jobs"]["run-tests"]["strategy"]["matrix"]

    assert matrix["python-version"] == ["3.10", "3.11", "3.12", "3.13"]
    assert "3.9" not in matrix["python-version"]


def test_tests_workflow_windows_include_uses_latest_python() -> None:
    """Test that the windows-latest matrix include entry uses Python 3.13."""
    workflow = _load_workflow("_tests.yml")
    matrix = workflow["jobs"]["run-tests"]["strategy"]["matrix"]

    windows_entries = [entry for entry in matrix["include"] if entry.get("os") == "windows-latest"]
    assert len(windows_entries) == 1
    assert windows_entries[0]["python-version"] == "3.13"


def test_tests_workflow_uses_updated_setup_pdm_action() -> None:
    """Test that the setup-pdm GitHub Action was bumped to v4.2."""
    workflow = _load_workflow("_tests.yml")
    steps = workflow["jobs"]["run-tests"]["steps"]

    pdm_steps = [step for step in steps if str(step.get("uses", "")).startswith("pdm-project/setup-pdm")]
    assert len(pdm_steps) == 1
    assert pdm_steps[0]["uses"] == "pdm-project/setup-pdm@v4.2"


@pytest.mark.parametrize(  # type: ignore
    "workflow_name,job_name",
    [
        ("ci-dev.yml", "generate-docs"),
        ("ci-prod.yml", "generate-docs"),
        ("generate-dev-docs.yml", "generate-docs-test"),
    ],
)
def test_docs_workflows_generate_social_cards_enabled(workflow_name: str, job_name: str) -> None:
    """Test that documentation workflows enable social card generation via env var."""
    workflow = _load_workflow(workflow_name)
    env = workflow["jobs"][job_name]["env"]

    assert "MKDOCS_GENERATE_SOCIAL_CARDS" in env
    assert env["MKDOCS_GENERATE_SOCIAL_CARDS"] is True


def test_manual_tests_workflow_is_valid_yaml() -> None:
    """Test that manual_tests.yml is syntactically valid YAML."""
    workflow = _load_workflow("manual_tests.yml")
    assert workflow is not None
    assert "jobs" in workflow


def test_manual_tests_workflow_newest_matrix_python_versions() -> None:
    """Test that the newest-dependencies job matrix drops Python 3.9 and adds 3.13."""
    workflow = _load_workflow("manual_tests.yml")
    matrix = workflow["jobs"]["run-tests-newest"]["strategy"]["matrix"]

    assert matrix["python-version"] == ["3.10", "3.11", "3.12", "3.13"]
    assert "3.9" not in matrix["python-version"]


def test_manual_tests_workflow_newest_matrix_include_entries() -> None:
    """Test that macos/windows include entries in the newest job use Python 3.13."""
    workflow = _load_workflow("manual_tests.yml")
    matrix = workflow["jobs"]["run-tests-newest"]["strategy"]["matrix"]

    for entry in matrix["include"]:
        assert entry["python-version"] == "3.13"


def test_manual_tests_workflow_oldest_matrix_python_version() -> None:
    """Test that the oldest-dependencies job now uses Python 3.10 as the floor."""
    workflow = _load_workflow("manual_tests.yml")
    matrix = workflow["jobs"]["run-tests-oldest"]["strategy"]["matrix"]

    assert matrix["python-version"] == ["3.10"]
    assert "3.9" not in matrix["python-version"]


def test_all_workflow_files_are_parseable() -> None:
    """Test that every workflow file referenced in this PR parses as valid YAML."""
    for workflow_name in [
        "_tests.yml",
        "ci-dev.yml",
        "ci-prod.yml",
        "generate-dev-docs.yml",
        "manual_tests.yml",
    ]:
        workflow = _load_workflow(workflow_name)
        assert isinstance(workflow, dict)
        assert "jobs" in workflow