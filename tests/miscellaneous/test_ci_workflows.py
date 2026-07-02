"""Tests validating GitHub Actions workflow files and pre-commit configuration."""

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# GitHub Actions files use `on:` as a trigger key. PyYAML (following YAML 1.1) interprets the
# unquoted `on` scalar as the boolean `True`, so loaded documents expose the trigger definition
# under the `True` key instead of the string `"on"`.
ON_KEY = True


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_matrix_python_versions(job: dict) -> list[str]:
    return job["strategy"]["matrix"]["python-version"]


def _find_setup_pdm_steps(steps: list[dict]) -> list[dict]:
    prefix = "pdm-project/setup-pdm"
    return [step for step in steps if str(step.get("uses", "")).startswith(prefix)]


class TestReusableTestsWorkflow:
    """Tests for `.github/workflows/_tests.yml`."""

    @pytest.fixture
    def workflow(self) -> dict:
        """Load and parse the `_tests.yml` workflow file."""
        return _load_yaml(WORKFLOWS_DIR / "_tests.yml")

    def test_is_valid_yaml(self, workflow: dict) -> None:
        """Test that the workflow file parses to a non-empty mapping."""
        assert isinstance(workflow, dict)
        assert workflow

    def test_is_reusable_workflow(self, workflow: dict) -> None:
        """Test that the workflow is exposed as a reusable `workflow_call`."""
        assert ON_KEY in workflow
        assert workflow[ON_KEY] == "workflow_call"

    def test_python_version_matrix_excludes_39_includes_313(self, workflow: dict) -> None:
        """Test that the matrix drops Python 3.9 support and adds Python 3.13."""
        job = workflow["jobs"]["run-tests"]
        python_versions = _find_matrix_python_versions(job)

        assert "3.9" not in python_versions
        assert python_versions == ["3.10", "3.11", "3.12", "3.13"]

    def test_windows_runner_uses_python_313(self, workflow: dict) -> None:
        """Test that the Windows-specific matrix include uses Python 3.13."""
        job = workflow["jobs"]["run-tests"]
        includes = job["strategy"]["matrix"]["include"]

        windows_entries = [entry for entry in includes if entry.get("os") == "windows-latest"]
        assert len(windows_entries) == 1
        assert windows_entries[0]["python-version"] == "3.13"

    def test_setup_pdm_action_pinned_version(self, workflow: dict) -> None:
        """Test that the `pdm-project/setup-pdm` action is pinned to v4.2."""
        steps = workflow["jobs"]["run-tests"]["steps"]
        pdm_steps = _find_setup_pdm_steps(steps)

        assert len(pdm_steps) == 1
        assert pdm_steps[0]["uses"] == "pdm-project/setup-pdm@v4.2"


class TestManualTestsWorkflow:
    """Tests for `.github/workflows/manual_tests.yml`."""

    @pytest.fixture
    def workflow(self) -> dict:
        """Load and parse the `manual_tests.yml` workflow file."""
        return _load_yaml(WORKFLOWS_DIR / "manual_tests.yml")

    def test_is_valid_yaml(self, workflow: dict) -> None:
        """Test that the workflow file parses to a non-empty mapping."""
        assert isinstance(workflow, dict)
        assert workflow

    def test_newest_dependencies_matrix(self, workflow: dict) -> None:
        """Test that the newest-dependencies job matrix excludes Python 3.9."""
        job = workflow["jobs"]["run-tests-newest"]
        python_versions = _find_matrix_python_versions(job)

        assert "3.9" not in python_versions
        assert python_versions == ["3.10", "3.11", "3.12", "3.13"]

    def test_newest_dependencies_extra_os_use_313(self, workflow: dict) -> None:
        """Test that macOS/Windows matrix includes all pin Python 3.13."""
        job = workflow["jobs"]["run-tests-newest"]
        includes = job["strategy"]["matrix"]["include"]

        assert len(includes) == 3
        assert all(entry["python-version"] == "3.13" for entry in includes)

    def test_oldest_dependencies_matrix_uses_310(self, workflow: dict) -> None:
        """Test that the oldest-dependencies job now targets Python 3.10 (not 3.9)."""
        job = workflow["jobs"]["run-tests-oldest"]
        python_versions = _find_matrix_python_versions(job)

        assert python_versions == ["3.10"]
        assert "3.9" not in python_versions


@pytest.mark.parametrize(
    "workflow_file",
    [
        "ci-dev.yml",
        "ci-prod.yml",
        "generate-dev-docs.yml",
    ],
)
def test_social_cards_env_flag_present(workflow_file: str) -> None:
    """Test that doc-generation workflows opt into social card generation."""
    workflow = _load_yaml(WORKFLOWS_DIR / workflow_file)

    generate_docs_jobs = [
        job for name, job in workflow["jobs"].items() if "env" in job and "generate-docs" in name
    ]
    assert generate_docs_jobs, f"No job with an `env` block found in {workflow_file}"

    for job in generate_docs_jobs:
        assert job["env"].get("MKDOCS_GENERATE_SOCIAL_CARDS") is True


@pytest.mark.parametrize(
    "workflow_file",
    [
        "_tests.yml",
        "ci-dev.yml",
        "ci-prod.yml",
        "generate-dev-docs.yml",
        "manual_tests.yml",
    ],
)
def test_workflow_files_are_parseable(workflow_file: str) -> None:
    """Test that all touched workflow files are syntactically valid YAML mappings."""
    workflow = _load_yaml(WORKFLOWS_DIR / workflow_file)

    assert isinstance(workflow, dict)
    assert "name" in workflow
    assert "jobs" in workflow
    assert workflow["jobs"]


def test_ci_prod_still_references_older_pdm_setup_action() -> None:
    """Regression test: only `_tests.yml` was bumped to `setup-pdm@v4.2` in this change.

    The `build-n-publish` job in `ci-prod.yml` was left untouched by this PR and should still be
    on the previous `pdm-project/setup-pdm@v3` pin. If this ever changes it should be a deliberate
    decision, not an accidental omission.
    """
    workflow = _load_yaml(WORKFLOWS_DIR / "ci-prod.yml")
    steps = workflow["jobs"]["build-n-publish"]["steps"]
    pdm_steps = _find_setup_pdm_steps(steps)

    assert len(pdm_steps) == 1
    assert pdm_steps[0]["uses"] == "pdm-project/setup-pdm@v3"


class TestPreCommitConfig:
    """Tests for `.pre-commit-config.yaml`."""

    @pytest.fixture
    def config(self) -> dict:
        """Load and parse the pre-commit configuration file."""
        return _load_yaml(REPO_ROOT / ".pre-commit-config.yaml")

    def _find_hook(self, config: dict, repo_substring: str, hook_id: str) -> dict:
        for repo in config["repos"]:
            if repo_substring in repo["repo"]:
                for hook in repo["hooks"]:
                    if hook["id"] == hook_id:
                        return hook
        raise AssertionError(f"Hook {hook_id!r} from repo containing {repo_substring!r} not found")

    def test_is_valid_yaml(self, config: dict) -> None:
        """Test that the pre-commit config parses to a non-empty mapping."""
        assert isinstance(config, dict)
        assert config["repos"]

    def test_refurb_targets_python_310(self, config: dict) -> None:
        """Test that refurb is configured for the new Python 3.10 minimum."""
        refurb_hook = self._find_hook(config, "refurb", "refurb")
        args = refurb_hook["args"]

        assert "--python-version" in args
        version_index = args.index("--python-version") + 1
        assert args[version_index] == "3.10"
        assert "3.9" not in args