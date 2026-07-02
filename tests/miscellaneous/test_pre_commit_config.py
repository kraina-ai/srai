"""Tests validating the .pre-commit-config.yaml configuration file."""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_COMMIT_CONFIG_PATH = REPO_ROOT / ".pre-commit-config.yaml"


def _load_pre_commit_config() -> dict[str, Any]:
    """Load and parse the pre-commit configuration file."""
    with PRE_COMMIT_CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _find_hook(config: dict[str, Any], repo_substring: str, hook_id: str) -> dict[str, Any]:
    """Find a specific hook definition by repo URL substring and hook id."""
    for repo in config["repos"]:
        if repo_substring in repo["repo"]:
            for hook in repo["hooks"]:
                if hook["id"] == hook_id:
                    return hook
    raise AssertionError(f"Hook {hook_id} not found for repo containing {repo_substring!r}")


def test_pre_commit_config_is_valid_yaml() -> None:
    """Test that .pre-commit-config.yaml is syntactically valid YAML."""
    config = _load_pre_commit_config()
    assert config is not None
    assert "repos" in config


def test_refurb_hook_targets_python_3_10() -> None:
    """Test that the refurb hook was updated to target Python 3.10 (was 3.9)."""
    config = _load_pre_commit_config()
    refurb_hook = _find_hook(config, "dosisod/refurb", "refurb")

    args = refurb_hook["args"]
    assert "--python-version" in args
    version_index = args.index("--python-version") + 1
    assert args[version_index] == "3.10"
    assert args[version_index] != "3.9"


def test_refurb_hook_keeps_other_args() -> None:
    """Test that unrelated refurb args were preserved during the version bump."""
    config = _load_pre_commit_config()
    refurb_hook = _find_hook(config, "dosisod/refurb", "refurb")

    args = refurb_hook["args"]
    assert "--disable" in args
    assert "FURB184" in args
    assert "--format" in args
    assert "github" in args