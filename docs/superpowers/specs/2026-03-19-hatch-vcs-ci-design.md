# Design: Migrate to hatch-vcs + CI/CD + Pre-commit

## Overview

Migrate zndraw-socketio from static versioning to git-tag-based versioning via hatch-vcs, add GitHub Actions for testing and publishing, and add pre-commit hooks for code quality.

## 1. hatch-vcs Migration (pyproject.toml)

Replace static `version = "0.1.5"` with dynamic VCS-based versioning.

**Changes to `pyproject.toml`:**
- Remove `version = "0.1.5"` from `[project]`
- Add `dynamic = ["version"]` to `[project]`
- Change `build-system.requires` from `["hatchling"]` to `["hatchling", "hatch-vcs"]`
- Add sections:
  ```toml
  [tool.hatch.version]
  source = "vcs"

  [tool.hatch.build.hooks.vcs]
  version-file = "src/zndraw_socketio/_version.py"
  ```

**Other changes:**
- Add `src/zndraw_socketio/_version.py` to `.gitignore` (generated at build time, not committed)

**Release workflow:** Tag with `v*` (e.g., `git tag v0.2.0`) to set the version. hatch-vcs derives the version from the most recent git tag.

## 2. Pytest Workflow (`.github/workflows/pytest.yaml`)

- **Triggers:** push to `main`, all pull requests
- **Matrix:** `ubuntu-latest`, Python 3.10, 3.11, 3.12, 3.13
- **Steps:**
  1. `actions/checkout@v4`
  2. `astral-sh/setup-uv@v5`
  3. `uv run --all-extras pytest`

## 3. Publish Workflow (`.github/workflows/publish.yaml`)

- **Triggers:** tags matching `v*`
- **Permissions:** `id-token: write` (for OIDC trusted publishing)
- **Environment:** `pypi`
- **Steps:**
  1. `actions/checkout@v4` with `fetch-depth: 0` (full history needed for hatch-vcs)
  2. `astral-sh/setup-uv@v5`
  3. `uv build`
  4. `pypa/gh-action-pypi-publish@release/v1`

**Prerequisite:** Configure trusted publishing on pypi.org to allow this GitHub repo/workflow.

## 4. Pre-commit Config (`.pre-commit-config.yaml`)

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

## Files Changed

| File | Action |
|---|---|
| `pyproject.toml` | Modify (hatch-vcs config) |
| `.gitignore` | Modify (add `_version.py`) |
| `.github/workflows/pytest.yaml` | Create |
| `.github/workflows/publish.yaml` | Create |
| `.pre-commit-config.yaml` | Create |
