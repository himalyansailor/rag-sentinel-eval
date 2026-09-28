"""Detect git / CI metadata for a run. Never raises: provenance is best-effort."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping

from rag_sentinel.models import Provenance


def _git(*args: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [git, *args], capture_output=True, text=True, timeout=5, check=True
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip() or None


def detect_provenance(
    env: Mapping[str, str] | None = None, *, trigger: str | None = None
) -> Provenance:
    """Build :class:`Provenance` from GitHub Actions variables, falling back to local git.

    On pull requests GitHub sets ``GITHUB_HEAD_REF`` (the source branch); on pushes
    ``GITHUB_REF_NAME`` is the branch name.
    """
    env = os.environ if env is None else env
    if env.get("GITHUB_ACTIONS") == "true":
        run_url = None
        if all(k in env for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")):
            run_url = (
                f"{env['GITHUB_SERVER_URL']}/{env['GITHUB_REPOSITORY']}/actions/runs/"
                f"{env['GITHUB_RUN_ID']}"
            )
        return Provenance(
            git_sha=env.get("GITHUB_SHA"),
            git_ref=env.get("GITHUB_HEAD_REF") or env.get("GITHUB_REF_NAME"),
            trigger=trigger or f"ci:{env.get('GITHUB_EVENT_NAME', 'unknown')}",
            ci_run_url=run_url,
        )
    return Provenance(
        git_sha=_git("rev-parse", "HEAD"),
        git_ref=_git("rev-parse", "--abbrev-ref", "HEAD"),
        trigger=trigger or "local",
    )
