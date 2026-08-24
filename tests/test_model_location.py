"""Model weights live in <repo>/models by default (see app/__init__.py).

HF_HOME is read by huggingface_hub at IMPORT time, so the default can only be
proven with a clean subprocess: the current test process already has `app`
imported (and HF_HOME set) by the time these tests run, so an in-process
assertion would just be re-checking this same process's already-set env var,
not the behaviour a fresh interpreter gets.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run(code: str, env: dict) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_hf_home_defaults_into_the_project_when_unset(monkeypatch):
    import os

    env = os.environ.copy()
    env.pop("HF_HOME", None)
    out = _run("import os, app; print(os.environ['HF_HOME'])", env)
    assert out == str(ROOT / "models")


def test_explicit_hf_home_is_respected(monkeypatch):
    """The Docker image sets HF_HOME=/opt/hf; our default must not override it."""
    import os

    env = os.environ.copy()
    env["HF_HOME"] = "/opt/hf"
    out = _run("import os, app; print(os.environ['HF_HOME'])", env)
    assert out == "/opt/hf"


def test_models_dir_points_at_the_repo_models_directory():
    import app

    assert app.MODELS_DIR == ROOT / "models"


def test_weights_are_present_at_the_expected_cache_path():
    """Fast existence/size check -- no model load, so it stays in the fast suite."""
    matches = list(
        ROOT.glob("models/hub/models--hexgrad--Kokoro-82M/snapshots/*/kokoro-v1_0.pth")
    )
    if not matches:
        pytest.skip("weights not baked yet; run scripts/bake_assets.py")
    weights = matches[0]
    size_mb = weights.stat().st_size / (1024 * 1024)
    assert size_mb > 300, f"expected >300MB, got {size_mb:.1f}MB"
