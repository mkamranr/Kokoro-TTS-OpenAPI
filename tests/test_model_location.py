"""Model weights live in <repo>/models by default (see app/__init__.py).

HF_HOME is read by huggingface_hub at IMPORT time, so the default can only be
proven with a clean subprocess: the current test process already has `app`
imported (and HF_HOME set) by the time these tests run, so an in-process
assertion would just be re-checking this same process's already-set env var,
not the behaviour a fresh interpreter gets.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# What huggingface_hub will actually use, which is what matters. HF_HOME being
# set is only a means to this end.
PRINT_CACHE = "import huggingface_hub.constants as c; print(c.HF_HUB_CACHE)"


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


def _clean_env() -> dict:
    """A child environment with HF_HOME removed.

    This process already has HF_HOME set (importing `app` did it), so inheriting
    it would test nothing: the child would "pass" on the parent's leftovers
    rather than on the default the code establishes.
    """
    env = os.environ.copy()
    env.pop("HF_HOME", None)
    env.pop("HF_HUB_CACHE", None)
    return env


def test_hf_home_defaults_into_the_project_when_unset(monkeypatch):
    out = _run("import os, app; print(os.environ['HF_HOME'])", _clean_env())
    assert out == str(ROOT / "models")


def test_importing_the_engine_resolves_the_cache_into_the_repo():
    """The invariant that keeps weights out of ~/.cache/huggingface.

    A stray ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M with one
    re-downloaded voice pack in it is what prompted this test: some process had
    imported huggingface_hub without importing `app` first. Asserting on the
    resolved HF_HUB_CACHE (not merely on HF_HOME) is what actually pins it.
    """
    out = _run(f"import app.engine; {PRINT_CACHE}", _clean_env())
    assert out == str(ROOT / "models" / "hub"), out
    assert Path(out).is_relative_to(ROOT)


def test_importing_the_bake_script_resolves_the_cache_into_the_repo():
    """Pins the fragile import order inside scripts/bake_assets.py.

    That script must import `app` before `huggingface_hub`. Reorder those two
    imports and it silently downloads 326 MB into the shared user cache -- which
    is exactly the bug this suite exists to catch, so it is tested through the
    real module rather than by eyeballing the import block.
    """
    out = _run(f"import scripts.bake_assets; {PRINT_CACHE}", _clean_env())
    assert out == str(ROOT / "models" / "hub"), out


def test_importing_huggingface_hub_alone_does_not_resolve_into_the_repo():
    """The leak, demonstrated: no `app` import means no project cache.

    Without this, the two tests above could pass for the wrong reason (e.g. a
    stale HF_HOME leaking in) and nobody would know.
    """
    out = _run(PRINT_CACHE, _clean_env())
    assert not Path(out).is_relative_to(ROOT), out


def test_an_explicit_hf_home_beats_the_project_default():
    """Docker sets HF_HOME=/opt/hf. A legitimate override must survive."""
    env = _clean_env()
    env["HF_HOME"] = "/opt/hf"
    out = _run(f"import app.engine; {PRINT_CACHE}", env)
    assert out == "/opt/hf/hub", out


def test_explicit_hf_home_is_respected(monkeypatch):
    """The Docker image sets HF_HOME=/opt/hf; our default must not override it."""
    env = os.environ.copy()
    env["HF_HOME"] = "/opt/hf"
    out = _run("import os, app; print(os.environ['HF_HOME'])", env)
    assert out == "/opt/hf"


def test_models_dir_points_at_the_repo_models_directory():
    import app

    assert app.MODELS_DIR == ROOT / "models"


def test_resolved_cache_dir_reports_the_real_cache():
    import app

    assert app.resolved_cache_dir() == str(ROOT / "models" / "hub")


def test_health_reports_the_resolved_models_dir(client):
    """The guarantee is observable instead of silent -- one curl proves it."""
    import app

    body = client.get("/health").json()
    assert body["models_dir"] == app.resolved_cache_dir()
    assert Path(body["models_dir"]).is_relative_to(ROOT)


def test_a_cache_outside_the_project_warns(monkeypatch, caplog):
    """The leak signature: outside the project, and nobody asked for that."""
    import app

    monkeypatch.setattr(app, "HF_HOME_WAS_EXPLICIT", False)
    monkeypatch.setattr(
        app, "resolved_cache_dir", lambda: "/home/someone/.cache/huggingface/hub"
    )
    with caplog.at_level("INFO", logger="app"):
        app.log_cache_location()
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "OUTSIDE the project" in warnings[0].getMessage()


def test_an_operator_set_hf_home_outside_the_project_does_not_warn(
    monkeypatch, caplog
):
    """Docker sets HF_HOME=/opt/hf on purpose. Warning there would be noise."""
    import app

    monkeypatch.setattr(app, "HF_HOME_WAS_EXPLICIT", True)
    monkeypatch.setattr(app, "resolved_cache_dir", lambda: "/opt/hf/hub")
    with caplog.at_level("INFO", logger="app"):
        app.log_cache_location()
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_the_normal_in_project_cache_does_not_warn(monkeypatch, caplog):
    import app

    monkeypatch.setattr(app, "HF_HOME_WAS_EXPLICIT", False)
    with caplog.at_level("INFO", logger="app"):
        cache = app.log_cache_location()
    assert cache == str(ROOT / "models" / "hub")
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("HuggingFace cache" in r.getMessage() for r in caplog.records)


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
