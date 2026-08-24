"""Kokoro TTS API.

Model weights live inside the project by default: importing `app` points the
HuggingFace cache at <repo>/models unless HF_HOME is already set (the Docker
image sets it to /opt/hf). This runs before anything imports huggingface_hub,
which reads HF_HOME at import time.

That ordering is the whole guarantee, and it is easy to break: a process that
imports `huggingface_hub` or `kokoro` WITHOUT importing `app` first silently
falls back to ~/.cache/huggingface/hub and re-downloads weights there. So the
resolved cache is logged at engine startup, reported by GET /health as
`models_dir`, and a WARNING is emitted when it lands outside the project
without the operator having asked for that.
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Captured BEFORE the setdefault below. Afterwards HF_HOME is always set, so
# "did the operator choose this?" becomes unanswerable -- and that is exactly
# what tells a legitimate override (Docker's HF_HOME=/opt/hf) apart from the
# leak we warn about.
HF_HOME_WAS_EXPLICIT = "HF_HOME" in os.environ
os.environ.setdefault("HF_HOME", str(MODELS_DIR))


def resolved_cache_dir() -> str:
    """Where huggingface_hub will actually store weights.

    Read from huggingface_hub rather than recomputed here, so it reports the
    truth instead of our guess at it. The import is deliberately lazy: it must
    happen after the setdefault above, and module import of `app` must stay
    cheap for the fast test suite.
    """
    from huggingface_hub import constants

    return str(constants.HF_HUB_CACHE)


def _is_inside_project(path: str) -> bool:
    try:
        return Path(path).resolve().is_relative_to(PROJECT_ROOT)
    except (OSError, ValueError):
        return False


def log_cache_location() -> str:
    """Log the resolved cache dir, WARNING if it looks like the leak.

    Returns the path so callers can report it without resolving it twice.
    """
    cache = resolved_cache_dir()
    logger.info("HuggingFace cache: %s (HF_HOME=%s)", cache, os.environ["HF_HOME"])
    if not HF_HOME_WAS_EXPLICIT and not _is_inside_project(cache):
        logger.warning(
            "HuggingFace cache %s is OUTSIDE the project and HF_HOME was not set "
            "by the operator: something imported huggingface_hub before `app`, so "
            "weights will be re-downloaded there instead of into %s",
            cache,
            MODELS_DIR,
        )
    return cache
