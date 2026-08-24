"""Kokoro TTS API.

Model weights live inside the project by default: importing `app` points the
HuggingFace cache at <repo>/models unless HF_HOME is already set (the Docker
image sets it to /opt/hf). This runs before anything imports huggingface_hub,
which reads HF_HOME at import time.
"""
import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
os.environ.setdefault("HF_HOME", str(MODELS_DIR))
