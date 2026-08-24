#!/usr/bin/env bash
# Idempotent CPU-only setup for Intel macOS. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3.10}"

echo "==> checking the environment"
arch="$(uname -m)"
echo "    arch:   $arch"
# WARN, deliberately not a hard failure. The pins in requirements-mac-cpu.txt
# (torch 2.2.2, the last Intel-Mac wheel, and its numpy<2 companion) simply do
# not apply on Apple Silicon: that machine can run a current torch perfectly
# well and should not be blocked by a script whose constraints are not its own.
if [ "$arch" != "x86_64" ]; then
  echo
  echo "    WARNING: this script targets Intel macOS (x86_64), not $arch."
  echo "    WARNING: requirements-mac-cpu.txt pins torch==2.2.2 + numpy<2 for the"
  echo "    WARNING: last Intel-Mac wheel. On $arch, install a current torch"
  echo "    WARNING: instead (pip install torch) and skip those two pins."
  echo "    WARNING: continuing anyway -- the rest of the setup is arch-agnostic."
  echo
fi
"$PYTHON" -c 'import sys; v=sys.version_info; \
  sys.exit(0 if (3,10) <= (v.major,v.minor) < (3,13) else 1)' || {
  echo "ERROR: need Python 3.10-3.12, got $("$PYTHON" -V 2>&1)"; exit 1; }
echo "    python: $("$PYTHON" -V 2>&1)"

free_gb="$(df -g . | awk 'NR==2 {print $4}')"
echo "    free:   ${free_gb}Gi"
if [ "$free_gb" -lt 4 ]; then
  echo "ERROR: need at least 4Gi free for torch + weights, have ${free_gb}Gi"
  exit 1
fi

echo "==> creating .venv"
[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip

echo "==> installing dependencies (torch 2.2.2 is the last Intel-Mac wheel)"
.venv/bin/pip install -r requirements-mac-cpu.txt
.venv/bin/pip install -r requirements-dev.txt

echo "==> downloading the spaCy tagger misaki's English G2P needs"
.venv/bin/python -m spacy download en_core_web_sm

echo "==> pre-downloading Kokoro weights and all 28 English voices"
.venv/bin/python scripts/bake_assets.py

echo "==> verifying with a real synthesis"
.venv/bin/python - <<'PY'
import time
import numpy as np
import soundfile as sf
from app.engine import KokoroEngine

engine = KokoroEngine(device="cpu")
text = "Kokoro is running locally on this machine."
started = time.perf_counter()
segments = list(engine.iter_segments(text, "af_heart", "a", 1.0))
elapsed = time.perf_counter() - started

audio = np.concatenate([s.audio for s in segments])
duration = len(audio) / engine.sample_rate
sf.write("/tmp/kokoro_hello.wav", audio, engine.sample_rate)
words = [w.word for s in segments for w in s.words]

print(f"    audio:   {duration:.2f}s written to /tmp/kokoro_hello.wav")
print(f"    compute: {elapsed:.2f}s  (real-time factor {elapsed / duration:.2f})")
print(f"    words:   {len(words)} timed -> {' '.join(words)}")
assert np.max(np.abs(audio)) > 0.05, "synthesis produced silence"
PY

echo
echo "Setup complete. Listen:  afplay /tmp/kokoro_hello.wav"
echo "Start the API:           .venv/bin/python -m app"
echo "Open the UI:             http://127.0.0.1:8080/"
