# Kokoro TTS API

Self-hosted English text-to-speech built on [Kokoro-82M](https://github.com/hexgrad/kokoro),
with an OpenAI-compatible endpoint, word-level timestamps, streaming, and a
browser UI. Runs CPU-native on Intel macOS or GPU-accelerated in Docker.

## Quick start — macOS (CPU)

```bash
./scripts/setup_mac.sh
.venv/bin/python -m app
```

Open <http://127.0.0.1:8080/>.

The pins in `requirements-mac-cpu.txt` are deliberate: `torch==2.2.2` is the last
release with an Intel-Mac wheel, and `numpy<2` is what that torch was built
against. Do not float them, and do not move `numpy<2` up into
`requirements-base.txt` — the CUDA image ships numpy 2.x and must stay uncapped.

Expect a real-time factor around 0.65 on a 2016 quad-core CPU — measured
0.62–0.68 across short and long inputs, i.e. **faster than real time**: roughly
6–7 seconds of compute per 10 seconds of audio. Add about 2.5 seconds of one-off
model load and warm-up at startup. The streaming endpoint and the UI still
matter, because they start playing on the first segment instead of waiting for
the whole request.

## Quick start — Windows 11 + NVIDIA GPU (Docker)

Prerequisites: Docker Desktop with the WSL2 backend, and a current NVIDIA
driver. No CUDA toolkit is needed on the host.

```powershell
cd docker
docker compose -f docker-compose.gpu.yml up --build -d
curl http://localhost:8080/health
```

`"device": "cuda"` in the response means the GPU is in use. Open
<http://localhost:8080/> for the UI.

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/speech` | OpenAI-compatible. Drop-in for OpenAI TTS clients. |
| `POST /tts` | Native. Voice blending, word timestamps. |
| `POST /tts/stream` | NDJSON stream of PCM chunks + timings. |
| `GET /voices` | The 28 English voices with quality grades. |
| `GET /health` | Device, backend, warm-up time, resolved `models_dir`. |
| `GET /docs`, `GET /redoc` | Swagger UI and ReDoc, served offline. |

Interactive docs are served from assets vendored under `web/vendor/`, not from a
CDN, so `/docs` and `/redoc` work inside the offline container. Refresh them
with `.venv/bin/python scripts/fetch_docs_assets.py` (it pins exact versions of
`swagger-ui-dist` and `redoc`, and the downloaded files are committed on
purpose).

### OpenAI-compatible

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","input":"Hello from Kokoro.","voice":"nova","response_format":"mp3"}' \
  --output hello.mp3
```

OpenAI voice names map onto Kokoro voices: `alloy`, `echo`, `fable`, `onyx`,
`nova`, `shimmer`. Any real Kokoro id works too. `model` is accepted and ignored.

### Native, with word timestamps

```bash
curl -X POST http://127.0.0.1:8080/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello there.","voice":"af_heart","include_timestamps":true}'
```

```json
{
  "audio": "<base64 wav>",
  "format": "wav",
  "sample_rate": 24000,
  "duration": 1.525,
  "voice": "af_heart",
  "segments": 1,
  "phonemes": "həlˈO ðˈɛɹ.",
  "words": [{"word": "Hello", "start": 0.375, "end": 0.675},
            {"word": "there", "start": 0.675, "end": 1.25},
            {"word": ".", "start": 1.25, "end": 1.425}]
}
```

`segments` is how many chunks Kokoro split the text into, and `phonemes` is the
G2P output for the whole request. Sentence-final punctuation gets its own timed
entry in `words`, because misaki times it — nothing filters it out.

Without `include_timestamps`, the response is raw audio bytes.

### Voice blending

```bash
curl -X POST http://127.0.0.1:8080/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"A blended voice.","voice":"af_bella:0.6,af_sky:0.4"}' \
  --output blend.wav
```

Up to four voices. Weights are normalized; omitting them averages equally
(`af_bella,af_sky`).

### Streaming

```bash
curl -N -X POST http://127.0.0.1:8080/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"First line.\nSecond line."}'
```

One JSON object per line: a `meta` line, a `chunk` line per segment (base64
float32 PCM at 24 kHz plus absolute word timings), then `done`. The `format`
field is ignored here — chunks are always raw PCM.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `KOKORO_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `KOKORO_DEFAULT_VOICE` | `af_heart` | Voice when a request omits one |
| `KOKORO_MAX_CHARS` | `5000` | Per-request input cap |
| `KOKORO_MAX_CONCURRENCY` | 1 (CPU) / 2 (CUDA) | Concurrent synthesis permits |
| `KOKORO_TORCH_THREADS` | torch default | `torch.set_num_threads` |
| `KOKORO_VOICE_CACHE_SIZE` | `32` | Blended-voice tensors kept in memory (LRU) |
| `KOKORO_API_KEY` | unset | Requires `Authorization: Bearer` when set |
| `KOKORO_HOST` / `KOKORO_PORT` | `127.0.0.1` / `8080` | Bind address |
| `KOKORO_ALLOW_ORIGINS` | unset | Comma-separated CORS origins |
| `HF_HOME` | `<repo>/models` | Weights/voices cache location |

`KOKORO_DEFAULT_VOICE` is validated at startup: a typo aborts the boot naming
the bad value, rather than turning every voice-less request into a 400.

Binding to the LAN? Set `KOKORO_API_KEY` as well — `/health` stays public, every
synthesis route requires the bearer token.

## Model weights

Weights live inside the project, at `models/`, not in the shared
`~/.cache/huggingface`. That's `kokoro-v1_0.pth` (312 MB) plus 28 voice packs
under `voices/*.pt` — about 326 MB total. Importing `app` points
`HF_HOME` at `<repo>/models` before anything imports `huggingface_hub`
(see `app/__init__.py`), so this happens automatically — no manual step is
normally needed.

Check where they actually landed: `GET /health` reports the resolved cache as
`models_dir`, and startup logs it. If it resolves outside the project and you
did not set `HF_HOME` yourself, startup logs a WARNING — that means something
imported `huggingface_hub` before `app`, and weights would be re-downloaded
into `~/.cache/huggingface` instead.

**Automatic download.** The first time the app starts, or the first time it
synthesizes, `huggingface_hub` fetches whatever isn't already on disk into
`models/`. Expect a one-time delay of a minute or more on a normal connection.

**Pre-download explicitly** (recommended before the first real request, and
what CI/Docker builds do):

```bash
.venv/bin/python scripts/bake_assets.py
```

`scripts/setup_mac.sh` already runs this for you, so macOS setups get it for
free.

**Manual download (air-gapped machines).** Fetch these files from
<https://huggingface.co/hexgrad/Kokoro-82M>:

- `config.json`
- `kokoro-v1_0.pth`
- `voices/*.pt` (all 28 English voice packs)

Place them under a HuggingFace-cache-shaped directory so the loader finds
them — `<snapshot-id>` can be any string, e.g. `manual`:

```
models/hub/models--hexgrad--Kokoro-82M/
└── snapshots/
    └── <snapshot-id>/
        ├── config.json
        ├── kokoro-v1_0.pth
        └── voices/
            ├── af_heart.pt
            └── ... (28 files total)
```

**Relocating the cache.** Set `HF_HOME` to move weights elsewhere — it is read
at import time, so export it before starting the app:

```bash
export HF_HOME=/path/to/cache
.venv/bin/python -m app
```

`models/` is gitignored and excluded from the Docker build context
(`.dockerignore`) — it is 326 MB of binary weights that should never be
committed, and the Docker image bakes its own copy into `/opt/hf` at build
time instead.

## Tests

```bash
.venv/bin/pytest                              # fast: no model load, seconds
KOKORO_RUN_SLOW=1 .venv/bin/pytest -m slow    # loads the real model
```

## Troubleshooting

- **`KOKORO_DEVICE=cuda but torch reports no CUDA device`** — the container did
  not get the GPU. Check `nvidia-smi` on the host and that Docker Desktop's WSL2
  backend is enabled.
- **Synthesis feels slow on the Mac** — CPU synthesis runs at about 0.65× real
  time, so a long request still takes seconds of wall clock even though compute
  is faster than the audio it produces. Use `/tts/stream` so playback starts on
  the first segment, or point the UI at the GPU box. The first request after
  startup also pays a one-off model load and warm-up.
- **`OSError: [E050] Can't find model 'en_core_web_sm'`** — run
  `.venv/bin/python -m spacy download en_core_web_sm`.
- **Timestamps drift slightly** — Kokoro derives them from predicted phoneme
  durations, so they are approximate by design. Whole-word highlighting absorbs it.
