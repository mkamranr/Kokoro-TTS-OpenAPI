# Kokoro TTS API — Design

**Date:** 2026-08-24
**Status:** Approved (design), pending implementation plan

## 1. Purpose

A self-hosted HTTP API that turns English text into speech using
[Kokoro-82M](https://github.com/hexgrad/kokoro), plus a browser UI for
driving it by hand. One codebase runs in two places:

1. Natively, CPU-only, on the author's 2016 Intel MacBook Pro (development).
2. As a CUDA Docker container on a Windows 11 host with an RTX 2070 (fast path).

Application code is identical in both; only the requirements file and the
launcher differ.

## 2. Non-goals

- Languages other than English. Kokoro ships 54 voices across 9 languages;
  this service exposes only the 28 English ones (`a*` US, `b*` UK). Other
  languages need different misaki G2P extras and are out of scope.
- Voice cloning or fine-tuning.
- Multi-tenancy, quotas, or usage accounting.
- Horizontal scaling / job queues. Single process, in-process model.

## 3. Verified environment facts

Everything below was checked on the target Mac, not assumed.

| Fact | Value | Consequence |
|---|---|---|
| CPU | Intel i7-6820HQ, x86_64 | No MPS. CPU inference only. |
| macOS | 12.7.6 (Monterey) | Caps several wheel choices — see below. |
| RAM / free disk | 16 GB / 19 GB free | ~2.5 GB budget for venv + weights is fine. |
| Python | 3.10.4 | Satisfies `kokoro`'s `>=3.10,<3.13`. |
| `torch` | `2.2.2` is the **last** release with a `macosx_10_9_x86_64` cp310 wheel | Pin exactly `torch==2.2.2` on the Mac. |
| `onnxruntime` | macOS wheels moved to `macosx_13_0` at 1.20; newest Monterey-installable is 1.19.2, but `kokoro-onnx` needs `>=1.20.1` | **Rejected the ONNX backend.** PyTorch is the only clean Mac path. |
| `espeakng-loader` | ships `macosx_10_12_x86_64` wheel | No Homebrew espeak-ng install needed. |
| `soundfile` / `lameenc` | `macosx_10_9_x86_64` / `macosx_10_9_universal2` cp310 wheels | WAV + MP3 with **no ffmpeg** on either host. |
| Docker | CLI present, daemon not running | Mac path must not depend on Docker. |

## 4. Architecture

```
kokoro-api/
├── app/
│   ├── main.py              # app factory, lifespan model load, router mount, static mount
│   ├── config.py            # Settings (pydantic-settings), env-driven
│   ├── schemas.py           # pydantic request/response models
│   ├── engine.py            # KokoroEngine — the only module that imports kokoro/torch
│   ├── voices.py            # 28-voice catalog, OpenAI aliases, blend spec parsing
│   ├── audio.py             # WAV/MP3 encoding, PCM concat, silence padding
│   ├── errors.py            # OpenAI-shaped error envelope + handlers
│   └── routes/
│       ├── openai.py        # POST /v1/audio/speech
│       ├── native.py        # POST /tts, POST /tts/stream, GET /voices
│       └── health.py        # GET /health
├── web/                     # static UI: index.html, app.js, styles.css (no build step)
├── tests/
├── scripts/setup_mac.sh
├── docker/Dockerfile.cuda, docker/docker-compose.gpu.yml
├── requirements-mac-cpu.txt
├── requirements-gpu.txt
└── README.md
```

The boundary that matters: **`engine.py` is the only module that imports
`torch` or `kokoro`.** Everything else deals in plain numpy arrays,
dataclasses, and dicts, so the entire route/validation/encoding layer is
testable without the 350 MB model.

## 5. Engine

### 5.1 Loading and device selection

One `KokoroEngine` per process, built during FastAPI lifespan startup and
stored on `app.state`. Device from `KOKORO_DEVICE` = `auto|cpu|cuda`;
`auto` resolves to `cuda` when `torch.cuda.is_available()` else `cpu`.

Two `KPipeline`s are held, one per English lang code: `a` (US) and `b` (UK).
They share a single `KModel` instance (`KPipeline(lang_code=..., model=model)`)
so weights are loaded once, not twice.

A warm-up synthesis of a short fixed string runs at startup so the first real
request doesn't pay lazy-init cost; its wall time is recorded and reported by
`/health`.

### 5.2 Concurrency

Torch inference here is not safely reentrant and the Mac has 4 cores, so:

- One `asyncio.Semaphore`, size `KOKORO_MAX_CONCURRENCY` (default 1 on CPU, 2 on CUDA).
- Each synthesis runs in a worker thread via `asyncio.to_thread`, so the event
  loop keeps serving `/health` and static assets while a long synthesis runs.
- `torch.set_num_threads(KOKORO_TORCH_THREADS)` (default: physical core count)
  is set once at startup.

### 5.3 Voices and blending

`voices.py` holds the static catalog of all 28 English voices — id, display
name, gender, accent (`American`/`British`), lang code, and the quality grade
from upstream `VOICES.md`. Note: `af_heart` exists as `voices/af_heart.pt` on
HuggingFace but is absent from the `VOICES.md` grade table; the catalog records
it as grade `A` and marks it the service default, matching upstream's own README
example.

Voice specs accepted by the API:

- `af_bella` — single voice.
- `af_bella,af_sky` — unweighted average (Kokoro's `load_voice` handles this natively).
- `af_bella:0.6,af_sky:0.4` — **weighted** blend, which Kokoro does not support.
  The engine loads each pack via `pipeline.load_single_voice`, normalizes the
  weights to sum to 1, computes the weighted sum of the `[510, 1, 256]` tensors,
  and passes the resulting tensor to `KPipeline.__call__(voice=<tensor>)`.

  **Gotcha to encode in the implementation:** `KPipeline.load_voice` returns a
  passed-in tensor only when `isinstance(voice, torch.FloatTensor)`, which is
  false for CUDA tensors. Blend tensors must therefore always be built and
  passed as **CPU** float tensors; `__call__` does its own `.to(model.device)`.

Blend results are cached by their canonical spec string, bounded to
`KOKORO_VOICE_CACHE_SIZE` (default 32) entries.

### 5.4 Segmentation

No custom chunker. `KPipeline.__call__` already splits input on `\n+`, and
`en_tokenize` already packs segments to ≤510 phoneme tokens, warning and
truncating past that. The engine consumes the generator segment by segment,
which is exactly the granularity needed for both streaming and timestamps.

`KOKORO_MAX_CHARS` (default 5000) bounds total input length; over that is a
400, not a silent truncation.

### 5.5 Word-level timestamps

Kokoro gives these away for English and the implementation must not
reinvent them:

- `KPipeline.infer` always calls the model with `return_output=True`, so every
  `Result` carries `output.pred_dur`.
- For `lang_code in 'ab'`, `__call__` then calls `KPipeline.join_timestamps(tks,
  pred_dur)`, which writes `.start_ts` / `.end_ts` onto each `misaki.en.MToken`.

Two adaptations are required:

1. **Segment offsetting.** Each `Result`'s timestamps restart near zero. The
   engine accumulates `offset += len(segment_audio) / 24000` (plus any inserted
   inter-segment silence) and adds `offset` to every token's start/end before
   emitting. This is the single most important correctness detail in the feature
   and gets a dedicated test.
2. **Token filtering.** Tokens with no phonemes (punctuation, whitespace) get no
   timestamps. Emit only tokens where `start_ts` and `end_ts` are both non-None,
   as `{"word": t.text, "start": float, "end": float}`.

The engine returns a `Synthesis` dataclass: `audio` (float32 numpy, 24 kHz),
`sample_rate`, `duration`, `words: list[WordTiming]`, `voice`, `segments`.

## 6. HTTP API

Sample rate is always 24000. Errors always use the OpenAI envelope
`{"error": {"message": ..., "type": ...}}` so both client styles handle
failures identically.

### 6.1 `POST /v1/audio/speech` — OpenAI-compatible

Body: `{model, input, voice, speed, response_format}`. `model` is accepted and
ignored (any value; `kokoro` is conventional). `response_format` ∈ `wav|mp3`;
OpenAI's other values (`opus`, `aac`, `flac`, `pcm`) return 400 with a message
naming the supported set, rather than silently substituting a format.
Returns raw audio bytes with the matching content type — no timestamps, because
OpenAI's schema has no place for them.

OpenAI voice names are aliased onto real Kokoro voices so existing clients work
untouched: `alloy→af_alloy`, `echo→am_echo`, `fable→bm_fable`, `onyx→am_onyx`,
`nova→af_nova`, `shimmer→af_sky`. Any real Kokoro id is also accepted here.

### 6.2 `POST /tts` — native

```json
{"text": "...", "voice": "af_heart", "lang": "a", "speed": 1.0,
 "format": "wav", "include_timestamps": false}
```

- `include_timestamps: false` → raw audio bytes (`audio/wav` | `audio/mpeg`).
- `include_timestamps: true` → `application/json`:
  `{"audio": "<base64>", "format", "sample_rate", "duration", "voice",
    "words": [{"word","start","end"}], "phonemes": "..."}`

Carrying audio as base64 in JSON is deliberate: it keeps one round trip for the
common "play it and highlight the words" case, and these are short clips.

### 6.3 `POST /tts/stream` — streaming NDJSON

Same request body. Responds `application/x-ndjson`, one JSON object per line,
flushed as each segment finishes:

```
{"type":"meta","sample_rate":24000,"voice":"af_heart","format":"pcm_f32le"}
{"type":"chunk","index":0,"audio":"<base64>","duration":1.84,"words":[...]}
{"type":"chunk","index":1,"audio":"<base64>","duration":2.10,"words":[...]}
{"type":"done","duration":3.94,"segments":2}
```

Chunk `words` are already absolute-offset. Chunk audio is **always** raw
float32 PCM at 24 kHz, base64-encoded: the request's `format` field is ignored
here, because per-segment WAV headers or MP3 frames would have to be stripped
and re-joined by the client for no gain. Clients wanting an encoded file use
`/tts` instead.

This endpoint exists because the Mac synthesizes at roughly real time — waiting
for a whole paragraph before the first byte would make the UI feel broken.

### 6.4 `GET /voices`

`{"voices": [{id, name, gender, accent, lang, grade, default}], "count": 28}`

### 6.5 `GET /health`

`{"status", "device", "backend", "torch_version", "model_loaded",
  "warmup_seconds", "voices": 28, "max_concurrency"}`

### 6.6 Errors

| Status | Condition |
|---|---|
| 400 | empty text, text over `KOKORO_MAX_CHARS`, unknown voice id, malformed blend spec, bad weights, `speed` outside 0.5–2.0, unsupported format, `lang` not `a`/`b` |
| 401 | `KOKORO_API_KEY` set and `Authorization: Bearer` missing/wrong |
| 503 | request arrives before warm-up completes |
| 500 | unexpected failure; logged with a request id that is echoed in the body |

### 6.7 Auth

Optional. Unset `KOKORO_API_KEY` (the default) means no auth, which is right
for `127.0.0.1`. When set, all endpoints except `/health` and the static UI
require the bearer token. The UI reads a key from a field in its own settings
panel and stores it in `localStorage`.

## 7. Web UI

Served at `/` from `web/` via `StaticFiles`. Vanilla HTML/CSS/JS — no npm, no
build step, no CDN, so it works offline inside the container.

Controls: text area (with character counter against `KOKORO_MAX_CHARS`), voice
picker grouped by accent+gender showing grades, US/UK toggle, speed slider
(0.5–2.0), format select, Speak button, and an "Advanced" disclosure holding the
weighted-blend builder (two voice pickers + a mix slider that writes the
`a:0.6,b:0.4` spec) and the API key field.

Output: an audio player, a Download button, a synthesis-stats line (time to
first audio, total time, audio duration, computed real-time factor), and the
**karaoke panel** — the input text re-rendered as word spans, highlighting the
active word as playback advances.

Playback and highlighting: the UI calls `/tts/stream`, decodes each chunk's
float32 PCM into a WebAudio `AudioBuffer`, and schedules chunks back-to-back on
the AudioContext timeline so playback starts as soon as chunk 0 lands. A
`requestAnimationFrame` loop maps `audioContext.currentTime` to the accumulated
word list by binary search and moves the highlight. Scheduled-chunk playback
also yields an exact clock for highlighting, which `<audio>` element timing does
not reliably give.

Fallback: if streaming fails or WebAudio is unavailable, the UI retries against
`/tts` with `include_timestamps: true` and plays a blob URL, driving the
highlight from `audio.currentTime`. This keeps the UI functional in the degraded
case and is worth the ~30 extra lines.

## 8. Configuration

All via environment, all with defaults, read once into a `Settings` object.

| Variable | Default | Meaning |
|---|---|---|
| `KOKORO_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `KOKORO_DEFAULT_VOICE` | `af_heart` | Used when a request omits `voice` |
| `KOKORO_MAX_CHARS` | `5000` | Per-request input cap |
| `KOKORO_MAX_CONCURRENCY` | `1` cpu / `2` cuda | Concurrent synthesis permits |
| `KOKORO_TORCH_THREADS` | physical cores | `torch.set_num_threads` |
| `KOKORO_API_KEY` | unset | Enables bearer auth when set |
| `KOKORO_VOICE_CACHE_SIZE` | `32` | Cached voice/blend packs |
| `KOKORO_HOST` / `KOKORO_PORT` | `127.0.0.1` / `8080` | Bind address |
| `KOKORO_ALLOW_ORIGINS` | `` | CORS origins, comma-separated |
| `HF_HOME` | platform default | Where weights/voices cache |

## 9. Testing

TDD throughout. The hard requirement: **the default suite must not load the
model** and must finish in seconds.

- **Fake engine.** The engine is provided through a FastAPI dependency, so tests
  override it with a fake that returns a deterministic sine wave and canned word
  timings. All route behaviour — schemas, aliases, auth, errors, content types,
  NDJSON framing — is tested against it.
- **Unit tests, no torch:** blend spec parsing (valid, malformed, weights not
  summing to 1, unknown ids); OpenAI voice aliasing; WAV header correctness and
  round-trip through `soundfile`; MP3 frame-sync validation; the timestamp
  offset accumulator (fed synthetic per-segment token lists, asserting
  monotonically increasing absolute times across segment boundaries); config
  precedence and defaults.
- **Integration test, marked slow:** loads the real model, synthesizes a known
  sentence, asserts non-silent audio, duration within a plausible band, word
  count matching the input, and monotonic non-overlapping timings. Runs only
  with `KOKORO_RUN_SLOW=1`, and is the acceptance gate on both machines.
- **UI:** a smoke test that `/` serves and referenced assets resolve. Interactive
  behaviour is verified by hand in a browser at the end of implementation; no JS
  test runner is being introduced for this.

## 10. Deployment

### 10.1 This Mac (CPU)

`scripts/setup_mac.sh` is idempotent and: verifies Python is 3.10–3.12 and the
arch is x86_64; creates `.venv`; installs `requirements-mac-cpu.txt` (pinned
`torch==2.2.2`, `kokoro==0.9.4`, `misaki[en]>=0.9.4`, `fastapi`, `uvicorn`,
`soundfile`, `lameenc`, `pydantic-settings`); runs `python -m spacy download
en_core_web_sm` (misaki's English G2P needs it for POS tagging); pre-downloads
weights and voices; then runs one synthesis to prove the install and prints the
measured real-time factor.

`transformers` is pinned to a release known to work against torch 2.2 rather
than floating, since current releases assume newer torch.

### 10.2 Windows 11 + RTX 2070 (CUDA, Docker)

`docker/Dockerfile.cuda` on `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
(Turing SM 7.5 is fully supported). `apt-get install espeak-ng` as belt and
braces alongside `espeakng-loader`. Model weights, voices, and the spacy model
are baked in at build time so the container starts with no network. Non-root
user, `HF_HOME` inside the image, healthcheck hitting `/health`.

`docker/docker-compose.gpu.yml` reserves the GPU via
`deploy.resources.reservations.devices` with `capabilities: [gpu]`, publishes
8080, and sets `KOKORO_DEVICE=cuda` plus `KOKORO_HOST=0.0.0.0` — the default
`127.0.0.1` bind would leave the published port unreachable from the Windows
host, so the compose file must override it. Host prerequisites: Docker Desktop with the
WSL2 backend and a current NVIDIA driver — no CUDA toolkit on the host.

The README documents both paths, including binding to `0.0.0.0` plus setting
`KOKORO_API_KEY` when exposing the service to the LAN.

## 11. Performance expectations

Setting these up front so the Mac's numbers don't read as a bug: Kokoro is 82M
parameters, and on a 2016 4-core Skylake laptop CPU a real-time factor near or
somewhat above 1.0 is expected — roughly 10–25 s of compute for 10 s of audio.
The streaming endpoint hides most of that latency. The RTX 2070 should be an
order of magnitude faster. `/health` reports warm-up time and the UI shows a
per-request RTF, so both machines are measurable rather than guessed at.

## 12. Risks

| Risk | Mitigation |
|---|---|
| `kokoro==0.9.4` or `transformers` needs torch > 2.2 at runtime on the Mac | Pin `transformers`; the setup script's warm-up synthesis catches it immediately. If unfixable, the Mac falls back to being a client of the GPU box — the API contract is unchanged. |
| Disk: 19 GB free, ~2.5 GB needed | Setup script checks free space before installing and fails early with a clear message. |
| `join_timestamps` carries an upstream `TODO` about its `-3` offset | Timings are approximate by nature; the UI highlights whole words, which tolerates tens of milliseconds of drift. Documented, not worked around. |
| Weighted blends aren't an upstream feature | Confined to the engine, guarded by the CPU-tensor gotcha above, and covered by tests. |
| WebAudio chunk scheduling is the fiddliest code in the project | Explicit non-streaming fallback path, kept working. |
