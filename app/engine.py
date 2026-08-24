"""Kokoro-82M inference. The ONLY module permitted to import torch or kokoro."""
import logging
import time
from collections import OrderedDict
from typing import Iterator

import numpy as np

from app.timing import words_from_tokens
from app.types import SAMPLE_RATE, Segment
from app.voices import DEFAULT_VOICE_ID, VOICES, parse_voice_spec

logger = logging.getLogger(__name__)

REPO_ID = "hexgrad/Kokoro-82M"
WARMUP_TEXT = "Kokoro is ready."
LANG_CODES = ("a", "b")


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "KOKORO_DEVICE=cuda but torch reports no CUDA device available"
        )
    return requested


class KokoroEngine:
    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        device: str = "auto",
        torch_threads: int = 0,
        voice_cache_size: int = 32,
    ):
        import torch
        from kokoro import KModel, KPipeline

        self._torch = torch
        self.device = resolve_device(device)
        if torch_threads > 0:
            torch.set_num_threads(torch_threads)

        self._model = KModel(repo_id=REPO_ID).to(self.device).eval()
        # One model, two pipelines: US and UK G2P share the same weights.
        self._pipelines = {
            code: KPipeline(lang_code=code, repo_id=REPO_ID, model=self._model)
            for code in LANG_CODES
        }
        self._cache_size = voice_cache_size
        self._blend_cache: "OrderedDict[str, object]" = OrderedDict()
        self.warmup_seconds = self._warmup()

    def _warmup(self) -> float:
        started = time.perf_counter()
        for _ in self.iter_segments(WARMUP_TEXT, DEFAULT_VOICE_ID, "a", 1.0):
            pass
        elapsed = time.perf_counter() - started
        logger.info("warmup synthesis took %.2fs on %s", elapsed, self.device)
        return elapsed

    def info(self) -> dict:
        return {
            "device": self.device,
            "backend": "pytorch",
            "torch_version": self._torch.__version__,
            "warmup_seconds": round(self.warmup_seconds, 3),
            "voices": len(VOICES),
        }

    def _voice_argument(self, spec: str, lang: str):
        """A plain voice id, or a CPU float32 blend tensor.

        Kokoro's load_voice averages voices equally; weighted blends are ours.
        The tensor MUST be CPU float32: KPipeline.load_voice only passes a
        tensor through when isinstance(voice, torch.FloatTensor), which is
        False for CUDA tensors.
        """
        components = parse_voice_spec(spec)
        if len(components) == 1:
            return components[0].voice_id

        cached = self._blend_cache.get(spec)
        if cached is not None:
            self._blend_cache.move_to_end(spec)
            return cached

        pipeline = self._pipelines[lang]
        packs = [
            pipeline.load_single_voice(c.voice_id).detach().cpu().float()
            for c in components
        ]
        stacked = self._torch.stack(packs)
        weights = self._torch.tensor(
            [c.weight for c in components], dtype=self._torch.float32
        ).view(-1, 1, 1, 1)
        blend = (stacked * weights).sum(dim=0).cpu().float()

        self._blend_cache[spec] = blend
        while len(self._blend_cache) > self._cache_size:
            self._blend_cache.popitem(last=False)
        return blend

    def iter_segments(
        self, text: str, voice: str, lang: str | None, speed: float
    ) -> Iterator[Segment]:
        code = lang if lang in LANG_CODES else "a"
        pipeline = self._pipelines[code]
        voice_argument = self._voice_argument(voice, code)

        index = 0
        for result in pipeline(text, voice=voice_argument, speed=speed):
            if result.audio is None:
                continue
            audio = result.audio.detach().cpu().numpy().astype(np.float32)
            # Timestamps here are segment-relative; SynthesisService offsets them.
            yield Segment(
                index=index,
                audio=audio,
                words=words_from_tokens(result.tokens or []),
                phonemes=result.phonemes or "",
                sample_rate=self.sample_rate,
            )
            index += 1
