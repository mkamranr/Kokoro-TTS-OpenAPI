"""A deterministic stand-in for KokoroEngine. No torch, no model, no downloads."""
import time
from typing import Iterator

import numpy as np

from app.types import SAMPLE_RATE, Segment, WordTiming

WORD_SECONDS = 0.3
GAP_SECONDS = 0.05


class FakeEngine:
    """Splits text on newlines into segments and on whitespace into timed words.

    Emits a 220 Hz tone whose length matches the words, so audio duration and
    word timings stay consistent with each other.
    """

    sample_rate = SAMPLE_RATE

    def __init__(self, segment_delay: float = 0.0, device: str = "fake"):
        self.segment_delay = segment_delay
        self.device = device
        self.calls: list[tuple] = []

    def info(self) -> dict:
        return {
            "device": self.device,
            "backend": "fake",
            "torch_version": "n/a",
            "warmup_seconds": 0.0,
            "voices": 28,
        }

    def iter_segments(
        self, text: str, voice: str, lang: str | None, speed: float
    ) -> Iterator[Segment]:
        self.calls.append((text, voice, lang, speed))
        lines = [line for line in text.split("\n") if line.strip()]
        for index, line in enumerate(lines):
            if self.segment_delay:
                time.sleep(self.segment_delay)
            words: list[WordTiming] = []
            cursor = 0.0
            for word in line.split():
                # Segment-relative on purpose: the service must make these absolute.
                words.append(WordTiming(word, cursor, cursor + WORD_SECONDS))
                cursor += WORD_SECONDS + GAP_SECONDS
            samples = max(1, int(cursor * SAMPLE_RATE))
            t = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
            audio = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
            yield Segment(
                index=index, audio=audio, words=words, phonemes=f"seg{index}"
            )
