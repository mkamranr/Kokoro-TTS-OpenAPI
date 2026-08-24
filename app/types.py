"""Plain data types shared by every layer. Deliberately torch-free."""
from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np

SAMPLE_RATE = 24000


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float

    def as_dict(self) -> dict:
        return {"word": self.word, "start": round(self.start, 3), "end": round(self.end, 3)}


@dataclass
class Segment:
    index: int
    audio: np.ndarray  # float32, mono, SAMPLE_RATE
    words: list[WordTiming] = field(default_factory=list)
    phonemes: str = ""

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


@dataclass
class Synthesis:
    audio: np.ndarray
    sample_rate: int
    duration: float
    words: list[WordTiming]
    phonemes: str
    voice: str
    segments: int


class EngineProtocol(Protocol):
    """What the routes need from a synthesizer. KokoroEngine and FakeEngine both satisfy it."""

    sample_rate: int

    def info(self) -> dict: ...

    def iter_segments(
        self, text: str, voice: str, lang: str | None, speed: float
    ) -> Iterator[Segment]: ...
