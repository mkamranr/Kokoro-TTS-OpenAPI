"""Plain data types shared by every layer. Deliberately torch-free."""
from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np

SAMPLE_RATE = 24000


def audio_duration(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Samples to seconds. The ONE place this division happens.

    Segment.duration and Synthesis.duration used to divide independently --
    Segment by the module constant, Synthesis by engine.sample_rate. Equal
    today at 24000, but a divergence would silently skew every word offset,
    because Segment.duration is what TimelineAccumulator shifts by. Both now
    call this, and both pass the producing engine's rate (EngineProtocol.
    sample_rate), with SAMPLE_RATE as the sole default.
    """
    return len(audio) / sample_rate


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
    audio: np.ndarray  # float32, mono, at sample_rate
    words: list[WordTiming] = field(default_factory=list)
    phonemes: str = ""
    # Stamped by the producing engine, so duration and word offsets can never
    # be computed against a different rate than the audio was generated at.
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self) -> float:
        return audio_duration(self.audio, self.sample_rate)


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
