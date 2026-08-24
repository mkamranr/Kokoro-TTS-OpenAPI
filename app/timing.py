"""Turning Kokoro's segment-relative token timings into one absolute timeline.

Kokoro populates MToken.start_ts / .end_ts via KPipeline.join_timestamps, but
each yielded segment starts its clock over near zero. TimelineAccumulator adds
the elapsed audio duration so the words form a single monotonic timeline.
"""
from typing import Iterable

from app.types import WordTiming


def words_from_tokens(tokens: Iterable) -> list[WordTiming]:
    """Extract timed words from misaki MTokens (duck-typed for testability).

    The rule is timestamps, nothing else: a token is skipped when start_ts or
    end_ts is None, or when its text is blank after stripping. Whether it has
    phonemes is irrelevant.

    In particular, misaki DOES time sentence-final punctuation, so a token like
    "." legitimately carries timestamps and legitimately appears in the API's
    `words` array. Filtering punctuation out here (or in the engine, or in the
    route) would throw away real timing data and contradict the spec.
    """
    words: list[WordTiming] = []
    for token in tokens:
        start = getattr(token, "start_ts", None)
        end = getattr(token, "end_ts", None)
        if start is None or end is None:
            continue
        text = (getattr(token, "text", "") or "").strip()
        if not text:
            continue
        start = float(start)
        end = float(end)
        words.append(WordTiming(text, start, max(start, end)))
    return words


class TimelineAccumulator:
    """Shifts each segment's timings by the audio duration already emitted."""

    def __init__(self) -> None:
        self._offset = 0.0

    @property
    def total(self) -> float:
        return self._offset

    def add(self, words: list[WordTiming], duration: float) -> list[WordTiming]:
        offset = self._offset
        shifted = [
            WordTiming(w.word, w.start + offset, w.end + offset) for w in words
        ]
        self._offset = offset + duration
        return shifted
