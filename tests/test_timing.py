from dataclasses import dataclass

import pytest

from app.timing import TimelineAccumulator, words_from_tokens
from app.types import WordTiming


@dataclass
class StubToken:
    """Stands in for misaki.en.MToken — same attribute names, no torch."""

    text: str
    phonemes: str | None = "x"
    start_ts: float | None = None
    end_ts: float | None = None
    whitespace: str = " "


def test_words_from_tokens_extracts_word_and_bounds():
    tokens = [
        StubToken("Hello", start_ts=0.0, end_ts=0.4),
        StubToken("world", start_ts=0.45, end_ts=0.9),
    ]
    assert words_from_tokens(tokens) == [
        WordTiming("Hello", 0.0, 0.4),
        WordTiming("world", 0.45, 0.9),
    ]


def test_tokens_without_timings_are_dropped():
    """Missing timestamps is the ONLY reason a token is dropped."""
    tokens = [
        StubToken("Hi", start_ts=0.0, end_ts=0.3),
        StubToken(".", phonemes=None),
        StubToken("there", start_ts=None, end_ts=0.9),
    ]
    assert [w.word for w in words_from_tokens(tokens)] == ["Hi"]


def test_a_timed_token_with_no_phonemes_is_still_emitted():
    """misaki times sentence-final punctuation, and the spec keeps it.

    The "." here has phonemes=None but real timestamps, which is exactly what
    misaki produces for sentence-final punctuation -- so it belongs in the
    `words` array. Filtering on phonemes instead of on timestamps would silently
    drop it and lose real timing data.
    """
    tokens = [
        StubToken("Hello", start_ts=0.0, end_ts=0.4),
        StubToken(".", phonemes=None, start_ts=0.4, end_ts=0.55),
    ]
    assert words_from_tokens(tokens) == [
        WordTiming("Hello", 0.0, 0.4),
        WordTiming(".", 0.4, 0.55),
    ]


def test_surrounding_whitespace_is_stripped_from_words():
    tokens = [StubToken("  Hello\n", start_ts=0.0, end_ts=0.4)]
    assert words_from_tokens(tokens)[0].word == "Hello"


def test_blank_text_tokens_are_dropped():
    tokens = [StubToken("   ", start_ts=0.0, end_ts=0.4)]
    assert words_from_tokens(tokens) == []


def test_end_never_precedes_start():
    tokens = [StubToken("Odd", start_ts=0.5, end_ts=0.2)]
    assert words_from_tokens(tokens) == [WordTiming("Odd", 0.5, 0.5)]


def test_accumulator_leaves_the_first_segment_alone():
    acc = TimelineAccumulator()
    words = acc.add([WordTiming("one", 0.0, 0.5)], duration=1.0)
    assert words == [WordTiming("one", 0.0, 0.5)]
    assert acc.total == pytest.approx(1.0)


def test_accumulator_shifts_later_segments_by_elapsed_audio():
    acc = TimelineAccumulator()
    acc.add([WordTiming("one", 0.0, 0.5)], duration=1.0)
    second = acc.add([WordTiming("two", 0.0, 0.6)], duration=1.5)
    assert second == [WordTiming("two", 1.0, 1.6)]
    assert acc.total == pytest.approx(2.5)


def test_timings_stay_monotonic_across_many_segments():
    """The regression this whole class exists to prevent."""
    acc = TimelineAccumulator()
    collected: list[WordTiming] = []
    for _ in range(5):
        # Every segment reports the same segment-relative timings.
        collected += acc.add(
            [WordTiming("a", 0.0, 0.4), WordTiming("b", 0.5, 0.9)], duration=1.0
        )

    assert len(collected) == 10
    for earlier, later in zip(collected, collected[1:]):
        assert later.start >= earlier.start
        assert earlier.end <= later.end
    assert collected[-1].end == pytest.approx(4.9)
    assert acc.total == pytest.approx(5.0)


def test_accumulator_counts_duration_even_with_no_words():
    acc = TimelineAccumulator()
    assert acc.add([], duration=0.8) == []
    assert acc.add([WordTiming("x", 0.0, 0.2)], duration=0.5) == [
        WordTiming("x", 0.8, 1.0)
    ]
