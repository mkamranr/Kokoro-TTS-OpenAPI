"""Real-model tests. Gated: run with KOKORO_RUN_SLOW=1."""
import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("KOKORO_RUN_SLOW") != "1",
        reason="set KOKORO_RUN_SLOW=1 to run model-loading tests",
    ),
]

SENTENCE = "The quick brown fox jumps over the lazy dog."


@pytest.fixture(scope="module")
def engine():
    from app.engine import KokoroEngine

    return KokoroEngine(device="auto")


def test_engine_reports_its_backend(engine):
    info = engine.info()
    assert info["backend"] == "pytorch"
    assert info["device"] in {"cpu", "cuda"}
    assert info["voices"] == 28
    assert info["warmup_seconds"] > 0


def test_synthesis_produces_non_silent_audio_of_plausible_length(engine):
    segments = list(engine.iter_segments(SENTENCE, "af_heart", "a", 1.0))
    assert segments
    audio = np.concatenate([s.audio for s in segments])
    duration = len(audio) / engine.sample_rate

    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) > 0.05, "audio is silent"
    assert 1.5 < duration < 6.0, f"implausible duration {duration:.2f}s"


def test_word_timings_cover_the_sentence_in_order(engine):
    segments = list(engine.iter_segments(SENTENCE, "af_heart", "a", 1.0))
    words = [w for s in segments for w in s.words]

    # Real misaki output legitimately differs from a naive split(): the
    # trailing "." arrives as its own timed MToken (see task-10-report.md for
    # the exact observed token list: ..., 'dog', '.'). Per the pre-authorized
    # ruling, this is relaxed from strict equality to order-preserving
    # containment -- punctuation-only tokens are dropped before comparing --
    # while the ordering/monotonicity checks below are unchanged.
    spoken = [w.word.lower().strip(".,") for w in words if w.word.strip(".,")]
    assert spoken == SENTENCE.lower().rstrip(".").split()

    for earlier, later in zip(words, words[1:]):
        assert earlier.start <= earlier.end
        assert later.start >= earlier.start


def test_timings_stay_inside_the_audio_duration(engine):
    segments = list(engine.iter_segments(SENTENCE, "af_heart", "a", 1.0))
    duration = sum(s.duration for s in segments)
    words = [w for s in segments for w in s.words]
    # Segment-relative here; the service adds offsets. Per-segment bound only.
    for segment in segments:
        for word in segment.words:
            assert 0 <= word.start <= segment.duration + 0.2
    assert duration > 0


def test_multi_paragraph_text_yields_multiple_segments(engine):
    text = "First paragraph here.\nSecond paragraph here.\nThird one too."
    segments = list(engine.iter_segments(text, "af_heart", "a", 1.0))
    assert len(segments) >= 2
    assert [s.index for s in segments] == list(range(len(segments)))


def test_british_voice_and_lang_code(engine):
    segments = list(engine.iter_segments("Good afternoon.", "bm_george", "b", 1.0))
    audio = np.concatenate([s.audio for s in segments])
    assert np.max(np.abs(audio)) > 0.05


def test_weighted_blend_synthesizes(engine):
    segments = list(
        engine.iter_segments("Blended voice test.", "af_bella:0.7000,af_sky:0.3000", "a", 1.0)
    )
    audio = np.concatenate([s.audio for s in segments])
    assert np.max(np.abs(audio)) > 0.05


def test_blend_differs_from_either_component(engine):
    text = "Comparing voices now."
    def render(voice):
        return np.concatenate(
            [s.audio for s in engine.iter_segments(text, voice, "a", 1.0)]
        )

    bella = render("af_bella")
    blend = render("af_bella:0.5000,af_sky:0.5000")
    shortest = min(len(bella), len(blend))
    assert not np.allclose(bella[:shortest], blend[:shortest], atol=1e-4)


def test_speed_affects_duration(engine):
    slow = list(engine.iter_segments(SENTENCE, "af_heart", "a", 0.8))
    fast = list(engine.iter_segments(SENTENCE, "af_heart", "a", 1.5))
    assert sum(s.duration for s in fast) < sum(s.duration for s in slow)


@pytest.mark.asyncio
async def test_service_end_to_end_with_the_real_engine(engine):
    from app.service import SynthesisService

    service = SynthesisService(engine, max_concurrency=1)
    result = await service.synthesize(
        "One two three.\nFour five six.", "af_heart", "a", 1.0
    )
    assert result.segments >= 2
    starts = [w.start for w in result.words]
    assert starts == sorted(starts)
    assert result.words[-1].end <= result.duration + 0.5
