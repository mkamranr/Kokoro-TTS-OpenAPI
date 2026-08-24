import asyncio

import pytest

from app.service import SynthesisService
from tests.fakes import FakeEngine


@pytest.mark.asyncio
async def test_synthesize_concatenates_segments_and_words():
    service = SynthesisService(FakeEngine(), max_concurrency=1)
    result = await service.synthesize("one two\nthree four", "af_heart", "a", 1.0)

    assert result.segments == 2
    assert [w.word for w in result.words] == ["one", "two", "three", "four"]
    assert result.sample_rate == 24000
    assert result.duration == pytest.approx(len(result.audio) / 24000)
    assert result.voice == "af_heart"


@pytest.mark.asyncio
async def test_word_timings_are_absolute_across_segments():
    service = SynthesisService(FakeEngine(), max_concurrency=1)
    result = await service.synthesize("one two\nthree four", "af_heart", "a", 1.0)

    starts = [w.start for w in result.words]
    assert starts == sorted(starts)
    # The second segment's first word must start after the first segment's audio.
    assert result.words[2].start >= result.words[1].end


@pytest.mark.asyncio
async def test_stream_segments_yields_progressively():
    service = SynthesisService(FakeEngine(), max_concurrency=1)
    seen = [seg.index async for seg in service.stream_segments("a\nb\nc", "af_heart", "a", 1.0)]
    assert seen == [0, 1, 2]


@pytest.mark.asyncio
async def test_engine_receives_the_arguments_it_was_given():
    engine = FakeEngine()
    service = SynthesisService(engine, max_concurrency=1)
    await service.synthesize("hello", "af_bella:0.5,af_sky:0.5", "b", 1.25)
    assert engine.calls == [("hello", "af_bella:0.5,af_sky:0.5", "b", 1.25)]


@pytest.mark.asyncio
async def test_semaphore_serializes_concurrent_synthesis():
    engine = FakeEngine(segment_delay=0.05)
    service = SynthesisService(engine, max_concurrency=1)

    async def run():
        return await service.synthesize("x", "af_heart", "a", 1.0)

    started = asyncio.get_running_loop().time()
    await asyncio.gather(run(), run())
    elapsed = asyncio.get_running_loop().time() - started
    # Serialized: two 50ms synthesis calls cannot finish in under 100ms.
    assert elapsed >= 0.09


@pytest.mark.asyncio
async def test_the_event_loop_stays_responsive_during_synthesis():
    """Synthesis must run in a thread, not block the loop."""
    engine = FakeEngine(segment_delay=0.2)
    service = SynthesisService(engine, max_concurrency=1)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await service.synthesize("x", "af_heart", "a", 1.0)
    task.cancel()
    assert ticks > 5


def test_info_includes_engine_details_and_concurrency():
    service = SynthesisService(FakeEngine(), max_concurrency=3)
    info = service.info()
    assert info["device"] == "fake"
    assert info["max_concurrency"] == 3
