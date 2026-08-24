"""Concurrency and threading around any EngineProtocol implementation.

Torch inference here is not safely reentrant and the Mac has four cores, so
synthesis is serialized by a semaphore and executed in worker threads, keeping
the event loop free to serve /health and static assets.
"""
import asyncio
from typing import AsyncIterator

from app.audio import concat
from app.timing import TimelineAccumulator
from app.types import EngineProtocol, Segment, Synthesis


class SynthesisService:
    def __init__(self, engine: EngineProtocol, max_concurrency: int = 1):
        self._engine = engine
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def sample_rate(self) -> int:
        return self._engine.sample_rate

    def info(self) -> dict:
        return {**self._engine.info(), "max_concurrency": self._max_concurrency}

    async def stream_segments(
        self, text: str, voice: str, lang: str | None, speed: float
    ) -> AsyncIterator[Segment]:
        """Yield segments as they finish, with absolute word timings."""
        sentinel = object()
        async with self._semaphore:
            generator = self._engine.iter_segments(text, voice, lang, speed)
            timeline = TimelineAccumulator()

            def pull():
                return next(generator, sentinel)

            while True:
                segment = await asyncio.to_thread(pull)
                if segment is sentinel:
                    return
                segment.words = timeline.add(segment.words, segment.duration)
                yield segment

    async def synthesize(
        self, text: str, voice: str, lang: str | None, speed: float
    ) -> Synthesis:
        chunks = []
        words = []
        phonemes = []
        count = 0
        async for segment in self.stream_segments(text, voice, lang, speed):
            chunks.append(segment.audio)
            words.extend(segment.words)
            if segment.phonemes:
                phonemes.append(segment.phonemes)
            count += 1

        audio = concat(chunks)
        return Synthesis(
            audio=audio,
            sample_rate=self.sample_rate,
            duration=len(audio) / self.sample_rate,
            words=words,
            phonemes=" ".join(phonemes),
            voice=voice,
            segments=count,
        )
