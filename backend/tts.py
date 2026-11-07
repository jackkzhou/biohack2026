"""ElevenLabs streaming text-to-speech in the cloned voice.

`speak()` starts a job that streams MP3 from ElevenLabs into memory as it is
generated. The browser plays it from GET /tts/{id}.mp3, which tails the job, so
playback starts with the first chunk instead of after the whole sentence.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import OrderedDict

import httpx

log = logging.getLogger("tts")

API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"
OUTPUT_FORMAT = "mp3_44100_128"
BYTES_PER_SECOND = 128_000 / 8  # for estimating playback length
KEEP_JOBS = 20


class TTSJob:
    def __init__(self, job_id: int, text: str) -> None:
        self.id = job_id
        self.text = text
        self.chunks: list[bytes] = []
        self.done = False
        self.error: str | None = None
        self.cond = asyncio.Condition()
        self.started = time.monotonic()

    @property
    def size(self) -> int:
        return sum(len(c) for c in self.chunks)

    async def stream(self):
        """Yield chunks as they arrive; safe for several simultaneous listeners."""
        i = 0
        while True:
            async with self.cond:
                await self.cond.wait_for(lambda: i < len(self.chunks) or self.done)
            while i < len(self.chunks):
                yield self.chunks[i]
                i += 1
            if self.done:
                return

    async def wait_done(self) -> None:
        async with self.cond:
            await self.cond.wait_for(lambda: self.done)


class TTS:
    def __init__(self, api_key: str, voice_id: str, model: str) -> None:
        self.api_key, self.voice_id, self.model = api_key, voice_id, model
        self._client: httpx.AsyncClient | None = None
        self._jobs: OrderedDict[int, TTSJob] = OrderedDict()
        self._ids = itertools.count(int(time.time()))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.voice_id)

    async def start(self) -> None:
        if self.enabled:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, read=30.0))
        else:
            log.info("ElevenLabs not configured: phrases will be shown but not spoken")

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()

    def get(self, job_id: int) -> TTSJob | None:
        return self._jobs.get(job_id)

    def speak(self, text: str) -> TTSJob | None:
        """Begin streaming `text`; returns the job immediately (None if TTS is off)."""
        if not (self.enabled and self._client):
            return None
        job = TTSJob(next(self._ids), text)
        self._jobs[job.id] = job
        while len(self._jobs) > KEEP_JOBS:
            self._jobs.popitem(last=False)
        asyncio.create_task(self._run(job), name=f"tts-{job.id}")
        return job

    async def _run(self, job: TTSJob) -> None:
        assert self._client
        try:
            async with self._client.stream(
                "POST",
                API.format(voice=self.voice_id),
                params={"output_format": OUTPUT_FORMAT},
                headers={"xi-api-key": self.api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                json={"text": job.text, "model_id": self.model},
            ) as res:
                if res.status_code != 200:
                    body = (await res.aread()).decode("utf-8", "ignore")[:200]
                    raise RuntimeError(f"ElevenLabs HTTP {res.status_code}: {body}")
                async for chunk in res.aiter_bytes(4096):
                    if chunk:
                        async with job.cond:
                            job.chunks.append(chunk)
                            job.cond.notify_all()
        except Exception as e:  # noqa: BLE001
            job.error = str(e)
            log.warning("TTS failed: %s", e)
        finally:
            async with job.cond:
                job.done = True
                job.cond.notify_all()

    @staticmethod
    def playback_seconds(job: TTSJob) -> float:
        return job.size / BYTES_PER_SECOND
