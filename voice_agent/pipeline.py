"""Per-connection conversation session: VAD -> STT -> LLM (+tools) -> chunker -> TTS, with barge-in.

Concurrency model (single asyncio loop, blocking GPU work pushed to threads):

    websocket recv ──► on_audio() ──► Endpointer (inline, <1 ms/frame)
                                         │ speech_start ──► interrupt()  (cancel turn + tell client to flush)
                                         │ speech_end   ──► create_task(_turn(audio))
    _turn:  STT (thread) ─► TaskGroup[ LLM producer ─► asyncio.Queue[sentence] ─► TTS consumer ─► send_audio ]

Interruption rules:
  * The user starts talking while a turn is running -> cancel the turn task. Cancellation propagates
    into the TaskGroup: the LLM stream is closed (server aborts generation) and TTS stops after its
    current chunk. The client is told to drop its queued playback audio immediately.
  * If no audio had been played yet, the user was just pausing mid-thought: the half-finished turn is
    rolled back and its audio is prepended to the next utterance, so nothing they said is lost.
  * If audio had been played, the assistant message stored in history holds only the sentences that
    were actually sent, so the LLM knows where it was cut off.
"""
import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from .chunker import SentenceChunker
from .config import Config
from .llm import LLM
from .vad import Endpointer


@dataclass
class Models:
    stt: object
    llm: LLM
    tts: object


def float_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class Session:
    def __init__(
        self,
        models: Models,
        cfg: Config,
        send_audio: Callable[[bytes], Awaitable[None]],
        send_event: Callable[[dict], Awaitable[None]],
    ):
        self.m = models
        self.cfg = cfg
        self.send_audio = send_audio
        self.send_event = send_event
        self.endpointer = Endpointer(cfg)
        self.history: list[dict] = [{"role": "system", "content": cfg.system_prompt}]
        self.turn_task: asyncio.Task | None = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._carry: np.ndarray | None = None   # audio of a turn the user cut off before we spoke

    # ------------------------------------------------------------------ input
    async def on_audio(self, pcm: np.ndarray):
        self._pending = np.concatenate([self._pending, pcm])
        n = self.cfg.vad_frame
        while len(self._pending) >= n:
            frame, self._pending = self._pending[:n], self._pending[n:]
            for event, audio in self.endpointer.process(frame):
                if event == "speech_start":
                    await self.interrupt()
                    await self.send_event({"type": "user_speaking"})
                else:
                    if self._carry is not None:
                        audio, self._carry = np.concatenate([self._carry, audio]), None
                    self.turn_task = asyncio.create_task(self._turn(audio, time.perf_counter()))

    async def interrupt(self):
        task, self.turn_task = self.turn_task, None
        if task is None or task.done():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await self.send_event({"type": "interrupt"})

    async def close(self):
        await self.interrupt()

    # ------------------------------------------------------------------ turn
    async def _turn(self, audio: np.ndarray, t_end: float):
        """t_end = moment the endpointer fired (the user actually stopped `end_silence_ms` earlier)."""
        metrics = {"endpoint_ms": self.cfg.end_silence_ms}
        turn_start = len(self.history)
        spoken: list[str] = []
        audio_started = False
        sentences: asyncio.Queue[str | None] = asyncio.Queue()

        async def produce():
            chunker = SentenceChunker(self.cfg.first_chunk_min_words, self.cfg.chunk_min_words)
            first_token = True
            async for delta in self.m.llm.respond(self.history):
                if first_token:
                    metrics["llm_ttft_ms"] = round((time.perf_counter() - t_llm) * 1000)
                    first_token = False
                for s in chunker.push(delta):
                    metrics.setdefault("first_phrase_ms", round((time.perf_counter() - t_llm) * 1000))
                    await sentences.put(s)
            for s in chunker.flush():
                await sentences.put(s)
            await sentences.put(None)

        async def consume():
            nonlocal audio_started
            while (sentence := await sentences.get()) is not None:
                t_tts = time.perf_counter()
                gen = self.m.tts.synth(sentence)
                while (chunk := await asyncio.to_thread(next, gen, None)) is not None:
                    if not audio_started:
                        audio_started = True
                        metrics["tts_first_chunk_ms"] = round((time.perf_counter() - t_tts) * 1000)
                        metrics["voice_to_voice_ms"] = round(
                            (time.perf_counter() - t_end) * 1000 + self.cfg.end_silence_ms
                        )
                        await self.send_event({"type": "metrics", **metrics})
                    await self.send_audio(float_to_pcm16(chunk))
                spoken.append(sentence)
                await self.send_event({"type": "assistant_text", "text": sentence})

        try:
            t_stt = time.perf_counter()
            text = await asyncio.to_thread(self.m.stt.transcribe, audio)
            metrics["stt_ms"] = round((time.perf_counter() - t_stt) * 1000)
            if not text:
                return
            await self.send_event({"type": "user_text", "text": text})
            self.history.append({"role": "user", "content": text})

            t_llm = time.perf_counter()
            async with asyncio.TaskGroup() as tg:
                tg.create_task(produce())
                tg.create_task(consume())
        except asyncio.CancelledError:
            if not audio_started:
                del self.history[turn_start:]
                self._carry = audio
            raise
        finally:
            if spoken:
                self.history.append({"role": "assistant", "content": " ".join(spoken)})
            elif len(self.history) > turn_start:
                # user turn kept but nothing fully spoken: keep user/assistant alternation well-formed
                self.history.append({"role": "assistant", "content": ""})
