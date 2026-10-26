"""Offline latency + accuracy benchmark for every stage. Prints p50/p90 tables you can paste in the README.

    python -m voice_agent.bench --runs 10

Test audio is synthesised with the TTS itself, so it is clean speech: real-world WER will be higher.
Reported numbers exclude network and the endpointing silence (END_SILENCE_MS), which add on top.
"""
import argparse
import asyncio
import json
import statistics
import time

import numpy as np
import torch

from .chunker import SentenceChunker
from .config import load_config
from .llm import LLM
from .stt import load_stt
from .tools import registry
from .tts import load_tts

PROMPTS = [
    "What's a good name for a golden retriever puppy?",
    "Explain in simple words why the sky is blue.",
    "Give me one quick tip to sleep better tonight.",
    "What is the capital of Australia, and is it the largest city there?",
]
TOOL_CASES = [
    ("What's the weather like in Pune right now?", "get_weather"),
    ("What time is it in Tokyo?", "get_current_time"),
    ("What is 347 times 29?", "calculate"),
    ("Tell me a fun fact about octopuses.", None),
]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def row(name, xs):
    return f"| {name:<26} | {statistics.median(xs):>8.0f} | {pct(xs, 90):>8.0f} |"


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    idx = np.linspace(0, len(audio) - 1, int(len(audio) * dst / src))
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


async def main(runs: int):
    cfg = load_config()
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"STT={cfg.stt_backend}  LLM={cfg.llm_model}  TTS=kokoro/{cfg.tts_voice}\n")
    stt, tts, llm = load_stt(cfg), load_tts(cfg), LLM(cfg, registry)
    res = {k: [] for k in ["tts_first_chunk", "tts_rtf", "stt", "llm_ttft", "llm_first_phrase", "llm_tok_s"]}

    # --- TTS + STT ---------------------------------------------------------
    wers = []
    try:
        import jiwer
    except ImportError:
        jiwer = None
    for i in range(runs):
        text = PROMPTS[i % len(PROMPTS)]
        t0 = time.perf_counter()
        chunks, first = [], None
        for c in tts.synth(text):
            first = first or time.perf_counter()
            chunks.append(c)
        total = time.perf_counter() - t0
        audio24 = np.concatenate(chunks)
        res["tts_first_chunk"].append((first - t0) * 1000)
        res["tts_rtf"].append(total / (len(audio24) / tts.sample_rate) * 1000)  # ms of compute per s of audio

        audio16 = resample(audio24, tts.sample_rate, 16000)
        t0 = time.perf_counter()
        hyp = stt.transcribe(audio16)
        res["stt"].append((time.perf_counter() - t0) * 1000)
        if jiwer:
            norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ")
            wers.append(jiwer.wer(norm(text), norm(hyp)))

    # --- LLM ---------------------------------------------------------------
    for i in range(runs):
        msgs = [{"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]
        chunker = SentenceChunker(cfg.first_chunk_min_words, cfg.chunk_min_words)
        t0 = time.perf_counter()
        ttft = phrase = None
        n_deltas = 0
        async for delta in llm.respond(msgs):
            ttft = ttft or time.perf_counter()
            n_deltas += 1
            if phrase is None and chunker.push(delta):
                phrase = time.perf_counter()
        end = time.perf_counter()
        res["llm_ttft"].append((ttft - t0) * 1000)
        res["llm_first_phrase"].append(((phrase or end) - t0) * 1000)
        res["llm_tok_s"].append(n_deltas / max(end - ttft, 1e-6))  # deltas ~= tokens on vLLM/Ollama

    # --- Tool-calling accuracy ----------------------------------------------
    tool_hits = []
    for prompt, expected in TOOL_CASES:
        msgs = [{"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": prompt}]
        async for _ in llm.respond(msgs):
            pass
        called = [tc["function"]["name"] for m in msgs if m.get("tool_calls") for tc in m["tool_calls"]]
        ok = (expected in called) if expected else not called
        tool_hits.append(ok)
        print(f"tool-call {'OK ' if ok else 'BAD'}  {prompt!r} -> {called or 'no tool'}")

    print("\n| stage                      | p50 (ms) | p90 (ms) |\n|---|---|---|")
    print(row("TTS time-to-first-chunk", res["tts_first_chunk"]))
    print(row("TTS compute per 1 s audio", res["tts_rtf"]))
    print(row("STT (full utterance)", res["stt"]))
    print(row("LLM time-to-first-token", res["llm_ttft"]))
    print(row("LLM time-to-first-phrase", res["llm_first_phrase"]))
    est = [cfg.end_silence_ms + s + p + t for s, p, t in zip(res["stt"], res["llm_first_phrase"], res["tts_first_chunk"])]
    print(row("EST. voice-to-voice*", est))
    print(f"\nLLM decode speed: {statistics.median(res['llm_tok_s']):.0f} tok/s (median)")
    if wers:
        print(f"STT WER on synthetic speech: {100 * statistics.mean(wers):.1f}%")
    print(f"Tool-call accuracy: {sum(tool_hits)}/{len(tool_hits)}")
    print(f"* includes END_SILENCE_MS={cfg.end_silence_ms}; excludes network/tunnel.")
    print(json.dumps({k: round(statistics.median(v), 1) for k, v in res.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=8)
    asyncio.run(main(ap.parse_args().runs))
