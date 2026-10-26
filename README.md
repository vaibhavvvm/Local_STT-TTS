# Local Voice Agent

A real-time voice-to-voice assistant built entirely from open-weight models: **you speak, it answers out loud in under a second, it can call tools, and you can interrupt it mid-sentence.** It is benchmarked on Google Colab GPUs (T4, L4, A100) with one notebook.

```
mic ─► Silero VAD ─► Parakeet-TDT 0.6B ─► Qwen 4B (vLLM, tool calling) ─► phrase chunker ─► Kokoro-82M ─► speaker
          ▲                                                                                         │
          └──────────────────── barge-in: user speech cancels the LLM and flushes audio ◄─────────────┘
```

> Status: architecture and code are complete. The latency figures below are **targets and vendor-reported numbers**; run `voice_agent/bench.py` on your GPU and replace them with measured results before publishing.

---

## 1. Spec sheet

| Stage | Pick | Format / precision | Settings | Why |
|---|---|---|---|---|
| Transport | Browser AudioWorklet over WebSocket (FastAPI) + Cloudflare quick tunnel | int16 PCM | mic 16 kHz up, TTS 24 kHz down | Colab can't reach your mic directly; this gives full control over flush/interrupt (WebRTC/FastRTC is the upgrade path) |
| VAD | **Silero VAD v5+** | ONNX/JIT, CPU | 512-sample frames (32 ms @ 16 kHz), threshold 0.5/0.35 hysteresis, 96 ms to trigger, 320 ms pre-roll | Accurate, under 1 ms per frame on CPU, and the default front end for Pipecat's Smart Turn |
| Endpointing | Silence hangover (400 ms) → **Smart Turn v3** upgrade | ONNX, CPU | VAD `stop_secs=0.2` + semantic turn model | Silence wait is the single biggest latency cost. A semantic turn model lets you cut it to about 200 ms |
| STT | **NVIDIA Parakeet-TDT 0.6B v3** (NeMo) | bf16 autocast (fp32 on T4) | full-utterance transcribe at end of turn, greedy | ~6.3% avg WER on Open ASR, extremely fast. A 3-5 s utterance takes tens of ms on a GPU |
| STT fallback | faster-whisper **large-v3-turbo** | `float16` (`int8_float16` to save VRAM) | `beam_size=1`, `without_timestamps` | 99 languages (Hindi etc.) |
| LLM | **Qwen3-4B-Instruct-2507** (default) / **Qwen3.5-4B** (candidate) | bf16 on L4/A100. Q4_K_M GGUF on T4 | `temperature 0.6`, `max_tokens 300`, thinking **off** | Strong tool calling at 4B. Non-thinking mode keeps TTFT low |
| LLM engine | **vLLM** (L4/A100) / **Ollama** (T4) | OpenAI-compatible API | prefix caching on, `--enable-auto-tool-choice --tool-call-parser hermes` | Prefix cache reuses the system prompt and tool schemas every turn. Aborts generation when the client disconnects |
| Text→TTS bridge | Phrase chunker | — | 1st chunk at first clause boundary ≥ 4 words, then sentence boundaries | Audio starts after about 5-8 tokens while prosody stays natural |
| TTS | **Kokoro-82M** | fp32, GPU | 24 kHz, streamed per phrase | Lowest time-to-first-audio among quality open TTS, Apache-2.0, 0.5 GB VRAM |
| Orchestration | Single `asyncio` loop. GPU work via `asyncio.to_thread`. Per-turn `TaskGroup` (LLM producer → queue → TTS consumer) | — | — | Cancelling one task tears down the whole turn cleanly |

### Latency budget (target, L4/A100, measured from when the user stops talking)

| Component | Target |
|---|---|
| Endpoint silence (`END_SILENCE_MS`) | 400 ms (→ 200 ms with Smart Turn) |
| STT (Parakeet, 3-5 s utterance) | 30-80 ms |
| LLM TTFT (prefix-cached) | 30-80 ms |
| LLM to first phrase (~6 tokens) | +60-100 ms |
| TTS first chunk (Kokoro) | 50-120 ms |
| Network (Cloudflare tunnel) | 50-150 ms |
| **Voice-to-voice** | **~650-950 ms** (≈ 450-750 ms with Smart Turn) |

Note that **speculative decoding barely matters here.** A 4B model on an L4 decodes faster than anyone can speak. What you hear as latency is endpointing, prefill and time to first phrase. Spend your optimisation effort there.

---

## 2. Trade-off matrix (models evaluated, October 2026)

### VAD / turn detection
| Model | Accuracy | Latency | Notes | Verdict |
|---|---|---|---|---|
| **Silero VAD v5/v6** | High | <1 ms/32 ms frame, CPU | MIT, de facto standard | ✅ default |
| TEN VAD | Claims > Silero (self-reported) | Lower compute than Silero | Benchmarked only on its own test set | 🔬 A/B it |
| WebRTC VAD | Low (noise-sensitive) | ~0 | Legacy | ❌ |
| **Smart Turn v3** (Pipecat) | Semantic end-of-turn | Fast CPU ONNX | Sits *on top of* VAD | ✅ phase-2 upgrade |
| Kyutai STT semantic VAD | Built into STT | — | Only with Kyutai STT | 🔬 |

### STT
| Model | Size | Avg WER (Open ASR) | Streaming | Languages | Verdict |
|---|---|---|---|---|---|
| **Parakeet-TDT 0.6B v3** | 0.6B | ~6.3% | Chunked (utterance) | 25 EU | ✅ default (speed + accuracy) |
| Nemotron Speech Streaming 0.6B | 0.6B | ~6.9% (1.12 s chunks) | **True cache-aware**, 80-1120 ms chunks | EN (3.5: multilingual) | 🔬 upgrade for live partials |
| Cohere Transcribe 2B / Granite Speech 4.1 2B / Canary-Qwen 2.5B | 2-2.5B | ~5.3-5.6% (top of board) | No | Varies | Accuracy-first, 3-4× the compute |
| Whisper large-v3-turbo (faster-whisper) | 0.8B | ~7.8% | No (re-runs windows) | **99** | ✅ multilingual fallback |
| Kyutai STT 1B | 1B | — | **Native** (0.5 s delay) | EN/FR | 🔬 via Unmute |

### LLM (voice + tool calling, ≤ 9B)
| Model | Size | Tool calling | Notes | Verdict |
|---|---|---|---|---|
| **Qwen3-4B-Instruct-2507** | 4B | Strong, `hermes` parser | Non-thinking variant, mature in vLLM/Ollama | ✅ default |
| **Qwen3.5-4B** | 4B | Yes, `qwen3_coder` parser | Newer (Mar 2026), hybrid thinking: disable it | 🔬 benchmark vs default |
| Qwen3.5-9B / Qwen3-8B | 8-9B | Stronger | ~2× prefill/decode cost | A100 option |
| Gemma 4 E4B | ~4B effective | Function calling | Native audio input, Apache-2.0 | 🔬 also an STT-less path |
| Llama 3.2 3B | 3B | Weak at multi-step | Older | ❌ |

### TTS
| Model | Size | First audio (reported) | Expressive / cloning | License | Verdict |
|---|---|---|---|---|---|
| **Kokoro-82M** | 82M | Lowest of the group, ~30× realtime | Fixed voices, no cloning | Apache-2.0 | ✅ default |
| Qwen3-TTS | 0.6-1.7B | ~97 ms (vendor) | Yes, cloning, multilingual | Apache-2.0 | 🔬 best "expressive" upgrade |
| Chatterbox-Turbo | 350M | "<200 ms" (unverified) | Emotion control, cloning | MIT | 🔬 |
| Orpheus | 150M-3B | ~100-200 ms (vendor) | Very expressive, tags | Apache-2.0 | 🔬 heavier |
| VibeVoice-Realtime-0.5B | 0.5B | ~300 ms | EN, single speaker | MIT | ⚠️ |
| Kyutai TTS / Pocket TTS | 1.6B / 100M | ~350 ms batched / CPU | Cloning (10 s) | CC-BY / MIT | 🔬 |

### Cascaded vs end-to-end speech-to-speech
End-to-end models (Moshi, Qwen-Omni family) can beat cascades on raw latency. A cascade is the right call for this project because **tool calling, swappable parts, inspectable transcripts and per-stage benchmarks** are the point. Kyutai **Unmute** (their STT + TTS around any LLM, ~450 ms claimed) is the closest open reference design to compare against.

---

## 3. Interruption (barge-in) logic

```
on every 32 ms mic frame:
    prob = silero(frame)
    if not in_speech:
        preroll.append(frame)
        if 3 consecutive frames ≥ 0.5:        # 96 ms debounce: coughs/clicks don't interrupt
            in_speech = True
            interrupt()                       # ◄── barge-in
    else:
        if 400 ms of frames < 0.35: emit speech_end(preroll + utterance)

interrupt():
    turn_task.cancel()                        # propagates into TaskGroup:
        ├─ LLM producer: async-for raises CancelledError → `await stream.close()` → vLLM aborts the request
        └─ TTS consumer: stops after the current ~50 ms chunk
    send {"type":"interrupt"}                 # browser AudioWorklet drops its whole playback queue at once
    history:
        if no audio was played yet → roll back the turn and prepend its audio to the next utterance
                                      (the user only paused, and their first words are kept)
        else → store only the sentences actually sent, so the LLM knows where it was cut off
```

Code: `voice_agent/pipeline.py` (`Session.interrupt`, `Session._turn`). **Use headphones.** Without them, the bot hears itself and interrupts itself. Browser echo cancellation helps but doesn't fix this reliably.

---

## 4. Tool calling

Tools live in `voice_agent/tools.py`. Each one is an async function plus a JSON schema:

```python
@registry.tool("Get the current weather for a city.",
               {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
async def get_weather(city: str) -> str: ...
```

The loop in `voice_agent/llm.py` streams text, collects `tool_calls` deltas, executes them, appends results and re-queries, up to 3 rounds. The system prompt tells the model to say a short filler ("Let me check.") *before* calling a tool. That filler is spoken immediately, which hides the extra round-trip. Bundled tools: weather (Open-Meteo, no key), timezone clock, safe calculator. `bench.py` reports tool-call accuracy.

To integrate your own projects, add tools that call your APIs, or wrap an MCP client so the agent can use any MCP server's tools.

---

## 5. Run it on Colab

1. Push this folder to GitHub and set `REPO` in `notebooks/colab_quickstart.ipynb`.
2. Open the notebook in Colab. In **Runtime → Change runtime type**, pick **L4** (best value) or **A100**. T4 works too.
3. Run all cells:
   - installs NeMo, Kokoro, Silero and FastAPI (~5 min)
   - starts **vLLM in a separate venv** on Ampere+ GPUs (avoids torch conflicts with NeMo), or **Ollama** on a T4
   - runs `python -m voice_agent.bench` and prints p50/p90 per stage, STT WER and tool-call accuracy
   - launches the app and prints a `https://…trycloudflare.com` link. Open it in Chrome, wear headphones and talk
4. The metrics line in the UI shows `stt_ms`, `llm_ttft_ms`, `first_phrase_ms`, `tts_first_chunk_ms` and `voice_to_voice_ms` for every turn.

**VRAM (L4, 24 GB):** vLLM 55% (~13 GB: 8 GB weights + KV cache), Parakeet ~2.5 GB, Kokoro ~0.5 GB.
**Colab compute:** the T4 is free. L4 and A100 use compute units. Check how many units your Google AI Pro plan includes on Colab's pricing page, and disconnect the runtime when you're done.

Local run (any Linux box with a GPU): `pip install -r requirements.txt`, start an OpenAI-compatible LLM server, set `LLM_BASE_URL`/`LLM_MODEL`, then run `python -m voice_agent.server` and open `http://localhost:8000`.

---

## 6. Roadmap

- [ ] Smart Turn v3 endpointing (cut silence wait from 400 ms to about 200 ms)
- [ ] Nemotron Speech Streaming for live partial transcripts and speculative LLM prefill
- [ ] Qwen3-TTS backend for expressive, cloned voices
- [ ] MCP client so any MCP server's tools plug in
- [ ] FastRTC/WebRTC transport (better than a WebSocket on lossy networks)
- [ ] Publish a measured benchmark table per GPU (T4 / L4 / A100)

## Project layout

```
voice_agent/
  config.py     all latency knobs (env-overridable)
  vad.py        Silero + endpointing state machine
  stt.py        Parakeet (NeMo) / faster-whisper
  llm.py        streaming OpenAI-compatible client + tool loop
  tools.py      tool registry + example tools
  chunker.py    LLM text stream → speakable phrases
  tts.py        Kokoro streaming synth
  pipeline.py   per-session orchestration + barge-in
  server.py     FastAPI WebSocket server
  bench.py      per-stage latency/accuracy benchmark
  static/client.html   browser mic capture + gapless playback with flush
notebooks/colab_quickstart.ipynb
```

## Sources

- STT: [Open ASR leaderboard roundup (Northflank)](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks) · [Parakeet vs Whisper vs Nemotron](https://openwhispr.com/blog/parakeet-vs-whisper-vs-nemotron) · [Open-weight STT 2026](https://presenc.ai/research/best-open-weight-speech-to-text-models-2026) · [Nemotron Speech Streaming model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) · [Kyutai STT](https://kyutai.org/stt) · [Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling)
- VAD / turn: [TEN VAD](https://github.com/TEN-framework/ten-vad) · [Pipecat Smart Turn](https://docs.pipecat.ai/server/utilities/turn-detection/smart-turn-overview) · [Pipecat speech input](https://docs.pipecat.ai/guides/learn/speech-input)
- LLM: [Qwen3.5 on Baseten (vLLM flags)](https://docs.baseten.co/examples/models/llm/qwen3.5.md) · [vLLM forum: Qwen3.5-4B](https://discuss.vllm.ai/t/deployment-parameters-for-qwen3-5-4b/2488) · [Gemma 4 E4B vs Qwen3.5-9B](https://www.llmreference.com/compare/gemma-4-e4b/qwen3.5-9b) · [Small LMs for tool calling (KDnuggets)](https://www.kdnuggets.com/5-small-language-models-for-agentic-tool-calling)
- TTS: [Open-source TTS comparison (Dograh)](https://www.dograh.com/feeds/blog/open-source-tts-comparison) · [Open-source TTS guide 2026](https://tts.ai/blog/open-source-text-to-speech-guide-2026/?lang=es) · [Qwen3-TTS streaming](https://mintlify.com/QwenLM/Qwen3-TTS/guides/streaming) · [VibeVoice-Realtime](https://huggingface.co/spaces/anycoderapps/VibeVoice-Realtime-0.5B/blob/26e0cd317c5bd8b23ae01aa6fe9f281ce86c69d9/docs/vibevoice-realtime-0.5b.md) · [Kyutai TTS + Unmute](https://kyutai.org/2025/07/03/tts-unmute-open-source.html) · [OCDevel TTS notes](https://dev.ocdevel.com/blog/20250714-open-source-tts)
- Colab: [Colab GPUs & pricing](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/) · [Colab CLI / GPU list 2026](https://dev.to/jovan_chan_9500711396d4e6/google-colab-cli-run-ai-agents-on-cloud-gpus-2026-4c7j) · [AI Pro Colab units report](https://www.inside.com.tw/article/42497-google-ai-pro-colab-200-compute-units-a100-cli)
- Transport: [FastRTC](https://pypi.org/project/fastrtc/)

## License

Code: MIT (add a `LICENSE` file). Check each model's own license before redistributing weights.
