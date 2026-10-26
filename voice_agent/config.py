"""Central config. Every knob that affects latency lives here so benchmarks are reproducible."""
import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Config:
    # --- Input / VAD -------------------------------------------------------
    sample_rate_in: int = 16000          # Silero, Parakeet and Whisper all want 16 kHz mono
    vad_frame: int = 512                 # Silero v5+ requires exactly 512 samples @16 kHz (32 ms)
    vad_threshold: float = 0.5           # enter-speech probability
    vad_neg_threshold: float = 0.35      # exit-speech probability (hysteresis)
    min_speech_ms: int = 96              # 3 frames of speech before we call it speech (barge-in debounce)
    end_silence_ms: int = int(_env("END_SILENCE_MS", "400"))  # the single biggest latency knob
    preroll_ms: int = 320                # audio kept from before speech start so the first syllable is not clipped
    max_utterance_s: float = 30.0

    # --- STT ---------------------------------------------------------------
    stt_backend: str = _env("STT_BACKEND", "parakeet")           # parakeet | whisper
    parakeet_model: str = _env("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    whisper_model: str = _env("WHISPER_MODEL", "large-v3-turbo")
    whisper_compute_type: str = _env("WHISPER_COMPUTE", "float16")  # float16 | int8_float16
    language: str = _env("LANGUAGE", "en")

    # --- LLM (any OpenAI-compatible server: vLLM, Ollama, llama.cpp) --------
    llm_base_url: str = _env("LLM_BASE_URL", "http://localhost:8001/v1")
    llm_api_key: str = _env("LLM_API_KEY", "local")
    llm_model: str = _env("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    llm_temperature: float = 0.6
    llm_max_tokens: int = 300
    max_tool_rounds: int = 3
    # Passed through to the server, e.g. {"chat_template_kwargs": {"enable_thinking": False}} for hybrid-thinking models
    llm_extra_body: dict = field(default_factory=dict)
    system_prompt: str = (
        "You are a friendly, fast voice assistant. Your replies are spoken aloud, so: "
        "answer in one to three short sentences, use plain conversational words, "
        "never use markdown, lists, emojis or URLs, and spell out symbols. "
        "Use the available tools whenever the user asks for live information or calculations. "
        "If a tool is needed, first say a very short phrase like 'Let me check.'"
    )

    # --- TTS ---------------------------------------------------------------
    tts_voice: str = _env("TTS_VOICE", "af_heart")
    tts_lang: str = _env("TTS_LANG", "a")    # Kokoro: a = American English, b = British, h = Hindi, ...
    tts_speed: float = 1.05
    tts_sample_rate: int = 24000
    first_chunk_min_words: int = 4           # first TTS chunk may cut at a comma once it has this many words
    chunk_min_words: int = 3


def load_config() -> Config:
    cfg = Config()
    if _env("LLM_DISABLE_THINKING", "0") == "1":
        cfg.llm_extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    return cfg
