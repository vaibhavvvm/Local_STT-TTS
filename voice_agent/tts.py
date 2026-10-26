"""Text-to-speech. Kokoro-82M: the fastest high-quality open TTS, 24 kHz output.

`synth()` is a generator so the pipeline can ship the first audio chunk the moment it exists.
Any other engine (Qwen3-TTS, Chatterbox-Turbo, VibeVoice-Realtime) just needs the same two members.
"""
from typing import Iterator

import numpy as np
import torch

from .config import Config


class KokoroTTS:
    def __init__(self, cfg: Config):
        from kokoro import KPipeline

        self.sample_rate = cfg.tts_sample_rate
        self.voice = cfg.tts_voice
        self.speed = cfg.tts_speed
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = KPipeline(lang_code=cfg.tts_lang, repo_id="hexgrad/Kokoro-82M", device=device)
        for _ in self.synth("Warming up."):
            pass

    def synth(self, text: str) -> Iterator[np.ndarray]:
        for _graphemes, _phonemes, audio in self.pipe(text, voice=self.voice, speed=self.speed):
            if audio is None:
                continue
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().float().cpu().numpy()
            yield audio.astype(np.float32)


def load_tts(cfg: Config):
    return KokoroTTS(cfg)
