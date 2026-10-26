"""Silero VAD + an endpointing state machine.

Feed it 512-sample float32 frames (32 ms @ 16 kHz). It emits:
  ("speech_start", None)    -> after `min_speech_ms` of sustained speech (used for barge-in)
  ("speech_end",  audio)    -> after `end_silence_ms` of silence, with the whole utterance incl. pre-roll
"""
import math
from collections import deque

import numpy as np
import torch
from silero_vad import load_silero_vad

from .config import Config

torch.set_num_threads(1)  # per-frame inference is ~0.3 ms on CPU; extra threads only add jitter


class Endpointer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = load_silero_vad()
        frame_ms = cfg.vad_frame / cfg.sample_rate_in * 1000
        self.start_frames = max(1, math.ceil(cfg.min_speech_ms / frame_ms))
        self.end_frames = max(1, math.ceil(cfg.end_silence_ms / frame_ms))
        self.max_frames = math.ceil(cfg.max_utterance_s * 1000 / frame_ms)
        self.preroll: deque[np.ndarray] = deque(maxlen=math.ceil(cfg.preroll_ms / frame_ms))
        self.reset()

    def reset(self):
        self.model.reset_states()
        self.preroll.clear()
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.buf: list[np.ndarray] = []

    def process(self, frame: np.ndarray) -> list[tuple[str, np.ndarray | None]]:
        prob = self.model(torch.from_numpy(frame), self.cfg.sample_rate_in).item()
        events: list[tuple[str, np.ndarray | None]] = []

        if not self.in_speech:
            self.preroll.append(frame)
            self.speech_run = self.speech_run + 1 if prob >= self.cfg.vad_threshold else 0
            if self.speech_run >= self.start_frames:
                self.in_speech = True
                self.buf = list(self.preroll)
                self.preroll.clear()
                self.silence_run = 0
                events.append(("speech_start", None))
            return events

        self.buf.append(frame)
        self.silence_run = self.silence_run + 1 if prob < self.cfg.vad_neg_threshold else 0
        if self.silence_run >= self.end_frames or len(self.buf) >= self.max_frames:
            audio = np.concatenate(self.buf)
            self.in_speech = False
            self.speech_run = 0
            self.buf = []
            events.append(("speech_end", audio))
        return events
