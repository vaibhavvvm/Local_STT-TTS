"""Speech-to-text backends. Both take a float32 16 kHz mono array and return text.

Design choice: we transcribe the *whole utterance* once the endpointer fires. With Parakeet-TDT a
3-5 s utterance takes tens of milliseconds on a GPU, so it is faster end-to-end than rolling partial
transcripts (and gives better WER). See README for the true-streaming upgrade (Nemotron Speech Streaming).
"""
import numpy as np
import torch

from .config import Config


class ParakeetSTT:
    """NVIDIA Parakeet-TDT 0.6B v3 via NeMo (25 European languages, top-tier WER, very high throughput)."""

    def __init__(self, cfg: Config):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(cfg.parakeet_model)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.transcribe(np.zeros(16000, dtype=np.float32))  # warm-up: first call compiles kernels

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray) -> str:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            out = self.model.transcribe([audio], batch_size=1, verbose=False)
        if isinstance(out, tuple):  # older NeMo RNNT models return (best_hyps, all_hyps)
            out = out[0]
        hyp = out[0]
        return (hyp.text if hasattr(hyp, "text") else str(hyp)).strip()


class WhisperSTT:
    """faster-whisper (CTranslate2). Use for the 99-language case or when NeMo will not install."""

    def __init__(self, cfg: Config):
        from faster_whisper import WhisperModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = cfg.whisper_compute_type if device == "cuda" else "int8"
        self.model = WhisperModel(cfg.whisper_model, device=device, compute_type=compute)
        self.language = cfg.language or None
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=1,                      # greedy: biggest speed win, tiny WER cost
            vad_filter=False,                 # we already ran VAD
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(s.text.strip() for s in segments).strip()


def load_stt(cfg: Config):
    return WhisperSTT(cfg) if cfg.stt_backend == "whisper" else ParakeetSTT(cfg)
