"""Bridges the LLM token stream to TTS.

Strategy: phrase-based buffering with an eager first chunk.
  * First chunk: cut at the first clause boundary (, ; : . ! ?) once it has >= N words, so audio
    starts after ~5-8 tokens instead of a full sentence.
  * Later chunks: cut at sentence boundaries, which gives the TTS enough context for natural prosody.
Boundaries require trailing whitespace, so "3.5" or "e.g." mid-token never split early.
"""
import re

_FIRST = re.compile(r"[.!?…,;:—]+[\"')\]]*\s")
_LATER = re.compile(r"[.!?…;:]+[\"')\]]*\s|\n+")
_MARKDOWN = re.compile(r"[*_#`>|~]+")
_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")


def clean_for_speech(text: str) -> str:
    text = _URL.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceChunker:
    def __init__(self, first_min_words: int = 4, min_words: int = 3):
        self.first_min_words = first_min_words
        self.min_words = min_words
        self._buf = ""
        self._first = True

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        out = []
        while (cut := self._find_cut()) is not None:
            piece, self._buf = self._buf[:cut], self._buf[cut:]
            piece = clean_for_speech(piece)
            if piece:
                out.append(piece)
                self._first = False
        return out

    def flush(self) -> list[str]:
        piece, self._buf = clean_for_speech(self._buf), ""
        return [piece] if piece else []

    def _find_cut(self) -> int | None:
        pattern, need = (_FIRST, self.first_min_words) if self._first else (_LATER, self.min_words)
        for m in pattern.finditer(self._buf):
            if len(self._buf[: m.end()].split()) >= need:
                return m.end()
        return None
