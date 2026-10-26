"""FastAPI WebSocket server + browser client.

Protocol on /ws:
  client -> server : binary = int16 PCM, 16 kHz mono (any frame size)   | text = {"type": "reset"}
  server -> client : binary = int16 PCM, 24 kHz mono (TTS audio)        | text = JSON events:
                     user_speaking, interrupt, user_text, assistant_text, metrics

Run:  python -m voice_agent.server   (expects an OpenAI-compatible LLM at LLM_BASE_URL)
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .config import load_config
from .llm import LLM
from .pipeline import Models, Session, pcm16_to_float
from .stt import load_stt
from .tools import registry
from .tts import load_tts

log = logging.getLogger("voice_agent")
cfg = load_config()
MODELS: Models | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global MODELS
    log.info("Loading STT (%s) and TTS ...", cfg.stt_backend)
    MODELS = Models(stt=load_stt(cfg), llm=LLM(cfg, registry), tts=load_tts(cfg))
    log.info("Models ready. LLM endpoint: %s (%s)", cfg.llm_base_url, cfg.llm_model)
    yield


app = FastAPI(lifespan=lifespan)
CLIENT_HTML = (Path(__file__).parent / "static" / "client.html").read_text(encoding="utf-8")


@app.get("/")
async def index():
    return HTMLResponse(CLIENT_HTML)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    send_lock = asyncio.Lock()  # turn task and receive loop both send; keep frames whole

    async def send_audio(data: bytes):
        async with send_lock:
            await ws.send_bytes(data)

    async def send_event(event: dict):
        async with send_lock:
            await ws.send_text(json.dumps(event))

    session = Session(MODELS, cfg, send_audio, send_event)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await session.on_audio(pcm16_to_float(msg["bytes"]))
            elif msg.get("text") and json.loads(msg["text"]).get("type") == "reset":
                await session.interrupt()
                session = Session(MODELS, cfg, send_audio, send_event)
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_ping_interval=20)
