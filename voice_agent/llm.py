"""Streaming LLM client with an in-loop tool-calling cycle.

Works against any OpenAI-compatible server (vLLM, Ollama, llama.cpp server). Text deltas are
yielded as they arrive; tool calls are executed between rounds and their messages appended to
`messages`. Cancelling the consuming task closes the HTTP stream, and vLLM/Ollama then abort
generation server-side, which frees the GPU for the next turn immediately.
"""
from typing import AsyncIterator

from openai import AsyncOpenAI

from .config import Config
from .tools import ToolRegistry


class LLM:
    def __init__(self, cfg: Config, registry: ToolRegistry):
        self.cfg = cfg
        self.registry = registry
        self.client = AsyncOpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)

    async def respond(self, messages: list[dict]) -> AsyncIterator[str]:
        tools = self.registry.schemas()
        for _ in range(self.cfg.max_tool_rounds + 1):
            kwargs = dict(
                model=self.cfg.llm_model,
                messages=messages,
                stream=True,
                temperature=self.cfg.llm_temperature,
                max_tokens=self.cfg.llm_max_tokens,
            )
            if tools:
                kwargs["tools"] = tools
            if self.cfg.llm_extra_body:
                kwargs["extra_body"] = self.cfg.llm_extra_body

            stream = await self.client.chat.completions.create(**kwargs)
            text_parts: list[str] = []
            calls: dict[int, dict] = {}
            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        text_parts.append(delta.content)
                        yield delta.content
                    for tc in delta.tool_calls or []:
                        slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
            finally:
                await stream.close()

            if not calls:
                return

            ordered = [calls[i] | {"id": calls[i]["id"] or f"call_{i}"} for i in sorted(calls)]
            messages.append({
                "role": "assistant",
                "content": "".join(text_parts) or None,
                "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                    for c in ordered
                ],
            })
            for c in ordered:
                result = await self.registry.call(c["name"], c["arguments"])
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
