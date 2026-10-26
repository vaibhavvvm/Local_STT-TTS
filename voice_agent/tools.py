"""Tool registry for LLM function calling.

Add a tool:

    @registry.tool("Describe what it does", {"type": "object", "properties": {...}, "required": [...]})
    async def my_tool(arg1: str) -> str: ...

Tools return short strings: the LLM rephrases them into speech. Keep them fast; every tool call
adds a full LLM round-trip before the answer is spoken.
"""
import ast
import json
import operator
from datetime import datetime
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

import httpx

ToolFn = Callable[..., Awaitable[str]]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, tuple[ToolFn, dict]] = {}

    def tool(self, description: str, parameters: dict):
        def wrap(fn: ToolFn) -> ToolFn:
            self._tools[fn.__name__] = (
                fn,
                {"type": "function", "function": {"name": fn.__name__, "description": description, "parameters": parameters}},
            )
            return fn

        return wrap

    def schemas(self) -> list[dict]:
        return [schema for _, schema in self._tools.values()]

    async def call(self, name: str, arguments: str) -> str:
        if name not in self._tools:
            return f"Error: unknown tool {name!r}"
        try:
            args = json.loads(arguments or "{}")
            return str(await self._tools[name][0](**args))
        except Exception as e:  # tool errors go back to the LLM, which can apologise or retry
            return f"Error: {type(e).__name__}: {e}"


registry = ToolRegistry()


@registry.tool(
    "Get the current date and time in a given IANA timezone.",
    {
        "type": "object",
        "properties": {"timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Kolkata"}},
        "required": ["timezone"],
    },
)
async def get_current_time(timezone: str) -> str:
    return datetime.now(ZoneInfo(timezone)).strftime("%A %d %B %Y, %I:%M %p")


@registry.tool(
    "Get the current weather for a city.",
    {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name, e.g. Pune"}},
        "required": ["city"],
    },
)
async def get_weather(city: str) -> str:
    # Open-Meteo: free, no API key.
    async with httpx.AsyncClient(timeout=5) as client:
        geo = (await client.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1})).json()
        if not geo.get("results"):
            return f"No location found for {city}."
        place = geo["results"][0]
        wx = (
            await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": place["latitude"], "longitude": place["longitude"],
                        "current": "temperature_2m,apparent_temperature,wind_speed_10m,precipitation"},
            )
        ).json()["current"]
    return (
        f"{place['name']}, {place.get('country', '')}: {wx['temperature_2m']}°C "
        f"(feels like {wx['apparent_temperature']}°C), wind {wx['wind_speed_10m']} km/h, "
        f"precipitation {wx['precipitation']} mm."
    )


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@registry.tool(
    "Evaluate an arithmetic expression exactly. Use for any maths.",
    {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. (23*17)/4"}},
        "required": ["expression"],
    },
)
async def calculate(expression: str) -> str:
    return str(_safe_eval(ast.parse(expression, mode="eval").body))
