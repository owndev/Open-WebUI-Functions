"""
Tool server mocks for the native tool calling scenarios (``gemini.tools``).

These are tools Open WebUI runs itself, not provider APIs: they are NOT behind
``E2E_MOCK_FAULT`` (a provider fault must not break the tools).

  OpenAPI  127.0.0.1:9111   GET  /openapi.json   the spec Open WebUI loads
                            GET  /weather?city=  operationId get_weather
                            POST /convert        operationId convert_units (JSON body)
                            GET  /lookup?key=    operationId lookup.v2 (a name
                                                 Gemini does not accept: the pipe
                                                 maps it)
           GET /__requests, POST /__reset       record of the OpenAPI calls and of
                                                 the MCP tool calls (path mcp:<tool>)
  MCP      127.0.0.1:9112   /mcp                 FastMCP streamable HTTP (the image's
                                                 ``mcp`` package): mcp_echo, mcp_sum.
                                                 Optional: when it cannot start,
                                                 serve_all.py says so and the MCP
                                                 scenario reports it.

usage: python mock_tools.py [--port 9111] [--mcp-port 9112]
"""

import argparse
import asyncio
import logging
import threading
import time
import warnings

from aiohttp import web

from common import REQUESTS_KEY, new_app, record

HOST = "127.0.0.1"
PORT = 9111
MCP_PORT = 9112
# operationId that is not a valid Gemini function name (dot): Open WebUI keeps it
# as the tool name, the pipe declares it as _gemini_function_name("lookup.v2").
LOOKUP_OPERATION = "lookup.v2"

# record of the running process (shared by the OpenAPI app and the MCP thread)
_CALLS: list = []


def spec(port: int = PORT) -> dict:
    return {
        "openapi": "3.0.3",
        "info": {"title": "E2E Tools", "version": "1.0.0"},
        "servers": [{"url": f"http://{HOST}:{port}"}],
        "paths": {
            "/weather": {
                "get": {
                    "operationId": "get_weather",
                    "summary": "Current weather for a city",
                    "parameters": [
                        {
                            "name": "city",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "City name",
                        }
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            },
            "/convert": {
                "post": {
                    "operationId": "convert_units",
                    "summary": "Convert a value between units",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "value": {"type": "number"},
                                        "unit_from": {"type": "string"},
                                        "unit_to": {"type": "string"},
                                    },
                                    "required": ["value", "unit_from", "unit_to"],
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "ok"}},
                }
            },
            "/lookup": {
                "get": {
                    "operationId": LOOKUP_OPERATION,
                    "summary": "Look a key up (operationId with a dot)",
                    "parameters": [
                        {
                            "name": "key",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            },
        },
    }


async def openapi_json(request: web.Request) -> web.Response:
    await record(request)
    return web.json_response(spec(request.app["port"]))


async def weather(request: web.Request) -> web.Response:
    await record(request)
    city = request.query.get("city")
    return web.json_response({"city": city, "temp_c": 21.5, "sky": "clear"})


async def convert(request: web.Request) -> web.Response:
    body = await record(request)
    body = body if isinstance(body, dict) else {}
    try:
        value = float(body.get("value", 0))
    except (TypeError, ValueError):
        value = 0.0
    pair = (body.get("unit_from"), body.get("unit_to"))
    result = round(value * 2.54, 4) if pair == ("in", "cm") else value
    return web.json_response({"result": result, "echo": body})


async def lookup(request: web.Request) -> web.Response:
    await record(request)
    return web.json_response({"key": request.query.get("key"), "found": True})


def make_app(port: int = PORT) -> web.Application:
    app = new_app(fault_injection=False)
    app[REQUESTS_KEY] = _CALLS  # the MCP tools record into the same list
    app["port"] = port
    app.router.add_get("/openapi.json", openapi_json)
    app.router.add_get("/weather", weather)
    app.router.add_post("/convert", convert)
    app.router.add_get("/lookup", lookup)
    return app


def _record_mcp(tool: str, arguments: dict) -> None:
    _CALLS.append(
        {"t": time.time(), "method": "MCP", "path": f"mcp:{tool}", "body": arguments}
    )


def start_mcp(port: int = MCP_PORT) -> threading.Thread:
    """FastMCP streamable-HTTP server in a daemon thread (own event loop).
    Raises ImportError when the image has no ``mcp`` package."""
    from mcp.server.fastmcp import FastMCP

    # FastMCP installs a rich handler on the root logger: the other mocks' log
    # lines (aiohttp's "Error handling request" blocks that e2e.py parses in
    # mocks.txt) must keep their plain format.
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    server = FastMCP("e2emcp", host=HOST, port=port, log_level="WARNING")
    root.handlers[:] = handlers
    root.setLevel(level)
    # The container runs with PYTHONWARNINGS=always::ResourceWarning (a mock that
    # leaks a session fails run.mocks-log); anyio's memory streams inside the MCP
    # library leak by design and are not the mocks' fault.
    warnings.filterwarnings("ignore", category=ResourceWarning, module=r"anyio\.")

    @server.tool()
    def mcp_echo(text: str, times: int = 1) -> str:
        """Echo a text a number of times."""
        _record_mcp("mcp_echo", {"text": text, "times": times})
        return " ".join([text] * int(times))

    @server.tool()
    def mcp_sum(values: list[float]) -> dict:
        """Sum a list of numbers."""
        _record_mcp("mcp_sum", {"values": values})
        return {"sum": sum(values), "n": len(values)}

    thread = threading.Thread(
        target=lambda: asyncio.run(server.run_streamable_http_async()),
        name="mock-mcp",
        daemon=True,
    )
    thread.start()
    return thread


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool server mocks")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--mcp-port", type=int, default=MCP_PORT)
    args = parser.parse_args()
    start_mcp(args.mcp_port)
    web.run_app(make_app(args.port), host=args.host, port=args.port)
