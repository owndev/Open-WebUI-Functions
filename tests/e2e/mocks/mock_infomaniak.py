"""
Mock of the Infomaniak AI API (OpenAI-compatible chat completions).

Set the pipe's ``INFOMANIAK_BASE_URL`` valve to ``http://127.0.0.1:<port>``.

Routes
  GET  /1/ai/models                                    model list (llm + one stt)
  POST /2/ai/{product_id}/openai/v1/chat/completions   chat completions

Models
  mixtral, llama3  normal answers; streaming sends one content event per write,
                   then finish + usage chunk (``stream_options.include_usage``) +
                   [DONE] together in one write
  burst            the whole SSE stream in ONE write (coalesced events, what a
                   fast upstream or a buffering proxy produces)
  split            the SSE stream cut into 23-byte writes (events split mid-JSON)
  bad-model        HTTP 400 with an OpenAI-style error body
  whisper          type "stt", must not be listed by the pipe

Open WebUI task prompts (title/tags/follow-ups) get the JSON they expect.

usage: python mock_infomaniak.py [--port 9104]
"""

import argparse
import asyncio
import json
import time

from aiohttp import web

from common import last_user_text, new_app, record, task_answer

MODELS = [
    {"name": "mixtral", "type": "llm", "description": "Mixtral Mock"},
    {"name": "llama3", "type": "llm", "description": "Llama 3 Mock"},
    {"name": "whisper", "type": "stt", "description": "not an llm"},
    {"name": "bad-model", "type": "llm", "description": "Error trigger"},
    {"name": "burst", "type": "llm", "description": "All SSE events in one write"},
    {"name": "split", "type": "llm", "description": "SSE events split mid-JSON"},
]
USAGE_STREAM = {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
USAGE = {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14}


async def list_models(request: web.Request) -> web.Response:
    await record(request)
    return web.json_response({"result": "success", "data": MODELS})


def _sse_events(model: str, pieces: list, include_usage: bool) -> list:
    base = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }
    events = []
    for i, piece in enumerate(pieces):
        delta = (
            {"role": "assistant", "content": piece} if i == 0 else {"content": piece}
        )
        events.append(
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        )
    events.append(
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    )
    if include_usage:
        events.append({**base, "choices": [], "usage": USAGE_STREAM})
    return [f"data: {json.dumps(e)}\n\n" for e in events] + ["data: [DONE]\n\n"]


async def chat_completions(request: web.Request) -> web.StreamResponse:
    body = await record(request)
    body = body if isinstance(body, dict) else {}
    model = body.get("model", "")
    if model == "bad-model":
        return web.json_response(
            {
                "result": "error",
                "error": {
                    "code": "model_not_found",
                    "message": "Mock: model not found",
                },
            },
            status=400,
        )
    task = task_answer(last_user_text(body.get("messages")))
    if not body.get("stream"):
        content = task or "Hello from Infomaniak (non-stream)."
        return web.json_response(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": USAGE,
            }
        )
    label = model if model in ("burst", "split") else "stream"
    pieces = [task] if task else ["Hello ", "from ", "Infomaniak ", f"({label})."]
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    events = _sse_events(model, pieces, include_usage)

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    if model == "burst":
        await resp.write("".join(events).encode())
    elif model == "split":
        raw = "".join(events).encode()
        for start in range(0, len(raw), 23):
            await resp.write(raw[start : start + 23])
            await asyncio.sleep(0.02)
    else:
        # Content events one per write; the closing events (finish, usage,
        # [DONE]) in one write, as many OpenAI-compatible servers flush them.
        tail = 3 if include_usage else 2
        for event in events[:-tail]:
            await resp.write(event.encode())
            await asyncio.sleep(0.05)
        await resp.write("".join(events[-tail:]).encode())
    await resp.write_eof()
    return resp


async def fallback(request: web.Request) -> web.Response:
    await record(request)
    return web.json_response({"error": {"message": "mock: no route"}}, status=404)


def make_app() -> web.Application:
    app = new_app()
    app.router.add_get("/1/ai/models", list_models)
    app.router.add_post("/2/ai/{pid}/openai/v1/chat/completions", chat_completions)
    app.router.add_route("*", "/{tail:.*}", fallback)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infomaniak AI API mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9104)
    args = parser.parse_args()
    web.run_app(make_app(), host=args.host, port=args.port)
