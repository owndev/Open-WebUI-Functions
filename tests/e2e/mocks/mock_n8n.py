"""
Mock of n8n webhooks for pipelines/n8n/n8n.py.

Set the pipe's ``N8N_URL`` valve to ``http://127.0.0.1:<port>/webhook/<scenario>``.

Non-streaming scenarios (``Content-Type: application/json`` unless noted)
  json             {"output": "Hello from n8n (json)."}
  json-tools       [{"output": ..., "intermediateSteps": [2 tool calls]}]  (list form)
  json-tools-dict  {"output": ..., "intermediateSteps": [...]}            (dict form)
  json-usage       {"output": ..., "usage": {prompt/completion/total tokens}}
  json-think       {"output": "<think>...</think>Final answer ..."}
  json-custom      {"reply": ..., "output": ...}  (for RESPONSE_FIELD="reply")
  text             text/plain body
  error            HTTP 500 {"message": ..., "hint": ...}
  slow-json        {"output": ...} after 20 s (Stop while waiting for the reply)

Streaming scenarios (n8n "Respond to Webhook: streaming" and friends). NDJSON
streams are ``application/json`` + chunked + ``Cache-Control: no-cache`` (what
n8n sends), SSE streams ``text/event-stream``; every piece is its own write.
  stream-ndjson         n8n native streaming: one JSON object per line
                        ({"type": "begin"|"item"|"end", "content": ...})
  stream-sse-mixed      text/event-stream mixing "data: {json}" events and plain
                        lines, every piece written as its own network chunk
  stream-sse-coalesced  the same bytes in a single write
  stream-plain          text/event-stream of plain text lines only
  stream-utf8-split     NDJSON with raw (unescaped) UTF-8 content, the bytes cut
                        right after every multi-byte lead byte, 0.3 s per write
  stream-braces         n8n items with braces and quotes inside the strings,
                        written back to back ('{..}{..}', no line break between
                        them, one write each), so only a string-aware scanner
                        finds where an object ends
  stream-sse-fields     SSE with event:/id:/retry: fields and a multi-line data:
  stream-openai         OpenAI chat.completion.chunk events (deltas, finish and
                        usage chunk, [DONE]) in one write
  stream-error-chunk    NDJSON begin / {"type": "error", ...} / end
  stream-midfail        NDJSON begin + one item, then the connection is cut
  stream-large-flat     one flat 400 KB JSON object ({"output": "x" * 400000})
                        in 1 KiB writes, 30 ms apart (a slow upstream: every
                        write reaches the pipe as its own network chunk)
  slow-ndjson           40 NDJSON items 0.5 s apart (Stop during the stream)

Any request whose ``chatInput`` is an Open WebUI task prompt (title/tags/
follow-ups) gets ``{"output": "<task JSON>"}`` regardless of the scenario.

usage: python mock_n8n.py [--port 9103]
"""

import argparse
import asyncio
import json
import time

from aiohttp import web

from common import annotate, new_app, record, task_answer

TOOL_STEPS = [
    {
        "action": {
            "tool": "Calculator",
            "toolInput": {"input": "2+2"},
            "toolCallId": "call_calc_1",
            "log": "Invoking Calculator with 2+2",
        },
        "observation": "4",
    },
    {
        "action": {
            "tool": "Wikipedia",
            "toolInput": {"query": "Open WebUI"},
            "toolCallId": "call_wiki_2",
            "log": "Invoking Wikipedia",
        },
        "observation": '{"title": "Open WebUI", "summary": "Self-hosted AI UI"}',
    },
]
USAGE = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
NDJSON = "application/json; charset=utf-8"
SSE = "text/event-stream"
NDJSON_META = {"nodeId": "agent-1", "nodeName": "AI Agent", "itemIndex": 0}
SSE_PIECES = [
    'data: {"content": "SSE part one. "}\n\n',
    "plain line between events\n",
    'data: {"text": "SSE part two."}\n\n',
    ": keep-alive comment\n\n",
    "data: [DONE]\n\n",
]
JSON_ANSWERS = {
    "json": {"output": "Hello from n8n (json)."},
    "json-tools": [{"output": "The answer is 4.", "intermediateSteps": TOOL_STEPS}],
    "json-tools-dict": {"output": "Dict answer is 4.", "intermediateSteps": TOOL_STEPS},
    "json-usage": {"output": "Hello with usage.", "usage": USAGE},
    "json-think": {"output": "<think>Step one\nStep two</think>Final answer."},
    "json-custom": {"reply": "Custom field answer.", "output": "Fallback output."},
}
UTF8_ITEMS = ["Größe ", "naïve ", "日本 ", "🙂"]
BRACE_ITEMS = ["a } b ", "{ c } ", '"q" {']
ERROR_CHUNK = "Tool node failed: quota exceeded"
LARGE_FLAT_SIZE = 400_000
SLOW_JSON_DELAY = 20
STREAM_DELAY = 0.05


def _ndjson(items: list, ensure_ascii: bool = True) -> list:
    """n8n native streaming: begin, one item per content piece, end."""
    lines = [{"type": "begin", "metadata": NDJSON_META}]
    lines += [{"type": "item", "content": c, "metadata": NDJSON_META} for c in items]
    lines.append({"type": "end", "metadata": NDJSON_META})
    return [json.dumps(x, ensure_ascii=ensure_ascii) + "\n" for x in lines]


def _split_after_lead_bytes(raw: bytes) -> list:
    """Cut ``raw`` right after every UTF-8 lead byte, so every multi-byte
    character is split across two writes."""
    pieces, start = [], 0
    for i, byte in enumerate(raw):
        if byte >= 0xC0:
            pieces.append(raw[start : i + 1])
            start = i + 1
    pieces.append(raw[start:])
    return pieces


def _openai_events() -> str:
    base = {
        "id": "chatcmpl-n8n",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "n8n-mock",
    }
    deltas = [{"role": "assistant", "content": "OpenAI "}, {"content": "style."}]
    events = [
        {**base, "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
        for d in deltas
    ]
    events.append(
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    )
    events.append({**base, "choices": [], "usage": USAGE})
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def _large_flat() -> list:
    raw = json.dumps({"output": "x" * LARGE_FLAT_SIZE}) + "\n"
    return [raw[i : i + 1024] for i in range(0, len(raw), 1024)]


def _back_to_back(items: list) -> list:
    """Begin and end lines, the items as JSON objects back to back on one line
    (each object its own write)."""
    lines = _ndjson(items)
    objects = [line.rstrip("\n") for line in lines[1:-1]]
    objects[-1] += "\n"
    return [lines[0], *objects, lines[-1]]


def _error_chunk() -> list:
    lines = [
        {"type": "begin", "metadata": NDJSON_META},
        {"type": "error", "content": ERROR_CHUNK, "metadata": NDJSON_META},
        {"type": "end", "metadata": NDJSON_META},
    ]
    return [json.dumps(x) + "\n" for x in lines]


def _stream(scenario: str):
    """(content type, body pieces, pause after each write), or None."""
    if scenario == "stream-ndjson":
        items = ["<think>Pondering</think>", "Hello ", "from n8n ", "(ndjson stream)."]
        return NDJSON, _ndjson(items), STREAM_DELAY
    if scenario == "stream-sse-mixed":
        return SSE, SSE_PIECES, STREAM_DELAY
    if scenario == "stream-sse-coalesced":
        return SSE, ["".join(SSE_PIECES)], STREAM_DELAY
    if scenario == "stream-plain":
        return SSE, ["Plain line one\n", "Plain line two\n", "Tail"], STREAM_DELAY
    if scenario == "stream-utf8-split":
        raw = "".join(_ndjson(UTF8_ITEMS, ensure_ascii=False)).encode()
        return NDJSON, _split_after_lead_bytes(raw), 0.3
    if scenario == "stream-braces":
        return NDJSON, _back_to_back(BRACE_ITEMS), STREAM_DELAY
    if scenario == "stream-sse-fields":
        pieces = [
            'event: message\nid: 1\nretry: 1000\ndata: {"content": "Event one. "}\n\n',
            "data: multi\ndata: line\n\n",
            "data: [DONE]\n\n",
        ]
        return SSE, pieces, STREAM_DELAY
    if scenario == "stream-openai":
        return SSE, [_openai_events()], STREAM_DELAY
    if scenario == "stream-error-chunk":
        return NDJSON, _error_chunk(), STREAM_DELAY
    if scenario == "stream-large-flat":
        return NDJSON, _large_flat(), 0.03
    if scenario == "slow-ndjson":
        return NDJSON, _ndjson([f"tick{i} " for i in range(40)]), 0.5
    return None


async def _write_pieces(
    request: web.Request, content_type: str, pieces: list, delay: float
) -> web.StreamResponse:
    resp = web.StreamResponse(
        headers={"Content-Type": content_type, "Cache-Control": "no-cache"}
    )
    resp.enable_chunked_encoding()
    await resp.prepare(request)
    try:
        for piece in pieces:
            await resp.write(piece if isinstance(piece, bytes) else piece.encode())
            await asyncio.sleep(delay)
        await resp.write_eof()
    except ConnectionResetError:  # stopped: the pipe closed the connection
        annotate(request, client_gone=True)
    return resp


async def _midfail(request: web.Request) -> web.StreamResponse:
    """NDJSON begin + one item, then the connection is closed without the
    terminating chunk (an upstream that breaks off mid-stream)."""
    resp = web.StreamResponse(
        headers={"Content-Type": NDJSON, "Cache-Control": "no-cache"}
    )
    resp.enable_chunked_encoding()
    await resp.prepare(request)
    for piece in _ndjson(["partial "])[:2]:
        await resp.write(piece.encode())
        await asyncio.sleep(STREAM_DELAY)
    await asyncio.sleep(0.2)
    request.transport.close()
    return resp


async def webhook(request: web.Request) -> web.StreamResponse:
    scenario = request.match_info["scenario"]
    body = await record(request, scenario=scenario)
    chat_input = body.get("chatInput", "") if isinstance(body, dict) else ""
    task = task_answer(chat_input)
    annotate(request, task=bool(task))
    if task:
        return web.json_response({"output": task})
    if scenario in JSON_ANSWERS:
        return web.json_response(JSON_ANSWERS[scenario])
    if scenario == "text":
        return web.Response(text="Plain text answer from n8n.")
    if scenario == "error":
        return web.json_response(
            {"message": "Workflow could not be started!", "hint": "Activate it."},
            status=500,
        )
    if scenario == "slow-json":
        await asyncio.sleep(SLOW_JSON_DELAY)
        return web.json_response({"output": "Slow answer."})
    if scenario == "stream-midfail":
        return await _midfail(request)
    stream = _stream(scenario)
    if not stream:
        return web.json_response(
            {"message": f"unknown scenario {scenario}"}, status=404
        )
    return await _write_pieces(request, *stream)


async def fallback(request: web.Request) -> web.Response:
    await record(request)
    return web.json_response({"message": "not found"}, status=404)


def make_app() -> web.Application:
    app = new_app()
    app.router.add_post("/webhook/{scenario}", webhook)
    app.router.add_route("*", "/{tail:.*}", fallback)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="n8n webhook mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9103)
    args = parser.parse_args()
    web.run_app(make_app(), host=args.host, port=args.port)
