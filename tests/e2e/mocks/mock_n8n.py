"""
Mock of n8n webhooks for pipelines/n8n/n8n.py.

Set the pipe's ``N8N_URL`` valve to ``http://127.0.0.1:<port>/webhook/<scenario>``.

Non-streaming scenarios (``Content-Type: application/json`` unless noted)
  json             {"output": "Hello from n8n (json)."}
  json-tools       [{"output": ..., "intermediateSteps": [2 tool calls]}]  (list form)
  json-tools-dict  {"output": ..., "intermediateSteps": [...]}            (dict form)
  json-usage       {"output": ..., "usage": {prompt/completion/total tokens}}
  json-think       {"output": "<think>...</think>Final answer ..."}
  text             text/plain body
  error            HTTP 500 {"message": ..., "hint": ...}

Streaming scenarios (n8n "Respond to Webhook: streaming" and friends)
  stream-ndjson         n8n native streaming: one JSON object per line
                        ({"type": "begin"|"item"|"end", "content": ...})
  stream-sse-mixed      text/event-stream mixing "data: {json}" events and plain
                        lines, every piece written as its own network chunk
  stream-sse-coalesced  the same bytes in a single write
  stream-plain          text/event-stream of plain text lines only

Any request whose ``chatInput`` is an Open WebUI task prompt (title/tags/
follow-ups) gets ``{"output": "<task JSON>"}`` regardless of the scenario.

usage: python mock_n8n.py [--port 9103]
"""

import argparse
import asyncio
import json

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
}


def _stream_pieces(scenario: str):
    """(content type, list of body pieces) for a streaming scenario."""
    if scenario == "stream-ndjson":
        items = ["<think>Pondering</think>", "Hello ", "from n8n ", "(ndjson stream)."]
        lines = [{"type": "begin", "metadata": NDJSON_META}]
        lines += [
            {"type": "item", "content": c, "metadata": NDJSON_META} for c in items
        ]
        lines.append({"type": "end", "metadata": NDJSON_META})
        return "application/json; charset=utf-8", [json.dumps(x) + "\n" for x in lines]
    if scenario == "stream-sse-mixed":
        return "text/event-stream", SSE_PIECES
    if scenario == "stream-sse-coalesced":
        return "text/event-stream", ["".join(SSE_PIECES)]
    if scenario == "stream-plain":
        return "text/event-stream", ["Plain line one\n", "Plain line two\n", "Tail"]
    return None, None


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
    content_type, pieces = _stream_pieces(scenario)
    if not pieces:
        return web.json_response(
            {"message": f"unknown scenario {scenario}"}, status=404
        )
    resp = web.StreamResponse(
        headers={"Content-Type": content_type, "Cache-Control": "no-cache"}
    )
    resp.enable_chunked_encoding()
    await resp.prepare(request)
    for piece in pieces:
        await resp.write(piece.encode())
        await asyncio.sleep(0.05)
    await resp.write_eof()
    return resp


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
