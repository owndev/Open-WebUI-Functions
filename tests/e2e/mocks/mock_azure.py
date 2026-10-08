"""
Mock of Azure OpenAI / Azure AI Foundry chat completions.

Point ``AZURE_AI_ENDPOINT`` at any URL on this mock whose path ends in
``/chat/completions`` (deployment path, Foundry ``/models`` path, ...).

Auth: ``api-key: mock-key-123`` or ``Authorization: Bearer mock-key-123``,
anything else -> HTTP 401 (Azure's error text).

Behaviour
  - model: ``x-ms-model-mesh-model-name`` header, else body ``model``, else the
    deployment name from the path; the plain answer echoes it:
    "Hello from mock Azure (<model>)."
  - streaming: like Azure, a first chunk with ``choices: []`` and
    ``prompt_filter_results``; a final usage chunk when
    ``stream_options.include_usage`` is set
  - ``data_sources`` in the body -> "On Your Data" (Azure AI Search) answer with
    ``context.citations`` (3 documents, the answer references [doc1] and [doc2]),
    ``context.all_retrieved_documents`` when requested via include_contexts.
    ``stream_options`` together with ``data_sources`` -> HTTP 400 (Azure rejects
    that combination)
  - Open WebUI task prompts (title/tags/follow-ups) get the JSON they expect;
    with ``data_sources`` the citations context is still attached
  - last user message containing ``force-400`` -> HTTP 400 JSON (content filter),
    ``force-500-text`` -> HTTP 500 text/plain, ``split-tokens`` -> streamed
    citation markers split across deltas ("[", "doc", "1", "]")

Recorded entries carry ``auth_mode``, ``model_header``, ``resolved_model`` and
``message_problems`` (Azure-style validation of the ``messages`` list).

usage: python mock_azure.py [--port 9102]
"""

import argparse
import asyncio
import json
import time

from aiohttp import web

from common import annotate, last_user_text, new_app, record, task_answer

API_KEY = "mock-key-123"
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
CITATIONS = [
    {
        "content": "The X100 charges via USB-C at up to 65 W.",
        "title": "X100 Product Manual",
        "url": "https://docs.example.com/x100/manual.pdf",
        "filepath": "manual.pdf",
        "chunk_id": "0",
    },
    {
        "content": "All devices come with a two-year limited warranty.",
        "title": "Warranty FAQ",
        "url": None,
        "filepath": "faq/warranty.html",
        "chunk_id": "1",
    },
    {
        "content": "Unrelated retrieved chunk that the answer does not cite.",
        "title": "Release Notes",
        "url": "https://docs.example.com/x100/release-notes",
        "filepath": "release-notes.md",
        "chunk_id": "2",
    },
]
SCORES = [
    {"original_search_score": 42.5, "rerank_score": 3.2, "filter_reason": "rerank"},
    {"original_search_score": 12.0, "filter_reason": "score"},
    {"original_search_score": 3.1, "rerank_score": 0.4, "filter_reason": "rerank"},
]
OYD_TOKENS = [
    "The X100 ",
    "charges via USB-C ",
    "[doc1]",
    ". It has a ",
    "two-year warranty ",
    "[doc2]",
    ".",
]
OYD_TOKENS_SPLIT = [
    "The X100 ",
    "charges via USB-C ",
    "[",
    "doc",
    "1",
    "]",
    ". It has a ",
    "two-year warranty ",
    "[doc",
    "2]",
    ".",
]
VALID_ROLES = {"system", "developer", "user", "assistant", "tool"}
VALID_PART_TYPES = {"text", "image_url", "input_audio", "refusal", "file"}


def _error(status: int, code: str, message: str) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message}}, status=status
    )


def _resolve_model(request: web.Request, body: dict):
    header = request.headers.get("x-ms-model-mesh-model-name")
    if header:
        return header
    if body.get("model"):
        return body["model"]
    if "/deployments/" in request.path:
        return request.path.split("/deployments/", 1)[1].split("/", 1)[0]
    return None


def _auth_mode(request: web.Request):
    if request.headers.get("api-key") == API_KEY:
        return "api-key"
    if request.headers.get("authorization") == f"Bearer {API_KEY}":
        return "bearer"
    return None


def validate_messages(messages) -> list:
    """Approximate Azure OpenAI's validation of the ``messages`` list."""
    problems, open_calls = [], set()
    for i, message in enumerate(messages):
        role = message.get("role")
        if role not in VALID_ROLES:
            problems.append(f"messages[{i}].role={role!r} not allowed")
            continue
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            problems.append(f"messages[{i}].content is {type(content).__name__}")
        for j, part in enumerate(content if isinstance(content, list) else []):
            if not isinstance(part, dict) or part.get("type") not in VALID_PART_TYPES:
                problems.append(f"messages[{i}].content[{j}] invalid part")
        for call in message.get("tool_calls") or []:
            open_calls.add(call.get("id"))
        if role == "tool" and message.get("tool_call_id") not in open_calls:
            problems.append(f"messages[{i}] tool_call_id without a tool call")
        if role in ("user", "system") and content is None:
            problems.append(f"messages[{i}] ({role}) content is null")
    return problems


def _context(body: dict) -> dict:
    context = {"citations": CITATIONS, "intent": '["x100 charging warranty"]'}
    for source in body.get("data_sources") or []:
        include = (source.get("parameters") or {}).get("include_contexts") or []
        if "all_retrieved_documents" in include:
            context["all_retrieved_documents"] = [
                {**doc, **score, "search_queries": ["x100"], "data_source_index": 0}
                for doc, score in zip(CITATIONS, SCORES)
            ]
    return context


async def chat_completions(request: web.Request) -> web.StreamResponse:
    body = await record(request)
    if not isinstance(body, dict):
        return _error(400, "BadRequest", "invalid JSON")
    model = _resolve_model(request, body)
    auth = _auth_mode(request)
    problems = validate_messages(body.get("messages") or [])
    annotate(
        request,
        auth_mode=auth,
        model_header=request.headers.get("x-ms-model-mesh-model-name"),
        resolved_model=model,
        message_problems=problems,
    )
    if auth is None:
        return _error(
            401,
            "401",
            "Access denied due to invalid subscription key or wrong API endpoint.",
        )
    text = last_user_text(body.get("messages"))
    oyd = bool(body.get("data_sources"))
    if "force-400" in text:
        return _error(
            400,
            "content_filter",
            "The response was filtered due to the prompt triggering Azure OpenAI's "
            "content management policy.",
        )
    if "force-500-text" in text:
        return web.Response(status=500, text="upstream exploded (mock)")
    if problems:
        return _error(400, "BadRequest", "Invalid messages: " + "; ".join(problems))
    if oyd and "stream_options" in body:
        return _error(
            400,
            "400",
            "Validation error at #/stream_options: Extra inputs are not permitted",
        )

    task = task_answer(text)
    if task:
        tokens = [task]
    elif oyd:
        tokens = OYD_TOKENS_SPLIT if "split-tokens" in text else OYD_TOKENS
    else:
        tokens = ["Hello ", "from ", "mock ", "Azure ", f"({model})", "."]

    if body.get("stream"):
        return await _stream(request, body, model, tokens, oyd)

    message = {"role": "assistant", "content": "".join(tokens)}
    if oyd:
        message["context"] = _context(body)
    return web.json_response(
        {
            "id": f"chatcmpl-mock-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": USAGE,
        }
    )


async def _stream(request, body, model, tokens, oyd) -> web.StreamResponse:
    resp = web.StreamResponse(
        headers={"Content-Type": "text/event-stream", "apim-request-id": "mock-apim"}
    )
    await resp.prepare(request)
    base = {
        "id": f"chatcmpl-mock-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }

    async def send(obj) -> None:
        payload = obj if isinstance(obj, str) else json.dumps(obj)
        await resp.write(f"data: {payload}\n\n".encode())
        await asyncio.sleep(0.02)

    def delta(d, finish=None) -> dict:
        return {**base, "choices": [{"index": 0, "finish_reason": finish, "delta": d}]}

    if oyd:
        await send(delta({"role": "assistant", "context": _context(body)}))
    else:
        filter_results = [
            {
                "prompt_index": 0,
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"}
                },
            }
        ]
        await send({**base, "choices": [], "prompt_filter_results": filter_results})
        await send(delta({"role": "assistant", "content": ""}))
    for token in tokens:
        await send(delta({"content": token}))
    await send(delta({}, "stop"))
    if (body.get("stream_options") or {}).get("include_usage"):
        await send({**base, "choices": [], "usage": USAGE})
    await send("[DONE]")
    await resp.write_eof()
    return resp


async def fallback(request: web.Request) -> web.Response:
    await record(request)
    return _error(404, "404", "Resource not found")


def make_app() -> web.Application:
    app = new_app()
    app.router.add_post("/{path:.*}/chat/completions", chat_completions)
    app.router.add_route("*", "/{tail:.*}", fallback)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Azure OpenAI / AI Foundry mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9102)
    args = parser.parse_args()
    web.run_app(make_app(), host=args.host, port=args.port)
