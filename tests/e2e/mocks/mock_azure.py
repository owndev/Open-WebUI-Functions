"""
Mock of Azure OpenAI / Azure AI Foundry chat completions.

Point ``AZURE_AI_ENDPOINT`` at any URL on this mock whose path ends in
``/chat/completions`` (deployment path, Foundry ``/models`` path, ...) and that
carries an ``api-version`` query parameter (without one -> HTTP 404 "Resource
not found", like Azure).

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
  - ``data_sources`` together with ``tools`` (``tool_choice`` not "none") -> the
    data sources are ignored and the plain answer comes back without context.
    Azure documents this for On Your Data function calling, and Open WebUI
    0.10+ adds its built-in tools to every browser chat
  - Open WebUI task prompts (title/tags/follow-ups) get the JSON they expect;
    with ``data_sources`` the citations context is still attached
  - trigger words in the last user message:
      ``force-400``      HTTP 400 JSON (content filter)
      ``force-500-text`` HTTP 500 text/plain
      ``split-tokens``   (On Your Data) citation markers split across deltas
                         ("[", "doc", "1", "]" and "[doc", "2]")
      ``split-link``     (On Your Data) an already linked reference split
                         across deltas ("[[", "doc", "1", "]](", url, ")")
      ``no-finish``      stream without a finish_reason chunk that ends on a
                         reference ("... [doc2]" without the final ".")
      ``no-refs``        (On Your Data) answer without any [docX] reference
      ``paren-url``      (On Your Data) doc1's URL contains "(v2)"
      ``big-context``    (On Your Data) context event of ~300 KB (one SSE line)
      ``huge-context``   (On Your Data) context event of ~5 MiB (one SSE line)
      ``content-null``   (On Your Data, non-stream) ``content: null`` with
                         finish_reason content_filter and the citations context

Recorded entries carry ``auth_mode``, ``model_header``, ``resolved_model``,
``api_version``, ``data_sources_ignored`` and ``message_problems`` (Azure-style
validation of the ``messages`` list).

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
OYD_TOKENS_SPLIT_LINK = [
    "See ",
    "[[",
    "doc",
    "1",
    "]](",
    "https://docs.example.com/x100/manual.pdf",
    ")",
    " and ",
    "[doc",
    "2",
    "]",
    ".",
]
OYD_TOKENS_NO_REFS = [
    "The requested information ",
    "is not available ",
    "in the retrieved data.",
]
PAREN_URL = "https://docs.example.com/x100/manual_(v2).pdf"
# Filler of the big/huge context documents. Each of the 3 citations carries it
# once in "citations" and once in "all_retrieved_documents" (the suites request
# that via include_contexts), so the context event is ~6x its size.
FILLER_WORD = "bigdoc "
BIG_DOC_CHARS = 50_000  # ~300 KB context event (> aiohttp's 128 KiB default)
HUGE_DOC_CHARS = 900_000  # ~5.4 MB context event (> the 4 MiB the pipe reads)
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


def _filler(chars: int) -> str:
    return (FILLER_WORD * (chars // len(FILLER_WORD) + 1))[:chars]


def _citations(text: str) -> list:
    """The 3 citations, changed by the trigger words of the question."""
    if "paren-url" in text:
        return [{**CITATIONS[0], "url": PAREN_URL}, *CITATIONS[1:]]
    if "huge-context" in text:
        return [{**c, "content": _filler(HUGE_DOC_CHARS)} for c in CITATIONS]
    if "big-context" in text:
        return [{**c, "content": _filler(BIG_DOC_CHARS)} for c in CITATIONS]
    return CITATIONS


def _context(body: dict, text: str) -> dict:
    citations = _citations(text)
    context = {"citations": citations, "intent": '["x100 charging warranty"]'}
    for source in body.get("data_sources") or []:
        include = (source.get("parameters") or {}).get("include_contexts") or []
        if "all_retrieved_documents" in include:
            context["all_retrieved_documents"] = [
                {**doc, **score, "search_queries": ["x100"], "data_source_index": 0}
                for doc, score in zip(citations, SCORES)
            ]
    return context


def _oyd_tokens(text: str) -> list:
    if "split-tokens" in text:
        return OYD_TOKENS_SPLIT
    if "split-link" in text:
        return OYD_TOKENS_SPLIT_LINK
    if "no-refs" in text:
        return OYD_TOKENS_NO_REFS
    return OYD_TOKENS


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
        api_version=request.query.get("api-version"),
        message_problems=problems,
    )
    if not request.query.get("api-version"):
        return _error(404, "404", "Resource not found")
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
    if oyd and body.get("tools") and body.get("tool_choice") != "none":
        # Azure OpenAI On Your Data: with tools (tool_choice not "none") the
        # data sources are ignored and the model answers on its own.
        oyd = False
        annotate(request, data_sources_ignored=True)

    task = task_answer(text)
    if task:
        tokens = [task]
    elif oyd:
        tokens = _oyd_tokens(text)
    else:
        tokens = ["Hello ", "from ", "mock ", "Azure ", f"({model})", "."]

    if body.get("stream"):
        return await _stream(request, body, model, tokens, oyd, text)

    message = {"role": "assistant", "content": "".join(tokens)}
    finish = "stop"
    if oyd:
        message["context"] = _context(body, text)
        if "content-null" in text:
            message["content"] = None  # e.g. a filtered completion
            finish = "content_filter"
    return web.json_response(
        {
            "id": f"chatcmpl-mock-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": USAGE,
        }
    )


async def _stream(request, body, model, tokens, oyd, text) -> web.StreamResponse:
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
        await send(delta({"role": "assistant", "context": _context(body, text)}))
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
    no_finish = "no-finish" in text
    if no_finish and tokens and tokens[-1] == ".":
        tokens = tokens[:-1]  # end on a reference that is held back until [DONE]
    for token in tokens:
        await send(delta({"content": token}))
    if not no_finish:
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
