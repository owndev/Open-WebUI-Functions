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
  - On Your Data retirement (emulated): ``data_sources`` for a model outside
    OYD_MODELS (gpt-4o, gpt-4o-mini, gpt-4.1) -> HTTP 400 "Azure OpenAI On
    Your Data is retired (mock): data_sources rejected for model <model>"
    (the real text after 2026-10-14 is unknown; this is only a marker)
  - Foundry ``/models/...`` path without the model header and without body
    ``model`` -> HTTP 400 "model is required"
  - grounded prompt of the pipeline mode (AZURE_AI_SEARCH_MODE=pipeline, no
    ``data_sources``): the current turn's user message starts with a
    ``<documents>`` block -> the On Your Data answer tokens (trigger words are
    read from the user text after the block) WITHOUT ``context`` (a plain
    model returns none, so the pipe has to add the citations itself); the
    "No documents were found" block -> the no-references answer
  - query generation: a first system message starting with "Write search
    queries for Azure AI Search." -> non-stream JSON answer
    {"queries": ["x100 charging", "x100 warranty"]}; trigger words in the
    transcript: ``qgen-bad-json`` (prose, no JSON), ``qgen-400`` (content
    filter 400), ``qgen-partial`` (["x100 charging", "qfail-x100"]),
    ``qgen-think`` (a <think> block with braces before the JSON),
    ``qgen-many`` (6 queries with a case duplicate and a 405-character one),
    ``qgen-slow`` (answers after 12 s, past the pipe's 10 s timeout)
  - ``use-tool`` with ``tools`` and no ``role: tool`` message yet -> a
    ``tool_calls`` answer for ``get_current_timestamp`` (Open WebUI's built-in
    tool) or else the first tool, arguments ``{}``, finish_reason tool_calls;
    once a tool message is there -> the normal (grounded) answer
  - embeddings: ``POST .../openai/v1/embeddings`` (model in the body,
    api-version optional) and ``POST /<path>/embeddings`` (api-version
    required); api-key / Bearer ``mock-key-123`` or api-key
    ``mock-embed-key-321``; one 8-float vector per input (``dimensions``
    floats when given); an input containing ``embed-fail`` -> HTTP 500
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
      ``huge-context``   (On Your Data) context event of ~9 MB (one SSE line)
      ``content-null``   (On Your Data, non-stream) ``content: null`` with
                         finish_reason content_filter and the citations context
      ``context-too-long`` HTTP 400 context_length_exceeded

Recorded entries carry ``auth_mode``, ``model_header``, ``resolved_model``,
``api_version``, ``data_sources_ignored`` and ``message_problems`` (Azure-style
validation of the ``messages`` list); chat entries also carry what the
pipeline mode injected: ``grounded``, ``prompt_docs`` (the [docN] labels of
the block), ``documents_blocks`` (``<documents>`` blocks outside the
system / developer messages, whose rules mention the tag),
``docs_indices`` / ``current_user_index`` (messages with a block / the current
turn's user message), ``docs_chars``, ``no_docs_block``, ``system_messages``
(system + developer), ``rules_in_system``, ``rules_count`` (rule blocks in
them), ``in_scope_rule`` ("true", "false" or None), ``tool_results_clause``,
``role_information_in_system``,
``tools_present``, ``model_in_body``, ``question`` (user text after the
block); query-generation entries carry ``query_generation`` and
``transcript``; embeddings entries ``embeddings``, ``input``, ``model``,
``dimensions``.

usage: python mock_azure.py [--port 9102]
"""

import argparse
import asyncio
import json
import re
import time

from aiohttp import web

from common import annotate, last_user_text, new_app, record, task_answer

API_KEY = "mock-key-123"
EMBED_KEY = "mock-embed-key-321"  # embedding_dependency.authentication.key
EMBED_DIMS = 8  # vector length of the Search mock's vector fields
# Models Azure OpenAI On Your Data still accepts (emulated retirement).
OYD_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4.1"}
# Pipeline mode (2.9.0): first line of the query-generation system message,
# the system rules and the empty documents block.
QG_MARKER = "Write search queries for Azure AI Search."
QG_QUERIES = ["x100 charging", "x100 warranty"]
# qgen-many: a case duplicate, a 405-character query, more than 3 queries
QG_MANY = [
    "x100 charging",
    "X100 CHARGING",
    "x100 " + "long " * 80,
    "x100 warranty",
    "x100 release notes",
    "x100 manual",
]
QG_SLOW_SECONDS = 12
RULES_MARKER = "## Retrieved documents"
IN_SCOPE_TRUE = "Answer only with information from the documents"
IN_SCOPE_FALSE = "you may answer from your own knowledge"
TOOL_RESULTS_CLAUSE = "and from tool results"
NO_DOCS_TEXT = "No documents were found for this question."
ROLE_INFORMATION = "You are the X100 support assistant (e2e role information)."
# Open WebUI's user message after a tool returned images (tool round).
TOOL_IMAGES_PROMPT = "Here are the images from the tool results above."
DOC_LABEL = re.compile(r"(?m)^\[doc(\d+)\]")
TOOL_CALL_ID = "call_e2e_1"
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
# in "citations" and, when include_contexts asks for them, again in
# "all_retrieved_documents". The citations alone already exceed the limits, so
# the scenarios do not depend on AZURE_AI_INCLUDE_SEARCH_SCORES.
FILLER_WORD = "bigdoc "
# 150 KB of citations (> aiohttp's default 128 KiB line), ~300 KB event
BIG_DOC_CHARS = 50_000
# 4.5 MB of citations (> the 4 MiB the 2.8.0 pipe reads), ~9 MB event
HUGE_DOC_CHARS = 1_500_000
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


def _auth_mode(request: web.Request, embed: bool = False):
    if request.headers.get("api-key") == API_KEY:
        return "api-key"
    if request.headers.get("authorization") == f"Bearer {API_KEY}":
        return "bearer"
    if embed and request.headers.get("api-key") == EMBED_KEY:
        return "embed-key"
    return None


def _text(content) -> str:
    """Text of a message content (string or list of parts)."""
    if isinstance(content, list):
        return " ".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return content if isinstance(content, str) else ""


def _current_user_index(messages: list):
    """The current turn's user message: the last user message before the
    trailing tool-round messages (tool messages, assistant messages with
    tool_calls, Open WebUI's user message for tool images)."""
    i = len(messages) - 1
    while i >= 0:
        message = messages[i]
        role = message.get("role")
        if role == "tool" or (role == "assistant" and message.get("tool_calls")):
            i -= 1
        elif role == "user" and _text(message.get("content")).startswith(
            TOOL_IMAGES_PROMPT
        ):
            i -= 1
        else:
            break
    return i if i >= 0 and messages[i].get("role") == "user" else None


def grounding(body: dict) -> dict:
    """What the pipeline mode injected into the chat request (annotations)."""
    messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
    texts = [_text(m.get("content")) for m in messages]
    current = _current_user_index(messages)
    user_text = texts[current] if current is not None else ""
    grounded = user_text.lstrip().startswith("<documents>")
    block, question = "", user_text
    if grounded:
        end = user_text.rfind("</documents>")
        cut = end + len("</documents>") if end >= 0 else len(user_text)
        block, question = user_text[:cut], user_text[cut:]
    is_system = [m.get("role") in ("system", "developer") for m in messages]
    system = [text for text, flag in zip(texts, is_system) if flag]
    system_text = "\n".join(system)
    # blocks outside the system-like messages (the rules mention "<documents>")
    others = [(i, t) for i, (t, flag) in enumerate(zip(texts, is_system)) if not flag]
    in_scope = None
    if IN_SCOPE_TRUE in system_text:
        in_scope = "true"
    elif IN_SCOPE_FALSE in system_text:
        in_scope = "false"
    return {
        "grounded": grounded,
        "prompt_docs": [int(n) for n in DOC_LABEL.findall(block)],
        "documents_blocks": sum(t.count("<documents>") for _, t in others),
        "docs_indices": [i for i, t in others if "<documents>" in t],
        "current_user_index": current,
        "docs_chars": len(block),
        "no_docs_block": NO_DOCS_TEXT in block,
        "system_messages": len(system),
        "rules_in_system": RULES_MARKER in system_text,
        "rules_count": system_text.count(RULES_MARKER),
        "in_scope_rule": in_scope,
        "tool_results_clause": TOOL_RESULTS_CLAUSE in system_text,
        "role_information_in_system": ROLE_INFORMATION in system_text,
        "tools_present": bool(body.get("tools")),
        "model_in_body": body.get("model"),
        "question": question.strip()[:300],
    }


def _is_query_generation(body: dict) -> bool:
    messages = body.get("messages") or []
    first = messages[0] if messages and isinstance(messages[0], dict) else {}
    return first.get("role") in ("system", "developer") and _text(
        first.get("content")
    ).lstrip().startswith(QG_MARKER)


def _tool_name(body: dict):
    """get_current_timestamp when the request offers it, else the first tool."""
    names = [
        (tool.get("function") or {}).get("name")
        for tool in body.get("tools") or []
        if isinstance(tool, dict)
    ]
    names = [n for n in names if n]
    if not names:
        return None
    return "get_current_timestamp" if "get_current_timestamp" in names else names[0]


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


def _completion(model, message: dict, finish: str = "stop") -> web.Response:
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


async def _query_generation(request: web.Request, body: dict, model):
    """Answer of the pipe's query-generation request (non-stream JSON)."""
    messages = body.get("messages") or []
    transcript = "\n".join(
        _text(m.get("content"))
        for m in messages[1:]
        if isinstance(m, dict) and m.get("role") == "user"
    )
    annotate(request, transcript=transcript[:6000])
    if "qgen-400" in transcript:
        return _error(
            400,
            "content_filter",
            "The response was filtered due to the prompt triggering Azure OpenAI's "
            "content management policy.",
        )
    if "qgen-slow" in transcript:
        await asyncio.sleep(QG_SLOW_SECONDS)
    if "qgen-bad-json" in transcript:
        content = "Sure! Search for the X100 charging specs and the warranty terms."
    elif "qgen-partial" in transcript:
        content = json.dumps({"queries": ["x100 charging", "qfail-x100"]})
    elif "qgen-think" in transcript:
        content = '<think>{"draft": 1}</think>{"queries": ["x100 charging"]}'
    elif "qgen-many" in transcript:
        content = json.dumps({"queries": QG_MANY})
    else:
        content = json.dumps({"queries": QG_QUERIES})
    return _completion(model, {"role": "assistant", "content": content})


async def chat_completions(request: web.Request) -> web.StreamResponse:
    body = await record(request)
    if not isinstance(body, dict):
        return _error(400, "BadRequest", "invalid JSON")
    model = _resolve_model(request, body)
    auth = _auth_mode(request)
    problems = validate_messages(body.get("messages") or [])
    query_generation = _is_query_generation(body)
    info = {} if query_generation else grounding(body)
    annotate(
        request,
        auth_mode=auth,
        model_header=request.headers.get("x-ms-model-mesh-model-name"),
        resolved_model=model,
        api_version=request.query.get("api-version"),
        message_problems=problems,
        query_generation=query_generation,
        **info,
    )
    if not request.query.get("api-version"):
        return _error(404, "404", "Resource not found")
    if auth is None:
        return _error(
            401,
            "401",
            "Access denied due to invalid subscription key or wrong API endpoint.",
        )
    if request.path.startswith("/models/") and not model:
        return _error(
            400,
            "BadRequest",
            "model is required (mock: no x-ms-model-mesh-model-name header and no "
            "model in the body)",
        )
    if body.get("data_sources") and model not in OYD_MODELS:
        return _error(
            400,
            "BadRequest",
            "Azure OpenAI On Your Data is retired (mock): data_sources rejected for "
            f"model {model}",
        )
    if query_generation:
        return await _query_generation(request, body, model)
    grounded = info.get("grounded")
    text = last_user_text(body.get("messages"))
    if grounded:
        # trigger words come from the user's text after the documents block
        messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
        current = _text(messages[info["current_user_index"]].get("content"))
        text = current[current.rfind("</documents>") + len("</documents>") :]
    oyd = bool(body.get("data_sources"))
    if "force-400" in text:
        return _error(
            400,
            "content_filter",
            "The response was filtered due to the prompt triggering Azure OpenAI's "
            "content management policy.",
        )
    if "context-too-long" in text:
        return _error(
            400,
            "context_length_exceeded",
            "This model's maximum context length is 128000 tokens. However, your "
            "messages resulted in 131072 tokens. Please reduce the length of the "
            "messages.",
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

    tool = None
    if "use-tool" in text and not any(
        isinstance(m, dict) and m.get("role") == "tool"
        for m in body.get("messages") or []
    ):
        tool = _tool_name(body)
    task = task_answer(text)
    if task:
        tokens = [task]
    elif oyd:
        tokens = _oyd_tokens(text)
    elif info.get("no_docs_block"):
        tokens = OYD_TOKENS_NO_REFS
    elif grounded:
        tokens = _oyd_tokens(text)
    else:
        tokens = ["Hello ", "from ", "mock ", "Azure ", f"({model})", "."]

    if body.get("stream"):
        return await _stream(request, body, model, tokens, oyd, text, tool)

    if tool:
        call = {
            "id": TOOL_CALL_ID,
            "type": "function",
            "function": {"name": tool, "arguments": "{}"},
        }
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        return _completion(model, message, "tool_calls")
    message = {"role": "assistant", "content": "".join(tokens)}
    finish = "stop"
    if oyd:
        message["context"] = _context(body, text)
        if "content-null" in text:
            message["content"] = None  # e.g. a filtered completion
            finish = "content_filter"
    return _completion(model, message, finish)


async def _stream(
    request, body, model, tokens, oyd, text, tool=None
) -> web.StreamResponse:
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
        if tool:
            call = {
                "index": 0,
                "id": TOOL_CALL_ID,
                "type": "function",
                "function": {"name": tool, "arguments": ""},
            }
            await send(
                delta({"role": "assistant", "content": None, "tool_calls": [call]})
            )
            arguments = {"index": 0, "function": {"arguments": "{}"}}
            await send(delta({"tool_calls": [arguments]}))
            await send(delta({}, "tool_calls"))
            if (body.get("stream_options") or {}).get("include_usage"):
                await send({**base, "choices": [], "usage": USAGE})
            await send("[DONE]")
            await resp.write_eof()
            return resp
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


async def embeddings(request: web.Request) -> web.Response:
    """Azure OpenAI embeddings: the v1 route (model in the body, api-version
    optional) and the dated deployments route (api-version required)."""
    body = await record(request)
    data = body if isinstance(body, dict) else {}
    v1 = request.path.endswith("/openai/v1/embeddings")
    inputs = data.get("input")
    if isinstance(inputs, str):
        inputs = [inputs]
    auth = _auth_mode(request, embed=True)
    annotate(
        request,
        embeddings=True,
        v1=v1,
        auth_mode=auth,
        api_version=request.query.get("api-version"),
        model_header=request.headers.get("x-ms-model-mesh-model-name"),
        input=inputs,
        model=data.get("model"),
        dimensions=data.get("dimensions"),
    )
    if not v1 and not request.query.get("api-version"):
        return _error(404, "404", "Resource not found")
    if auth is None:
        return _error(
            401,
            "401",
            "Access denied due to invalid subscription key or wrong API endpoint.",
        )
    if (
        not isinstance(inputs, list)
        or not inputs
        or not all(isinstance(i, str) and i for i in inputs)
    ):
        return _error(400, "BadRequest", "input must be a non-empty list of strings")
    if v1 and not data.get("model"):
        return _error(400, "BadRequest", "model is required")
    if any("embed-fail" in i for i in inputs):
        return _error(500, "InternalServerError", "embedding backend failed (mock)")
    dims = data.get("dimensions") or EMBED_DIMS
    vectors = [
        {
            "object": "embedding",
            "index": i,
            "embedding": [round(0.01 * (i + 1) + 0.001 * j, 4) for j in range(dims)],
        }
        for i in range(len(inputs))
    ]
    deployment = request.path.split("/deployments/", 1)[-1].split("/", 1)[0]
    return web.json_response(
        {
            "object": "list",
            "data": vectors,
            "model": data.get("model") or deployment,
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
    )


async def fallback(request: web.Request) -> web.Response:
    await record(request)
    return _error(404, "404", "Resource not found")


def make_app() -> web.Application:
    app = new_app()
    app.router.add_post("/{path:.*}/chat/completions", chat_completions)
    app.router.add_post("/{path:.*}/embeddings", embeddings)
    app.router.add_route("*", "/{tail:.*}", fallback)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Azure OpenAI / AI Foundry mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9102)
    args = parser.parse_args()
    web.run_app(make_app(), host=args.host, port=args.port)
