"""
Group ``rag`` of the azure suite (suites/azure.py): the Azure AI Search
retrieval of pipelines/azure/azure_ai_foundry.py (3.0.0, #187) against
mocks/mock_search.py (Azure AI Search, managed identity tokens) and
mocks/mock_azure.py (chat, query generation, embeddings, tool calls).

The pipe no longer uses Azure OpenAI On Your Data: it never sends
``data_sources``, and a request that carries them (API stream and non-stream,
browser path as from an inlet filter) ends with an error that names the
removal and a terminal error status (``client-data-sources.*``); it never logs
the 2.8.1 retirement notice (``notice.none``). The checks of the former
``oyd`` group whose behaviour still exists (``[docX]`` links, history
unlinking, sources and the show-all valve, scores, large and too large stream
events, ``content: null``, background tasks also without a websocket session)
run here against the pipe's own retrieval.

Order: every check that uses the function as installed, then ``mode.*``
(saves the function again, first with the AZURE_AI_SEARCH_MODE valve of a
2.9.0 pre-release), ``rag.log.debug``, which runs the staged file in the
driver process with every logger at DEBUG, and last ``notice.none`` over the
server log of the whole azure suite run.

Loaded by suites/azure.py when the group runs (modules starting with ``_`` are
not suites).
"""

import asyncio
import importlib.util
import json
import logging
import os
import re
import sys
import time
import types
import uuid
from typing import Optional

from harness import Suite, short
from harness.config import FUNCTIONS_DIR
from harness.owui import completion_text
from suites.azure import (
    ALL_SOURCES,
    FID,
    FILLER,
    KEY,
    LINE_TOO_LONG,
    LINKED,
    NO_REFS,
    PAREN_LINKED,
    PATH,
    RAG_EMBED_KEY,
    RAG_EMBED_TOKEN,
    RAG_JSON_KEY,
    RAG_SEARCH_KEY,
    RAG_SEARCH_TOKEN,
    REFERENCED_SOURCES,
    SEARCH_MODE,
    SHOW_ALL,
    SPLIT_LINK,
    STREAM_ERROR,
    TASKS,
    TOOLS,
    UNLINKED,
    _answered,
    _is_task,
    _last_status,
    _status_sequence,
    _statuses,
    _wait_tasks,
)

DEPLOYMENT = "gpt-5-mini"
MODEL = f"{FID}.{DEPLOYMENT}"
PAUSE_DEPLOYMENT = "gpt-5-pause"  # query generation pause (per model)
RESET_DEPLOYMENT = "gpt-5-reset"  # timeouts that are not in a row
DOTTED_DEPLOYMENT = "gpt-4.1"  # a model name with a dot
QUESTION = "x100 charging and warranty?"
AGAIN = "x100 charging and warranty again?"
INDEX = "x100-docs"
CUSTOM_INDEX = "x100-custom"
SEARCH_PATH = f"/indexes/{INDEX}/docs/search"
API_VERSION = "2026-04-01"
EMBED_API_VERSION = "2024-10-21"
EMBED_DEPLOYMENT = "text-embedding-3-small"
QG_QUERIES = ["x100 charging", "x100 warranty"]
QG_MARKER = "Write search queries for Azure AI Search."
ROLE_INFORMATION = "You are the X100 support assistant (e2e role information)."
TITLES = ["X100 Product Manual", "Warranty FAQ", "Release Notes"]
DOC1_BLOCK = (
    "[doc1] Title: X100 Product Manual\nFile: manual.pdf\n"
    "The X100 charges via USB-C at up to 65 W."
)
DOC2_BLOCK = (
    "[doc2] Title: Warranty FAQ\nFile: faq/warranty.html\n"
    "All devices come with a two-year limited warranty."
)
REFUSAL = "The requested information isn't present in the retrieved documents."
HELLO = f"Hello from mock Azure ({DEPLOYMENT})."
STATUS_SEARCH = "Searching Azure AI Search..."
STATUS_QGEN = "Generating search queries..."
STATUS_NO_DOCS = "No documents found in Azure AI Search"
STATUS_SENDING = "Sending request to Azure AI..."
STATUS_RAG_STREAM = [
    STATUS_SEARCH,
    STATUS_SENDING,
    "Streaming response from Azure AI...",
    "Streaming completed",
]
STATUS_RAG_NONSTREAM = [STATUS_SEARCH, STATUS_SENDING, "Request completed"]
ERROR_PREFIX = "Error: Azure AI Search:"
CLIENT_DS_ERROR = (
    "Error: Azure AI Search: data_sources in the request is not supported: this "
    "pipeline no longer uses Azure OpenAI On Your Data (removed in 3.0.0)"
)
# The whole answer (and final status) of a request with client data_sources
CLIENT_DS_ANSWER = (
    CLIENT_DS_ERROR
    + "; remove data_sources, the search is configured by AZURE_AI_DATA_SOURCES"
)
# The pipe's ERROR line for it (also required, not only tolerated)
CLIENT_DS_LOG = (
    "Error in Azure AI request: Azure AI Search: data_sources in the request"
)
CONTEXT_HINT = (
    "lower AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS or top_n_documents, or start a new chat"
)
# The whole answer to the mock's context-too-long: Azure's message and the
# hint, nothing else (e.g. no On Your Data hint as in the 2.9.0 pre-releases)
CONTEXT_LENGTH_ANSWER = (
    "Error: This model's maximum context length is 128000 tokens. However, your "
    "messages resulted in 131072 tokens. Please reduce the length of the "
    f"messages. ({CONTEXT_HINT})"
)
# mock_azure bad-ref / mixed-ref: references to a document that does not
# exist ([doc9], the search returns 3), alone and next to [doc1]
BAD_REF = "The X100 charges via USB-C [doc9]."
MIXED_REF_LINKED = (
    "The X100 charges via USB-C [[doc1]](https://docs.example.com/x100/manual.pdf) "
    "and [doc9]."
)
# The 2.8.1 On Your Data retirement notice (a WARNING of the pipe): every
# server-log line with all of these
OYD_NOTICE = ("On Your Data", "October 14, 2026")
# Server-log signatures of provoked errors.
SEARCH_ERROR = ("function_azure:pipe", "Azure AI Search")
CHAT_ERROR = ("function_azure:pipe", "Error in Azure AI request")
# A valve of the 2.9.0 pre-releases (stored in Open WebUI's database or set
# as an environment variable) that 3.0.0 no longer has
STALE_MODE_VALVE = f'        {SEARCH_MODE}: str = Field(default="pipeline")\n'
MI_RESOURCE_ID = (
    "/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/e2e/"
    "providers/Microsoft.ManagedIdentity/userAssignedIdentities/e2e-uami"
)
MI_FAIL_ID = MI_RESOURCE_ID.replace("e2e-uami", "mi-fail-e2e")
SEARCH_RESOURCE = "https://search.azure.com"
SECRET_GROUP = "grp-e2e-secret-77"  # in a filter: must not reach answer or log
PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "2mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)
IMAGE_PART = {"type": "image_url", "image_url": {"url": PNG}}
# Open WebUI's user message after a tool returned images (a tool round)
TOOL_IMAGES_PROMPT = (
    "Here are the images from the tool results above. Please analyze them."
)
TOOL_CALL = {
    "id": "call_e2e_cache",
    "type": "function",
    "function": {"name": "get_current_timestamp", "arguments": "{}"},
}
# A tool round as Open WebUI sends it to the pipe: the question, the tool
# call and its result after it.
TOOL_ROUND = [
    {"role": "user", "content": QUESTION},
    {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
    {"role": "tool", "tool_call_id": TOOL_CALL["id"], "content": "2026-10-09T12:00Z"},
]
# Order of the merged documents of qgen-order (mock_search: "x100 order one"
# -> docs 3, 1; "x100 order two" -> docs 2, 1): reciprocal rank fusion for
# BM25, the best reranker score for semantic (first appearance: 3, 1, 2).
ORDER_RRF = ["X100 Product Manual", "Release Notes", "Warranty FAQ"]
ORDER_RERANK = ["Release Notes", "Warranty FAQ", "X100 Product Manual"]
_BLOCK_TITLES = re.compile(r"(?m)^\[doc\d+\] Title: (.*)$")
FULL_MAPPING = {
    "content_fields": ["content"],
    "title_field": "title",
    "url_field": "url",
    "filepath_field": "filepath",
}
CUSTOM_MAPPING = {
    "content_fields": ["body"],
    "title_field": "doc_title",
    "url_field": "source_url",
    "filepath_field": "source_file",
}
DETAILS_BLOCK = (
    '<details type="reasoning" done="true">\n<summary>Thought</summary>\n'
    "secret-thoughts-e2e\n</details>\n"
)
_DOC_SECTION = re.compile(r"(?ms)^\[doc(\d+)\] (.*?)(?=^\[doc\d+\] |\Z)")


# ------------------------------------------------------------------ helpers
def _text(content) -> str:
    """Text of a message content (string or list of parts)."""
    if isinstance(content, list):
        return " ".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return content if isinstance(content, str) else ""


def _messages(entry: dict) -> list:
    body = entry.get("body") or {}
    return [m for m in body.get("messages") or [] if isinstance(m, dict)]


def _current_text(entry: dict) -> str:
    """Text of the current turn's user message of a recorded chat request."""
    index = entry.get("current_user_index")
    messages = _messages(entry)
    if index is None or not 0 <= index < len(messages):
        return ""
    return _text(messages[index].get("content"))


def _block(entry: dict) -> str:
    """The <documents> block of the current user message ('' without one)."""
    text = _current_text(entry)
    if not text.lstrip().startswith("<documents>"):
        return ""
    end = text.rfind("</documents>")
    return text[: end + len("</documents>")] if end >= 0 else text


def _block_docs(entry: dict) -> dict:
    """``{n: document text}`` of the block (header lines removed)."""
    block = _block(entry)
    inner = block[len("<documents>") :] if block.startswith("<documents>") else block
    end = inner.rfind("</documents>")
    inner = inner[:end] if end >= 0 else inner
    docs = {}
    for match in _DOC_SECTION.finditer(inner):
        lines = match.group(2).strip("\n").split("\n")
        rest = lines[1:]
        if rest and rest[0].startswith("File: "):
            rest = rest[1:]
        docs[int(match.group(1))] = "\n".join(rest).strip()
    return docs


def _system_text(entry: dict) -> str:
    return "\n".join(
        _text(m.get("content"))
        for m in _messages(entry)
        if m.get("role") in ("system", "developer")
    )


def _grounded(entry: dict, docs: Optional[list] = None) -> bool:
    """Grounded prompt: one <documents> block in the current user message with
    the documents ``docs`` (default 1, 2, 3), the rules in the one system
    message, no data_sources."""
    return (
        entry.get("grounded") is True
        and entry.get("prompt_docs") == (docs or [1, 2, 3])
        and entry.get("documents_blocks") == 1
        and entry.get("docs_indices") == [entry.get("current_user_index")]
        and entry.get("system_messages") == 1
        and entry.get("rules_in_system") is True
        and entry.get("rules_count") == 1
        and "data_sources" not in (entry.get("body") or {})
    )


def _chat_brief(entry: dict) -> str:
    if not entry:
        return "chat: none"
    body = entry.get("body") or {}
    return (
        f"chat: path={entry.get('path')} grounded={entry.get('grounded')} "
        f"docs={entry.get('prompt_docs')} blocks={entry.get('documents_blocks')} "
        f"at={entry.get('docs_indices')}/{entry.get('current_user_index')} "
        f"system={entry.get('system_messages')} rules={entry.get('rules_count')} "
        f"keys={sorted(body)}"
    )


def _search_brief(entries: list) -> str:
    return "searches: " + short(
        [
            f"{e.get('status')} {e.get('path')} q={e.get('search')!r} "
            f"type={e.get('queryType')} top={e.get('top')} auth={e.get('auth_mode')}"
            for e in entries
        ],
        300,
    )


def _answer(content) -> str:
    """The answer first (an error text shows at once)."""
    return f"answer={short(content or '', 120)}"


def _after_done(res) -> str:
    """SSE events an API stream sent after its first [DONE] (an OpenAI-style
    client stops reading there and never sees them)."""
    return (
        f"after [DONE]: {res.after_done} events, "
        f"content={short(res.after_done_content, 80)!r}"
    )


def _context(data) -> dict:
    """``choices[0].message.context`` of a non-stream API answer."""
    if isinstance(data, dict) and data.get("choices"):
        message = data["choices"][0].get("message") or {}
        return message.get("context") or {}
    return {}


def _sse_events(raw: str) -> list:
    events = []
    for line in (raw or "").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except ValueError:
            continue
    return events


def _stream_context(raw: str) -> tuple:
    """(index of the first event with delta.context, index of the first event
    with content, that context)."""
    ctx_index, content_index, context = None, None, {}
    for i, event in enumerate(_sse_events(raw)):
        for choice in (event.get("choices") if isinstance(event, dict) else None) or []:
            delta = choice.get("delta") or {}
            if ctx_index is None and isinstance(delta.get("context"), dict):
                ctx_index, context = i, delta["context"]
            if content_index is None and delta.get("content"):
                content_index = i
    return ctx_index, content_index, context


def _intent(context: dict):
    try:
        return json.loads(context.get("intent") or "null")
    except (TypeError, ValueError):
        return None


def _titles(citations: list) -> list:
    return [c.get("title") for c in citations if isinstance(c, dict)]


def _close(actual, expected, tolerance: float = 0.005) -> bool:
    try:
        return len(actual) == len(expected) and all(
            len(a) == len(e)
            and all(abs(float(x) - y) <= tolerance for x, y in zip(a, e))
            for a, e in zip(actual, expected)
        )
    except (TypeError, ValueError):
        return False


def _followup(latest: str, earlier: str = LINKED) -> list:
    return [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": earlier},
        {"role": "user", "content": latest},
    ]


def _client_ds_errors(t: Suite, mark: int) -> list:
    """The pipe's ERROR lines for a request with client data_sources since
    ``mark``."""
    return [
        line
        for line in t.log.lines(mark, CLIENT_DS_LOG)
        if "| ERROR" in line and "function_azure" in line
    ]


def _warnings(t: Suite, mark: int, *needles: str) -> list:
    """WARNING lines of the Azure pipe since ``mark`` containing all needles."""
    return [
        line
        for line in t.log.since(mark).splitlines()
        if "| WARNING" in line
        and "function_azure" in line
        and all(n in line for n in needles)
    ]


class Rag:
    """Mocks, valves and checks of the group."""

    def __init__(self, t: Suite, mock, search, base_valves: dict):
        self.t, self.mock, self.search, self.base = t, mock, search, base_valves

    def ds(self, auth="key", **parameters) -> dict:
        """One azure_search data source on the Search mock (endpoint with a
        trailing "/", which the pipe has to strip)."""
        base = {"endpoint": self.search.url + "/", "index_name": INDEX}
        if auth == "key":
            base["authentication"] = {"type": "api_key", "key": RAG_SEARCH_KEY}
        elif auth is not None:
            base["authentication"] = auth
        return {"type": "azure_search", "parameters": {**base, **parameters}}

    def valves(self, ds=None, **changes) -> dict:
        """The rag valves (``ds``: a data source dict / list, or the raw
        AZURE_AI_DATA_SOURCES string)."""
        if ds is None:
            ds = self.ds()
        return {
            **self.base,
            "AZURE_AI_ENDPOINT": f"{self.mock.url}/openai/deployments/{DEPLOYMENT}"
            "/chat/completions?api-version=2025-04-01-preview",
            "AZURE_AI_MODEL": DEPLOYMENT,
            "AZURE_AI_MODEL_IN_BODY": False,
            "USE_AUTHORIZATION_HEADER": False,
            "AZURE_AI_INCLUDE_SEARCH_SCORES": True,
            SHOW_ALL: True,
            "AZURE_AI_DATA_SOURCES": ds if isinstance(ds, str) else json.dumps(ds),
            "AZURE_AI_SEARCH_KEY": "",
            "AZURE_AI_SEARCH_API_VERSION": API_VERSION,
            "AZURE_AI_SEARCH_QUERY_GENERATION": "auto",
            "AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS": -1,
            **changes,
        }

    async def set(self, ds=None, **changes) -> None:
        await self.t.owui.update_valves(FID, **self.valves(ds, **changes))

    async def reset(self) -> int:
        """Clear both mock records; returns a log mark."""
        await self.mock.reset()
        await self.search.reset()
        return self.t.mark()

    async def chats(self) -> list:
        """Chat requests of answers (no task, no query generation)."""
        return await self.mock.requests(
            lambda e: str(e.get("path", "")).endswith("/chat/completions")
            and not e.get("query_generation")
            and not _is_task(e)
        )

    async def chat(self) -> dict:
        entries = await self.chats()
        return entries[-1] if entries else {}

    async def tasks(self) -> list:
        return await self.mock.requests(
            lambda e: str(e.get("path", "")).endswith("/chat/completions")
            and _is_task(e)
        )

    async def qgens(self) -> list:
        return await self.mock.requests(lambda e: bool(e.get("query_generation")))

    async def embeds(self) -> list:
        return await self.mock.requests(lambda e: bool(e.get("embeddings")))

    async def searches(self) -> list:
        """Requests that reached the Search mock (token requests excluded)."""
        return await self.search.requests(lambda e: not e.get("msi"))

    async def tokens(self) -> list:
        return await self.search.requests(lambda e: bool(e.get("msi")))

    def check(self, sid: str, title: str, ok, detail: str):
        return self.t.check(f"rag.{sid}", title, ok, detail)

    async def settle_errors(self, mark: int, *signatures) -> None:
        """Errors the scenario provoked on purpose (after a short settle)."""
        await self.t.log.settle(0.5)
        self.t.expect_errors(mark, *signatures)


# -------------------------------------------------------------------- group
async def rag(t: Suite, mock, base_valves: dict) -> None:
    r = Rag(t, mock, t.mock("search"), base_valves)
    await rag_api(r)
    await rag_browser(r)
    await rag_tool_round(r)
    await rag_cache(r)
    await rag_links(r)
    await rag_links_browser(r)
    await rag_history(r)
    await rag_query_text(r)
    await rag_no_refs(r)
    await rag_scores(r)
    await rag_events(r)
    await rag_content_null(r)
    await rag_query_types(r)
    await rag_vector(r)
    await rag_fields(r)
    await rag_selection(r)
    await rag_budget(r)
    await rag_prompt(r)
    await rag_sanitize(r)
    await rag_list_content(r)
    await rag_no_hits(r)
    await rag_search_errors(r)
    await rag_config(r)
    await rag_not_configured(r)
    await rag_tasks(r)
    await rag_no_session(r)
    await rag_client_data_sources(r)
    await rag_qgen(r)
    await rag_auth(r)
    await rag_context_length(r)
    await rag_stop(r)
    # saves the function again (a fresh module)
    await rag_mode_removed(r)
    await rag_debug_log(r)
    rag_notice_none(r)
    await t.owui.update_valves(FID, **base_valves)


# ---------------------------------------------------------------- API path
async def rag_api(r: Rag) -> None:
    t = r.t
    await r.set()
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    chat, searches = await r.chat(), await r.searches()
    s = searches[0] if searches else {}
    body = chat.get("body") or {}
    context = _context(res.json)
    citations = context.get("citations") or []
    block = _block(chat)
    prompt_ok = (
        _grounded(chat)
        and DOC1_BLOCK in block
        and DOC2_BLOCK in block
        and "http" not in block
        and _current_text(chat).endswith("\n\n" + QUESTION)
    )
    search_ok = (
        len(searches) == 1
        and s.get("path") == SEARCH_PATH
        and s.get("api_version") == API_VERSION
        and s.get("auth_mode") == "api-key"
        and not s.get("both_auth")
        and str((s.get("headers") or {}).get("content-type")).startswith(
            "application/json"
        )
        and s.get("body") == {"search": QUESTION, "queryType": "simple", "top": 10}
    )
    chat_ok = (
        chat.get("path") == f"/openai/deployments/{DEPLOYMENT}/chat/completions"
        and chat.get("api_version") == "2025-04-01-preview"
    )
    r.check(
        "api.nonstream",
        "API non-stream: [docX] links; chat request to the path and api-version "
        "of AZURE_AI_ENDPOINT (Azure OpenAI deployment) without data_sources, "
        "documents [doc1]-[doc3] in a <documents> block (title / file, no URL) "
        "before the user text, rules in the one system message; one search "
        "(path, api-version 2026-04-01, api-key, body {search, queryType "
        "simple, top 10}); message.context citations and intent",
        res.status == 200
        and res.content == LINKED
        and "stream_options" not in body
        and chat_ok
        and prompt_ok
        and search_ok
        and _titles(citations) == TITLES
        and _intent(context) == [QUESTION],
        f"{_answer(res.content)} HTTP {res.status} {_chat_brief(chat)} "
        f"api_version={chat.get('api_version')} "
        f"block_format={DOC1_BLOCK in block and DOC2_BLOCK in block} "
        f"url_in_block={'http' in block} {_search_brief(searches)} "
        f"search_body={short(s.get('body'), 120)} citations={_titles(citations)} "
        f"intent={context.get('intent')!r}",
    )

    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=True)
    chat, searches = await r.chat(), await r.searches()
    body = chat.get("body") or {}
    ctx_index, content_index, context = _stream_context(res.raw)
    r.check(
        "api.stream",
        "API stream: [docX] links and [DONE]; a first event with "
        "delta.context (3 citations) before any content; stream_options."
        "include_usage forwarded, usage 18",
        res.status == 200
        and res.content == LINKED
        and res.done
        and (res.usage or {}).get("total_tokens") == 18
        and ctx_index is not None
        and (content_index is None or ctx_index < content_index)
        and len(context.get("citations") or []) == 3
        and body.get("stream_options") == {"include_usage": True}
        and _grounded(chat)
        and len(searches) == 1,
        f"{_answer(res.content)} done={res.done} usage={res.usage} "
        f"context_event={ctx_index} first_content={content_index} "
        f"citations={len(context.get('citations') or [])} "
        f"stream_options={body.get('stream_options')} {_chat_brief(chat)} "
        f"{_search_brief(searches)}",
    )

    # Foundry endpoint (model in the header, not in the URL): #187 asks for
    # retrieval on non-*.openai.azure.com endpoints.
    await r.set(
        AZURE_AI_ENDPOINT=f"{r.mock.url}/models/chat/completions"
        "?api-version=2024-05-01-preview"
    )
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=True)
    chat, searches = await r.chat(), await r.searches()
    r.check(
        "api.foundry",
        "Foundry /models/chat/completions endpoint (model header): grounded "
        "request, linked stream answer, one search",
        res.status == 200
        and res.content == LINKED
        and res.done
        and chat.get("path") == "/models/chat/completions"
        and chat.get("model_header") == DEPLOYMENT
        and _grounded(chat)
        and len(searches) == 1,
        f"{_answer(res.content)} model_header={chat.get('model_header')} "
        f"{_chat_brief(chat)} {_search_brief(searches)}",
    )

    await r.set()
    for stream in (False, True):
        kind = "stream" if stream else "nonstream"
        await r.reset()
        res = await t.owui.chat(
            MODEL, QUESTION, stream=stream, tools=TOOLS, tool_choice="auto"
        )
        chat = await r.chat()
        body = chat.get("body") or {}
        r.check(
            f"tools.api.{kind}",
            "client tools / tool_choice are forwarded together with the "
            f"retrieval (stream={stream})",
            res.status == 200
            and res.content == LINKED
            and body.get("tools") == TOOLS
            and body.get("tool_choice") == "auto"
            and _grounded(chat),
            f"{_answer(res.content)} tools={short(body.get('tools'), 80)} "
            f"tool_choice={body.get('tool_choice')!r} {_chat_brief(chat)}",
        )


# ------------------------------------------------------------ browser path
async def rag_browser(r: Rag) -> None:
    t = r.t
    await r.set()
    async with t.browser() as b:
        for stream in (True, False):
            kind = "stream" if stream else "nonstream"
            await r.reset()
            c = await b.chat(MODEL, QUESTION, stream=stream)
            chat, searches = await r.chat(), await r.searches()
            body = chat.get("body") or {}
            tools = [
                (tool.get("function") or {}).get("name")
                for tool in body.get("tools") or []
                if isinstance(tool, dict)
            ]
            distances = [s.get("distances") for s in c.sources]
            expected = STATUS_RAG_STREAM if stream else STATUS_RAG_NONSTREAM
            r.check(
                f"browser.{kind}",
                f"browser path stream={stream} (built-in tools on): linked answer, "
                "only the referenced sources with BM25 scores, usage, statusHistory "
                f"{expected}, Open WebUI's tools forwarded upstream",
                c.done
                and c.content == LINKED
                and c.source_names == REFERENCED_SOURCES
                and _close(distances, [[0.425], [0.12]])
                and (c.usage or {}).get("total_tokens") == 18
                and _status_sequence(c.status_history, expected)
                and "get_current_timestamp" in tools
                and _grounded(chat)
                and len(searches) == 1,
                f"{c.brief()} distances={distances} "
                f"statuses={_statuses(c.status_history)} tools={tools[:6]} "
                f"{_chat_brief(chat)} {_search_brief(searches)}",
            )


async def rag_tool_round(r: Rag) -> None:
    """Open WebUI's native tool loop calls the pipe again for the tool round:
    the retrieval is cached (one search), the documents are injected again
    without piling up, the sources are emitted once."""
    t = r.t
    await r.set()
    async with t.browser() as b:
        mark = await r.reset()
        c = await b.chat(MODEL, QUESTION + " use-tool", stream=True)
    chats, searches = await r.chats(), await r.searches()
    await t.log.settle(0.5)
    errors = t.log.errors(mark)
    first, last = (chats[0], chats[-1]) if chats else ({}, {})
    tool_at = [i for i, m in enumerate(_messages(last)) if m.get("role") == "tool"]
    searching = c.status_descriptions.count(STATUS_SEARCH)
    r.check(
        "tool-round",
        "tool round (use-tool, built-in get_current_timestamp): final answer "
        "linked, referenced sources once each, one search (cached), two "
        "grounded chat requests with the same documents in the original user "
        "message (one block, one system message each), one 'Searching' status",
        c.done
        and LINKED in c.content
        and c.source_names == REFERENCED_SOURCES
        and len(searches) == 1
        and len(chats) == 2
        and all(_grounded(e) for e in chats)
        and bool(first.get("tools_present"))
        and bool(tool_at)
        and (last.get("current_user_index") or 0) < tool_at[0]
        and searching == 1
        and (_last_status(c.status_history).get("done") is True)
        and not errors,
        f"{c.brief()} chats={len(chats)} "
        + " | ".join(_chat_brief(e) for e in chats)
        + f" tool_messages_at={tool_at} searching_statuses={searching} "
        f"{_search_brief(searches)} log_errors={errors[:2]}",
    )

    # The tool round itself references [doc1] before its tool call: doc1 is
    # emitted in that round and must not be emitted again in the final one.
    async with t.browser() as b:
        await r.reset()
        c = await b.chat(MODEL, QUESTION + " use-tool-ref", stream=True)
    chats, searches = await r.chats(), await r.searches()
    doc1 = REFERENCED_SOURCES[0]
    r.check(
        "tool-round.ref",
        "tool round whose text references [doc1] before the tool call: doc1 "
        "saved once (emitted in the tool round, skipped in the final round), "
        "doc2 from the final round",
        c.done
        and LINKED in c.content
        and c.source_names == REFERENCED_SOURCES
        and c.source_names.count(doc1) == 1
        and len(searches) == 1
        and len(chats) == 2
        and all(_grounded(e) for e in chats),
        f"{c.brief()} chats={len(chats)} {_search_brief(searches)}",
    )

    # Non-stream answer with tool calls and no content (streaming off in the
    # browser): a tool round, so no "show all" fallback.
    async with t.browser() as b:
        await r.reset()
        c = await b.chat(MODEL, QUESTION + " use-tool", stream=False)
    chats, searches = await r.chats(), await r.searches()
    first = chats[0] if chats else {}
    r.check(
        "tool-round.nonstream",
        "non-stream answer with tool calls and no text (browser, streaming "
        "off): no sources (a tool round never uses the show-all fallback), "
        "terminal status",
        c.source_names == []
        and _last_status(c.status_history).get("done") is True
        and len(searches) == 1
        and bool(chats)
        and _grounded(first)
        and bool(first.get("tools_present")),
        f"{c.brief()} {_chat_brief(first)} {_search_brief(searches)}",
    )


async def rag_cache(r: Rag) -> None:
    """The documents of a message are reused only by a tool round of the same
    user and message (API requests with a fixed ``id``, which Open WebUI
    passes on as the message id), and never after a valve change."""
    t = r.t
    msg_id = f"e2e-cache-{int(time.time())}"
    await r.set(AZURE_AI_SEARCH_QUERY_GENERATION="off")
    await r.reset()
    res1 = await t.owui.chat(MODEL, QUESTION, stream=False, id=msg_id)
    first = len(await r.searches())
    res2 = await t.owui.chat(MODEL, TOOL_ROUND, stream=False, id=msg_id)
    second, chat2 = len(await r.searches()), await r.chat()
    res3 = await t.owui.chat(MODEL, QUESTION, stream=False, id=msg_id)
    third = len(await r.searches())
    r.check(
        "cache.tool-round",
        "same message id: the tool round reuses the documents (no new search, "
        "the same documents injected again); a request that is no tool round "
        "searches again",
        res1.content == LINKED
        and res2.content == LINKED
        and res3.content == LINKED
        and [first, second, third] == [1, 1, 2]
        and _grounded(chat2)
        and chat2.get("current_user_index") == 1,  # after the added system message
        f"{_answer(res1.content)} round2={short(res2.content, 60)} "
        f"plain={short(res3.content, 60)} searches after each={[first, second, third]} "
        f"{_chat_brief(chat2)}",
    )

    # Another index (a valve change) between the rounds: no reuse.
    await r.set(
        r.ds(index_name=CUSTOM_INDEX, fields_mapping=CUSTOM_MAPPING),
        AZURE_AI_SEARCH_QUERY_GENERATION="off",
    )
    await r.reset()
    res4 = await t.owui.chat(MODEL, TOOL_ROUND, stream=False, id=msg_id)
    searches, chat4 = await r.searches(), await r.chat()
    r.check(
        "cache.valve-change",
        "tool round after AZURE_AI_DATA_SOURCES changed (another index): the "
        "cached documents of the old index are not reused, a new search on "
        "the new index",
        res4.content == LINKED
        and len(searches) == 1
        and (searches[0] if searches else {}).get("index") == CUSTOM_INDEX
        and _grounded(chat4),
        f"{_answer(res4.content)} {_search_brief(searches)} {_chat_brief(chat4)}",
    )


async def rag_links(r: Rag) -> None:
    """[docX] handling of the 2.8.0 code with the pipeline's citations."""
    t = r.t
    await r.set()
    for sid, text, expected in (
        ("split-tokens", QUESTION + " split-tokens", LINKED),
        ("split-link", QUESTION + " split-link", SPLIT_LINK),
        ("no-finish", QUESTION + " split-tokens no-finish", LINKED[:-1]),
        ("paren-url", QUESTION + " paren-url", PAREN_LINKED),
    ):
        await r.reset()
        res = await t.owui.chat(MODEL, text, stream=True)
        chat = await r.chat()
        r.check(
            sid,
            f"API stream '{text}': references linked once (held back text "
            "included), all of it before [DONE] and nothing after it",
            res.status == 200
            and res.content == expected
            and res.done
            and not res.after_done
            and _grounded(chat),
            f"{_answer(res.content)} done={res.done} {_after_done(res)} "
            f"{_chat_brief(chat)}",
        )

    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION + " paren-url", stream=False)
    chat = await r.chat()
    r.check(
        "paren-url.nonstream",
        "API non-stream: parentheses in a citation URL are percent-encoded in the link",
        res.status == 200 and res.content == PAREN_LINKED and _grounded(chat),
        f"{_answer(res.content)} {_chat_brief(chat)}",
    )

    await r.set(AZURE_AI_MODEL=f"{DEPLOYMENT};{DOTTED_DEPLOYMENT}")
    await r.reset()
    res = await t.owui.chat(f"{FID}.{DOTTED_DEPLOYMENT}", QUESTION, stream=True)
    chat, searches = await r.chat(), await r.searches()
    body_model = (chat.get("body") or {}).get("model")
    r.check(
        "dotted.stream",
        f"API stream with a dotted model name ({DOTTED_DEPLOYMENT}): the name "
        "reaches upstream intact (header and body), linked answer, [DONE]",
        res.status == 200
        and res.content == LINKED
        and res.done
        and chat.get("model_header") == DOTTED_DEPLOYMENT
        and body_model == DOTTED_DEPLOYMENT
        and _grounded(chat)
        and len(searches) == 1,
        f"{_answer(res.content)} header={chat.get('model_header')!r} "
        f"body.model={body_model!r} {_chat_brief(chat)} {_search_brief(searches)}",
    )

    await r.set(AZURE_AI_SEARCH_QUERY_GENERATION="off")
    await r.reset()
    res = await t.owui.chat(MODEL, _followup(AGAIN, PAREN_LINKED), stream=False)
    chat = await r.chat()
    messages = _messages(chat)
    sent = next(
        (_text(m.get("content")) for m in messages if m.get("role") == "assistant"),
        None,
    )
    r.check(
        "history-unlink",
        "links of earlier answers go back as plain [docX]; the documents only "
        "in the last user message",
        res.status == 200
        and res.content == LINKED
        and sent == UNLINKED
        and _grounded(chat)
        and chat.get("current_user_index") == len(messages) - 1,
        f"{_answer(res.content)} sent={sent!r} {_chat_brief(chat)}",
    )


async def rag_links_browser(r: Rag) -> None:
    """[docX] references split across stream deltas in the browser path (as
    the web UI sends it, Open WebUI's built-in tools on)."""
    t = r.t
    await r.set()
    async with t.browser() as b:
        for sid, text, expected in (
            ("split-tokens.browser", QUESTION + " split-tokens", LINKED),
            ("split-link.browser", QUESTION + " split-link", SPLIT_LINK),
            ("no-finish.browser", QUESTION + " split-tokens no-finish", LINKED[:-1]),
        ):
            await r.reset()
            c = await b.chat(MODEL, text, stream=True)
            chat = await r.chat()
            r.check(
                sid,
                f"browser stream '{text}': linked answer saved, only the "
                "referenced sources",
                c.done
                and c.content == expected
                and c.source_names == REFERENCED_SOURCES
                and _grounded(chat),
                f"{c.brief()} {_chat_brief(chat)}",
            )


def _sent_assistant(entry: dict):
    """Text of the first assistant message of a recorded chat request."""
    return next(
        (
            _text(m.get("content"))
            for m in _messages(entry)
            if m.get("role") == "assistant"
        ),
        None,
    )


async def rag_history(r: Rag) -> None:
    """Links of earlier answers saved before 2.8.0 (unencoded parentheses in
    the URL) go back as plain [docX]; a hostile history line stays linear."""
    t = r.t
    await r.set(AZURE_AI_SEARCH_QUERY_GENERATION="off")
    earlier = "y [[doc1]](https://docs.example.com/a_(b).pdf) z"
    await r.reset()
    res = await t.owui.chat(MODEL, _followup(AGAIN, earlier), stream=False)
    chat = await r.chat()
    sent = _sent_assistant(chat)
    r.check(
        "legacy-history-paren",
        "links saved before 2.8.0 with ')' in the URL go back as plain [docX]",
        res.status == 200
        and res.content == LINKED
        and sent == "y [doc1] z"
        and _grounded(chat),
        f"{_answer(res.content)} sent={sent!r} {_chat_brief(chat)}",
    )

    # Any API client can send this history: a long line of unclosed links must
    # not make the unlinking quadratic (it runs in the event loop).
    hostile = "[[doc1]](" * 9000 + "x"
    await r.reset()
    started = time.monotonic()
    res = await t.owui.chat(MODEL, _followup(AGAIN, hostile), stream=False)
    elapsed = time.monotonic() - started
    chat = await r.chat()
    sent = _sent_assistant(chat)
    r.check(
        "history-hostile",
        "81 KB history line of unclosed [[docX]]( links: passed through "
        "unchanged in under 3 s (unlinking stays linear)",
        res.status == 200
        and res.content == LINKED
        and sent == hostile
        and elapsed < 3.0
        and _grounded(chat),
        f"{_answer(res.content)} elapsed={elapsed:.2f}s "
        f"sent_unchanged={sent == hostile} {_chat_brief(chat)}",
    )


async def rag_query_text(r: Rag) -> None:
    """Search text from the user text: whitespace collapsed, a leading "-"
    (the simple parser's NOT) replaced, cut at 1,000 characters on a word
    boundary."""
    t = r.t
    await r.set()
    await r.reset()
    res = await t.owui.chat(MODEL, "x100   charging\n and -warranty?", stream=False)
    searches = await r.searches()
    first = str((searches[0] if searches else {}).get("search") or "")
    long_text = "x100 charging and warranty " + " ".join(f"word{i}" for i in range(300))
    await r.reset()
    res2 = await t.owui.chat(MODEL, long_text, stream=False)
    searches2 = await r.searches()
    second = str((searches2[0] if searches2 else {}).get("search") or "")
    boundary = long_text.startswith(second) and (
        len(second) == len(long_text) or long_text[len(second)] == " "
    )
    r.check(
        "query-text",
        "search text: whitespace collapsed, a leading '-' (NOT operator) "
        "replaced, at most 1,000 characters cut on a word boundary",
        res.content == LINKED
        and res2.content == LINKED
        and "-warranty" not in first
        and "\n" not in first
        and " ".join(first.split()) == QUESTION
        and 0 < len(second) <= 1000
        and boundary,
        f"{_answer(res.content)} search={first!r} long: {_answer(res2.content)} "
        f"len={len(second)} word_boundary={boundary}",
    )

    # Open WebUI's tool round after a tool returned images ends with its own
    # user message: the question stays the current user message.
    messages = TOOL_ROUND + [
        {
            "role": "user",
            "content": [{"type": "text", "text": TOOL_IMAGES_PROMPT}, IMAGE_PART],
        }
    ]
    await r.reset()
    res = await t.owui.chat(MODEL, messages, stream=False)
    chat, searches, qgens = await r.chat(), await r.searches(), await r.qgens()
    r.check(
        "query-text.tool-images",
        "tool round ending with Open WebUI's 'Here are the images from the "
        "tool results above' user message: search text, documents and the "
        "first-turn rule of query generation use the question before it",
        res.content == LINKED
        and [s.get("search") for s in searches] == [QUESTION]
        and _grounded(chat)
        and chat.get("current_user_index") == 1  # after the added system message
        and not qgens,
        f"{_answer(res.content)} {_search_brief(searches)} qgen={len(qgens)} "
        f"{_chat_brief(chat)}",
    )

    # A file attached in the browser: Open WebUI puts an <attached_files>
    # block (native function calling) and its RAG template with the file
    # around the last user message; __metadata__ keeps the typed prompt (after
    # the block) as base_user_prompt (Open WebUI 0.12) or user_prompt (0.11).
    async with t.browser() as b:
        earlier = await b.chat(MODEL, "x100 earlier chat e2e?", stream=True)
        await r.reset()
        attached = {
            "type": "chat",
            "id": earlier.chat_id,
            "name": "earlier chat e2e",
            "context": "full",
        }
        c = await b.chat(MODEL, QUESTION, stream=True, files=[attached])
    chat, searches = await r.chat(), await r.searches()
    question = _current_text(chat)
    question = question[question.rfind("</documents>") + len("</documents>") :]
    r.check(
        "query-text.attached",
        "browser chat with a file attached (another chat, full context): "
        "Open WebUI's <attached_files> block and RAG template change the user "
        "message, the search text is still the typed prompt",
        c.done
        and LINKED in c.content
        and "<attached_files>" in question
        and [s.get("search") for s in searches] == [QUESTION]
        and _grounded(chat),
        f"{_search_brief(searches)} attached_block={'<attached_files>' in question} "
        f"question={short(question, 120)} {c.brief()} {_chat_brief(chat)}",
    )

    # The same rule for both metadata keys on every Open WebUI version: the
    # staged file runs in this process (as rag.log.debug) with a user message
    # wrapped like the one above. Open WebUI stores the prompt before adding
    # its RAG template as base_user_prompt since 0.12
    # (utils/middleware.py, process_chat_payload) and as user_prompt up to 0.11.
    block = '<attached_files>\n<file type="chat" id="c1" name="e2e"/>\n</attached_files>\n\n'
    wrapped = (
        f"{block}### Task:\nRespond to the user query using the provided "
        f"context.\n<context>\nearlier chat\n</context>\n\n<user_query>\n"
        f"{QUESTION}\n</user_query>"
    )
    variants = (
        ("base_user_prompt", {"base_user_prompt": block + QUESTION}),
        ("user_prompt", {"user_prompt": block + QUESTION}),
        ("both", {"base_user_prompt": QUESTION, "user_prompt": "stale prompt"}),
        ("base None", {"base_user_prompt": None, "user_prompt": block + QUESTION}),
    )
    seen, failure = [], ""
    try:
        module = _load_staged_pipe()
        for n, (label, prompts) in enumerate(variants, start=900):
            await r.reset()
            text = await asyncio.wait_for(
                _pipe_in_process(
                    module.Pipe(),
                    r.valves(),
                    MODEL,
                    [{"role": "user", "content": wrapped}],
                    n,
                    prompts=prompts,
                ),
                120,
            )
            searched = [s.get("search") for s in await r.searches()]
            seen.append((label, text == LINKED, searched))
    except Exception as exc:  # the check reports it
        failure = f"{type(exc).__name__}: {short(str(exc), 200)} "
    finally:
        logging.getLogger("azure_ai.pipe").setLevel(logging.NOTSET)
    r.check(
        "query-text.metadata",
        "staged file in the driver process, user message wrapped in Open "
        "WebUI's <attached_files> block and RAG template: the search text is "
        "the typed prompt from __metadata__ base_user_prompt (Open WebUI 0.12) "
        "or user_prompt (0.11); base_user_prompt wins when both are set",
        not failure
        and len(seen) == len(variants)
        and all(answered and searched == [QUESTION] for _, answered, searched in seen),
        f"{failure}"
        + " ".join(
            f"{label}: answered={answered} search={searched!r}"
            for label, answered, searched in seen
        ),
    )


async def rag_no_refs(r: Rag) -> None:
    """Sources of an answer without references, and of one whose references
    point only to documents that do not exist ([doc9] with 3 documents: no
    reference either) or partly ([doc1] and [doc9]: only doc1)."""
    t = r.t
    async with t.browser() as b:
        no_refs = "answer without [docX]"
        bad_ref = "answer citing only [doc9] of 3 documents (counts as no reference)"
        for sid, show_all, trigger, content, expected, title in (
            ("no-refs.default", True, "no-refs", NO_REFS, ALL_SOURCES, no_refs),
            ("no-refs.valve-false", False, "no-refs", NO_REFS, [], no_refs),
            ("bad-ref.default", True, "bad-ref", BAD_REF, ALL_SOURCES, bad_ref),
            ("bad-ref.valve-false", False, "bad-ref", BAD_REF, [], bad_ref),
            (
                "mixed-ref",
                True,
                "mixed-ref",
                MIXED_REF_LINKED,
                REFERENCED_SOURCES[:1],
                "answer citing [doc1] and [doc9] of 3 documents: doc1 linked, "
                "[doc9] left as it is",
            ),
        ):
            await r.set(**{SHOW_ALL: show_all})
            await r.reset()
            c = await b.chat(MODEL, QUESTION + " " + trigger, stream=True)
            chat = await r.chat()
            r.check(
                sid,
                f"{title}, show-all valve {show_all} -> {len(expected)} sources",
                c.done
                and c.content == content
                and c.source_names == expected
                and _grounded(chat),
                f"{c.brief()} {_chat_brief(chat)}",
            )


async def rag_scores(r: Rag) -> None:
    """Relevance shown on the source cards: BM25 / 100, reranker / 4, RRF
    rescaled by rank (score x 60 / number of fused lists)."""
    t = r.t
    two_vectors = {**FULL_MAPPING, "vector_fields": ["contentVector", "titleVector"]}
    cases = (
        ("scores.on.simple", "simple (BM25 42.5 / 12 of 100)", {}, [[0.425], [0.12]]),
        (
            "scores.on.semantic",
            "semantic (reranker 3.2 / 2.4 of 4)",
            {"query_type": "semantic", "semantic_configuration": "x100-semantic"},
            [[0.8], [0.6]],
        ),
        (
            "scores.on.hybrid",
            "vector_simple_hybrid (RRF 0.0328 / 0.0323 x 60 / 2)",
            {"query_type": "vector_simple_hybrid"},
            [[0.98], [0.97]],
        ),
        (
            "scores.on.vector",
            "vector over one field (cosine 0.91 / 0.85 as is)",
            {"query_type": "vector"},
            [[0.91], [0.85]],
        ),
        (
            "scores.on.vector-fields",
            "vector over two fields (RRF x 60 / 2)",
            {"query_type": "vector", "fields_mapping": two_vectors},
            [[0.98], [0.97]],
        ),
    )
    async with t.browser() as b:
        for sid, label, parameters, expected in cases:
            await r.set(r.ds(**parameters))
            await r.reset()
            c = await b.chat(MODEL, QUESTION, stream=True)
            distances = [s.get("distances") for s in c.sources]
            searches = await r.searches()
            r.check(
                sid,
                f"AZURE_AI_INCLUDE_SEARCH_SCORES=true, {label}: relevance "
                f"{expected} for the referenced sources",
                c.done
                and c.content == LINKED
                and _close(distances, expected, 0.011)
                and len(searches) == 1,
                f"{c.brief()} distances={distances} {_search_brief(searches)}",
            )

        await r.set(AZURE_AI_INCLUDE_SEARCH_SCORES=False)
        await r.reset()
        c = await b.chat(MODEL, QUESTION, stream=True)
    distances = [s.get("distances") for s in c.sources]
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    citations = _context(res.json).get("citations") or []
    score_keys = sorted(
        {
            key
            for citation in citations
            for key in ("original_search_score", "rerank_score", "relevance")
            if key in citation
        }
    )
    r.check(
        "scores.off",
        "AZURE_AI_INCLUDE_SEARCH_SCORES=false: cards show 0, API citations "
        "without score keys",
        c.done
        and c.content == LINKED
        and distances == [[0.0], [0.0]]
        and res.content == LINKED
        and len(citations) == 3
        and not score_keys,
        f"{c.brief()} distances={distances} API {_answer(res.content)} "
        f"citations={len(citations)} score keys={score_keys}",
    )

    await r.set()
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    citations = _context(res.json).get("citations") or []
    first = citations[0] if citations else {}
    r.check(
        "scores.on.api",
        "API citations carry original_search_score and relevance, no filter_reason",
        res.content == LINKED
        and len(citations) == 3
        and all("relevance" in c and "filter_reason" not in c for c in citations)
        and first.get("original_search_score") == 42.5
        and abs(float(first.get("relevance") or 0) - 0.425) < 0.005,
        f"{_answer(res.content)} first citation keys={sorted(first)} "
        f"relevance={first.get('relevance')} "
        f"score={first.get('original_search_score')}",
    )


def _context_event(raw: str) -> tuple:
    """(bytes of the SSE line with delta.context, that context)."""
    for line in (raw or "").splitlines():
        if not line.startswith("data:") or '"context"' not in line:
            continue
        try:
            event = json.loads(line[5:].strip())
        except ValueError:
            continue
        for choice in (event.get("choices") if isinstance(event, dict) else None) or []:
            context = (choice.get("delta") or {}).get("context")
            if isinstance(context, dict):
                return len(line.encode("utf-8")), context
    return 0, {}


async def rag_events(r: Rag) -> None:
    """Large stream events: the context event of 3 long documents (no budget)
    is larger than aiohttp's default line of 128 KiB and reaches API clients,
    and an upstream event of ~300 KB (as On Your Data's context event of 2.8.x)
    is read; an upstream event larger than the 4 MiB the pipe reads ends the
    stream with an error message, without its text in the answer or the log."""
    t = r.t
    await r.set(AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS=0)
    mark = await r.reset()
    res = await t.owui.chat(MODEL, QUESTION + " big-context big-event", stream=True)
    await t.log.settle(0.5)
    too_long = len(t.log.lines(mark, LINE_TOO_LONG))
    size, context = _context_event(res.raw)
    lengths = [len(str(c.get("content") or "")) for c in context.get("citations") or []]
    padded = max(
        (
            len(line.encode("utf-8"))
            for line in (res.raw or "").splitlines()
            if "x_mock_padding" in line
        ),
        default=0,
    )
    chat = await r.chat()
    r.check(
        "big-context",
        "a context event of more than 128 KiB (one SSE line: 3 documents of "
        "50,000 characters, AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS=0) reaches the API "
        "client, and an upstream event of ~300 KB (one SSE line) is read and "
        "passed on: linked answer and [DONE] (nothing after it), the citations "
        "with the full text",
        res.status == 200
        and res.content == LINKED
        and res.done
        and not res.after_done
        and size > 128 * 1024
        and lengths == [50000, 50000, 50000]
        and padded > 256 * 1024
        and not too_long
        and _grounded(chat),
        f"{_answer(res.content)} done={res.done} {_after_done(res)} "
        f"context_event_bytes={size} citation_lengths={lengths} "
        f"upstream_event_bytes={padded} line_too_long_log={too_long} "
        f"{_chat_brief(chat)}",
    )

    await r.set()
    mark = await r.reset()
    res = await t.owui.chat(MODEL, QUESTION + " huge-event", stream=True)
    await t.log.settle(0.5)
    leaked = FILLER in t.log.since(mark)
    # the stream fails on purpose (an event larger than the pipe reads)
    t.expect_errors(mark, STREAM_ERROR)
    chat = await r.chat()
    r.check(
        "huge-event.api",
        "an upstream event over 4 MiB (one SSE line) ends the stream with an "
        "'Error: ... larger than 4 MiB ...' delta and [DONE] (nothing after "
        "it), without its text in the answer or the log",
        res.status == 200
        and res.content.startswith("Error:")
        and "larger than 4 MiB" in res.content
        and FILLER not in res.content
        and res.done
        and not res.after_done
        and not leaked
        and _grounded(chat),
        f"{_answer(res.content)} done={res.done} {_after_done(res)} "
        f"event text in log={leaked} {_chat_brief(chat)}",
    )

    async with t.browser() as b:
        mark = await r.reset()
        c = await b.chat(MODEL, QUESTION + " huge-event", stream=True)
        await t.log.settle(0.5)
        t.expect_errors(mark, STREAM_ERROR)
    last = _last_status(c.status_history)
    description = str(last.get("description", ""))
    leaked = FILLER in t.log.since(mark)
    r.check(
        "huge-event.browser",
        "browser: an upstream event over 4 MiB -> 'Error: ...' saved and a final "
        "'Error: ...' status done, without its text",
        c.done
        and c.content.startswith("Error:")
        and FILLER not in c.content
        and description.startswith("Error:")
        and FILLER not in description
        and last.get("done") is True
        and not leaked,
        f"{c.brief()} last status={short(description, 120)} done={last.get('done')} "
        f"event text in log={leaked}",
    )


async def rag_content_null(r: Rag) -> None:
    """A non-stream answer whose content is null (e.g. a content filter)."""
    t = r.t
    await r.set()
    async with t.browser() as b:
        await r.reset()
        # Open WebUI 0.11 never marks a non-stream answer without content as
        # done (non_streaming_chat_response_handler), so do not wait for it.
        c = await b.chat(MODEL, QUESTION + " content-null", stream=False, wait=10)
    chat = await r.chat()
    last = _last_status(c.status_history)
    r.check(
        "content-null",
        "browser non-stream answer with content null (content filter): Azure's "
        "response, no pipe error, final status 'Request completed'",
        not c.content.startswith("Error")
        and not c.error
        and last.get("description") == "Request completed"
        and last.get("done") is True
        and _grounded(chat),
        f"{c.brief()} statuses={_statuses(c.status_history)} "
        f"sources={c.source_names} {_chat_brief(chat)}",
    )


async def rag_query_types(r: Rag) -> None:
    """Search request body per query_type (index vectorizer: kind text)."""
    t = r.t
    vq_text = {"kind": "text", "text": QUESTION, "fields": "contentVector"}
    cases = (
        (
            "semantic",
            {"query_type": "semantic", "semantic_configuration": "x100-semantic"},
            QUESTION,
            {
                "search": QUESTION,
                "queryType": "semantic",
                "semanticConfiguration": "x100-semantic",
                "semanticErrorHandling": "partial",
                "top": 10,
            },
        ),
        (
            "semantic.partial",
            {"query_type": "semantic", "semantic_configuration": "x100-semantic"},
            QUESTION + " semantic-partial",
            # HTTP 206 without reranker scores still answers: BM25 / 100
            ("partial", 0.425, "BM25 relevance (42.5 / 100)"),
        ),
        (
            "semantic.partial.hybrid",
            {
                "query_type": "vectorSemanticHybrid",
                "semantic_configuration": "x100-semantic",
            },
            QUESTION + " semantic-partial",
            # fused lists (search + one vector field) without reranker
            # scores: RRF rescaled by rank, 0.0328 x 60 / 2
            ("partial", 0.984, "RRF relevance (0.0328 x 60 / 2)"),
        ),
        (
            "query-type.vector",
            {"query_type": "vector"},
            QUESTION,
            {"vectorQueries": [{**vq_text, "k": 10}], "top": 10},
        ),
        (
            "query-type.vector-semantic-hybrid",
            {
                "query_type": "vectorSemanticHybrid",
                "semantic_configuration": "x100-semantic",
            },
            QUESTION,
            {
                "search": QUESTION,
                "queryType": "semantic",
                "semanticConfiguration": "x100-semantic",
                "semanticErrorHandling": "partial",
                "vectorQueries": [{**vq_text, "k": 50}],
                "top": 10,
            },
        ),
        (
            "vector.integrated",
            {"query_type": "vectorSimpleHybrid"},
            QUESTION,
            {
                "search": QUESTION,
                "queryType": "simple",
                "vectorQueries": [{**vq_text, "k": 10}],
                "top": 10,
            },
        ),
    )
    for sid, parameters, text, expected in cases:
        await r.set(r.ds(**parameters))
        await r.reset()
        res = await t.owui.chat(MODEL, text, stream=False)
        searches, embeds = await r.searches(), await r.embeds()
        s = searches[0] if searches else {}
        citations = _context(res.json).get("citations") or []
        if isinstance(expected, tuple):
            _, value, label = expected
            relevance = (citations[0] if citations else {}).get("relevance")
            ok = (
                s.get("status") == 206
                and len(citations) == 3
                and not any("rerank_score" in c for c in citations)
                and abs(float(relevance or 0) - value) < 0.005
            )
            title = (
                f"{parameters['query_type']} partial result (HTTP 206, no "
                "reranker scores): linked answer, citations without "
                f"rerank_score, {label}"
            )
        else:
            ok = s.get("body") == expected and s.get("status") == 200
            title = (
                f"query_type {parameters['query_type']!r}: search body "
                f"{short(expected, 200)}"
            )
        r.check(
            sid,
            title,
            res.status == 200
            and res.content == LINKED
            and ok
            and len(searches) == 1
            and not embeds,
            f"{_answer(res.content)} status={s.get('status')} "
            f"body={short(s.get('body'), 260)} embeddings={len(embeds)} "
            f"citations={len(citations)}",
        )


async def rag_vector(r: Rag) -> None:
    """Embeddings for vector queries (embedding_dependency)."""
    t = r.t
    deployment = {"type": "deployment_name", "deployment_name": EMBED_DEPLOYMENT}
    hybrid = {"query_type": "vector_simple_hybrid", "embedding_dependency": deployment}
    cases = (
        (
            "vector.deployment",
            "deployment_name: one /openai/v1/embeddings request on the chat host "
            "(model, input [question], chat api-key, no api-version, no model "
            "header, no dimensions); search vectorQueries kind vector, length 8, "
            "contentVector, k 10, plus search",
            {},
            "/openai/v1/embeddings",
            "api-key",
        ),
        (
            "vector.deployment.prefix",
            "deployment_name behind a gateway path prefix (/aoai/openai/...): "
            "embeddings at /aoai/openai/v1/embeddings",
            {
                "AZURE_AI_ENDPOINT": f"{r.mock.url}/aoai/openai/deployments/"
                f"{DEPLOYMENT}/chat/completions?api-version=2025-04-01-preview"
            },
            "/aoai/openai/v1/embeddings",
            "api-key",
        ),
        (
            "vector.deployment.foundry",
            "deployment_name with a Foundry /models/chat/completions endpoint: "
            "embeddings at /openai/v1/embeddings",
            {
                "AZURE_AI_ENDPOINT": f"{r.mock.url}/models/chat/completions"
                "?api-version=2024-05-01-preview"
            },
            "/openai/v1/embeddings",
            "api-key",
        ),
        (
            "vector.deployment.bearer",
            "deployment_name with USE_AUTHORIZATION_HEADER: embeddings with the "
            "chat call's Bearer header",
            {"USE_AUTHORIZATION_HEADER": True},
            "/openai/v1/embeddings",
            "bearer",
        ),
    )
    for sid, title, changes, path, auth in cases:
        await r.set(r.ds(**hybrid), **changes)
        await r.reset()
        res = await t.owui.chat(MODEL, QUESTION, stream=False)
        embeds, searches = await r.embeds(), await r.searches()
        e = embeds[0] if embeds else {}
        s = searches[0] if searches else {}
        vq = (s.get("vectorQueries") or [{}])[0]
        r.check(
            sid,
            title,
            res.status == 200
            and res.content == LINKED
            and len(embeds) == 1
            and e.get("path") == path
            and e.get("model") == EMBED_DEPLOYMENT
            and e.get("input") == [QUESTION]
            and e.get("auth_mode") == auth
            and not e.get("api_version")
            and not e.get("model_header")
            and "dimensions" not in (e.get("body") or {})
            and len(searches) == 1
            and s.get("search") == QUESTION
            and vq.get("kind") == "vector"
            and vq.get("vector_len") == 8
            and vq.get("fields") == "contentVector"
            and vq.get("k") == 10,
            f"{_answer(res.content)} embeddings={short(embeds and {k: e.get(k) for k in ('path', 'model', 'input', 'auth_mode', 'api_version', 'model_header')}, 220)} "
            f"vq={vq} {_search_brief(searches)}",
        )

    endpoint = f"{r.mock.url}/openai/deployments/{EMBED_DEPLOYMENT}/embeddings"
    cases = (
        (
            "vector.endpoint",
            "endpoint dependency: api-version=2024-10-21 appended, its own api-key, "
            "body {input, dimensions}",
            {
                "type": "endpoint",
                "endpoint": endpoint,
                "authentication": {"type": "api_key", "key": RAG_EMBED_KEY},
                "dimensions": 8,
            },
            EMBED_API_VERSION,
            "embed-key",
            {"input": [QUESTION], "dimensions": 8},
        ),
        (
            "vector.endpoint.api-version",
            "endpoint dependency with its own api-version: kept",
            {
                "type": "endpoint",
                "endpoint": endpoint + "?api-version=2023-05-15",
                "authentication": {"type": "api_key", "key": RAG_EMBED_KEY},
            },
            "2023-05-15",
            "embed-key",
            {"input": [QUESTION]},
        ),
        (
            "vector.endpoint.same-host",
            "endpoint dependency without authentication on the chat host: the "
            "chat key is reused",
            {"type": "endpoint", "endpoint": endpoint},
            EMBED_API_VERSION,
            "api-key",
            {"input": [QUESTION]},
        ),
        (
            "vector.endpoint.token",
            "endpoint dependency with authentication access_token: "
            "'Authorization: Bearer <token>', no api-key",
            {
                "type": "endpoint",
                "endpoint": endpoint,
                "authentication": {
                    "type": "access_token",
                    "access_token": RAG_EMBED_TOKEN,
                },
            },
            EMBED_API_VERSION,
            "embed-bearer",
            {"input": [QUESTION]},
        ),
    )
    for sid, title, dependency, api_version, auth, body in cases:
        await r.set(
            r.ds(query_type="vector_simple_hybrid", embedding_dependency=dependency)
        )
        await r.reset()
        res = await t.owui.chat(MODEL, QUESTION, stream=False)
        embeds, searches = await r.embeds(), await r.searches()
        e = embeds[0] if embeds else {}
        s = searches[0] if searches else {}
        vq = (s.get("vectorQueries") or [{}])[0]
        r.check(
            sid,
            title,
            res.status == 200
            and res.content == LINKED
            and len(embeds) == 1
            and e.get("path") == f"/openai/deployments/{EMBED_DEPLOYMENT}/embeddings"
            and e.get("api_version") == api_version
            and e.get("auth_mode") == auth
            and (auth != "embed-bearer" or "api-key" not in (e.get("headers") or {}))
            and e.get("body") == body
            and vq.get("kind") == "vector"
            and vq.get("vector_len") == 8,
            f"{_answer(res.content)} embeddings={short([{k: x.get(k) for k in ('path', 'api_version', 'auth_mode', 'body')} for x in embeds], 260)} "
            f"vq={vq}",
        )

    # another host (or scheme) without authentication: the chat key must not
    # go there
    other = endpoint.replace("127.0.0.1", "localhost")
    cases = (
        (
            "vector.endpoint.other-host",
            "endpoint dependency without authentication on another host: "
            "configuration error, nothing sent",
            {"type": "endpoint", "endpoint": other},
            "authentication is missing",
        ),
        (
            "vector.endpoint.other-scheme",
            "endpoint dependency without authentication on the chat host and "
            "port but another scheme (https instead of http): configuration "
            "error, the chat key is never sent",
            {"type": "endpoint", "endpoint": endpoint.replace("http://", "https://")},
            "authentication is missing",
        ),
        (
            "config-error.embedding-v1",
            "endpoint dependency with a v1 URL (needs model): configuration error "
            "that points to deployment_name",
            {
                "type": "endpoint",
                "endpoint": f"{r.mock.url}/openai/v1/embeddings",
                "authentication": {"type": "api_key", "key": RAG_EMBED_KEY},
            },
            "deployment_name",
        ),
        (
            "config-error.embedding-model-id",
            "embedding_dependency type model_id (Elasticsearch only): "
            "configuration error",
            {"type": "model_id", "model_id": "e2e-model"},
            "",
        ),
    )
    for sid, title, dependency, needle in cases:
        await r.set(
            r.ds(query_type="vector_simple_hybrid", embedding_dependency=dependency)
        )
        mark = await r.reset()
        res = await t.owui.chat(MODEL, QUESTION, stream=False)
        await r.settle_errors(mark, SEARCH_ERROR)
        embeds, searches, chats = await r.embeds(), await r.searches(), await r.chats()
        r.check(
            sid,
            title,
            res.status == 200
            and res.content.startswith(ERROR_PREFIX)
            and needle in res.content
            and not embeds
            and not searches
            and not chats,
            f"{_answer(res.content)} embeddings={len(embeds)} "
            f"searches={len(searches)} chats={len(chats)}",
        )

    await r.set(r.ds(**hybrid))
    mark = await r.reset()
    res = await t.owui.chat(MODEL, QUESTION + " embed-fail", stream=False)
    await r.settle_errors(mark, SEARCH_ERROR)
    embeds, searches, chats = await r.embeds(), await r.searches(), await r.chats()
    r.check(
        "vector.error",
        "embeddings HTTP 500: 'Error: Azure AI Search: embeddings request "
        "failed (HTTP 500) ... Check embedding_dependency.', no search, no chat",
        res.status == 200
        and res.content.startswith(ERROR_PREFIX)
        and "embeddings request failed" in res.content
        and "500" in res.content
        and "embedding_dependency" in res.content
        and len(embeds) == 1
        and not searches
        and not chats,
        f"{_answer(res.content)} embeddings={len(embeds)} searches={len(searches)} "
        f"chats={len(chats)}",
    )


async def rag_fields(r: Rag) -> None:
    t = r.t
    await r.set(r.ds(index_name=CUSTOM_INDEX, fields_mapping=CUSTOM_MAPPING))
    async with t.browser() as b:
        await r.reset()
        c = await b.chat(MODEL, QUESTION, stream=True)
    searches = await r.searches()
    chat = await r.chat()
    s = searches[0] if searches else {}
    select = sorted(f.strip() for f in str(s.get("select") or "").split(",") if f)
    r.check(
        "fields-mapping",
        "x100-custom with fields_mapping: select = the mapped fields, titles, "
        "files and URLs from the mapped fields",
        c.done
        and c.content == LINKED
        and c.source_names == REFERENCED_SOURCES
        and select
        == sorted(
            CUSTOM_MAPPING["content_fields"]
            + ["doc_title", "source_file", "source_url"]
        )
        and s.get("path") == f"/indexes/{CUSTOM_INDEX}/docs/search"
        and DOC1_BLOCK in _block(chat),
        f"{c.brief()} select={s.get('select')!r} {_search_brief(searches)} "
        f"{_chat_brief(chat)}",
    )

    mapping = {
        **CUSTOM_MAPPING,
        "content_fields": ["doc_title", "body"],
        "content_fields_separator": " | ",
    }
    await r.set(r.ds(index_name=CUSTOM_INDEX, fields_mapping=mapping))
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    citations = _context(res.json).get("citations") or []
    content = (citations[0] if citations else {}).get("content")
    r.check(
        "fields-mapping.separator",
        "two content_fields joined with content_fields_separator (prompt and citation)",
        res.content == LINKED
        and content
        == "X100 Product Manual | The X100 charges via USB-C at up to 65 W.",
        f"{_answer(res.content)} doc1 content={content!r}",
    )

    await r.set(r.ds(index_name=CUSTOM_INDEX))
    mark = await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    await t.log.settle(0.5)
    chat, searches = await r.chat(), await r.searches()
    docs = _block_docs(chat)
    warned = _warnings(t, mark, "content_fields")
    r.check(
        "fields-mapping.fallback",
        "no fields_mapping and no 'content' field: the other string fields "
        "become the document text, no select, one warning naming "
        "fields_mapping.content_fields",
        _answered(res.content)
        and _grounded(chat)
        and "The X100 charges via USB-C at up to 65 W." in docs.get(1, "")
        and len(searches) == 1
        and (searches[0].get("select") is None)
        and len(warned) == 1,
        f"{_answer(res.content)} doc1={short(docs.get(1), 120)} "
        f"select={(searches[0] if searches else {}).get('select')!r} "
        f"warnings={len(warned)} {_chat_brief(chat)}",
    )

    await r.set(r.ds(index_name=CUSTOM_INDEX, fields_mapping=CUSTOM_MAPPING))
    async with t.browser() as b:
        mark = await r.reset()
        c = await b.chat(MODEL, QUESTION + " list-title", stream=True)
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION + " list-title", stream=False)
    await t.log.settle(0.5)
    errors = t.log.errors(mark)
    citations = _context(res.json).get("citations") or []
    title = (citations[0] if citations else {}).get("title")
    r.check(
        "list-title",
        "a list-valued title field becomes 'a, b' (citation title and saved "
        "source name), no ERROR in the log",
        c.done
        and c.content == LINKED
        and c.source_names[:1] == ["[doc1] - X100 Product Manual, Chapter 3"]
        and title == "X100 Product Manual, Chapter 3"
        and not errors,
        f"{c.brief()} API title={title!r} log_errors={errors[:2]}",
    )


async def rag_selection(r: Rag) -> None:
    """filter, top_n_documents and strictness."""
    t = r.t
    flt = "category eq 'manuals' and year ge 2024"
    await r.set(r.ds(filter=flt, top_n_documents=2))
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    chat, searches = await r.chat(), await r.searches()
    s = searches[0] if searches else {}
    citations = _context(res.json).get("citations") or []
    r.check(
        "filter-topn",
        "filter sent verbatim; top_n_documents 2 -> top 10 candidates, "
        "documents [doc1] [doc2], 2 citations",
        res.content == LINKED
        and s.get("filter") == flt
        and s.get("top") == 10
        and chat.get("prompt_docs") == [1, 2]
        and len(citations) == 2,
        f"{_answer(res.content)} filter={s.get('filter')!r} top={s.get('top')} "
        f"docs={chat.get('prompt_docs')} citations={len(citations)}",
    )

    await r.set(r.ds(top_n_documents=100))
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    searches = await r.searches()
    s = searches[0] if searches else {}
    r.check(
        "filter-topn.clamp",
        "top_n_documents 100 is clamped to 50 (top 50 candidates)",
        res.content == LINKED and s.get("top") == 50,
        f"{_answer(res.content)} top={s.get('top')}",
    )

    await r.set(r.ds(top_n_documents=8))
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    searches = await r.searches()
    s = searches[0] if searches else {}
    r.check(
        "filter-topn.headroom",
        "top_n_documents 8 -> top 16 candidates (2 x top_n headroom for "
        "strictness and duplicates)",
        res.content == LINKED and s.get("top") == 16,
        f"{_answer(res.content)} top={s.get('top')}",
    )

    await r.set(AZURE_AI_SEARCH_API_VERSION="2024-07-01")
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    searches = await r.searches()
    s = searches[0] if searches else {}
    r.check(
        "api-version",
        "AZURE_AI_SEARCH_API_VERSION=2024-07-01: the search request uses that "
        "api-version",
        res.content == LINKED
        and len(searches) == 1
        and s.get("api_version") == "2024-07-01",
        f"{_answer(res.content)} api_version={s.get('api_version')!r}",
    )

    for kind, parameters in (
        ("simple", {}),
        (
            "semantic",
            {"query_type": "semantic", "semantic_configuration": "x100-semantic"},
        ),
    ):
        await r.set(r.ds(strictness=5, **parameters))
        await r.reset()
        res = await t.owui.chat(MODEL, QUESTION, stream=False)
        chat = await r.chat()
        citations = _context(res.json).get("citations") or []
        r.check(
            f"strictness.{kind}",
            f"strictness 5 ({kind}): only the best document is kept "
            "(BM25 below 75 % of the best / reranker below 2.5 dropped)",
            res.status == 200
            and res.content.startswith("The X100 charges via USB-C [[doc1]](")
            and chat.get("prompt_docs") == [1]
            and len(citations) == 1,
            f"{_answer(res.content)} docs={chat.get('prompt_docs')} "
            f"citations={len(citations)}",
        )


async def rag_budget(r: Rag) -> None:
    """Document text budget (characters / 4 tokens) with 3 x 50,000-character
    documents (big-context)."""
    t = r.t
    cases = (
        (
            "budget.auto",
            "auto (min(32000, 1600 x 5) tokens = 32,000 chars)",
            -1,
            32000,
            [1, 2, 3],
            {},
        ),
        (
            "budget.auto.top-n",
            "auto with top_n_documents 25 (min(32000, 1600 x 25) tokens = "
            "128,000 chars, more than 32,000, less than the 150,000 of the "
            "documents)",
            -1,
            128000,
            [1, 2, 3],
            {"top_n_documents": 25},
        ),
        (
            "budget.valve",
            "AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS=1000 (4,000 chars)",
            1000,
            4000,
            [1, 2, 3],
            {},
        ),
        (
            "budget.unlimited",
            "AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS=0 (no limit)",
            0,
            None,
            [1, 2, 3],
            {},
        ),
        (
            "budget.drop",
            "AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS=300 (1,200 chars: under 500 per "
            "document the lowest-ranked one is dropped)",
            300,
            1200,
            [1, 2],
            {},
        ),
    )
    for sid, label, tokens, chars, expected_docs, parameters in cases:
        await r.set(r.ds(**parameters), AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS=tokens)
        await r.reset()
        res = await t.owui.chat(MODEL, QUESTION + " big-context", stream=False)
        chat = await r.chat()
        docs = _block_docs(chat)
        citations = _context(res.json).get("citations") or []
        lengths = [len(docs.get(n, "")) for n in sorted(docs)]
        cards = [str(c.get("content") or "").strip() for c in citations]
        if chars is None:
            size_ok = lengths == [50000, 50000, 50000]
        else:
            size_ok = (
                sum(lengths) <= chars + 3 * 3
                and sum(lengths) > chars * 3 // 4  # the budget is used
                and all(500 <= n < 50000 for n in lengths)
                and chat.get("docs_chars", 0) <= chars + 1000
            )
        r.check(
            sid,
            f"document budget {label}: documents {expected_docs}, each cut "
            "within the budget, card content = the injected text",
            res.content == LINKED
            and chat.get("prompt_docs") == expected_docs
            and size_ok
            and cards == [docs.get(n, "") for n in sorted(docs)],
            f"{_answer(res.content)} docs={chat.get('prompt_docs')} lengths={lengths} "
            f"block_chars={chat.get('docs_chars')} cards={[len(c) for c in cards]} "
            f"cards_equal={cards == [docs.get(n, '') for n in sorted(docs)]}",
        )


async def rag_prompt(r: Rag) -> None:
    """System rules: in_scope, tool-results clause, role_information, one
    system-like message."""
    t = r.t
    cases = (
        (
            "in-scope.true",
            "in_scope true (default), no tools: 'answer only from the documents' "
            "rule with Microsoft's refusal sentence, no tool-results clause",
            {},
            {},
            lambda c: c.get("in_scope_rule") == "true"
            and not c.get("tool_results_clause")
            and REFUSAL in _system_text(c),
        ),
        (
            "in-scope.tools",
            "in_scope true with client tools: the rule names tool results",
            {},
            {"tools": TOOLS},
            lambda c: c.get("in_scope_rule") == "true" and c.get("tool_results_clause"),
        ),
        (
            "in-scope.false",
            "in_scope false: 'you may answer from your own knowledge' rule",
            {"in_scope": False},
            {},
            lambda c: c.get("in_scope_rule") == "false",
        ),
    )
    for sid, title, parameters, extra, predicate in cases:
        await r.set(r.ds(**parameters))
        await r.reset()
        res = await t.owui.chat(MODEL, QUESTION, stream=False, **extra)
        chat = await r.chat()
        r.check(
            sid,
            title,
            res.content == LINKED and _grounded(chat) and bool(predicate(chat)),
            f"{_answer(res.content)} in_scope_rule={chat.get('in_scope_rule')} "
            f"tool_clause={chat.get('tool_results_clause')} "
            f"refusal={REFUSAL in _system_text(chat)} {_chat_brief(chat)}",
        )

    await r.set(r.ds(role_information=ROLE_INFORMATION))
    results = []
    for label, messages in (
        ("no client system", [{"role": "user", "content": QUESTION}]),
        (
            "client system",
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": QUESTION},
            ],
        ),
        (
            "client developer",
            [
                {"role": "developer", "content": "Be brief."},
                {"role": "user", "content": QUESTION},
            ],
        ),
    ):
        await r.reset()
        res = await t.owui.chat(MODEL, messages, stream=False)
        chat = await r.chat()
        sent = _messages(chat)
        first = sent[0] if sent else {}
        role = messages[0]["role"] if messages[0]["role"] != "user" else "system"
        ok = (
            res.content == LINKED
            and _grounded(chat)
            and chat.get("role_information_in_system") is True
            and first.get("role") == role
            and (
                label == "no client system"
                or _text(first.get("content")).startswith("Be brief.")
            )
        )
        results.append(
            (
                label,
                ok,
                f"{label}: {_answer(res.content)} first_role={first.get('role')} role_info={chat.get('role_information_in_system')} system={chat.get('system_messages')}",
            )
        )
    r.check(
        "role-information",
        "role_information in the system message; without, with a client system "
        "and with a client developer message: exactly one system-like message, "
        "the client's text first",
        all(ok for _, ok, _ in results),
        " | ".join(detail for _, _, detail in results),
    )

    await r.set(r.ds(role_information="Z" * 20000))
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    chat = await r.chat()
    runs = [len(m) for m in re.findall(r"Z+", _system_text(chat))]
    r.check(
        "role-information.truncated",
        "role_information is cut to 16,000 characters",
        res.content == LINKED and runs and 15000 <= max(runs) <= 16000,
        f"{_answer(res.content)} longest run={max(runs) if runs else 0}",
    )


async def rag_sanitize(r: Rag) -> None:
    t = r.t
    await r.set()
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION + " doc-inject", stream=False)
    chat = await r.chat()
    text = _current_text(chat)
    block = _block(chat)
    lowered = block.lower()
    citations = _context(res.json).get("citations") or []
    card = str((citations[0] if citations else {}).get("content") or "")
    tags = re.findall(r"<\s*/?\s*documents\b[^>]*>", text, re.IGNORECASE)
    starts = re.findall(r"<\s*/?\s*documents", text, re.IGNORECASE)
    header = next((ln for ln in text.splitlines() if ln.startswith("[doc1] ")), "")
    file_line = next((ln for ln in text.splitlines() if ln.startswith("File: ")), "")
    marker = text.find("PWNED-E2E")
    r.check(
        "sanitize",
        "document text, title and file name that close the block (also with "
        "nested tags a single removal pass would rebuild, e.g. "
        "'</docu<documents>ments>') or fake labels are sanitized: one "
        "<documents> and one </documents> tag, the injected text inside the "
        "block, [doc9] shown as [ doc9] in the block, its header and the "
        "citation",
        res.content == LINKED
        and tags == ["<documents>", "</documents>"]
        and len(starts) == 2
        and 0 <= marker < text.find("</documents>")
        and "[ doc9]" in block
        and "[doc9]" not in lowered
        and "[doc7]" not in lowered
        and "[ doc9]" in header
        and "[ doc9]" in file_line
        and "[ doc9]" in card
        and not re.search(r"<\s*/?\s*documents", card, re.IGNORECASE)
        and chat.get("prompt_docs") == [1, 2, 3],
        f"{_answer(res.content)} tags={tags} tag_starts={len(starts)} "
        f"injected_text_at={marker}/{text.find('</documents>')} "
        f"header={short(header, 80)} file={short(file_line, 60)} "
        f"docs={chat.get('prompt_docs')} card={short(card, 160)}",
    )


async def rag_list_content(r: Rag) -> None:
    t = r.t
    await r.set()
    content = [{"type": "text", "text": QUESTION}, IMAGE_PART]
    await r.reset()
    res = await t.owui.chat(MODEL, [{"role": "user", "content": content}], stream=False)
    chat = await r.chat()
    sent = _messages(chat)
    index = chat.get("current_user_index")
    parts = (
        sent[index].get("content") if index is not None and index < len(sent) else None
    )
    parts = parts if isinstance(parts, list) else []
    first = parts[0] if parts else {}
    r.check(
        "list-content",
        "user content as [text, image_url] parts: the block is a new first text "
        "part, the original parts are kept, Azure-valid messages",
        res.content == LINKED
        and _grounded(chat)
        and first.get("type") == "text"
        and str(first.get("text") or "").startswith("<documents>")
        and parts[1:] == content
        and chat.get("message_problems") == [],
        f"{_answer(res.content)} part types={[p.get('type') for p in parts if isinstance(p, dict)]} "
        f"problems={chat.get('message_problems')} {_chat_brief(chat)}",
    )

    await r.reset()
    res = await t.owui.chat(
        MODEL, [{"role": "user", "content": [IMAGE_PART]}], stream=False
    )
    chat, searches = await r.chat(), await r.searches()
    r.check(
        "image-only",
        "image-only turn (no query text): no search, no <documents>, plain answer",
        res.content == HELLO
        and not searches
        and chat.get("documents_blocks") == 0
        and not chat.get("rules_in_system"),
        f"{_answer(res.content)} searches={len(searches)} {_chat_brief(chat)}",
    )


async def rag_no_hits(r: Rag) -> None:
    t = r.t
    await r.set()
    async with t.browser() as b:
        await r.reset()
        c = await b.chat(MODEL, QUESTION + " no-hits", stream=True)
    chat, searches = await r.chat(), await r.searches()
    expected = [STATUS_SEARCH, STATUS_NO_DOCS] + STATUS_RAG_STREAM[1:]
    r.check(
        "no-hits",
        "no hits (in_scope true): 'No documents were found' block plus rules, "
        "answer without sources, statuses with 'No documents found in Azure AI "
        "Search', terminal status",
        c.done
        and c.content == NO_REFS
        and c.source_names == []
        and chat.get("no_docs_block") is True
        and chat.get("prompt_docs") == []
        and chat.get("documents_blocks") == 1
        and chat.get("rules_in_system") is True
        and _status_sequence(c.status_history, expected)
        and len(searches) == 1,
        f"{c.brief()} statuses={_statuses(c.status_history)} {_chat_brief(chat)}",
    )

    await r.set(r.ds(in_scope=False))
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION + " no-hits", stream=False)
    chat, searches = await r.chat(), await r.searches()
    r.check(
        "no-hits.out-of-scope",
        "no hits with in_scope false: plain request (no block, no rules)",
        res.content == HELLO
        and chat.get("documents_blocks") == 0
        and not chat.get("rules_in_system")
        and len(searches) == 1,
        f"{_answer(res.content)} {_chat_brief(chat)} {_search_brief(searches)}",
    )


# ------------------------------------------------------------------ errors
async def _error_case(
    r: Rag,
    sid: str,
    title: str,
    text: str = QUESTION,
    needles: tuple = (),
    absent: tuple = (),
    searches_expected: Optional[int] = None,
    log_absent: tuple = (),
    max_seconds: Optional[float] = None,
) -> tuple:
    """One request that must end with 'Error: Azure AI Search: ...' and no
    chat request (the valves are set by the caller); the provoked error is
    logged without traceback, ``log_absent`` nowhere in the log; the answer
    within ``max_seconds`` when given."""
    t = r.t
    mark = await r.reset()
    started = time.monotonic()
    res = await t.owui.chat(MODEL, text, stream=False)
    elapsed = time.monotonic() - started
    await r.settle_errors(mark, SEARCH_ERROR)
    chats, searches = await r.chats(), await r.searches()
    blocks = [
        block
        for block in t.log.error_blocks(mark)
        if all(part in block for part in SEARCH_ERROR)
    ]
    logged = [value for value in log_absent if value in t.log.since(mark)]
    traceback = any("Traceback" in block for block in blocks)
    ok = (
        res.status == 200
        and res.content.startswith(ERROR_PREFIX)
        and all(n in res.content for n in needles)
        and not any(a in res.content for a in absent)
        and not chats
        and (searches_expected is None or len(searches) == searches_expected)
        and not traceback
        and not logged
        and (max_seconds is None or elapsed < max_seconds)
    )
    r.check(
        sid,
        title,
        ok,
        f"{_answer(res.content)} chats={len(chats)} {_search_brief(searches)} "
        f"elapsed={elapsed:.1f}s traceback={traceback} logged={len(logged)}",
    )
    return res, searches, mark


async def rag_search_errors(r: Rag) -> None:
    t = r.t
    await r.set()
    await _error_case(
        r,
        "search-error.500",
        "search HTTP 500: 'Error: Azure AI Search: ...', not retried, no chat "
        "request, logged without traceback",
        QUESTION + " search-500",
        ("500",),
        searches_expected=1,
    )
    await _error_case(
        r,
        "search-error.403",
        "search HTTP 403: error with the Search Index Data Reader hint",
        QUESTION + " search-403",
        ("403", "Search Index Data Reader"),
        searches_expected=1,
    )
    await _error_case(
        r,
        "search-error.402",
        "search HTTP 402: semantic ranker quota hint",
        QUESTION + " search-402",
        ("quota",),
        searches_expected=1,
    )
    await _error_case(
        r,
        "search-error.503",
        "search HTTP 503 every time: two retries, then the throttling error",
        QUESTION + " search-503",
        ("throttling", "503"),
        searches_expected=3,
    )
    await _error_case(
        r,
        "search-error.429-long",
        "search HTTP 429 with Retry-After 30 s: fails at once",
        QUESTION + " search-429-long",
        ("429",),
        searches_expected=1,
    )
    await _error_case(
        r,
        "search-error.302",
        "search HTTP 302: not followed (the api-key would go along; the target "
        "/redirected is never requested), error naming the redirect",
        QUESTION + " search-302",
        ("302",),
        searches_expected=1,
    )
    await _error_case(
        r,
        "search-error.too-large",
        "search answer of 17 MiB (chunked, no Content-Length): the read stops "
        "at 16 MB, error naming chunked documents and fields_mapping",
        QUESTION + " search-huge",
        ("larger than 16 MB", "chunked", "fields_mapping"),
        searches_expected=1,
    )
    await _error_case(
        r,
        "search-error.deadline",
        "search answers HTTP 503 after 20 s every time: retried, the third "
        "attempt is cut off by the 45 s retrieval limit (error naming it), "
        "within 52 s",
        QUESTION + " search-deadline",
        ("within the 45 s retrieval limit",),
        searches_expected=3,
        max_seconds=52,
    )

    await r.set(r.ds(index_name="no-such-index"))
    await _error_case(
        r,
        "search-error.404",
        "unknown index (HTTP 404): error naming the index",
        needles=("no-such-index", "not found"),
        searches_expected=1,
    )
    await r.set(r.ds(fields_mapping={**FULL_MAPPING, "title_field": "no_such_field"}))
    await _error_case(
        r,
        "search-error.400-select",
        "HTTP 400 for an unknown select field: error with the fields_mapping hint",
        needles=("400", "fields_mapping"),
        searches_expected=1,
    )
    flt = f"group_ids/any(g: g eq '{SECRET_GROUP}') and invalid-filter"
    await r.set(r.ds(filter=flt))
    await _error_case(
        r,
        "search-error.400-filter",
        "HTTP 400 whose message quotes the filter: 'the filter in "
        "AZURE_AI_DATA_SOURCES is invalid', the filter (group ids) neither in "
        "the answer nor in the log",
        needles=("the filter in AZURE_AI_DATA_SOURCES is invalid",),
        absent=(SECRET_GROUP,),
        searches_expected=1,
        log_absent=(SECRET_GROUP,),
    )
    await r.set(r.ds(endpoint="http://127.0.0.1:9199"))
    await _error_case(
        r,
        "search-error.connect",
        "endpoint that refuses connections: 'cannot connect to <host>'",
        needles=("cannot connect", "127.0.0.1"),
    )

    async with t.browser() as b:
        await r.set()
        mark = await r.reset()
        c = await b.chat(MODEL, QUESTION + " search-500", stream=True)
        await r.settle_errors(mark, SEARCH_ERROR)
    chats = await r.chats()
    last = _last_status(c.status_history)
    r.check(
        "search-error.browser",
        "browser stream, search HTTP 500: error saved, 'Searching' then a final "
        "'Error: Azure AI Search: ...' status done, no chat request",
        c.content.startswith(ERROR_PREFIX)
        and c.status_descriptions[:1] == [STATUS_SEARCH]
        and str(last.get("description", "")).startswith(ERROR_PREFIX)
        and last.get("done") is True
        and not chats,
        f"{c.brief()} statuses={_statuses(c.status_history)} chats={len(chats)}",
    )

    for sid, text, statuses in (
        ("search-retry", QUESTION + " search-503-once", [503, 200]),
        ("search-retry.429", QUESTION + " search-429-once", [429, 200]),
    ):
        await r.set()
        await r.reset()
        started = time.monotonic()
        res = await t.owui.chat(MODEL, text, stream=False)
        elapsed = time.monotonic() - started
        searches = await r.searches()
        r.check(
            sid,
            f"search {statuses[0]} once: retried, linked answer"
            + (" (Retry-After 1 s honored)" if statuses[0] == 429 else ""),
            res.content == LINKED
            and [s.get("status") for s in searches] == statuses
            and (statuses[0] != 429 or elapsed >= 1.0),
            f"{_answer(res.content)} statuses={[s.get('status') for s in searches]} "
            f"elapsed={elapsed:.1f}s",
        )


async def rag_config(r: Rag) -> None:
    """Configuration errors end the request before any I/O (fail closed)."""
    t = r.t
    search_url = r.search.url
    cases = [
        (
            "config-error.invalid-json",
            "AZURE_AI_DATA_SOURCES that is not valid JSON: error with line and "
            "column, the JSON (and its key) not echoed",
            '{"type": "azure_search", "parameters": {"endpoint": '
            f'"{search_url}", "authentication": {{"type": "api_key", "key": '
            f'"{RAG_SEARCH_KEY}"}}',
            "not valid JSON",
            {},
        ),
        (
            "config-error.elasticsearch",
            "data source type elasticsearch: error naming type azure_search and "
            "the removal of On Your Data in 3.0.0",
            {"type": "elasticsearch", "parameters": {"endpoint": search_url}},
            "not supported; use type azure_search (other types needed Azure "
            "OpenAI On Your Data, which this pipeline no longer uses since 3.0.0)",
            {},
        ),
        (
            "config-error.no-key",
            "no authentication and no AZURE_AI_SEARCH_KEY: 'no API key' error",
            r.ds(auth=None),
            "AZURE_AI_SEARCH_KEY",
            {},
        ),
        ("config-error.empty-object", "{} is a configuration error", {}, "", {}),
        (
            "config-error.endpoint",
            "endpoint that is not an absolute http(s) URL: configuration error",
            r.ds(endpoint="mock-search.search.windows.net"),
            "",
            {},
        ),
        (
            "config-error.query-type",
            "unknown query_type: configuration error",
            r.ds(query_type="fancy"),
            "",
            {},
        ),
        (
            "config-error.auth-type",
            "authentication type connection_string: configuration error",
            r.ds(auth={"type": "connection_string", "connection_string": "x"}),
            "",
            {},
        ),
    ]
    for sid, title, ds, needle, changes in cases:
        await r.set(ds, **changes)
        await _error_case(
            r,
            sid,
            title,
            needles=(needle,),
            absent=(RAG_SEARCH_KEY, '"azure_search"'),
            searches_expected=0,
        )

    await r.set({"type": "elasticsearch", "parameters": {"endpoint": search_url}})
    async with t.browser() as b:
        mark = await r.reset()
        c = await b.chat(MODEL, QUESTION, stream=True)
        await r.settle_errors(mark, SEARCH_ERROR)
    last = _last_status(c.status_history)
    r.check(
        "config-error.browser",
        "browser: a configuration error ends with one 'Error: Azure AI Search: "
        "...' status (done), no 'Searching' status",
        c.content.startswith(ERROR_PREFIX)
        and len(c.status_history) == 1
        and str(last.get("description", "")).startswith(ERROR_PREFIX)
        and last.get("done") is True,
        f"{c.brief()} statuses={_statuses(c.status_history)}",
    )

    mark = t.mark()
    await r.set(
        [r.ds(), {"type": "elasticsearch", "parameters": {"endpoint": search_url}}]
    )
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    res2 = await t.owui.chat(MODEL, QUESTION, stream=False)
    await t.log.settle(0.5)
    warned = _warnings(t, mark, "only the first azure_search entry")
    r.check(
        "config.several",
        "several data sources: the first azure_search entry is used, one warning",
        res.content == LINKED and res2.content == LINKED and len(warned) == 1,
        f"{_answer(res.content)} second={short(res2.content, 60)} warnings={len(warned)}",
    )

    mark = t.mark()
    await r.set(r.ds(foo_unknown="unknown-value-e2e"))
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    await t.log.settle(0.5)
    warned = _warnings(t, mark, "foo_unknown")
    r.check(
        "config.unknown-keys",
        "unknown parameter keys: one warning with the key names only (no values)",
        res.content == LINKED
        and len(warned) == 1
        and "unknown-value-e2e" not in t.log.since(mark),
        f"{_answer(res.content)} warnings={short(warned, 200)}",
    )


async def rag_not_configured(r: Rag) -> None:
    """Whitespace, [] and null count as 'not configured': plain chat as
    before."""
    t = r.t
    results = []
    for value in ("   ", "[]", "null"):
        await r.set(value)
        mark = await r.reset()
        res = await t.owui.chat(MODEL, QUESTION, stream=False)
        await t.log.settle(0.3)
        chat, searches = await r.chat(), await r.searches()
        errors = t.log.errors(mark)
        ok = (
            res.content == HELLO
            and not searches
            and chat.get("documents_blocks") == 0
            and not errors
        )
        results.append(
            (
                ok,
                f"{value!r}: {_answer(res.content)} searches={len(searches)} "
                f"blocks={chat.get('documents_blocks')} errors={errors[:1]}",
            )
        )
    r.check(
        "not-configured",
        "AZURE_AI_DATA_SOURCES '   ', '[]' or 'null': plain answer, no search, "
        "no <documents>, no ERROR",
        all(ok for ok, _ in results),
        " | ".join(detail for _, detail in results),
    )


async def rag_tasks(r: Rag) -> None:
    t = r.t
    await r.set()
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    status, answer, _ = await t.owui.title_task(
        MODEL, [{"role": "user", "content": QUESTION}]
    )
    searches, tasks = await r.searches(), await r.tasks()
    grounded_tasks = [e for e in tasks if e.get("documents_blocks")]
    with_ds = [e for e in tasks if "data_sources" in (e.get("body") or {})]
    r.check(
        "tasks.api",
        "title task with the rag valves (#123): answered without retrieval and "
        "without data_sources (the chat answer before it: grounded, one search)",
        res.content == LINKED
        and status == 200
        and "Mock Title" in str(answer)
        and len(searches) == 1
        and tasks
        and not grounded_tasks
        and not with_ds,
        f"{_answer(res.content)} task HTTP {status} answer={short(answer, 60)} "
        f"searches={len(searches)} task requests={len(tasks)} "
        f"grounded tasks={len(grounded_tasks)} with data_sources={len(with_ds)}",
    )

    async with t.browser() as b:
        await r.reset()
        c = await b.chat(
            MODEL, QUESTION, stream=True, background_tasks=TASKS, wait_title=True
        )
        waited = await _wait_tasks(t, r.mock, c.chat_id)
        c = await b.reload(c)
    searches, tasks = await r.searches(), await r.tasks()
    grounded_tasks = [e for e in tasks if e.get("documents_blocks")]
    with_ds = [e for e in tasks if "data_sources" in (e.get("body") or {})]
    r.check(
        "tasks.browser",
        "browser chat with background tasks (#123): one search (the answer), "
        "task prompts without <documents> and data_sources, only the "
        "referenced sources and the answer's statuses, also after the tasks",
        c.done
        and c.content == LINKED
        and c.source_names == REFERENCED_SOURCES
        and _status_sequence(c.status_history, STATUS_RAG_STREAM)
        and len(searches) == 1
        and len(tasks) >= len(TASKS)
        and not grounded_tasks
        and not with_ds,
        f"{c.brief()} searches={len(searches)} task requests={len(tasks)} "
        f"grounded tasks={len(grounded_tasks)} with data_sources={len(with_ds)} "
        f"{waited}",
    )


async def rag_no_session(r: Rag) -> None:
    """Saved chat without a websocket session (no built-in tools): background
    tasks run with the message's metadata, so their events must not reach it
    (#123: 9 sources and 7 statuses on 2.7.0)."""
    t = r.t
    await r.set()
    aid, uid = str(uuid.uuid4()), str(uuid.uuid4())
    body = {
        "model": MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": QUESTION}],
        "id": aid,
        "parent_id": None,
        "user_message": {
            "id": uid,
            "parentId": None,
            "childrenIds": [aid],
            "role": "user",
            "content": QUESTION,
            "timestamp": int(time.time()),
            "models": [MODEL],
        },
        "background_tasks": TASKS,
    }
    await r.reset()
    status, data = await t.owui.api("POST", "/api/chat/completions", body)
    answer = completion_text(data) or ""
    chat_id = data.get("chat_id") if isinstance(data, dict) else None
    if not chat_id:
        _, chats = await t.owui.api("GET", "/api/v1/chats/?page=1")
        for item in (chats if isinstance(chats, list) else [])[:10]:
            saved = await t.owui.get_chat(item["id"])
            messages = ((saved.get("chat") or {}).get("history") or {}).get(
                "messages"
            ) or {}
            if aid in messages:
                chat_id = item["id"]
                break
    waited = await _wait_tasks(t, r.mock, chat_id)
    saved = await t.owui.get_chat(chat_id) if chat_id else {}
    messages = ((saved.get("chat") or {}).get("history") or {}).get("messages") or {}
    message = messages.get(aid) or {}
    names = [(s.get("source") or {}).get("name") for s in message.get("sources") or []]
    history = message.get("statusHistory") or []
    searches, tasks = await r.searches(), await r.tasks()
    grounded_tasks = [e for e in tasks if e.get("documents_blocks")]
    with_ds = [e for e in tasks if "data_sources" in (e.get("body") or {})]
    answered = status == 200 and answer == LINKED
    r.check(
        "no-session.sources",
        "saved chat without websocket session + background tasks: linked "
        "answer, only the referenced sources and the answer's statuses, one "
        "search, task prompts without <documents> and data_sources (#123)",
        answered
        and message.get("content") == LINKED
        and names == REFERENCED_SOURCES
        and _status_sequence(history, STATUS_RAG_NONSTREAM)
        and len(searches) == 1
        and bool(tasks)
        and not grounded_tasks
        and not with_ds,
        f"answered={answered} task requests={len(tasks)} "
        f"grounded tasks={len(grounded_tasks)} with data_sources={len(with_ds)} "
        f"searches={len(searches)} sources={names} "
        f"statuses={_statuses(history)} content={short(message.get('content'))} "
        f"chat={chat_id} {waited}",
    )


async def rag_client_data_sources(r: Rag) -> None:
    """data_sources sent by the client (On Your Data, removed in 3.0.0) are
    refused (never fetched, never forwarded), with or without the valve, in
    streamed and non-streamed API requests and in the browser path (as an
    inlet filter adds them), with the pipe's ERROR line and a terminal error
    status; an empty list is ignored."""
    t = r.t
    client = [
        {
            "type": "azure_search",
            "parameters": {
                "endpoint": r.search.url,
                "index_name": "client-index",
                "filter": "group_ids/any(g: g eq 'grp-client')",
                "authentication": {"type": "api_key", "key": RAG_SEARCH_KEY},
            },
        }
    ]
    for sid, ds in (
        ("client-data-sources.valve", None),
        ("client-data-sources.no-valve", ""),
    ):
        await r.set(ds)
        for stream in (False, True):
            mark = await r.reset()
            res = await t.owui.chat(MODEL, QUESTION, stream=stream, data_sources=client)
            await r.settle_errors(mark, SEARCH_ERROR)
            chats, recorded = await r.chats(), await r.search.requests()
            logged = _client_ds_errors(t, mark)
            r.check(
                f"{sid}.stream" if stream else sid,
                f"client data_sources (stream={stream}, "
                f"{'valve set' if ds is None else 'no valve'}): the answer is "
                "exactly 'Error: Azure AI Search: data_sources in the request is "
                "not supported: this pipeline no longer uses Azure OpenAI On Your "
                "Data (removed in 3.0.0); ...'"
                + (" and [DONE]" if stream else "")
                + ", the pipe's ERROR line logged once, nothing searched, nothing "
                "sent upstream",
                res.content == CLIENT_DS_ANSWER
                and (res.done or not stream)
                and len(logged) == 1
                and not recorded
                and not chats,
                f"{_answer(res.content)} done={res.done} error lines={len(logged)} "
                f"search mock requests={len(recorded)} chats={len(chats)}",
            )

    # The browser path, with data_sources in the body as an inlet filter
    # adds them: the event emitter gets the terminal error status
    await r.set()
    async with t.browser() as b:
        mark = await r.reset()
        c = await b.chat(MODEL, QUESTION, stream=True, extra={"data_sources": client})
        await r.settle_errors(mark, SEARCH_ERROR)
    chats, recorded = await r.chats(), await r.search.requests()
    logged = _client_ds_errors(t, mark)
    r.check(
        "client-data-sources.browser",
        "browser path with data_sources in the body (as from an inlet filter): "
        "the error saved as the answer and as the only, final status (done), the "
        "pipe's ERROR line logged once, nothing searched, nothing sent upstream",
        c.done
        and c.content == CLIENT_DS_ANSWER
        and _statuses(c.status_history) == [(CLIENT_DS_ANSWER, True, False)]
        and len(logged) == 1
        and not recorded
        and not chats,
        f"{c.brief()} statuses={_statuses(c.status_history)} error "
        f"lines={len(logged)} search mock requests={len(recorded)} "
        f"chats={len(chats)}",
    )

    await r.set()
    mark = await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False, data_sources=[])
    await t.log.settle(0.3)
    chat, searches = await r.chat(), await r.searches()
    errors = t.log.errors(mark)
    r.check(
        "client-data-sources.empty",
        "an empty data_sources list from the client is ignored: grounded answer "
        "from the valve's search, no data_sources upstream, no ERROR",
        res.content == LINKED and _grounded(chat) and len(searches) == 1 and not errors,
        f"{_answer(res.content)} {_chat_brief(chat)} {_search_brief(searches)} "
        f"log_errors={errors[:1]}",
    )


# --------------------------------------------------------- query generation
async def rag_qgen(r: Rag) -> None:
    t = r.t
    await r.set()
    history = _followup("and the warranty?", DETAILS_BLOCK + LINKED)
    await r.reset()
    res = await t.owui.chat(MODEL, history, stream=False)
    qgens, searches, chat = await r.qgens(), await r.searches(), await r.chat()
    q = qgens[0] if qgens else {}
    body = q.get("body") or {}
    messages = body.get("messages") or []
    system = _text((messages[0] if messages else {}).get("content"))
    transcript = q.get("transcript") or ""
    context = _context(res.json)
    request_ok = (
        q.get("path") == chat.get("path")
        and q.get("auth_mode") == "api-key"
        and set(body) <= {"model", "messages", "stream"}
        and body.get("stream") is False
        and body.get("model") == DEPLOYMENT
        and len(messages) == 2
        and system.startswith(QG_MARKER)
        and "1 to 3" in system
        and "Latest user message: and the warranty?" in transcript
        and f"User: {QUESTION}" in transcript
        and "Assistant:" in transcript
        and "[doc1]" in transcript
        and "[[doc1]](" not in transcript
        and "secret-thoughts-e2e" not in transcript
    )
    r.check(
        "qgen.followup.api",
        "follow-up turn (auto): one query-generation request (same endpoint, "
        "model, no sampling or tool parameters; transcript with unlinked "
        "history, no <details>), two searches, 3 deduped citations, intent = "
        "the generated queries",
        res.content == LINKED
        and len(qgens) == 1
        and request_ok
        and sorted(s.get("search") for s in searches) == QG_QUERIES
        and _titles(context.get("citations") or []) == TITLES
        and _intent(context) == QG_QUERIES
        and _grounded(chat),
        f"{_answer(res.content)} qgen={len(qgens)} request_ok={request_ok} "
        f"qgen keys={sorted(body)} stream={body.get('stream')!r} "
        f"model={body.get('model')!r} transcript={short(transcript, 160)} "
        f"{_search_brief(searches)} intent={context.get('intent')!r}",
    )

    async with t.browser() as b:
        c1 = await b.chat(MODEL, QUESTION, stream=True)
        await r.reset()
        c = await b.chat(
            MODEL,
            "and the warranty?",
            stream=True,
            history=[
                {"role": "user", "content": QUESTION},
                {"role": "assistant", "content": c1.content},
            ],
            chat_id=c1.chat_id,
            parent_id=c1.message_id,
        )
    qgens, searches = await r.qgens(), await r.searches()
    distances = [s.get("distances") for s in c.sources]
    r.check(
        "qgen.followup.browser",
        "browser follow-up: 'Generating search queries...' first, two searches, "
        "linked answer, referenced sources with the kept occurrences' scores",
        c.done
        and c.content == LINKED
        and c.source_names == REFERENCED_SOURCES
        and _close(distances, [[0.425], [0.425]])
        and _status_sequence(c.status_history, [STATUS_QGEN] + STATUS_RAG_STREAM)
        and len(qgens) == 1
        and len(searches) == 2,
        f"{c.brief()} distances={distances} statuses={_statuses(c.status_history)} "
        f"qgen={len(qgens)} searches={len(searches)}",
    )

    await r.set(
        AZURE_AI_ENDPOINT=f"{r.mock.url}/models/chat/completions"
        "?api-version=2024-05-01-preview",
        AZURE_AI_MODEL_IN_BODY=True,
    )
    await r.reset()
    res = await t.owui.chat(MODEL, _followup("and the warranty?"), stream=False)
    qgens, searches = await r.qgens(), await r.searches()
    q = qgens[0] if qgens else {}
    r.check(
        "qgen.model-in-body",
        "Foundry /models endpoint with AZURE_AI_MODEL_IN_BODY: the generation "
        "request carries the body model (the mock answers 400 without), two "
        "searches",
        res.content == LINKED
        and len(qgens) == 1
        and (q.get("body") or {}).get("model") == DEPLOYMENT
        and not q.get("model_header")
        and q.get("path") == "/models/chat/completions"
        and len(searches) == 2,
        f"{_answer(res.content)} qgen={len(qgens)} body.model="
        f"{(q.get('body') or {}).get('model')!r} header={q.get('model_header')!r} "
        f"{_search_brief(searches)}",
    )

    for sid, valve, messages, generations, queries in (
        ("qgen.first-turn", "auto", QUESTION, 0, [QUESTION]),
        ("qgen.always", "always", QUESTION, 1, QG_QUERIES),
        ("qgen.off", "off", _followup(AGAIN), 0, [AGAIN]),
    ):
        await r.set(AZURE_AI_SEARCH_QUERY_GENERATION=valve)
        await r.reset()
        res = await t.owui.chat(MODEL, messages, stream=False)
        qgens, searches = await r.qgens(), await r.searches()
        r.check(
            sid,
            f"AZURE_AI_SEARCH_QUERY_GENERATION={valve} "
            f"({'first turn' if isinstance(messages, str) else 'follow-up'}): "
            f"{generations} generation request(s), searches {queries}",
            res.content == LINKED
            and len(qgens) == generations
            and sorted(s.get("search") for s in searches) == sorted(queries),
            f"{_answer(res.content)} qgen={len(qgens)} {_search_brief(searches)}",
        )

    await r.set()
    for sid, trigger, expect_line in (
        ("qgen.fallback.bad-json", "qgen-bad-json", False),
        ("qgen.fallback.400", "qgen-400", True),
        ("qgen.fallback.deep", "qgen-deep", False),  # RecursionError in json
    ):
        latest = f"{AGAIN} {trigger}"
        mark = await r.reset()
        res = await t.owui.chat(MODEL, _followup(latest), stream=False)
        await t.log.settle(0.5)
        qgens, searches = await r.qgens(), await r.searches()
        errors = t.log.errors(mark)
        lines = [
            line
            for line in t.log.since(mark).splitlines()
            if "query generation failed" in line
        ]
        line_ok = (not expect_line) or (
            len(lines) == 1 and "400" in lines[0] and trigger not in lines[0]
        )
        r.check(
            sid,
            f"query generation fails ({trigger}): one search with the user text, "
            "linked answer, no ERROR"
            + (
                ", one 'query generation failed (HTTP 400)' line without the user text"
                if expect_line
                else ""
            ),
            res.content == LINKED
            and len(qgens) == 1
            and [s.get("search") for s in searches] == [latest]
            and not errors
            and line_ok,
            f"{_answer(res.content)} qgen={len(qgens)} {_search_brief(searches)} "
            f"log_errors={errors[:1]} lines={short(lines, 200)}",
        )

    await r.reset()
    res = await t.owui.chat(
        MODEL, _followup("and the warranty? qgen-think"), stream=False
    )
    searches = await r.searches()
    r.check(
        "qgen.think",
        "generation answer with a <think> block holding its own draft queries: "
        "the JSON after it is used (one search 'x100 charging', none for the "
        "draft)",
        res.content == LINKED
        and [s.get("search") for s in searches] == ["x100 charging"],
        f"{_answer(res.content)} {_search_brief(searches)}",
    )

    # [CONVERSATION SUMMARY] of Open WebUI's compaction in the system message
    summary = "[CONVERSATION SUMMARY] summary-marker-e2e: the X100 charger."
    await r.reset()
    res = await t.owui.chat(
        MODEL,
        [{"role": "system", "content": f"Be brief.\n\n{summary}"}]
        + _followup("and the warranty?"),
        stream=False,
    )
    qgens = await r.qgens()
    transcript = (qgens[0] if qgens else {}).get("transcript") or ""
    r.check(
        "qgen.summary",
        "the conversation summary Open WebUI's compaction put into the system "
        "message is part of the generation transcript",
        res.content == LINKED
        and len(qgens) == 1
        and "[CONVERSATION SUMMARY]" in transcript
        and "summary-marker-e2e" in transcript,
        f"{_answer(res.content)} qgen={len(qgens)} transcript={short(transcript, 200)}",
    )

    # Merge order of several queries: reciprocal rank fusion (BM25), the best
    # reranker score (semantic); never the order of first appearance.
    for kind, parameters, expected in (
        ("simple", {}, ORDER_RRF),
        (
            "semantic",
            {"query_type": "semantic", "semantic_configuration": "x100-semantic"},
            ORDER_RERANK,
        ),
    ):
        await r.set(r.ds(**parameters))
        await r.reset()
        res = await t.owui.chat(
            MODEL, _followup("and the warranty? qgen-order"), stream=False
        )
        chat, searches = await r.chat(), await r.searches()
        citations = _context(res.json).get("citations") or []
        block_titles = _BLOCK_TITLES.findall(_block(chat))
        r.check(
            f"qgen.merge-order.{kind}",
            f"two generated queries ({kind}): documents ordered by "
            + (
                "reciprocal rank fusion"
                if kind == "simple"
                else "the best reranker score"
            )
            + f" {expected}, in the prompt ([docN]) and the citations",
            res.status == 200
            and res.content.startswith("The X100 charges via USB-C [[doc1]](")
            and sorted(s.get("search") for s in searches)
            == ["x100 order one", "x100 order two"]
            and _titles(citations) == expected
            and block_titles == expected,
            f"{_answer(res.content)} citations={_titles(citations)} "
            f"block={block_titles} {_search_brief(searches)}",
        )

    # Strictness per query: the weaker query's best hit (BM25 8.0) is far
    # below the other query's best (42.5) but kept.
    await r.set()
    await r.reset()
    res = await t.owui.chat(
        MODEL, _followup("and the warranty? qgen-scale"), stream=False
    )
    searches = await r.searches()
    citations = _context(res.json).get("citations") or []
    r.check(
        "qgen.strictness-per-query",
        "strictness compares a hit with the best hit of its own query: the only "
        "hit of a query with low BM25 scores (8.0, the other query's best is "
        "42.5) is kept",
        res.content.startswith("The X100 charges via USB-C [[doc1]](")
        and sorted(s.get("search") for s in searches)
        == ["x100 charging", "x100 low scores"]
        and _titles(citations) == ORDER_RRF,
        f"{_answer(res.content)} citations={_titles(citations)} "
        f"{_search_brief(searches)}",
    )

    older = [
        {"role": "user", "content": "first-turn-marker-e2e?"},
        {"role": "assistant", "content": "first-answer-marker-e2e."},
        {"role": "user", "content": "second turn?"},
        {"role": "assistant", "content": "z" * 3000},
        {"role": "user", "content": "third turn?"},
        {"role": "assistant", "content": "third answer."},
    ]
    await r.reset()
    res = await t.owui.chat(MODEL, older + _followup("and the warranty?"), stream=False)
    qgens = await r.qgens()
    transcript = (qgens[0] if qgens else {}).get("transcript") or ""
    runs = [len(m) for m in re.findall(r"z+", transcript)]
    r.check(
        "qgen.transcript",
        "generation transcript: only the last 6 messages before the current one, "
        "each cut to 2,000 characters",
        res.content == LINKED
        and len(qgens) == 1
        and "first-turn-marker-e2e" not in transcript
        and "first-answer-marker-e2e" not in transcript
        and "second turn?" in transcript
        and runs
        and 1000 <= max(runs) <= 2000,
        f"{_answer(res.content)} qgen={len(qgens)} longest z run="
        f"{max(runs) if runs else 0} transcript={short(transcript[-300:], 200)}",
    )

    for sid, parameters, count in (
        ("qgen.max-queries", {}, 3),
        ("qgen.max-queries.valve", {"max_search_queries": 2}, 2),
    ):
        await r.set(r.ds(**parameters))
        await r.reset()
        res = await t.owui.chat(
            MODEL, _followup("and the warranty? qgen-many"), stream=False
        )
        searches = await r.searches()
        texts = [str(s.get("search") or "") for s in searches]
        lowered = [x.lower() for x in texts]
        r.check(
            sid,
            f"generated queries deduplicated case-insensitively, each cut to 300 "
            f"characters, at most max_search_queries ({count})",
            res.content == LINKED
            and len(texts) == count
            and len(set(lowered)) == len(lowered)
            and "x100 charging" in lowered
            and any(x.startswith("x100 long") and len(x) <= 300 for x in texts)
            and all(len(x) <= 300 for x in texts),
            f"{_answer(res.content)} searches={[short(x, 40) for x in texts]} "
            f"lengths={[len(x) for x in texts]}",
        )

    for sid, allow, ok_answer in (
        ("qgen.partial.strict", False, False),
        ("qgen.partial.allowed", True, True),
    ):
        await r.set(r.ds(allow_partial_result=allow))
        mark = await r.reset()
        res = await t.owui.chat(
            MODEL, _followup("and the warranty? qgen-partial"), stream=False
        )
        if not ok_answer:
            await r.settle_errors(mark, SEARCH_ERROR)
        searches, chats = await r.searches(), await r.chats()
        citations = _context(res.json).get("citations") or []
        if ok_answer:
            ok = res.content == LINKED and len(citations) == 2 and len(chats) == 1
        else:
            ok = res.content.startswith(ERROR_PREFIX) and not chats
        r.check(
            sid,
            f"two generated queries, one search fails (HTTP 500), "
            f"allow_partial_result {allow}: "
            + ("answer from the other query" if ok_answer else "the request fails"),
            ok and len(searches) == 2,
            f"{_answer(res.content)} citations={len(citations)} chats={len(chats)} "
            f"{_search_brief(searches)}",
        )

    await r.set(r.ds(allow_partial_result=True))
    mark = await r.reset()
    res = await t.owui.chat(
        MODEL, _followup("and the warranty? qgen-allfail"), stream=False
    )
    await r.settle_errors(mark, SEARCH_ERROR)
    searches, chats = await r.searches(), await r.chats()
    r.check(
        "qgen.partial.all-failed",
        "two generated queries, both searches fail (HTTP 500), "
        "allow_partial_result True: the request fails (no answer without "
        "documents)",
        res.content.startswith(ERROR_PREFIX)
        and "500" in res.content
        and len(searches) == 2
        and not chats,
        f"{_answer(res.content)} chats={len(chats)} {_search_brief(searches)}",
    )

    # Timeouts (10 s) on a separate model: after 3 in a row the generation is
    # paused for that model (15 minutes), so the other checks are not affected.
    await r.set(AZURE_AI_MODEL=f"{DEPLOYMENT};{PAUSE_DEPLOYMENT};{RESET_DEPLOYMENT}")
    model = f"{FID}.{PAUSE_DEPLOYMENT}"
    pause_mark = t.mark()
    rounds = []
    paused_at = time.monotonic()
    for i in range(4):
        latest = f"{AGAIN} qgen-slow {i}"
        await r.reset()
        started = time.monotonic()
        res = await t.owui.chat(model, _followup(latest), stream=False)
        elapsed = time.monotonic() - started
        if i == 2:
            paused_at = time.monotonic()  # the third timeout starts the pause
        qgens, searches = await r.qgens(), await r.searches()
        rounds.append(
            (res, elapsed, len(qgens), [s.get("search") for s in searches], latest)
        )
        if i == 0:
            r.check(
                "qgen.slow",
                "query generation slower than 10 s: fallback to the user text "
                "after ~10 s, linked answer",
                res.content == LINKED
                and len(qgens) == 1
                and [s.get("search") for s in searches] == [latest]
                and 9.0 <= elapsed < 30,
                f"{_answer(res.content)} elapsed={elapsed:.1f}s qgen={len(qgens)} "
                f"{_search_brief(searches)}",
            )
    await t.log.settle(0.5)
    warned = _warnings(
        t, pause_mark, PAUSE_DEPLOYMENT, "AZURE_AI_SEARCH_QUERY_GENERATION=off"
    )
    last = rounds[-1]
    r.check(
        "qgen.pause",
        "after 3 generation timeouts in a row the model's generation is paused: "
        "the 4th follow-up sends no generation request and answers at once; one "
        "warning naming the model and AZURE_AI_SEARCH_QUERY_GENERATION=off",
        all(x[0].content == LINKED for x in rounds)
        and [x[2] for x in rounds] == [1, 1, 1, 0]
        and last[3] == [last[4]]
        and last[1] < 8
        and len(warned) == 1,
        f"{_answer(last[0].content)} generations={[x[2] for x in rounds]} "
        f"elapsed={[round(x[1], 1) for x in rounds]} warnings={len(warned)}",
    )

    # The pause is per model: another deployment still generates queries.
    await r.reset()
    res = await t.owui.chat(MODEL, _followup(f"{AGAIN} other model"), stream=False)
    qgens = await r.qgens()
    r.check(
        "qgen.pause.other-model",
        f"while {PAUSE_DEPLOYMENT} is paused, {DEPLOYMENT} still sends its "
        "generation request (the pause is per model)",
        res.content == LINKED and len(qgens) == 1,
        f"{_answer(res.content)} qgen={len(qgens)}",
    )

    # Timeouts that are not in a row: timeout, success, timeout, timeout ->
    # the count was reset by the success, so the next follow-up still
    # generates queries.
    reset_model = f"{FID}.{RESET_DEPLOYMENT}"
    sequence = []
    first_answer = None
    for i, trigger in enumerate(("qgen-slow", "", "qgen-slow", "qgen-slow", "")):
        await r.reset()
        res = await t.owui.chat(
            reset_model, _followup(f"{AGAIN} {trigger} reset {i}"), stream=False
        )
        first_answer = res.content if first_answer is None else first_answer
        sequence.append((res.content == LINKED, len(await r.qgens())))
    r.check(
        "qgen.pause.not-in-a-row",
        "timeout, success, timeout, timeout on one model: a success resets the "
        "count, so the fifth follow-up still sends a generation request",
        all(ok for ok, _ in sequence) and [n for _, n in sequence] == [1] * 5,
        f"{_answer(first_answer)} answers linked={[ok for ok, _ in sequence]} "
        f"generations={[n for _, n in sequence]}",
    )

    # 15 minutes, not seconds: still paused after more than 16 s.
    wait = 16.5 - (time.monotonic() - paused_at)
    if wait > 0:
        await asyncio.sleep(wait)
    await r.reset()
    res = await t.owui.chat(model, _followup(f"{AGAIN} still paused"), stream=False)
    qgens = await r.qgens()
    r.check(
        "qgen.pause.lasts",
        f"{PAUSE_DEPLOYMENT} is still paused more than 16 s after the third "
        "timeout (the pause lasts 15 minutes): no generation request",
        res.content == LINKED and not qgens and time.monotonic() - paused_at > 16,
        f"{_answer(res.content)} qgen={len(qgens)} "
        f"since pause={time.monotonic() - paused_at:.0f}s",
    )


# --------------------------------------------------------------- auth, mode
async def rag_auth(r: Rag) -> None:
    t = r.t
    await r.set(r.ds(auth={"type": "access_token", "access_token": RAG_SEARCH_TOKEN}))
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    searches = await r.searches()
    s = searches[0] if searches else {}
    r.check(
        "auth.bearer",
        "authentication access_token: 'Authorization: Bearer <token>', no api-key",
        res.content == LINKED
        and s.get("auth_mode") == "bearer"
        and "api-key" not in (s.get("headers") or {}),
        f"{_answer(res.content)} auth={s.get('auth_mode')} "
        f"headers={sorted(s.get('headers') or {})}",
    )

    await r.set(r.ds(auth=None), AZURE_AI_SEARCH_KEY=RAG_SEARCH_KEY)
    stored = (await t.owui.get_valves(FID)).get("AZURE_AI_SEARCH_KEY", "")
    spec = (await t.owui.valves_spec(FID)).get("properties", {})
    password = ((spec.get("AZURE_AI_SEARCH_KEY") or {}).get("input") or {}).get("type")
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    searches = await r.searches()
    s = searches[0] if searches else {}
    r.check(
        "auth.key-valve",
        "AZURE_AI_SEARCH_KEY (encrypted password valve) and no key in the JSON: "
        "the valve's key in the api-key header",
        res.content == LINKED
        and (s.get("headers") or {}).get("api-key") == RAG_SEARCH_KEY
        and str(stored).startswith("encrypted:")
        and RAG_SEARCH_KEY not in str(stored)
        and password == "password",
        f"{_answer(res.content)} api-key ok={(s.get('headers') or {}).get('api-key') == RAG_SEARCH_KEY} "
        f"stored={short(stored, 30)} input={password}",
    )

    mark = t.mark()
    await r.set(
        r.ds(auth={"type": "api_key", "key": RAG_JSON_KEY}),
        AZURE_AI_SEARCH_KEY=RAG_SEARCH_KEY,
    )
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    res2 = await t.owui.chat(MODEL, QUESTION, stream=False)
    await t.log.settle(0.5)
    searches = await r.searches()
    keys = [(s.get("headers") or {}).get("api-key") for s in searches]
    warned = _warnings(t, mark, "remove the plaintext key")
    r.check(
        "auth.key-valve.wins",
        "AZURE_AI_SEARCH_KEY and a key in the JSON: the valve wins, one warning "
        "to remove the plaintext key",
        res.content == LINKED
        and res2.content == LINKED
        and keys == [RAG_SEARCH_KEY, RAG_SEARCH_KEY]
        and len(warned) == 1,
        f"{_answer(res.content)} valve key used={[k == RAG_SEARCH_KEY for k in keys]} "
        f"warnings={len(warned)}",
    )

    # Managed identity: run.sh emulates App Service managed identity
    # (IDENTITY_ENDPOINT -> mock_search /msi/token). App Service, Functions and
    # Container Apps select a user-assigned identity by its resource ID with
    # the query parameter mi_res_id (an unknown name such as resource_id is
    # ignored there: the token of the system-assigned identity comes back).
    for sid, auth, ids in (
        (
            "auth.mi.system",
            {"type": "system_assigned_managed_identity"},
            {},
        ),
        (
            "auth.mi.user",
            {
                "type": "user_assigned_managed_identity",
                "managed_identity_resource_id": MI_RESOURCE_ID,
            },
            {"mi_res_id": MI_RESOURCE_ID},
        ),
    ):
        await r.set(r.ds(auth=auth))
        await r.reset()
        res = await t.owui.chat(MODEL, QUESTION, stream=False)
        res2 = await t.owui.chat(MODEL, QUESTION, stream=False)
        tokens, searches = await r.tokens(), await r.searches()
        token = tokens[0] if tokens else {}
        r.check(
            sid,
            f"{auth['type']}: an Entra token of the Open WebUI host (scope "
            "search.azure.com"
            + (
                ", the identity selected with mi_res_id=<resource id>"
                if ids
                else ", no identity parameter"
            )
            + ") as Bearer; the second request uses the cached token",
            res.content == LINKED
            and res2.content == LINKED
            and len(tokens) == 1
            and str(token.get("resource", "")).rstrip("/") == SEARCH_RESOURCE
            and (token.get("identity_ids") or {}) == ids
            and [s.get("auth_mode") for s in searches] == ["bearer", "bearer"],
            f"{_answer(res.content)} token requests={len(tokens)} "
            f"resource={token.get('resource')!r} ids={token.get('identity_ids')} "
            f"auth={[s.get('auth_mode') for s in searches]}",
        )

    await r.set(
        r.ds(
            auth={
                "type": "user_assigned_managed_identity",
                "managed_identity_resource_id": MI_FAIL_ID,
            }
        )
    )
    res, searches, _ = await _error_case(
        r,
        "auth.mi-error",
        "managed identity token request fails (HTTP 400): 'could not get a "
        "Microsoft Entra token for the managed identity of the Open WebUI "
        "host', no search, no chat",
        needles=("could not get a Microsoft Entra token",),
        absent=(RAG_SEARCH_TOKEN,),
        searches_expected=0,
    )


async def rag_context_length(r: Rag) -> None:
    t = r.t
    await r.set()
    mark = await r.reset()
    res = await t.owui.chat(MODEL, QUESTION + " context-too-long", stream=False)
    await r.settle_errors(mark, CHAT_ERROR)
    chat = await r.chat()
    r.check(
        "context-length",
        "chat HTTP 400 context_length_exceeded on a request with documents: "
        "exactly Azure's message plus the budget / new chat hint (no other hint)",
        res.content == CONTEXT_LENGTH_ANSWER and _grounded(chat),
        f"answer={short(res.content, 300)} {_chat_brief(chat)}",
    )


async def rag_stop(r: Rag) -> None:
    """Stop while the search is running: the open status ends with done."""
    t = r.t
    await r.set()
    async with t.browser() as b:
        mark = await r.reset()
        c = await b.chat(
            MODEL, QUESTION + " search-slow", stream=True, stop_after_s=3, stop_wait=10
        )
        await asyncio.sleep(2)
        c = await b.reload(c)
    await t.log.settle(0.5)
    chats, searches = await r.chats(), await r.searches()
    errors = t.log.errors(mark)
    stop_ok = bool(c.stopped) and all(code == 200 for code, _ in c.stopped)
    last = _last_status(c.status_history)
    r.check(
        "stop",
        "Stop during a slow search: the last status is done, no chat request, "
        "no ERROR / Traceback",
        stop_ok
        and last.get("done") is True
        and c.status_descriptions[:1] == [STATUS_SEARCH]
        and not chats
        and len(searches) == 1
        and not errors,
        f"{c.brief()} stop_ok={stop_ok} last={last} chats={len(chats)} "
        f"searches={len(searches)} log_errors={errors[:2]}",
    )

    # Stop while the search queries are generated (follow-up, slow generation)
    async with t.browser() as b:
        c1 = await b.chat(MODEL, QUESTION, stream=True)
        mark = await r.reset()
        c = await b.chat(
            MODEL,
            "and the warranty? qgen-slow stop",
            stream=True,
            history=[
                {"role": "user", "content": QUESTION},
                {"role": "assistant", "content": c1.content},
            ],
            chat_id=c1.chat_id,
            parent_id=c1.message_id,
            stop_after_s=3,
            stop_wait=10,
        )
        await asyncio.sleep(2)
        c = await b.reload(c)
    await t.log.settle(0.5)
    chats, searches, qgens = await r.chats(), await r.searches(), await r.qgens()
    errors = t.log.errors(mark)
    stop_ok = bool(c.stopped) and all(code == 200 for code, _ in c.stopped)
    last = _last_status(c.status_history)
    r.check(
        "stop.qgen",
        "Stop while the search queries are generated: the 'Generating search "
        "queries...' status is ended (last status done), no search, no chat "
        "request, no ERROR / Traceback",
        stop_ok
        and last.get("done") is True
        and c.status_descriptions[:1] == [STATUS_QGEN]
        and len(qgens) == 1
        and not searches
        and not chats
        and not errors,
        f"{c.brief()} stop_ok={stop_ok} last={last} qgen={len(qgens)} "
        f"searches={len(searches)} chats={len(chats)} log_errors={errors[:2]}",
    )


# ------------------------------------------------------- removed mode valve
async def rag_mode_removed(r: Rag) -> None:
    """AZURE_AI_SEARCH_MODE existed only in 2.9.0 pre-releases (pipeline /
    on_your_data). A value stored by such a pre-release, or set in the
    environment, is ignored by 3.0.0 without an error."""
    t = r.t
    source = t.source(PATH)
    marker = "    class Valves(BaseModel):\n"
    patched = source.replace(marker, marker + STALE_MODE_VALVE, 1)
    installed, stored = 0, None
    try:
        installed, _ = await t.owui.install_function(FID, "Azure AI Foundry", patched)
        await r.set(**{SEARCH_MODE: "on_your_data"})
        stored = (await t.owui.get_valves(FID)).get(SEARCH_MODE)
    finally:
        # the real 3.0.0 file again, also when the steps above failed
        reinstalled, _ = await t.owui.install_function(FID, "Azure AI Foundry", source)
    await t.owui.models(refresh=True)
    kept = (await t.owui.get_valves(FID)).get(SEARCH_MODE)
    spec = (await t.owui.valves_spec(FID)).get("properties", {})
    mark = await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    await t.log.settle(0.5)
    chat, searches = await r.chat(), await r.searches()
    errors = t.log.errors(mark)
    r.check(
        "mode.stored",
        f"{SEARCH_MODE}=on_your_data stored by a 2.9.0 pre-release (the staged "
        "file with that valve added), then the 3.0.0 file saved again: no such "
        "valve in the spec, the stored value is ignored (grounded answer from "
        "the pipe's own search, no data_sources, no ERROR)",
        marker in source
        and installed == 200
        and stored == "on_your_data"
        and reinstalled == 200
        and kept == "on_your_data"
        and SEARCH_MODE not in spec
        and res.content == LINKED
        and _grounded(chat)
        and len(searches) == 1
        and not errors,
        f"{_answer(res.content)} valve added={marker in source} install HTTP "
        f"{installed} / {reinstalled} stored={stored!r} stored after the "
        f"update={kept!r} in spec={SEARCH_MODE in spec} {_chat_brief(chat)} "
        f"{_search_brief(searches)} log_errors={errors[:1]}",
    )

    # The environment: the staged file in this process (as rag.log.debug)
    failure, text, fields, attribute = "", "", [], True
    os.environ[SEARCH_MODE] = "on_your_data"
    await r.reset()
    try:
        module = _load_staged_pipe()
        fields = sorted(getattr(module.Pipe.Valves, "model_fields", {}) or {})
        pipe = module.Pipe()
        text = await asyncio.wait_for(
            _pipe_in_process(
                pipe,
                {**r.valves(), SEARCH_MODE: "on_your_data"},
                MODEL,
                [{"role": "user", "content": QUESTION}],
                0,
            ),
            120,
        )
        attribute = hasattr(pipe.valves, SEARCH_MODE)
    except Exception as exc:  # the check reports it
        failure = f"{type(exc).__name__}: {short(str(exc), 200)}"
    finally:
        os.environ.pop(SEARCH_MODE, None)
        logging.getLogger("azure_ai.pipe").setLevel(logging.NOTSET)
    chat, searches = await r.chat(), await r.searches()
    r.check(
        "mode.env",
        f"{SEARCH_MODE}=on_your_data in the environment (and passed as a "
        "valve): the staged file (run in the driver process) has no such valve "
        "and answers from its own search",
        not failure
        and bool(fields)
        and SEARCH_MODE not in fields
        and not attribute
        and text == LINKED
        and _grounded(chat)
        and len(searches) == 1,
        f"{failure or 'ran'} {_answer(text)} valve fields={len(fields)} "
        f"{SEARCH_MODE} in fields={SEARCH_MODE in fields} attribute={attribute} "
        f"{_chat_brief(chat)} {_search_brief(searches)}",
    )


# ------------------------------------------------------------- DEBUG log
# The server runs at INFO (logs.no-secrets reads its log). Open WebUI at DEBUG
# logs the stored valves itself (aiosqlite), so the pipe's own DEBUG output is
# checked here: the staged file runs in this driver process with every logger
# at DEBUG, against the same mocks.
class _LogCapture(logging.Handler):
    """Every log record of this process (message and traceback)."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        self.records: list = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except Exception as exc:  # the raw parts still count
            text = f"{record.msg!r} {record.args!r} ({exc!r})"
        self.records.append((record.name, record.levelno, text))


def _load_staged_pipe():
    """The staged pipe file as a module of this process.

    ``open_webui.env`` is a stub during the import (the real one sets the
    process up like a server); SRC_LOG_LEVELS OPENAI=DEBUG is what
    GLOBAL_LOG_LEVEL=DEBUG gives the pipe's logger in Open WebUI.
    """
    env = types.ModuleType("open_webui.env")
    env.AIOHTTP_CLIENT_TIMEOUT = 300
    env.SRC_LOG_LEVELS = {"OPENAI": "DEBUG"}
    names = ("open_webui", "open_webui.env")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.setdefault("open_webui", types.ModuleType("open_webui"))
    sys.modules["open_webui.env"] = env
    try:
        spec = importlib.util.spec_from_file_location(
            "e2e_azure_debug", os.path.join(FUNCTIONS_DIR, PATH)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


async def _pipe_in_process(
    pipe,
    valves: dict,
    model: str,
    messages,
    n: int,
    extra: Optional[dict] = None,
    prompts: Optional[dict] = None,
) -> str:
    """One pipe() call in this process: the answer text or the SSE stream
    (``extra``: more keys of the request body; ``prompts``: the prompt keys of
    ``__metadata__``, by default ``user_prompt`` with the last message's text)."""
    pipe.valves = pipe.Valves(**valves)
    stream = isinstance(messages, tuple)  # a tuple: stream=True

    async def emit(event):
        pass

    if prompts is None:
        prompts = {"user_prompt": _text(messages[-1]["content"])}
    result = await pipe.pipe(
        {"model": model, "stream": stream, "messages": list(messages), **(extra or {})},
        __event_emitter__=emit,
        __metadata__={
            "user_id": "debug-user",
            "chat_id": "debug-chat",
            "message_id": f"debug-{n}",
            **prompts,
        },
    )
    if hasattr(result, "body_iterator"):
        parts = []
        async for chunk in result.body_iterator:
            parts.append(
                chunk.decode("utf-8", "replace")
                if isinstance(chunk, bytes)
                else str(chunk)
            )
        return "".join(parts)
    if isinstance(result, dict):
        choice = (result.get("choices") or [{}])[0]
        return str((choice.get("message") or {}).get("content") or "")
    return str(result or "")


async def rag_debug_log(r: Rag) -> None:
    """The pipe at DEBUG logs no key or token."""
    secrets = {
        KEY: "AZURE_AI_API_KEY",
        RAG_SEARCH_KEY: "AZURE_AI_SEARCH_KEY",
        RAG_JSON_KEY: "key in AZURE_AI_DATA_SOURCES",
        RAG_SEARCH_TOKEN: "search token",
        RAG_EMBED_KEY: "embedding key",
        RAG_EMBED_TOKEN: "embedding token",
    }
    embed = f"{r.mock.url}/openai/deployments/{EMBED_DEPLOYMENT}/embeddings"

    def hybrid(auth: dict) -> dict:
        dependency = {"type": "endpoint", "endpoint": embed, "authentication": auth}
        return {
            "query_type": "vector_simple_hybrid",
            "embedding_dependency": dependency,
        }

    def mi(resource_id: str) -> dict:
        return {
            "type": "user_assigned_managed_identity",
            "managed_identity_resource_id": resource_id,
        }

    question = [{"role": "user", "content": QUESTION}]
    json_key = {"type": "api_key", "key": RAG_JSON_KEY}
    token = {"type": "access_token", "access_token": RAG_SEARCH_TOKEN}
    client = [
        {
            "type": "azure_search",
            "parameters": {
                "endpoint": r.search.url,
                "index_name": "client-index",
                "authentication": json_key,
            },
        }
    ]
    # (label, valves, model, messages (a tuple streams), error expected[,
    # more body keys])
    cases = (
        (
            "valve key over the JSON key, stream",
            r.valves(r.ds(auth=json_key), AZURE_AI_SEARCH_KEY=RAG_SEARCH_KEY),
            MODEL,
            tuple(question),
            False,
        ),
        (
            "valve key, follow-up with query generation",
            r.valves(r.ds(auth=None), AZURE_AI_SEARCH_KEY=RAG_SEARCH_KEY),
            MODEL,
            _followup(AGAIN),
            False,
        ),
        (
            "search token, embedding key",
            r.valves(
                r.ds(auth=token, **hybrid({"type": "api_key", "key": RAG_EMBED_KEY}))
            ),
            MODEL,
            question,
            False,
        ),
        (
            "embedding token, chat key as Bearer, stream",
            r.valves(
                r.ds(
                    **hybrid({"type": "access_token", "access_token": RAG_EMBED_TOKEN})
                ),
                USE_AUTHORIZATION_HEADER=True,
            ),
            MODEL,
            tuple(question),
            False,
        ),
        (
            "system-assigned managed identity",
            r.valves(r.ds(auth={"type": "system_assigned_managed_identity"})),
            MODEL,
            question,
            False,
        ),
        (
            "user-assigned managed identity",
            r.valves(r.ds(auth=mi(MI_RESOURCE_ID))),
            MODEL,
            question,
            False,
        ),
        (
            "managed identity token error",
            r.valves(r.ds(auth=mi(MI_FAIL_ID))),
            MODEL,
            question,
            True,
        ),
        (
            "search HTTP 403",
            r.valves(r.ds(auth=None), AZURE_AI_SEARCH_KEY=RAG_SEARCH_KEY),
            MODEL,
            [{"role": "user", "content": "x100 search-403"}],
            True,
        ),
        (
            "client data_sources with a key (refused)",
            r.valves(),
            MODEL,
            question,
            True,
            {"data_sources": client},
        ),
        (
            "client data_sources with a key, stream (refused)",
            r.valves(),
            MODEL,
            tuple(question),
            True,
            {"data_sources": client},
        ),
    )
    capture = _LogCapture()
    root = logging.getLogger()
    level = root.level
    root.addHandler(capture)
    root.setLevel(logging.DEBUG)
    results, failure = [], ""
    await r.reset()
    try:
        module = _load_staged_pipe()
        pipe = module.Pipe()
        for n, (label, valves, model, messages, error, *extra) in enumerate(cases):
            text = await asyncio.wait_for(
                _pipe_in_process(
                    pipe, valves, model, messages, n, extra[0] if extra else None
                ),
                120,
            )
            results.append((label, bool(text) and text.startswith("Error:") == error))
    except Exception as exc:  # the check reports it
        failure = f"{type(exc).__name__}: {short(str(exc), 200)}"
    finally:
        root.removeHandler(capture)
        root.setLevel(level)
        logging.getLogger("azure_ai.pipe").setLevel(logging.NOTSET)
    embeds, tokens, chats = await r.embeds(), await r.tokens(), await r.chats()
    with_ds = [c for c in chats if "data_sources" in (c.get("body") or {})]
    debug = sum(
        1
        for name, levelno, _ in capture.records
        if name.startswith("azure_ai") and levelno == logging.DEBUG
    )
    leaks = []
    for value, label in secrets.items():
        hits = [text for _, _, text in capture.records if value in text]
        if hits:
            line = hits[0]
            for other in secrets:
                line = line.replace(other, "***")
            leaks.append(f"{label} in {len(hits)} record(s): {short(line, 160)}")
    r.check(
        "log.debug",
        "the pipe with every logger at DEBUG (run in the test driver; the "
        "server logs at INFO): the API key, the search key of the valve and the "
        "JSON, the search and embedding tokens and the embedding key are in no "
        "log record (keys, tokens, managed identity, query generation, errors, "
        "client data_sources with a key, refused also when streamed)",
        not failure
        and len(results) == len(cases)
        and all(ok for _, ok in results)
        and len(embeds) == 2
        and len(tokens) >= 3
        and not with_ds
        and debug > 0
        and not leaks,
        f"{failure or 'ran'} cases_ok={[label for label, ok in results if ok]} "
        f"failed={[label for label, ok in results if not ok]} "
        f"embeds={len(embeds)} tokens={len(tokens)} "
        f"chats with data_sources={len(with_ds)} "
        f"records={len(capture.records)} azure_ai_debug={debug} leaks={leaks}",
    )


# ------------------------------------------------------ retirement notice
def rag_notice_none(r: Rag) -> None:
    """3.0.0 has no On Your Data path, so the 2.8.1 retirement notice (a
    WARNING once per loaded copy of the module) must never come back: no line
    of the pipe in the server log of the whole suite run names On Your Data
    together with its retirement date."""
    t = r.t
    notices = [
        line.strip()
        for line in t.log.since(t.log_start).splitlines()
        if "function_azure" in line and all(part in line for part in OYD_NOTICE)
    ]
    r.check(
        "notice.none",
        "no On Your Data retirement notice of 2.8.1 in the server log (no line "
        "of the pipe with 'On Your Data' and 'October 14, 2026', at any level), "
        "although every kind of search request ran",
        not notices,
        f"notices={len(notices)} first={short(notices[:1], 240)}",
    )
