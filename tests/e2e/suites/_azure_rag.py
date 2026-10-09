"""
Group ``rag`` of the azure suite (suites/azure.py): the pipeline-side Azure AI
Search retrieval of pipelines/azure/azure_ai_foundry.py 2.9.0 (#187,
``AZURE_AI_SEARCH_MODE=pipeline``) against mocks/mock_search.py (Azure AI
Search, managed identity tokens) and mocks/mock_azure.py (chat, query
generation, embeddings, emulated On Your Data retirement).

The rag valves use the deployment gpt-5-mini, for which mock_azure rejects
``data_sources`` ("On Your Data is retired (mock)"). Files older than 2.9.0
send ``data_sources``, so every behaviour check fails with that answer and is
KNOWN ``azure-oyd-retired``; from 2.9.0 on the marker is off and every check
must pass (a failure is a regression, a pass an obsolete marker). Checks that
would not fail with that evidence on older files (no ``data_sources`` sent,
``on_your_data`` mode on purpose) only run from 2.9.0 on and are not tagged.

Order: all pipeline-mode checks, then ``rag.notice.none`` (pipeline mode never
logs the On Your Data notice), then the ``on_your_data`` sub-checks, which log
it legitimately (the notice is logged once per loaded copy of the module).

Loaded by suites/azure.py when the group runs (modules starting with ``_`` are
not suites).
"""

import asyncio
import json
import re
import time
from typing import Optional

from harness import Suite, known, short
from harness.known import staged_version, version_tuple
from suites.azure import (
    ALL_SOURCES,
    FID,
    LINKED,
    NO_REFS,
    PAREN_LINKED,
    PATH,
    RAG_EMBED_KEY,
    RAG_JSON_KEY,
    RAG_SEARCH_KEY,
    RAG_SEARCH_TOKEN,
    REFERENCED_SOURCES,
    SEARCH_SINCE,
    SHOW_ALL,
    SPLIT_LINK,
    TASKS,
    TOOLS,
    UNLINKED,
    _answered,
    _is_task,
    _last_status,
    _oyd_notices,
    _status_sequence,
    _statuses,
    _wait_tasks,
)

KNOWN = known.AZURE_OYD_RETIRED
DEPLOYMENT = "gpt-5-mini"
MODEL = f"{FID}.{DEPLOYMENT}"
PAUSE_DEPLOYMENT = "gpt-5-pause"  # query generation pause (per model)
OYD_DEPLOYMENT = "gpt-4.1"  # still accepted with data_sources by the mock
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
    "Error: Azure AI Search: data_sources in the request is not supported in "
    "pipeline mode"
)
RETIRED_HINT = (
    "(Azure OpenAI On Your Data was retired on 2026-10-14; set "
    "AZURE_AI_SEARCH_MODE=pipeline)"
)
CONTEXT_HINT = (
    "lower AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS or top_n_documents, or start a new chat"
)
# Server-log signatures of provoked errors.
SEARCH_ERROR = ("function_azure:pipe", "Azure AI Search")
CHAT_ERROR = ("function_azure:pipe", "Error in Azure AI request")
OLD_FOUNDRY_ERROR = (
    "function_azure:pipe",
    "Error in Azure AI request: 400",
    "/models/chat/completions",
)
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
    """The answer first (the KNOWN evidence of older files is in it)."""
    return f"answer={short(content or '', 120)}"


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

    @property
    def fixed(self) -> bool:
        """The staged file has the pipeline mode (2.9.0+)."""
        return version_tuple(staged_version(PATH)) >= version_tuple(SEARCH_SINCE)

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
            "AZURE_AI_SEARCH_MODE": "pipeline",
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

    def check(self, sid: str, title: str, ok, detail: str, tagged: bool = True):
        return self.t.check(
            f"rag.{sid}", title, ok, detail, known=KNOWN if tagged else None
        )

    async def settle_errors(self, mark: int, *signatures) -> None:
        """Errors the scenario provoked on purpose (after a short settle)."""
        await self.t.log.settle(0.5)
        self.t.expect_errors(mark, *signatures)


# -------------------------------------------------------------------- group
async def rag(t: Suite, mock, base_valves: dict) -> None:
    r = Rag(t, mock, t.mock("search"), base_valves)
    notice_before = bool(_oyd_notices(t, t.log_start))
    group_mark = t.mark()
    await rag_api(r)
    await rag_browser(r)
    await rag_tool_round(r)
    await rag_links(r)
    await rag_query_text(r)
    await rag_no_refs(r)
    await rag_scores(r)
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
    await rag_client_data_sources(r)
    await rag_qgen(r)
    await rag_auth(r)
    await rag_mode_unknown(r)
    await rag_context_length(r)
    await rag_stop(r)
    await rag_notice_none(r, notice_before, group_mark)
    # on_your_data on purpose: these log the On Your Data notice
    await rag_oyd_mode(r)
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
    r.check(
        "api.nonstream",
        "API non-stream (pipeline mode): [docX] links; chat request without "
        "data_sources, documents [doc1]-[doc3] in a <documents> block (title / "
        "file, no URL) before the user text, rules in the one system message; "
        "one search (path, api-version 2026-04-01, api-key, body {search, "
        "queryType simple, top 10}); message.context citations and intent",
        res.status == 200
        and res.content == LINKED
        and "stream_options" not in body
        and prompt_ok
        and search_ok
        and _titles(citations) == TITLES
        and _intent(context) == [QUESTION],
        f"{_answer(res.content)} HTTP {res.status} {_chat_brief(chat)} "
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
        "API stream (pipeline mode): [docX] links and [DONE]; a first event with "
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
    mark = await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=True)
    if not r.fixed:
        await r.settle_errors(mark, OLD_FOUNDRY_ERROR)
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
            f"retrieval (pipeline mode; stream={stream})",
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
            f"API stream '{text}' (pipeline mode): references linked once, "
            "[DONE] forwarded",
            res.status == 200
            and res.content == expected
            and res.done
            and _grounded(chat),
            f"{_answer(res.content)} done={res.done} {_chat_brief(chat)}",
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


async def rag_no_refs(r: Rag) -> None:
    t = r.t
    async with t.browser() as b:
        for show_all, expected, sid in (
            (True, ALL_SOURCES, "no-refs.default"),
            (False, [], "no-refs.valve-false"),
        ):
            await r.set(**{SHOW_ALL: show_all})
            await r.reset()
            c = await b.chat(MODEL, QUESTION + " no-refs", stream=True)
            chat = await r.chat()
            r.check(
                sid,
                f"answer without [docX], show-all valve {show_all} -> "
                f"{len(expected)} sources (pipeline mode)",
                c.done
                and c.content == NO_REFS
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
        "API citations carry original_search_score and relevance, no "
        "filter_reason (pipeline mode)",
        res.content == LINKED
        and len(citations) == 3
        and all("relevance" in c and "filter_reason" not in c for c in citations)
        and first.get("original_search_score") == 42.5
        and abs(float(first.get("relevance") or 0) - 0.425) < 0.005,
        f"{_answer(res.content)} first citation keys={sorted(first)} "
        f"relevance={first.get('relevance')} "
        f"score={first.get('original_search_score')}",
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
            None,  # HTTP 206 without reranker scores still answers
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
        if expected is None:
            relevance = (citations[0] if citations else {}).get("relevance")
            ok = (
                s.get("status") == 206
                and len(citations) == 3
                and not any("rerank_score" in c for c in citations)
                and abs(float(relevance or 0) - 0.425) < 0.005
            )
            title = (
                "semantic partial result (HTTP 206, no reranker scores): linked "
                "answer, citations without rerank_score, BM25 relevance (0.425)"
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
            None,
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
            None,
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
            OLD_FOUNDRY_ERROR,
        ),
        (
            "vector.deployment.bearer",
            "deployment_name with USE_AUTHORIZATION_HEADER: embeddings with the "
            "chat call's Bearer header",
            {"USE_AUTHORIZATION_HEADER": True},
            "/openai/v1/embeddings",
            "bearer",
            None,
        ),
    )
    for sid, title, changes, path, auth, old_error in cases:
        await r.set(r.ds(**hybrid), **changes)
        mark = await r.reset()
        res = await t.owui.chat(MODEL, QUESTION, stream=False)
        if old_error and not r.fixed:
            await r.settle_errors(mark, old_error)
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
            and e.get("body") == body
            and vq.get("kind") == "vector"
            and vq.get("vector_len") == 8,
            f"{_answer(res.content)} embeddings={short([{k: x.get(k) for k in ('path', 'api_version', 'auth_mode', 'body')} for x in embeds], 260)} "
            f"vq={vq}",
        )

    # another host without authentication: the chat key must not go there
    other = endpoint.replace("127.0.0.1", "localhost")
    cases = (
        (
            "vector.endpoint.other-host",
            "endpoint dependency without authentication on another host: "
            "configuration error, nothing sent",
            {"type": "endpoint", "endpoint": other},
            "",
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
        ),
        (
            "budget.valve",
            "AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS=1000 (4,000 chars)",
            1000,
            4000,
            [1, 2, 3],
        ),
        (
            "budget.unlimited",
            "AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS=0 (no limit)",
            0,
            None,
            [1, 2, 3],
        ),
        (
            "budget.drop",
            "AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS=300 (1,200 chars: under 500 per "
            "document the lowest-ranked one is dropped)",
            300,
            1200,
            [1, 2],
        ),
    )
    for sid, label, tokens, chars, expected_docs in cases:
        await r.set(AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS=tokens)
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
    r.check(
        "sanitize",
        "document text that closes the block or fakes labels is sanitized: one "
        "<documents> and one </documents> tag, [doc9] shown as [ doc9] in the "
        "block and the citation",
        res.content == LINKED
        and tags == ["<documents>", "</documents>"]
        and "[ doc9]" in block
        and "[doc9]" not in lowered
        and "[doc7]" not in lowered
        and "[ doc9]" in card
        and not re.search(r"<\s*/?\s*documents", card, re.IGNORECASE)
        and chat.get("prompt_docs") == [1, 2, 3],
        f"{_answer(res.content)} tags={tags} docs={chat.get('prompt_docs')} "
        f"card={short(card, 160)}",
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
    tagged: bool = True,
    log_absent: tuple = (),
) -> tuple:
    """One request that must end with 'Error: Azure AI Search: ...' and no
    chat request (the valves are set by the caller); the provoked error is
    logged without traceback, ``log_absent`` nowhere in the log."""
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
    )
    r.check(
        sid,
        title,
        ok,
        f"{_answer(res.content)} chats={len(chats)} {_search_brief(searches)} "
        f"elapsed={elapsed:.1f}s traceback={traceback} logged={len(logged)}",
        tagged=tagged,
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
            "config-error.elasticsearch",
            "data source type elasticsearch: error pointing to "
            "AZURE_AI_SEARCH_MODE=on_your_data",
            {"type": "elasticsearch", "parameters": {"endpoint": search_url}},
            "AZURE_AI_SEARCH_MODE=on_your_data",
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
    if r.fixed:  # older files ignore invalid JSON silently (plain chat)
        cases.insert(
            0,
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
        )
    for sid, title, ds, needle, changes in cases:
        await r.set(ds, **changes)
        await _error_case(
            r,
            sid,
            title,
            needles=(needle,),
            absent=(RAG_SEARCH_KEY, '"azure_search"'),
            searches_expected=0,
            tagged=sid != "config-error.invalid-json",
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
    before (gated: passes on older files too)."""
    if not r.fixed:
        return
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
        tagged=False,
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
    r.check(
        "tasks.api",
        "title task with the rag valves: answered without retrieval (the chat "
        "answer before it: grounded, one search)",
        res.content == LINKED
        and status == 200
        and "Mock Title" in str(answer)
        and len(searches) == 1
        and tasks
        and not grounded_tasks,
        f"{_answer(res.content)} task HTTP {status} answer={short(answer, 60)} "
        f"searches={len(searches)} task requests={len(tasks)} "
        f"grounded tasks={len(grounded_tasks)}",
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
    r.check(
        "tasks.browser",
        "browser chat with background tasks: one search (the answer), task "
        "prompts without <documents>, only the referenced sources and the "
        "answer's statuses",
        c.done
        and c.content == LINKED
        and c.source_names == REFERENCED_SOURCES
        and _status_sequence(c.status_history, STATUS_RAG_STREAM)
        and len(searches) == 1
        and len(tasks) >= len(TASKS)
        and not grounded_tasks,
        f"{c.brief()} searches={len(searches)} task requests={len(tasks)} "
        f"grounded tasks={len(grounded_tasks)} {waited}",
    )


async def rag_client_data_sources(r: Rag) -> None:
    """data_sources sent by the client are refused in pipeline mode (never
    fetched, never forwarded), with or without the valve."""
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
        mark = await r.reset()
        res = await t.owui.chat(MODEL, QUESTION, stream=False, data_sources=client)
        await r.settle_errors(mark, SEARCH_ERROR)
        chats, recorded = await r.chats(), await r.search.requests()
        r.check(
            sid,
            "client data_sources in pipeline mode "
            f"({'valve set' if ds is None else 'no valve'}): 'data_sources in the "
            "request is not supported in pipeline mode', nothing searched, no "
            "chat request",
            res.content.startswith(CLIENT_DS_ERROR)
            and "AZURE_AI_SEARCH_MODE=on_your_data" in res.content
            and not recorded
            and not chats,
            f"{_answer(res.content)} search mock requests={len(recorded)} "
            f"chats={len(chats)}",
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
    mark = await r.reset()
    res = await t.owui.chat(MODEL, _followup("and the warranty?"), stream=False)
    if not r.fixed:
        await r.settle_errors(mark, OLD_FOUNDRY_ERROR)
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
        "generation answer with a <think> block holding braces: the JSON after "
        "it is used (one search 'x100 charging')",
        res.content == LINKED
        and [s.get("search") for s in searches] == ["x100 charging"],
        f"{_answer(res.content)} {_search_brief(searches)}",
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

    # Timeouts (10 s) on a separate model: after 3 in a row the generation is
    # paused for that model (15 minutes), so the other checks are not affected.
    await r.set(AZURE_AI_MODEL=f"{DEPLOYMENT};{PAUSE_DEPLOYMENT}")
    model = f"{FID}.{PAUSE_DEPLOYMENT}"
    pause_mark = t.mark()
    rounds = []
    for i in range(4):
        latest = f"{AGAIN} qgen-slow {i}"
        await r.reset()
        started = time.monotonic()
        res = await t.owui.chat(model, _followup(latest), stream=False)
        elapsed = time.monotonic() - started
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
    # (IDENTITY_ENDPOINT -> mock_search /msi/token).
    for sid, auth, ids in (
        (
            "auth.mi.system",
            {"type": "system_assigned_managed_identity"},
            [],
        ),
        (
            "auth.mi.user",
            {
                "type": "user_assigned_managed_identity",
                "managed_identity_resource_id": MI_RESOURCE_ID,
            },
            [MI_RESOURCE_ID],
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
            "search.azure.com) as Bearer; the second request uses the cached token",
            res.content == LINKED
            and res2.content == LINKED
            and len(tokens) == 1
            and str(token.get("resource", "")).rstrip("/") == SEARCH_RESOURCE
            and sorted((token.get("identity_ids") or {}).values()) == ids
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


async def rag_mode_unknown(r: Rag) -> None:
    t = r.t
    mark = t.mark()
    await r.set(AZURE_AI_SEARCH_MODE="bogus")
    await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    res2 = await t.owui.chat(MODEL, QUESTION, stream=False)
    await t.log.settle(0.5)
    chat, searches = await r.chat(), await r.searches()
    warned = _warnings(t, mark, "AZURE_AI_SEARCH_MODE")
    r.check(
        "mode.unknown",
        "AZURE_AI_SEARCH_MODE=bogus: pipeline mode, one warning per process",
        res.content == LINKED
        and res2.content == LINKED
        and _grounded(chat)
        and len(searches) == 2
        and len(warned) == 1,
        f"{_answer(res.content)} searches={len(searches)} warnings={len(warned)} "
        f"{_chat_brief(chat)}",
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
        "Azure's message plus the budget / new chat hint",
        res.content.startswith("Error:")
        and "maximum context length" in res.content
        and CONTEXT_HINT in res.content
        and _grounded(chat),
        f"{_answer(res.content)} {_chat_brief(chat)}",
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


async def rag_notice_none(r: Rag, notice_before: bool, group_mark: int) -> None:
    """Pipeline mode never logs the On Your Data notice. Provable only when
    no notice was logged before the group (oyd not selected): the notice is
    logged once per loaded copy of the module."""
    if not r.fixed or notice_before:
        return
    notices = _oyd_notices(r.t, group_mark)
    r.check(
        "notice.none",
        "pipeline mode never logs the On Your Data retirement notice",
        not notices,
        f"notices={len(notices)} first={short(notices[:1], 200)}",
        tagged=False,
    )


async def rag_oyd_mode(r: Rag) -> None:
    """on_your_data sub-checks (2.9.0+, untagged): key valve injection, the
    mode alias and the retirement hint."""
    if not r.fixed:
        return
    t = r.t
    oyd = {
        "AZURE_AI_SEARCH_MODE": "on_your_data",
        "AZURE_AI_ENDPOINT": f"{r.mock.url}/openai/deployments/{OYD_DEPLOYMENT}"
        "/chat/completions?api-version=2025-01-01-preview",
        "AZURE_AI_MODEL": OYD_DEPLOYMENT,
    }
    model = f"{FID}.{OYD_DEPLOYMENT}"
    await r.set(r.ds(auth=None), AZURE_AI_SEARCH_KEY=RAG_SEARCH_KEY, **oyd)
    await r.reset()
    res = await t.owui.chat(model, QUESTION, stream=False)
    chat, searches = await r.chat(), await r.searches()
    sources = (chat.get("body") or {}).get("data_sources") or [{}]
    auth = (sources[0].get("parameters") or {}).get("authentication")
    stored = (await t.owui.get_valves(FID)).get("AZURE_AI_DATA_SOURCES", "")
    r.check(
        "auth.key-valve.oyd",
        "on_your_data mode: AZURE_AI_SEARCH_KEY goes into (a copy of) the "
        "data source sent to Azure; the stored JSON keeps no key",
        res.content == LINKED
        and auth == {"type": "api_key", "key": RAG_SEARCH_KEY}
        and RAG_SEARCH_KEY not in str(stored)
        and not searches,
        f"{_answer(res.content)} auth={'key' if auth else auth} "
        f"stored_has_key={RAG_SEARCH_KEY in str(stored)} searches={len(searches)}",
        tagged=False,
    )

    await r.set(**{**oyd, "AZURE_AI_SEARCH_MODE": "On-Your-Data"})
    await r.reset()
    res = await t.owui.chat(model, QUESTION, stream=False)
    chat, searches = await r.chat(), await r.searches()
    r.check(
        "mode.alias",
        "AZURE_AI_SEARCH_MODE=On-Your-Data (case, '-'): the legacy mode "
        "(data_sources sent, no search by the pipe)",
        res.content == LINKED
        and bool((chat.get("body") or {}).get("data_sources"))
        and not searches,
        f"{_answer(res.content)} data_sources={bool((chat.get('body') or {}).get('data_sources'))} "
        f"searches={len(searches)}",
        tagged=False,
    )

    await r.set('{"type": "azure_search", "parameters": {', **oyd)
    mark = await r.reset()
    res = await t.owui.chat(model, QUESTION, stream=False)
    await r.settle_errors(
        mark, ("function_azure", "Error parsing AZURE_AI_DATA_SOURCES")
    )
    chat, searches = await r.chat(), await r.searches()
    r.check(
        "oyd.invalid-json",
        "on_your_data mode: AZURE_AI_DATA_SOURCES that is not valid JSON stays "
        "a plain chat (as 2.8.x), no 'Error: Azure AI Search'",
        res.content == f"Hello from mock Azure ({OYD_DEPLOYMENT})."
        and not (chat.get("body") or {}).get("data_sources")
        and not searches,
        f"{_answer(res.content)} {_chat_brief(chat)} searches={len(searches)}",
        tagged=False,
    )

    await r.set(AZURE_AI_SEARCH_MODE="on_your_data")
    mark = await r.reset()
    res = await t.owui.chat(MODEL, QUESTION, stream=False)
    await r.settle_errors(mark, CHAT_ERROR)
    await r.set(AZURE_AI_SEARCH_MODE="on_your_data", AZURE_AI_API_KEY="wrong-key-e2e")
    mark = await r.reset()
    res401 = await t.owui.chat(MODEL, QUESTION, stream=False)
    await r.settle_errors(mark, CHAT_ERROR)
    r.check(
        "oyd-retired-hint",
        "on_your_data mode, Azure answers 400 to data_sources: the retirement "
        "hint is appended; a 401 (wrong chat key) gets no hint",
        res.content.startswith("Error:")
        and "On Your Data is retired (mock)" in res.content
        and RETIRED_HINT in res.content
        and res401.content.startswith("Error:")
        and "Access denied" in res401.content
        and RETIRED_HINT not in res401.content,
        f"400: {short(res.content, 200)} 401: {short(res401.content, 160)}",
        tagged=False,
    )
