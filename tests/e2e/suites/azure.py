"""
Azure suite: pipelines/azure/azure_ai_foundry.py against mocks/mock_azure.py.

Groups (``--only azure.<group>``)
  valves   valve names are kept (public API), new valves have their defaults
  models   AZURE_AI_MODEL lists (; , space separated) with exact names, custom
           AZURE_AI_PIPELINE_PREFIX, model from an *.openai.azure.com URL,
           predefined models, fallback model
  api      API path non-stream / stream, api-key / Bearer header, path and
           api-version, allow-listed body (extra client keys dropped), tools
           forwarded, stream_options only for streams, errors (JSON 400 with
           exactly Azure's message, text/plain 500)
  dotted   model names containing dots reach upstream intact
  browser  browser path: saved answer, usage, full status sequence, error status
  tasks    background title task (with Azure AI Search valves: rag.tasks.*)
  rag      Azure AI Search retrieval of the pipe (3.0.0, #187; On Your Data
           and its data_sources are gone) against mocks/mock_search.py:
           request bodies per query_type, embeddings, auth (key, key valve,
           token, managed identity), query generation (fallback, pause),
           strictness, merge, top-n, budget, sanitizing, prompt placement,
           citations, scores, [docX] links (also split across stream deltas,
           already linked, URL with parentheses; API and browser; nothing
           after [DONE]), history unlinking (also links saved before 2.8.0, a
           hostile history line), only referenced sources (show-all valve,
           references to documents that do not exist), large and too large
           stream events, content null, tool rounds, tasks (also without a
           websocket session), client data_sources (API stream and
           non-stream, browser: an error naming the removal, ERROR line and
           terminal status), the removed AZURE_AI_SEARCH_MODE valve, no On
           Your Data retirement notice, fail-closed errors, Stop; see
           suites/_azure_rag.py
  logs     no API key, no search key or token and no citation or search text
           in the server log
"""

import asyncio
import json
import time

from harness import Suite, short
from harness.known import staged_version, version_tuple

# "rag" runs before "logs" (which scans what rag logged).
GROUPS = (
    "valves",
    "models",
    "api",
    "dotted",
    "browser",
    "tasks",
    "rag",
    "logs",
)
FID = "azure"
PATH = "pipelines/azure/azure_ai_foundry.py"
KEY = "mock-key-123"
MODELS = ("gpt-4o", "gpt-4.1", "Phi-3.5-mini-instruct")
PREFIX = "Azure AI"
# Valves of the 2.7.0 release: valve names are public API and must stay.
MAIN_VALVES = (
    "AZURE_AI_PIPELINE_PREFIX",
    "AZURE_AI_API_KEY",
    "AZURE_AI_ENDPOINT",
    "AZURE_AI_MODEL",
    "AZURE_AI_MODEL_IN_BODY",
    "USE_PREDEFINED_AZURE_AI_MODELS",
    "USE_AUTHORIZATION_HEADER",
    "AZURE_AI_DATA_SOURCES",
    "AZURE_AI_INCLUDE_SEARCH_SCORES",
    "BM25_SCORE_MAX",
    "RERANK_SCORE_MAX",
)
SHOW_ALL = "AZURE_AI_SHOW_ALL_CITATIONS_WITHOUT_REFERENCES"
SHOW_ALL_SINCE = "2.8.0"  # first version with the SHOW_ALL valve
# Azure AI Search retrieval of the pipe (#187): first version and its valves
# with their defaults (None: no default checked).
SEARCH_SINCE = "3.0.0"
SEARCH_VALVES = {
    "AZURE_AI_SEARCH_KEY": None,  # encrypted, password input
    "AZURE_AI_SEARCH_API_VERSION": "2026-04-01",
    "AZURE_AI_SEARCH_QUERY_GENERATION": "auto",
    "AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS": -1,
}
SEARCH_ENUMS = {
    "AZURE_AI_SEARCH_QUERY_GENERATION": {"auto", "always", "off"},
}
# The mode valve of the unreleased 2.9.0 (pipeline / on_your_data): never
# released, removed with On Your Data in 3.0.0.
SEARCH_MODE = "AZURE_AI_SEARCH_MODE"
# Secrets of the rag group (mock values); never in the server log.
RAG_SEARCH_KEY = "mock-search-key-456"
RAG_SEARCH_TOKEN = "mock-search-token-789"
RAG_EMBED_KEY = "mock-embed-key-321"
RAG_EMBED_TOKEN = "mock-embed-token-e2e"
RAG_JSON_KEY = "json-plaintext-key-000"
# Texts the rag group sends to the Search mock or gets back (generated
# queries, document titles and URLs): not at INFO in the server log.
SEARCH_TEXTS = ("x100 warranty", "Release Notes", "docs.example.com/x100")
# Client keys Open WebUI passes through to the pipe but the allow-list must drop.
NOT_ALLOWED = {"user": "u-e2e", "logit_bias": {"50256": -100}, "foo_not_allowed": "x"}
ALLOWED = {
    "model",
    "messages",
    "deployment",
    "frequency_penalty",
    "max_tokens",
    "max_citations",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "stop",
    "stream",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
    "stream_options",
}
USAGE_OPTIONS = {"include_usage": True}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Current time",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
MANUAL_URL = "https://docs.example.com/x100/manual.pdf"
LINKED = (
    f"The X100 charges via USB-C [[doc1]]({MANUAL_URL}). "
    "It has a two-year warranty [[doc2]](faq/warranty.html)."
)
UNLINKED = "The X100 charges via USB-C [doc1]. It has a two-year warranty [doc2]."
# mock_search trigger "paren-url": doc1's URL is .../manual_(v2).pdf
PAREN_LINKED = LINKED.replace(
    MANUAL_URL, "https://docs.example.com/x100/manual_%28v2%29.pdf"
)
SPLIT_LINK = f"See [[doc1]]({MANUAL_URL}) and [[doc2]](faq/warranty.html)."
NO_REFS = "The requested information is not available in the retrieved data."
REFERENCED_SOURCES = ["[doc1] - X100 Product Manual", "[doc2] - Warranty FAQ"]
ALL_SOURCES = REFERENCED_SOURCES + ["[doc3] - Release Notes"]
# text of doc1 (mock); must not appear in the server log at INFO
CITATION_TEXT = "The X100 charges via USB-C at up to 65 W."
FILLER = "bigdoc bigdoc bigdoc"  # filler text of the big documents and events
TASKS = {
    "title_generation": True,
    "tags_generation": True,
    "follow_up_generation": True,
}
STATUS_STREAM = [
    "Sending request to Azure AI...",
    "Streaming response from Azure AI...",
    "Streaming completed",
]
STATUS_NONSTREAM = ["Sending request to Azure AI...", "Request completed"]
LINE_TOO_LONG = "Got more than 131072 bytes"
STREAM_ERROR = ("function_azure:stream_processor_with_citations", "Error processing")
# The whole answer to the mock's force-400 (content filter): Azure's message
# and nothing appended (e.g. no On Your Data hint as in the 2.9.0 pre-releases)
FILTERED_ERROR = (
    "Error: The response was filtered due to the prompt triggering Azure OpenAI's "
    "content management policy."
)


# First line of the task in Open WebUI's RAG template (a file attached to a
# chat message): it starts with "### Task:" too, but is no background task.
RAG_TEMPLATE_TASK = "Respond to the user query using the provided context"


def _is_task(entry: dict) -> bool:
    """A background-task request: a "### Task:" that is not the RAG template."""
    text = json.dumps((entry.get("body") or {}).get("messages"))
    return any(
        not part.removeprefix("\\n").lstrip().startswith(RAG_TEMPLATE_TASK)
        for part in text.split("### Task:")[1:]
    )


def _statuses(history: list) -> list:
    """(description, done, hidden) of every saved status."""
    return [
        (s.get("description"), s.get("done"), bool(s.get("hidden"))) for s in history
    ]


def _status_sequence(history: list, expected: list) -> bool:
    """Exactly the ``expected`` descriptions, only the last one done, none
    hidden."""
    want = [(d, i == len(expected) - 1, False) for i, d in enumerate(expected)]
    return _statuses(history) == want


def _last_status(history: list) -> dict:
    return history[-1] if history else {}


def _answered(content: str) -> bool:
    """An answer came back (not an error text)."""
    return bool(content) and not content.startswith("Error")


def _allow_listed(body: dict) -> bool:
    """Only allow-listed keys went upstream, the extra client keys were dropped
    and an allow-listed optional parameter (temperature) was kept."""
    return (
        set(body) <= ALLOWED
        and not set(NOT_ALLOWED) & set(body)
        and body.get("temperature") == 0.3
    )


async def _models(t: Suite) -> dict:
    """``{id: name}`` of the azure.* models."""
    return {
        m["id"]: m.get("name")
        for m in await t.owui.models()
        if m["id"].startswith(FID + ".")
    }


async def run(t: Suite) -> None:
    mock = t.mock("azure")
    if not await t.install(FID, PATH, "Azure AI Foundry"):
        return
    base_valves = {
        "AZURE_AI_PIPELINE_PREFIX": PREFIX,
        "AZURE_AI_ENDPOINT": f"{mock.url}/models/chat/completions"
        "?api-version=2024-05-01-preview",
        "AZURE_AI_API_KEY": KEY,
        "AZURE_AI_MODEL": "gpt-4o;gpt-4.1, Phi-3.5-mini-instruct",
        "AZURE_AI_MODEL_IN_BODY": False,
        "USE_PREDEFINED_AZURE_AI_MODELS": False,
        "AZURE_AI_DATA_SOURCES": "",
        "USE_AUTHORIZATION_HEADER": False,
    }
    await t.owui.update_valves(FID, **base_valves)
    valves = await t.owui.get_valves(FID)
    spec = (await t.owui.valves_spec(FID)).get("properties", {})
    stored = valves.get("AZURE_AI_API_KEY", "")
    t.check(
        "valves.encrypted",
        "AZURE_AI_API_KEY stored encrypted and rendered as password input",
        stored.startswith("encrypted:")
        and KEY not in stored
        and (spec.get("AZURE_AI_API_KEY", {}).get("input") or {}).get("type")
        == "password",
        f"stored={short(stored, 40)}",
    )
    if t.selected("valves"):
        valves_compat(t, spec)
    if t.selected("models"):
        await models(t, base_valves)
    if t.selected("api"):
        await api(t, mock, base_valves)
    if t.selected("dotted"):
        await dotted(t, mock, base_valves)
    if t.selected("browser"):
        await browser(t, mock)
    if t.selected("tasks"):
        await tasks(t, mock)
    rag_mark = None
    if t.selected("rag"):
        from suites._azure_rag import rag  # needs this module's names

        rag_mark = t.mark()
        await rag(t, mock, base_valves)
    await t.owui.update_valves(FID, **base_valves)
    if t.selected("logs"):
        logs(t, rag_mark)
    t.scan_log()


def valves_compat(t: Suite, spec: dict) -> None:
    version = staged_version(PATH)
    required = list(MAIN_VALVES)
    if version_tuple(version) >= version_tuple(SHOW_ALL_SINCE):
        required.append(SHOW_ALL)
    search = version_tuple(version) >= version_tuple(SEARCH_SINCE)
    if search:
        required.extend(SEARCH_VALVES)
    missing = [name for name in required if name not in spec]
    show_all_default = (spec.get(SHOW_ALL) or {}).get("default")
    wrong = []
    if search:
        if SEARCH_MODE in spec:
            wrong.append(f"{SEARCH_MODE} still a valve")
        for name, default in SEARCH_VALVES.items():
            got = (spec.get(name) or {}).get("default")
            if default is not None and got != default:
                wrong.append(f"{name}.default={got!r}")
        for name, values in SEARCH_ENUMS.items():
            enum = (spec.get(name) or {}).get("enum")
            if set(enum or ()) != values:
                wrong.append(f"{name}.enum={enum}")
        key_input = ((spec.get("AZURE_AI_SEARCH_KEY") or {}).get("input") or {}).get(
            "type"
        )
        if key_input != "password":
            wrong.append(f"AZURE_AI_SEARCH_KEY.input={key_input!r}")
    t.check(
        "valves.compat",
        f"valve names of 2.7.0 kept; {SHOW_ALL} defaults to true "
        f"(since {SHOW_ALL_SINCE}); the Azure AI Search valves with their "
        f"defaults, enums and a password input for the key, no {SEARCH_MODE} "
        f"(since {SEARCH_SINCE})",
        not missing
        and (SHOW_ALL not in required or show_all_default is True)
        and not wrong,
        f"version={version} missing={missing} {SHOW_ALL}.default={show_all_default} "
        f"wrong={wrong}",
    )


async def models(t: Suite, base_valves: dict) -> None:
    listed = await _models(t)
    want = {f"{FID}.{m}": f"{PREFIX}: {m}" for m in MODELS}
    t.check(
        "models",
        "AZURE_AI_MODEL list (; , space separated) -> one model each, named "
        "exactly '<prefix>: <model>'",
        listed == want,
        f"listed={listed}",
    )
    cases = (
        (
            "models.exact.space",
            "space separated AZURE_AI_MODEL -> exact names",
            {"AZURE_AI_MODEL": "m-a m-b"},
            {f"{FID}.m-a": f"{PREFIX}: m-a", f"{FID}.m-b": f"{PREFIX}: m-b"},
        ),
        (
            "models.exact.prefix",
            "custom AZURE_AI_PIPELINE_PREFIX -> '<prefix>: <model>'",
            {"AZURE_AI_MODEL": "m-a m-b", "AZURE_AI_PIPELINE_PREFIX": "My AI"},
            {f"{FID}.m-a": "My AI: m-a", f"{FID}.m-b": "My AI: m-b"},
        ),
        (
            "models.url-extract",
            "no AZURE_AI_MODEL, *.openai.azure.com deployment URL -> the "
            "deployment is the model",
            {
                "AZURE_AI_MODEL": "",
                "AZURE_AI_ENDPOINT": "https://res.openai.azure.com/openai/"
                "deployments/dep-x/chat/completions?api-version=2024-10-21",
            },
            {f"{FID}.dep-x": f"{PREFIX}: dep-x"},
        ),
        (
            "models.fallback",
            "no AZURE_AI_MODEL, no model in the URL, no predefined models -> "
            "one 'azure_ai' model",
            {"AZURE_AI_MODEL": ""},
            {f"{FID}.azure_ai": f"{PREFIX}: {PREFIX}"},
        ),
    )
    for sid, title, changes, expected in cases:
        await t.owui.update_valves(FID, **{**base_valves, **changes})
        listed = await _models(t)
        t.check(sid, title, listed == expected, f"listed={listed}")

    await t.owui.update_valves(
        FID,
        **{**base_valves, "AZURE_AI_MODEL": "", "USE_PREDEFINED_AZURE_AI_MODELS": True},
    )
    listed = await _models(t)
    t.check(
        "models.predefined",
        "USE_PREDEFINED_AZURE_AI_MODELS -> the predefined list (gpt-4.1, Phi-4, ...)",
        listed.get(f"{FID}.gpt-4.1") == f"{PREFIX}: OpenAI GPT-4.1"
        and listed.get(f"{FID}.Phi-4") == f"{PREFIX}: Phi-4"
        and len(listed) > 20,
        f"{len(listed)} models, gpt-4.1={listed.get(f'{FID}.gpt-4.1')!r} "
        f"Phi-4={listed.get(f'{FID}.Phi-4')!r}",
    )
    await t.owui.update_valves(FID, **base_valves)


async def api(t: Suite, mock, base_valves: dict) -> None:
    model = f"{FID}.gpt-4o"
    hello = "Hello from mock Azure (gpt-4o)."
    await mock.reset()
    r = await t.owui.chat(
        model,
        "hello",
        stream=False,
        temperature=0.3,
        stream_options=USAGE_OPTIONS,
        **NOT_ALLOWED,
    )
    req = await mock.last()
    body = req.get("body") or {}
    answered = r.status == 200 and r.content == hello
    t.check(
        "api.nonstream",
        "API non-stream: answer and usage passed through",
        answered and (r.usage or {}).get("total_tokens") == 18,
        r.brief(),
    )
    t.check(
        "api.nonstream.request",
        "upstream: decrypted api-key, model header, only allow-listed body keys "
        "(extra client keys dropped)",
        req.get("auth_mode") == "api-key"
        and req.get("model_header") == "gpt-4o"
        and _allow_listed(body),
        f"auth={req.get('auth_mode')} header={req.get('model_header')} "
        f"sent extra={sorted(NOT_ALLOWED)} upstream keys={sorted(body)}",
    )
    t.check(
        "api.nonstream.stream-options",
        "non-stream request with a client stream_options: answered, "
        "stream_options not forwarded",
        answered and "stream_options" not in body,
        f"answered={answered} upstream stream_options={body.get('stream_options')} "
        f"upstream keys={sorted(body)}",
    )
    t.check(
        "api-version",
        "upstream URL keeps path and api-version of AZURE_AI_ENDPOINT (AI Foundry "
        "/models endpoint)",
        req.get("path") == "/models/chat/completions"
        and (req.get("query") or {}).get("api-version") == "2024-05-01-preview",
        f"path={req.get('path')} query={req.get('query')}",
    )

    await mock.reset()
    r = await t.owui.chat(model, "hello", stream=True, temperature=0.3, **NOT_ALLOWED)
    req = await mock.last()
    body = req.get("body") or {}
    t.check(
        "api.stream",
        "API stream: answer, usage chunk and [DONE]",
        r.status == 200
        and r.content == hello
        and (r.usage or {}).get("total_tokens") == 18
        and r.done,
        r.brief(),
    )
    t.check(
        "api.stream.request",
        "upstream: stream_options.include_usage requested, only allow-listed body "
        "keys (extra client keys dropped)",
        (body.get("stream_options") or {}).get("include_usage") is True
        and _allow_listed(body),
        f"stream_options={body.get('stream_options')} "
        f"sent extra={sorted(NOT_ALLOWED)} upstream keys={sorted(body)}",
    )

    await mock.reset()
    r = await t.owui.chat(model, "hello", stream=True, tools=TOOLS, tool_choice="auto")
    req = await mock.last()
    body = req.get("body") or {}
    t.check(
        "api.tools-forwarded",
        "without data_sources, client tools / tool_choice are forwarded",
        r.status == 200
        and r.content == hello
        and body.get("tools") == TOOLS
        and body.get("tool_choice") == "auto",
        f"{r.brief()} upstream tools={short(body.get('tools'), 120)} "
        f"tool_choice={body.get('tool_choice')!r}",
    )

    await t.owui.update_valves(FID, **{**base_valves, "USE_AUTHORIZATION_HEADER": True})
    await mock.reset()
    r = await t.owui.chat(model, "hello", stream=False)
    req = await mock.last()
    t.check(
        "bearer",
        "USE_AUTHORIZATION_HEADER=true -> Authorization: Bearer <decrypted key>, "
        "no api-key header",
        r.status == 200
        and r.content == hello
        and req.get("auth_mode") == "bearer"
        and "api-key" not in (req.get("headers") or {}),
        f"auth={req.get('auth_mode')} headers={sorted(req.get('headers') or {})} "
        f"{r.brief()}",
    )
    await t.owui.update_valves(FID, **base_valves)

    for stream in (False, True):
        mark = t.mark()
        r = await t.owui.chat(model, "force-400 please", stream=stream)
        await t.log.settle(0.5)
        # the pipe logs the provoked upstream 400 (anything else still fails)
        t.expect_errors(mark, ("function_azure:pipe", "Error in Azure AI request: 400"))
        t.check(
            f"api.error-400.{'stream' if stream else 'nonstream'}",
            f"upstream HTTP 400 -> readable error (stream={stream}): exactly "
            "'Error: <Azure's message>', no hint appended",
            r.status == 200 and r.content == FILTERED_ERROR,
            r.brief(),
        )

    mark = t.mark()
    r = await t.owui.chat(model, "force-500-text", stream=False)
    await t.log.settle(0.5)
    t.expect_errors(mark, ("function_azure:pipe", "Error in Azure AI request: 500"))
    t.check(
        "api.error-500-text",
        "upstream HTTP 500 with a text/plain body -> the body text in the answer",
        r.status == 200 and r.content == "Error: upstream exploded (mock)",
        r.brief(),
    )


async def dotted(t: Suite, mock, base_valves: dict) -> None:
    for in_body in (False, True):
        await t.owui.update_valves(
            FID, **{**base_valves, "AZURE_AI_MODEL_IN_BODY": in_body}
        )
        for name in ("gpt-4.1", "Phi-3.5-mini-instruct"):
            for stream in (False, True):
                await mock.reset()
                r = await t.owui.chat(f"{FID}.{name}", "hello", stream=stream)
                req = await mock.last()
                body_model = (req.get("body") or {}).get("model")
                header = req.get("model_header")
                expected_header = None if in_body else name
                t.check(
                    f"dotted.{name}.{'body' if in_body else 'header'}."
                    f"{'stream' if stream else 'nonstream'}",
                    f"{name} (model in {'body' if in_body else 'header'}, "
                    f"stream={stream}) reaches upstream intact",
                    r.status == 200
                    and body_model == name
                    and header == expected_header,
                    f"header={header!r} body.model={body_model!r} {r.brief()}",
                )
    await t.owui.update_valves(FID, **base_valves)


async def browser(t: Suite, mock) -> None:
    model = f"{FID}.gpt-4o"
    async with t.browser() as b:
        for stream in (True, False):
            await mock.reset()
            c = await b.chat(model, "hello", stream=stream)
            kind = "stream" if stream else "nonstream"
            t.check(
                f"browser.{kind}",
                f"browser path stream={stream}: answer, usage and done status saved",
                c.done
                and c.content == "Hello from mock Azure (gpt-4o)."
                and (c.usage or {}).get("total_tokens") == 18
                and _last_status(c.status_history).get("done") is True,
                c.brief(),
            )
            expected = STATUS_STREAM if stream else STATUS_NONSTREAM
            t.check(
                f"browser.status-sequence.{kind}",
                f"browser path stream={stream}: statusHistory is exactly "
                f"{expected}, only the last one done, none hidden",
                _status_sequence(c.status_history, expected),
                f"statuses={_statuses(c.status_history)}",
            )

        mark = t.mark()
        await mock.reset()
        c = await b.chat(model, "force-400 please", stream=True)
        await t.log.settle(0.5)
        t.expect_errors(mark, ("function_azure:pipe", "Error in Azure AI request: 400"))
    last = _last_status(c.status_history)
    t.check(
        "browser.error-status",
        "browser path upstream HTTP 400: error saved and final status done, both "
        "exactly 'Error: <Azure's message>' (no hint appended)",
        c.done
        and c.content == FILTERED_ERROR
        and last.get("description") == FILTERED_ERROR
        and last.get("done") is True,
        f"{c.brief()} statuses={_statuses(c.status_history)}",
    )


async def tasks(t: Suite, mock) -> None:
    await mock.reset()
    status, answer, raw = await t.owui.title_task(
        f"{FID}.gpt-4o", [{"role": "user", "content": "Hi"}]
    )
    t.check(
        "tasks.title",
        "title task via /api/v1/tasks/title/completions",
        status == 200 and answer and "Mock Title" in answer,
        f"HTTP {status} answer={short(answer)} raw={short(raw, 200)}",
    )


async def _wait_tasks(t: Suite, mock, chat_id, timeout: float = 60) -> str:
    """Wait until title, tags and follow-ups ran: 3 task requests and the tags
    saved (the last task), so their events are in the saved message."""
    deadline = time.time() + timeout
    count, tags = 0, []
    while time.time() < deadline:
        count = len(await mock.requests(_is_task))
        saved = await t.owui.get_chat(chat_id) if chat_id else {}
        tags = (saved.get("meta") or {}).get("tags") or []
        if count >= len(TASKS) and tags:
            break
        await asyncio.sleep(0.5)
    await asyncio.sleep(1.5)  # late events of the last task
    return f"tasks={count} tags={tags}"


def logs(t: Suite, rag_mark=None) -> None:
    t.assert_no_secrets(
        KEY,
        RAG_SEARCH_KEY,
        RAG_SEARCH_TOKEN,
        RAG_EMBED_KEY,
        RAG_EMBED_TOKEN,
        RAG_JSON_KEY,
        sid="logs.no-secrets",
        title="the API key, the search keys and tokens never appear in the "
        "server log (INFO and above; the pipe at DEBUG: rag.log.debug)",
    )
    if not t.selected("rag"):
        return  # no citations were fetched
    logged = t.log.since(t.log_start).count(CITATION_TEXT)
    t.check(
        "logs.no-citation-content",
        "citation content (document text) is not logged at INFO or above",
        not logged,
        f"citation text logged {logged}x",
    )
    if rag_mark is None:
        return
    text = t.log.since(rag_mark)
    found = {needle: text.count(needle) for needle in SEARCH_TEXTS if needle in text}
    t.check(
        "logs.no-search-text",
        "generated search queries, document titles and URLs are not logged at "
        "INFO or above",
        not found,
        f"logged: {found}",
    )
