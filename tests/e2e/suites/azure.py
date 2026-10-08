"""
Azure suite: pipelines/azure/azure_ai_foundry.py against mocks/mock_azure.py.

Groups (``--only azure.<group>``)
  valves   valve names are kept (public API), new valves have their defaults
  models   AZURE_AI_MODEL lists (; , space separated) with exact names, custom
           AZURE_AI_PIPELINE_PREFIX, model from an *.openai.azure.com URL,
           predefined models, fallback model
  api      API path non-stream / stream, api-key / Bearer header, path and
           api-version, allow-listed body (extra client keys dropped), tools
           forwarded, stream_options only for streams, errors (JSON 400,
           text/plain 500)
  dotted   model names containing dots reach upstream intact
  browser  browser path: saved answer, usage, full status sequence, error status
  tasks    background title task, also with Azure AI Search valves (#123)
  oyd      Azure AI Search "On Your Data": [docX] links (also split across
           stream deltas, already linked, URL with parentheses), links in the
           history sent back as [docX], only referenced sources saved (show-all
           valve), relevance scores, no data_sources for background tasks (#123,
           also without a websocket session), no tools / stream_options with
           data_sources, large context events, content null
  logs     no API key and no citation text in the server log

Browser chats of the oyd group that test the stream / citation handling send
``params.function_calling = "legacy"``, so Open WebUI does not add its built-in
tools (with tools Azure ignores data_sources, see the mock). The chats without
that parameter are the realistic web UI case and check that the pipe drops the
built-in tools.
"""

import asyncio
import json
import time
import uuid

from harness import Suite, known, short
from harness.known import staged_version, version_tuple
from harness.owui import completion_text

GROUPS = ("valves", "models", "api", "dotted", "browser", "tasks", "oyd", "logs")
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
    "data_sources",
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
DATA_SOURCES = [
    {
        "type": "azure_search",
        "parameters": {
            "endpoint": "https://mock-search.search.windows.net",
            "index_name": "x100-docs",
            "authentication": {"type": "api_key", "key": "search-key"},
        },
    }
]
MANUAL_URL = "https://docs.example.com/x100/manual.pdf"
LINKED = (
    f"The X100 charges via USB-C [[doc1]]({MANUAL_URL}). "
    "It has a two-year warranty [[doc2]](faq/warranty.html)."
)
UNLINKED = "The X100 charges via USB-C [doc1]. It has a two-year warranty [doc2]."
# mock trigger "paren-url": doc1's URL is .../manual_(v2).pdf
PAREN_LINKED = LINKED.replace(
    MANUAL_URL, "https://docs.example.com/x100/manual_%28v2%29.pdf"
)
SPLIT_LINK = f"See [[doc1]]({MANUAL_URL}) and [[doc2]](faq/warranty.html)."
NO_REFS = "The requested information is not available in the retrieved data."
REFERENCED_SOURCES = ["[doc1] - X100 Product Manual", "[doc2] - Warranty FAQ"]
ALL_SOURCES = REFERENCED_SOURCES + ["[doc3] - Release Notes"]
# text of doc1 (mock); must not appear in the server log at INFO
CITATION_TEXT = "The X100 charges via USB-C at up to 65 W."
FILLER = "bigdoc bigdoc bigdoc"  # document text of the big/huge contexts
INCLUDE_CONTEXTS = ["citations", "all_retrieved_documents"]
TASKS = {
    "title_generation": True,
    "tags_generation": True,
    "follow_up_generation": True,
}
# Browser chats without Open WebUI's built-in tools (see the module docstring).
NO_BUILTIN_TOOLS = {"function_calling": "legacy"}
STATUS_STREAM = [
    "Sending request to Azure AI...",
    "Streaming response from Azure AI...",
    "Streaming completed",
]
STATUS_NONSTREAM = ["Sending request to Azure AI...", "Request completed"]
LINE_TOO_LONG = "Got more than 131072 bytes"
STREAM_ERROR = ("function_azure:stream_processor_with_citations", "Error processing")


def _is_task(entry: dict) -> bool:
    return "### Task:" in json.dumps((entry.get("body") or {}).get("messages"))


def _is_answer(entry: dict) -> bool:
    return not _is_task(entry)


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


def _upstream(entry: dict) -> str:
    """Upstream body keys and whether the mock ignored data_sources."""
    body = entry.get("body") or {}
    return (
        f"upstream keys={sorted(body)} "
        f"data_sources_ignored={bool(entry.get('data_sources_ignored'))}"
    )


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


def _oyd_valves(mock, base_valves: dict) -> dict:
    return {
        **base_valves,
        "AZURE_AI_ENDPOINT": f"{mock.url}/openai/deployments/gpt-4.1/chat/completions"
        "?api-version=2025-01-01-preview",
        "AZURE_AI_MODEL": "gpt-4.1",
        "AZURE_AI_DATA_SOURCES": json.dumps(DATA_SOURCES),
        "AZURE_AI_INCLUDE_SEARCH_SCORES": True,
        SHOW_ALL: True,
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
        await tasks(t, mock, base_valves)
    if t.selected("oyd"):
        await oyd(t, mock, base_valves)
    await t.owui.update_valves(FID, **base_valves)
    if t.selected("logs"):
        logs(t)
    t.scan_log()


def valves_compat(t: Suite, spec: dict) -> None:
    version = staged_version(PATH)
    required = list(MAIN_VALVES)
    if version_tuple(version) >= version_tuple(SHOW_ALL_SINCE):
        required.append(SHOW_ALL)
    missing = [name for name in required if name not in spec]
    show_all_default = (spec.get(SHOW_ALL) or {}).get("default")
    t.check(
        "valves.compat",
        f"valve names of 2.7.0 kept; {SHOW_ALL} defaults to true "
        f"(since {SHOW_ALL_SINCE})",
        not missing and (SHOW_ALL not in required or show_all_default is True),
        f"version={version} missing={missing} {SHOW_ALL}.default={show_all_default}",
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
        known=known.AZURE_NONSTREAM_STREAM_OPTIONS,
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
            f"upstream HTTP 400 -> readable error (stream={stream})",
            r.status == 200 and "content management policy" in r.content,
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
                    known=None if stream else known.AZURE_DOUBLE_STRIP,
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
        "browser path upstream HTTP 400: error saved, final 'Error: ...' status done",
        c.done
        and "content management policy" in c.content
        and str(last.get("description", "")).startswith("Error:")
        and last.get("done") is True,
        f"{c.brief()} statuses={_statuses(c.status_history)}",
    )


async def tasks(t: Suite, mock, base_valves: dict) -> None:
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

    await t.owui.update_valves(FID, **_oyd_valves(mock, base_valves))
    await mock.reset()
    status, answer, raw = await t.owui.title_task(
        f"{FID}.gpt-4.1", [{"role": "user", "content": "x100 charging?"}]
    )
    requests = await mock.requests(_is_task)
    with_sources = [e for e in requests if (e.get("body") or {}).get("data_sources")]
    answered = status == 200 and bool(answer) and "Mock Title" in answer
    t.check(
        "tasks.title.oyd",
        "title task with Azure AI Search valves: answered, sent without "
        "data_sources (#123)",
        answered and requests and not with_sources,
        f"answered={answered} task requests={len(requests)} "
        f"with data_sources={len(with_sources)} HTTP {status} "
        f"answer={short(answer)}",
        known=known.AZURE_123,
    )
    await t.owui.update_valves(FID, **base_valves)


async def oyd(t: Suite, mock, base_valves: dict) -> None:
    oyd_valves = _oyd_valves(mock, base_valves)
    await t.owui.update_valves(FID, **oyd_valves)
    model = f"{FID}.gpt-4.1"
    await oyd_api(t, mock, model, oyd_valves)
    await oyd_history(t, mock, model)
    await oyd_big(t, mock, model)
    await oyd_browser(t, mock, model, oyd_valves)
    await oyd_browser_tools(t, mock, model)
    await oyd_no_session(t, mock, model)
    await oyd_usage_capability(t, mock, model)


async def oyd_api(t: Suite, mock, model: str, oyd_valves: dict) -> None:
    question = "x100 charging and warranty?"
    await mock.reset()
    r = await t.owui.chat(model, question, stream=False)
    req = await mock.last()
    parameters = (
        (((req.get("body") or {}).get("data_sources") or [{}])[0]).get("parameters")
        or {}
    )
    t.check(
        "oyd.api.nonstream",
        "On Your Data non-stream: [docX] rewritten to markdown links",
        r.status == 200 and r.content == LINKED,
        f"{r.brief()} include_contexts={parameters.get('include_contexts')}",
    )
    t.check(
        "oyd.api-version",
        "upstream URL keeps path and api-version of AZURE_AI_ENDPOINT (Azure "
        "OpenAI deployment endpoint)",
        req.get("path") == "/openai/deployments/gpt-4.1/chat/completions"
        and (req.get("query") or {}).get("api-version") == "2025-01-01-preview",
        f"path={req.get('path')} query={req.get('query')}",
    )

    await mock.reset()
    r = await t.owui.chat(model, question, stream=True)
    req = await mock.last()
    t.check(
        "oyd.api.stream",
        "On Your Data stream: [docX] links and [DONE], no stream_options upstream",
        r.status == 200
        and r.content == LINKED
        and r.done
        and "stream_options" not in (req.get("body") or {}),
        f"{r.brief()} done={r.done} upstream keys={sorted(req.get('body') or {})}",
    )
    body_model = (req.get("body") or {}).get("model")
    t.check(
        "oyd.dotted.stream",
        "On Your Data stream: dotted model name reaches upstream intact",
        body_model == "gpt-4.1",
        f"body.model={body_model!r} {r.brief()}",
        known=known.AZURE_DOUBLE_STRIP,
    )

    await mock.reset()
    mark = t.mark()
    r = await t.owui.chat(model, question, stream=True, stream_options=USAGE_OPTIONS)
    req = await mock.last()
    await t.log.settle(0.5)
    sent_options = (req.get("body") or {}).get("stream_options")
    t.check(
        "oyd.stream-options",
        "client stream_options is not forwarded together with data_sources",
        r.status == 200
        and r.content == LINKED
        and "stream_options" not in (req.get("body") or {}),
        f"{r.brief()} upstream stream_options={sent_options}",
        known=known.AZURE_STREAM_OPTIONS,
        since=mark,
    )

    for stream in (True, False):
        await mock.reset()
        r = await t.owui.chat(
            model, question, stream=stream, tools=TOOLS, tool_choice="auto"
        )
        req = await mock.last()
        body = req.get("body") or {}
        t.check(
            f"oyd.tools.api.{'stream' if stream else 'nonstream'}",
            "tools / tool_choice are not sent with data_sources, the answer is "
            f"grounded (stream={stream})",
            r.status == 200
            and r.content == LINKED
            and "tools" not in body
            and "tool_choice" not in body
            and bool(body.get("data_sources")),
            f"{r.brief()} {_upstream(req)}",
            known=known.AZURE_TOOLS_DATA_SOURCES,
        )

    for sid, text, expected, issue in (
        ("oyd.split-tokens.api", "x100 split-tokens", LINKED, known.AZURE_HOLDBACK),
        ("oyd.split-link.api", "x100 split-link", SPLIT_LINK, known.AZURE_HOLDBACK),
        (
            "oyd.no-finish.api",
            "x100 split-tokens no-finish",
            LINKED[:-1],
            known.AZURE_FLUSH_BEFORE_DONE,
        ),
    ):
        await mock.reset()
        r = await t.owui.chat(model, text, stream=True)
        t.check(
            sid,
            f"API stream '{text}': references split across deltas are linked "
            "once, [DONE] forwarded",
            r.status == 200 and r.content == expected and r.done,
            f"{r.brief()} done={r.done}",
            known=issue,
        )

    await mock.reset()
    r = await t.owui.chat(model, "x100 paren-url", stream=False)
    t.check(
        "oyd.paren-url",
        "parentheses in citation URLs are percent-encoded in the link",
        r.status == 200 and r.content == PAREN_LINKED,
        r.brief(),
        known=known.AZURE_PAREN_URL,
    )

    # data_sources sent by the client (no AZURE_AI_DATA_SOURCES valve)
    await t.owui.update_valves(FID, **{**oyd_valves, "AZURE_AI_DATA_SOURCES": ""})
    await mock.reset()
    mark = t.mark()
    r = await t.owui.chat(model, question, stream=True, data_sources=DATA_SOURCES)
    req = await mock.last()
    await t.log.settle(0.5)
    body = req.get("body") or {}
    t.check(
        "oyd.client-data-sources",
        "data_sources from the client: [docX] links and [DONE], no stream_options "
        "upstream",
        r.status == 200
        and r.content == LINKED
        and r.done
        and bool(body.get("data_sources"))
        and "stream_options" not in body,
        f"{r.brief()} done={r.done} upstream "
        f"stream_options={body.get('stream_options')} {_upstream(req)}",
        known=known.AZURE_CLIENT_DATA_SOURCES,
        since=mark,
    )
    await t.owui.update_valves(FID, **oyd_valves)


async def oyd_history(t: Suite, mock, model: str) -> None:
    cases = (
        (
            "oyd.history-unlink",
            "links of earlier answers go back to Azure as plain [docX]",
            PAREN_LINKED,
            UNLINKED,
        ),
        (
            "oyd.legacy-history-paren",
            "links saved before 2.8.0 with ')' in the URL go back as plain [docX]",
            "y [[doc1]](https://docs.example.com/a_(b).pdf) z",
            "y [doc1] z",
        ),
    )
    for sid, title, earlier, expected in cases:
        history = [
            {"role": "user", "content": "x100 charging?"},
            {"role": "assistant", "content": earlier},
            {"role": "user", "content": "and the warranty?"},
        ]
        await mock.reset()
        r = await t.owui.chat(model, history, stream=False)
        req = await mock.last()
        messages = (req.get("body") or {}).get("messages") or []
        sent = next(
            (m.get("content") for m in messages if m.get("role") == "assistant"), None
        )
        answered = r.status == 200 and r.content == LINKED
        t.check(
            sid,
            title,
            answered and sent == expected,
            f"answered={answered} sent={sent!r} {r.brief()}",
            known=known.AZURE_HISTORY_UNLINK,
        )


async def oyd_big(t: Suite, mock, model: str) -> None:
    mark = t.mark()
    await mock.reset()
    r = await t.owui.chat(model, "x100 big-context", stream=True)
    await t.log.settle(0.5)
    too_long = len(t.log.lines(mark, LINE_TOO_LONG))
    t.check(
        "oyd.big-context",
        "a ~300 KB context event (one SSE line) is read: linked answer and [DONE]",
        r.status == 200 and r.content == LINKED and r.done and not too_long,
        f"{r.brief()} done={r.done} line_too_long_log={too_long}",
        known=known.AZURE_LINETOOLONG,
        since=mark,
    )

    mark = t.mark()
    await mock.reset()
    r = await t.owui.chat(model, "x100 huge-context", stream=True)
    await t.log.settle(0.5)
    too_long = len(t.log.lines(mark, LINE_TOO_LONG))
    leaked = FILLER in t.log.since(mark)
    # the stream fails on purpose (event larger than the pipe reads)
    t.expect_errors(mark, STREAM_ERROR)
    t.check(
        "oyd.huge-context.api",
        "a ~5 MiB context event ends the stream with an 'Error: ...' delta and "
        "[DONE], without document text in the answer or the log",
        r.status == 200
        and r.content.startswith("Error:")
        and FILLER not in r.content
        and r.done
        and not leaked,
        f"{r.brief()} done={r.done} line_too_long_log={too_long} "
        f"document text in log={leaked}",
        known=known.AZURE_LINETOOLONG,
        since=mark,
    )


async def oyd_browser(t: Suite, mock, model: str, oyd_valves: dict) -> None:
    """Browser chats without Open WebUI's built-in tools."""
    params = NO_BUILTIN_TOOLS
    async with t.browser() as b:
        for sid, text, expected, issue in (
            (
                "oyd.split-tokens.browser",
                "x100 split-tokens",
                LINKED,
                known.AZURE_HOLDBACK,
            ),
            (
                "oyd.split-link.browser",
                "x100 split-link",
                SPLIT_LINK,
                known.AZURE_HOLDBACK,
            ),
            (
                "oyd.no-finish.browser",
                "x100 split-tokens no-finish",
                LINKED[:-1],
                known.AZURE_FLUSH_BEFORE_DONE,
            ),
        ):
            await mock.reset()
            c = await b.chat(model, text, stream=True, params=params)
            t.check(
                sid,
                f"browser stream '{text}': linked answer saved, referenced sources",
                c.done
                and c.content == expected
                and c.source_names == REFERENCED_SOURCES,
                c.brief(),
                known=issue,
            )

        await mock.reset()
        c = await b.chat(model, "x100 no-refs", stream=True, params=params)
        t.check(
            "oyd.no-refs.default",
            "answer without [docX], show-all valve default -> all 3 sources",
            c.done and c.content == NO_REFS and c.source_names == ALL_SOURCES,
            c.brief(),
        )
        await t.owui.update_valves(FID, **{**oyd_valves, SHOW_ALL: False})
        await mock.reset()
        c = await b.chat(model, "x100 no-refs", stream=True, params=params)
        t.check(
            "oyd.no-refs.valve-false",
            "answer without [docX], show-all valve false -> no sources",
            c.done and c.content == NO_REFS and c.source_names == [],
            c.brief(),
            known=known.AZURE_SHOW_ALL_VALVE,
        )

        for include in (True, False):
            await t.owui.update_valves(
                FID, **{**oyd_valves, "AZURE_AI_INCLUDE_SEARCH_SCORES": include}
            )
            await mock.reset()
            c = await b.chat(model, "x100 charging?", stream=True, params=params)
            req = await mock.last(_is_answer)
            parameters = (
                (((req.get("body") or {}).get("data_sources") or [{}])[0]).get(
                    "parameters"
                )
                or {}
            )
            distances = [s.get("distances") for s in c.sources]
            if include:
                ok = parameters.get("include_contexts") == INCLUDE_CONTEXTS and (
                    distances == [[0.8], [0.12]]
                )
                title = (
                    "AZURE_AI_INCLUDE_SEARCH_SCORES=true: include_contexts sent, "
                    "relevance scores saved (rerank 3.2/4.0, BM25 12/100)"
                )
            else:
                ok = "include_contexts" not in parameters and (
                    distances == [[0.0], [0.0]]
                )
                title = (
                    "AZURE_AI_INCLUDE_SEARCH_SCORES=false: no include_contexts, "
                    "scores 0"
                )
            t.check(
                f"oyd.scores.{'on' if include else 'off'}",
                title,
                c.done and c.content == LINKED and ok,
                f"include_contexts={parameters.get('include_contexts')} "
                f"distances={distances} {c.brief()}",
            )
        await t.owui.update_valves(FID, **oyd_valves)

        mark = t.mark()
        await mock.reset()
        # Open WebUI 0.11 never marks a non-stream answer without content as
        # done (non_streaming_chat_response_handler), so do not wait for it.
        c = await b.chat(
            model, "x100 content-null", stream=False, params=params, wait=10
        )
        last = _last_status(c.status_history)
        t.check(
            "oyd.content-null",
            "non-stream answer with content null (content filter): the Azure "
            "response, no pipe error, final status 'Request completed'",
            not c.content.startswith("Error")
            and not c.error
            and last.get("description") == "Request completed"
            and last.get("done") is True,
            f"{c.brief()} statuses={_statuses(c.status_history)}",
            known=known.AZURE_CONTENT_NULL,
            since=mark,
        )

        mark = t.mark()
        await mock.reset()
        c = await b.chat(model, "x100 huge-context", stream=True, params=params)
        await t.log.settle(0.5)
        too_long = len(t.log.lines(mark, LINE_TOO_LONG))
        t.expect_errors(mark, STREAM_ERROR)
    last = _last_status(c.status_history)
    description = str(last.get("description", ""))
    t.check(
        "oyd.huge-context.browser",
        "browser: a ~5 MiB context event -> 'Error: ...' saved and final "
        "'Error: ...' status done, without document text",
        c.done
        and c.content.startswith("Error:")
        and FILLER not in c.content
        and description.startswith("Error:")
        and FILLER not in description
        and last.get("done") is True,
        f"{c.brief()} last status={short(description, 120)} done={last.get('done')} "
        f"line_too_long_log={too_long}",
        known=known.AZURE_LINETOOLONG,
        since=mark,
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


async def oyd_browser_tools(t: Suite, mock, model: str) -> None:
    """Browser chats as the web UI sends them: Open WebUI adds its built-in
    tools, which must not reach Azure together with data_sources."""
    question = "x100 charging and warranty?"
    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(
            model, question, stream=True, background_tasks=TASKS, wait_title=True
        )
        waited = await _wait_tasks(t, mock, c.chat_id)
        c = await b.reload(c)
        answer_req = await mock.last(_is_answer)
        upstream = _upstream(answer_req)
        t.check(
            "oyd.browser.content",
            "browser path (built-in tools): linked answer saved, no tools upstream",
            c.done
            and c.content == LINKED
            and "tools" not in (answer_req.get("body") or {}),
            f"{c.brief()} {upstream}",
            known=known.AZURE_TOOLS_DATA_SOURCES,
        )
        task_requests = await mock.requests(_is_task)
        with_sources = [
            e for e in task_requests if (e.get("body") or {}).get("data_sources")
        ]
        t.check(
            "oyd.tasks.no-data-sources",
            "background tasks are sent without data_sources",
            task_requests and not with_sources,
            f"answered={_answered(c.content)} task requests={len(task_requests)} "
            f"with data_sources={len(with_sources)} {waited}",
            known=known.AZURE_123,
        )
        t.check(
            "oyd.browser.sources",
            "saved sources = only the referenced documents, also after background "
            "tasks",
            c.source_names == REFERENCED_SOURCES,
            f"{c.brief()} title={c.title!r} {waited} {upstream}",
            known=known.AZURE_TOOLS_DATA_SOURCES,
        )

        await mock.reset()
        c = await b.chat(model, question, stream=True)
        req = await mock.last(_is_answer)
        body = req.get("body") or {}
        t.check(
            "oyd.browser.stream-status",
            "browser stream (built-in tools): linked answer, upstream body only "
            "data_sources/messages/model/stream, statusHistory complete and done",
            c.done
            and c.content == LINKED
            and sorted(body) == ["data_sources", "messages", "model", "stream"]
            and _status_sequence(c.status_history, STATUS_STREAM),
            f"{c.brief()} statuses={_statuses(c.status_history)} {_upstream(req)}",
            known=known.AZURE_TOOLS_DATA_SOURCES,
        )

        await mock.reset()
        c = await b.chat(model, question, stream=False)
        req = await mock.last(_is_answer)
        t.check(
            "oyd.browser.nonstream",
            "browser non-stream (built-in tools): linked answer, referenced "
            "sources, usage, statusHistory complete and done",
            c.done
            and c.content == LINKED
            and c.source_names == REFERENCED_SOURCES
            and (c.usage or {}).get("total_tokens") == 18
            and _status_sequence(c.status_history, STATUS_NONSTREAM),
            f"{c.brief()} statuses={_statuses(c.status_history)} {_upstream(req)}",
            known=known.AZURE_TOOLS_DATA_SOURCES,
        )


async def oyd_no_session(t: Suite, mock, model: str) -> None:
    """Saved chat without a websocket session (no built-in tools): background
    tasks run with the message's metadata, so their events must not reach it
    (#123: 9 sources and 7 statuses on 2.7.0)."""
    text = "x100 charging and warranty?"
    aid, uid = str(uuid.uuid4()), str(uuid.uuid4())
    body = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": text}],
        "id": aid,
        "parent_id": None,
        "user_message": {
            "id": uid,
            "parentId": None,
            "childrenIds": [aid],
            "role": "user",
            "content": text,
            "timestamp": int(time.time()),
            "models": [model],
        },
        "background_tasks": TASKS,
    }
    await mock.reset()
    status, data = await t.owui.api("POST", "/api/chat/completions", body)
    answer = completion_text(data) or ""
    chat_id = data.get("chat_id") if isinstance(data, dict) else None
    if not chat_id:
        _, chats = await t.owui.api("GET", "/api/v1/chats/?page=1")
        for chat in (chats if isinstance(chats, list) else [])[:10]:
            saved = await t.owui.get_chat(chat["id"])
            messages = ((saved.get("chat") or {}).get("history") or {}).get(
                "messages"
            ) or {}
            if aid in messages:
                chat_id = chat["id"]
                break
    waited = await _wait_tasks(t, mock, chat_id)
    saved = await t.owui.get_chat(chat_id) if chat_id else {}
    messages = ((saved.get("chat") or {}).get("history") or {}).get("messages") or {}
    message = messages.get(aid) or {}
    names = [(s.get("source") or {}).get("name") for s in message.get("sources") or []]
    history = message.get("statusHistory") or []
    task_requests = await mock.requests(_is_task)
    with_sources = [
        e for e in task_requests if (e.get("body") or {}).get("data_sources")
    ]
    answered = status == 200 and answer == LINKED
    t.check(
        "oyd.no-session.sources",
        "saved chat without websocket session + background tasks: linked answer, "
        "only the referenced sources and the answer's statuses (#123)",
        answered
        and message.get("content") == LINKED
        and names == REFERENCED_SOURCES
        and _status_sequence(history, STATUS_NONSTREAM)
        and not with_sources,
        f"answered={answered} task requests={len(task_requests)} "
        f"with data_sources={len(with_sources)} sources={names} "
        f"statuses={_statuses(history)} content={short(message.get('content'))} "
        f"chat={chat_id} {waited}",
        known=known.AZURE_123,
    )


async def oyd_usage_capability(t: Suite, mock, model: str) -> None:
    """Open WebUI 0.11.1+ adds stream_options to streamed requests of models
    with the 'usage' capability; Azure On Your Data rejects it."""
    status = await t.owui.upsert_model(
        model, "Azure gpt-4.1 (usage)", capabilities={"usage": True}
    )
    try:
        async with t.browser() as b:
            mark = t.mark()
            await mock.reset()
            c = await b.chat(model, "x100 charging?", stream=True)
            req = await mock.last(_is_answer)
            await t.log.settle(0.5)
        body = req.get("body") or {}
        t.check(
            "oyd.usage-capability",
            "model with the usage capability (Open WebUI adds stream_options): "
            "stream_options not sent with data_sources, linked answer",
            c.done and c.content == LINKED and "stream_options" not in body,
            f"model upsert HTTP {status} {c.brief()} upstream "
            f"stream_options={body.get('stream_options')} {_upstream(req)}",
            known=known.AZURE_STREAM_OPTIONS,
            since=mark,
        )
    finally:
        await t.owui.delete_model(model)


def logs(t: Suite) -> None:
    t.assert_no_secrets(
        KEY,
        sid="logs.no-secrets",
        title="the API key never appears in the server log (any level)",
    )
    if not t.selected("oyd"):
        return  # no citations were fetched
    logged = t.log.since(t.log_start).count(CITATION_TEXT)
    t.check(
        "logs.no-citation-content",
        "citation content (document text) is not logged at INFO",
        not logged,
        f"citation text logged {logged}x",
        known=known.AZURE_CITATION_INFO_LOG,
    )
