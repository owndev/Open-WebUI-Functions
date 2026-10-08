"""
Azure suite: pipelines/azure/azure_ai_foundry.py against mocks/mock_azure.py.

Groups (``--only azure.<group>``)
  models   AZURE_AI_MODEL list (; , space separated) -> manifold models
  api      API path non-stream / stream, headers, allow-listed body (extra
           client keys dropped), errors
  dotted   model names containing dots reach upstream intact
  browser  browser path: saved answer, usage, terminal status
  tasks    background title task
  oyd      Azure AI Search "On Your Data": [docX] links, citations/sources,
           no data_sources for background tasks (#123), stream_options
"""

import json

from harness import Suite, known, short

GROUPS = ("models", "api", "dotted", "browser", "tasks", "oyd")
FID = "azure"
PATH = "pipelines/azure/azure_ai_foundry.py"
KEY = "mock-key-123"
MODELS = ("gpt-4o", "gpt-4.1", "Phi-3.5-mini-instruct")
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
LINKED = (
    "The X100 charges via USB-C [[doc1]](https://docs.example.com/x100/manual.pdf). "
    "It has a two-year warranty [[doc2]](faq/warranty.html)."
)
REFERENCED_SOURCES = ["[doc1] - X100 Product Manual", "[doc2] - Warranty FAQ"]


def _is_task(entry: dict) -> bool:
    return "### Task:" in json.dumps((entry.get("body") or {}).get("messages"))


async def run(t: Suite) -> None:
    mock = t.mock("azure")
    if not await t.install(FID, PATH, "Azure AI Foundry"):
        return
    base_valves = {
        "AZURE_AI_ENDPOINT": f"{mock.url}/models/chat/completions"
        "?api-version=2024-05-01-preview",
        "AZURE_AI_API_KEY": KEY,
        "AZURE_AI_MODEL": "gpt-4o;gpt-4.1, Phi-3.5-mini-instruct",
        "AZURE_AI_MODEL_IN_BODY": False,
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
    if t.selected("models"):
        listed = {m["id"]: m.get("name") for m in await t.owui.models()}
        ids = [f"{FID}.{m}" for m in MODELS]
        t.check(
            "models",
            "AZURE_AI_MODEL list (; , space separated) -> one model each",
            all(i in listed for i in ids)
            and all(str(listed[i]).startswith("Azure AI: ") for i in ids),
            f"listed={[(k, v) for k, v in listed.items() if k.startswith(FID + '.')]}",
        )
    if t.selected("api"):
        await api(t, mock)
    if t.selected("dotted"):
        await dotted(t, mock, base_valves)
    if t.selected("browser"):
        await browser(t, mock)
    if t.selected("tasks"):
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
    if t.selected("oyd"):
        await oyd(t, mock, base_valves)
    await t.owui.update_valves(FID, **base_valves)
    t.scan_log()


def _allow_listed(body: dict) -> bool:
    """Only allow-listed keys went upstream, the extra client keys were dropped
    and an allow-listed optional parameter (temperature) was kept."""
    return (
        set(body) <= ALLOWED
        and not set(NOT_ALLOWED) & set(body)
        and body.get("temperature") == 0.3
    )


async def api(t: Suite, mock) -> None:
    model = f"{FID}.gpt-4o"
    await mock.reset()
    r = await t.owui.chat(model, "hello", stream=False, temperature=0.3, **NOT_ALLOWED)
    req = await mock.last()
    t.check(
        "api.nonstream",
        "API non-stream: answer and usage passed through",
        r.status == 200
        and r.content == "Hello from mock Azure (gpt-4o)."
        and (r.usage or {}).get("total_tokens") == 18,
        r.brief(),
    )
    body = req.get("body") or {}
    t.check(
        "api.nonstream.request",
        "upstream: decrypted api-key, model header, only allow-listed body keys "
        "(extra client keys dropped)",
        req.get("auth_mode") == "api-key"
        and req.get("model_header") == "gpt-4o"
        and _allow_listed(body)
        and "stream_options" not in body,
        f"auth={req.get('auth_mode')} header={req.get('model_header')} "
        f"sent extra={sorted(NOT_ALLOWED)} upstream keys={sorted(body)}",
    )

    await mock.reset()
    r = await t.owui.chat(model, "hello", stream=True, temperature=0.3, **NOT_ALLOWED)
    req = await mock.last()
    body = req.get("body") or {}
    t.check(
        "api.stream",
        "API stream: answer, usage chunk and [DONE]",
        r.status == 200
        and r.content == "Hello from mock Azure (gpt-4o)."
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
    async with t.browser() as b:
        for stream in (True, False):
            await mock.reset()
            c = await b.chat(f"{FID}.gpt-4o", "hello", stream=stream)
            last_status = c.status_history[-1] if c.status_history else {}
            sid = "browser.stream" if stream else "browser.nonstream"
            t.check(
                sid,
                f"browser path stream={stream}: answer, usage and done status saved",
                c.done
                and c.content == "Hello from mock Azure (gpt-4o)."
                and (c.usage or {}).get("total_tokens") == 18
                and last_status.get("done") is True,
                c.brief(),
            )


async def oyd(t: Suite, mock, base_valves: dict) -> None:
    oyd_valves = {
        **base_valves,
        "AZURE_AI_ENDPOINT": f"{mock.url}/openai/deployments/gpt-4.1/chat/completions"
        "?api-version=2025-01-01-preview",
        "AZURE_AI_MODEL": "gpt-4.1",
        "AZURE_AI_DATA_SOURCES": json.dumps(DATA_SOURCES),
        "AZURE_AI_INCLUDE_SEARCH_SCORES": True,
    }
    await t.owui.update_valves(FID, **oyd_valves)
    model = f"{FID}.gpt-4.1"
    question = "x100 charging and warranty?"

    await mock.reset()
    r = await t.owui.chat(model, question, stream=False)
    req = await mock.last()
    sources = (req.get("body") or {}).get("data_sources") or [{}]
    t.check(
        "oyd.api.nonstream",
        "On Your Data non-stream: [docX] rewritten to markdown links",
        r.status == 200 and r.content == LINKED,
        r.brief()
        + f" include_contexts={(sources[0].get('parameters') or {}).get('include_contexts')}",
    )

    await mock.reset()
    r = await t.owui.chat(model, question, stream=True)
    req = await mock.last()
    t.check(
        "oyd.api.stream",
        "On Your Data stream: [docX] links, no stream_options upstream",
        r.status == 200
        and r.content == LINKED
        and "stream_options" not in (req.get("body") or {}),
        r.brief() + f" upstream_keys={sorted(req.get('body') or {})}",
    )
    t.check(
        "oyd.dotted.stream",
        "On Your Data stream: dotted model name reaches upstream intact",
        (req.get("body") or {}).get("model") == "gpt-4.1",
        f"body.model={(req.get('body') or {}).get('model')!r}",
        known=known.AZURE_DOUBLE_STRIP,
    )

    await mock.reset()
    mark = t.mark()
    r = await t.owui.chat(
        model, question, stream=True, stream_options={"include_usage": True}
    )
    req = await mock.last()
    await t.log.settle(0.5)
    sent_options = (req.get("body") or {}).get("stream_options")
    t.check(
        "oyd.stream-options",
        "client stream_options is not forwarded together with data_sources",
        r.status == 200
        and r.content == LINKED
        and "stream_options" not in (req.get("body") or {}),
        r.brief() + f" upstream stream_options={sent_options}",
        known=known.AZURE_STREAM_OPTIONS,
        since=mark,
    )

    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(
            model,
            question,
            stream=True,
            background_tasks={
                "title_generation": True,
                "tags_generation": True,
                "follow_up_generation": True,
            },
            wait_title=True,
        )
        await t.log.settle(3)  # tags / follow-ups run after the title
        c = await b.reload(c)
    t.check(
        "oyd.browser.content",
        "browser path: linked answer saved",
        c.done and c.content == LINKED,
        c.brief(),
    )
    task_requests = await mock.requests(_is_task)
    with_sources = [
        e for e in task_requests if (e.get("body") or {}).get("data_sources")
    ]
    t.check(
        "oyd.tasks.no-data-sources",
        "background tasks are sent without data_sources",
        task_requests and not with_sources,
        f"task requests={len(task_requests)} with data_sources={len(with_sources)}",
        known=known.AZURE_123,
    )
    t.check(
        "oyd.browser.sources",
        "saved sources = only the referenced documents, also after background tasks",
        c.source_names == REFERENCED_SOURCES,
        f"sources={c.source_names} title={c.title!r} events={c.event_types}",
    )
