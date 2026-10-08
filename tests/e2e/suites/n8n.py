"""
n8n suite: pipelines/n8n/n8n.py against mocks/mock_n8n.py.

The webhook scenario is selected through the N8N_URL valve
(``.../webhook/<scenario>``, see mocks/mock_n8n.py).

Groups (``--only n8n.<group>``)
  api      API path: JSON answers, usage, intermediateSteps tool display,
           <think> formatting, NDJSON / SSE streams, upstream error
  browser  browser path: saved answers (JSON, usage, NDJSON stream)
  tasks    background title task
"""

from harness import Suite, known, short

FID = "n8n"
MODEL = FID  # n8n.py is a single pipe: the model id is the function id
PATH = "pipelines/n8n/n8n.py"
TOKEN = "n8n-secret"
TOOL_HEADER = "Tool Calls (2 steps)"


async def run(t: Suite) -> None:
    mock = t.mock("n8n")
    if not await t.install(FID, PATH, "N8N"):
        return
    valves = {
        "N8N_URL": f"{mock.url}/webhook/json",
        "N8N_BEARER_TOKEN": TOKEN,
        "CF_ACCESS_CLIENT_ID": "cf-id",
        "CF_ACCESS_CLIENT_SECRET": "cf-secret",
        "TOOL_DISPLAY_VERBOSITY": "detailed",
    }
    await t.owui.update_valves(FID, **valves)
    stored = await t.owui.get_valves(FID)
    t.check(
        "valves.encrypted",
        "N8N_BEARER_TOKEN / CF_ACCESS_CLIENT_SECRET stored encrypted",
        str(stored.get("N8N_BEARER_TOKEN")).startswith("encrypted:")
        and str(stored.get("CF_ACCESS_CLIENT_SECRET")).startswith("encrypted:"),
        f"token={short(stored.get('N8N_BEARER_TOKEN'), 30)}",
    )
    ids = await t.owui.model_ids(FID)
    t.check("models", "n8n pipe listed as model 'n8n'", MODEL in ids, f"ids={ids}")

    async def scenario(name: str) -> None:
        await t.owui.update_valves(FID, N8N_URL=f"{mock.url}/webhook/{name}")
        await mock.reset()

    if t.selected("api"):
        await api(t, mock, scenario)
    if t.selected("browser"):
        await browser(t, scenario)
    if t.selected("tasks"):
        await scenario("json")
        status, answer, raw = await t.owui.title_task(
            MODEL, [{"role": "user", "content": "Hi"}]
        )
        t.check(
            "tasks.title",
            "title task via /api/v1/tasks/title/completions",
            status == 200 and answer and "Mock Title" in answer,
            f"HTTP {status} answer={short(answer)} raw={short(raw, 200)}",
        )
    await t.owui.update_valves(FID, **valves)
    t.scan_log()


async def api(t: Suite, mock, scenario) -> None:
    await scenario("json")
    for stream in (False, True):
        r = await t.owui.chat(MODEL, "Hello n8n", stream=stream)
        t.check(
            f"api.json.{'stream' if stream else 'nonstream'}",
            f"JSON webhook answer (stream={stream})",
            r.status == 200 and r.content == "Hello from n8n (json).",
            r.brief(),
        )
    req = await mock.last()
    t.check(
        "api.json.request",
        "webhook got chatInput + decrypted bearer token and Cloudflare headers",
        (req.get("body") or {}).get("chatInput") == "Hello n8n"
        and req.get("headers", {}).get("authorization") == f"Bearer {TOKEN}"
        and req.get("headers", {}).get("cf-access-client-secret") == "cf-secret",
        f"headers={req.get('headers')} body_keys={sorted(req.get('body') or {})}",
    )

    await scenario("json-usage")
    for stream in (False, True):
        r = await t.owui.chat(MODEL, "usage please", stream=stream)
        t.check(
            f"api.usage.{'stream' if stream else 'nonstream'}",
            f"usage from the webhook answer is reported (stream={stream})",
            r.status == 200
            and r.content == "Hello with usage."
            and (r.usage or {}).get("total_tokens") == 8,
            r.brief(),
        )

    for name, answer in (
        ("json-tools", "The answer is 4."),
        ("json-tools-dict", "Dict"),
    ):
        await scenario(name)
        r = await t.owui.chat(MODEL, "use tools", stream=False)
        t.check(
            f"api.{name}",
            f"intermediateSteps ({name}) rendered as tool-call details",
            r.status == 200
            and r.content.startswith(answer)
            and TOOL_HEADER in r.content
            and "Calculator" in r.content,
            r.brief(),
        )

    await scenario("json-think")
    r = await t.owui.chat(MODEL, "think", stream=False)
    t.check(
        "api.think",
        "<think> block converted to a collapsible thought",
        r.status == 200
        and "<details>" in r.content
        and "Step one" in r.content
        and r.content.endswith("Final answer."),
        r.brief(),
    )

    await scenario("stream-ndjson")
    for stream in (True, False):
        r = await t.owui.chat(MODEL, "stream please", stream=stream)
        t.check(
            f"api.ndjson.{'stream' if stream else 'nonstream'}",
            f"n8n NDJSON stream assembled (client stream={stream})",
            r.status == 200
            and "Hello from n8n (ndjson stream)." in r.content
            and "<details>" in r.content
            and '"type"' not in r.content,
            r.brief(),
        )

    for name in ("stream-sse-mixed", "stream-sse-coalesced"):
        await scenario(name)
        r = await t.owui.chat(MODEL, "sse please", stream=True)
        t.check(
            f"api.{name}",
            f"SSE stream ({name}): JSON events and plain lines assembled",
            r.status == 200
            and "SSE part one." in r.content
            and "SSE part two." in r.content,
            r.brief(),
        )
        t.check(
            f"api.{name}.control-lines",
            f"SSE stream ({name}): comments and [DONE] are not part of the answer",
            "[DONE]" not in r.content and "keep-alive" not in r.content,
            r.brief(),
            known=known.N8N_SSE_CONTROL_LINES,
        )

    await scenario("error")
    mark = t.mark()
    r = await t.owui.chat(MODEL, "fail please", stream=False)
    await t.log.settle(0.5)
    t.expect_errors(mark)
    t.check(
        "api.error",
        "webhook HTTP 500 -> readable error with n8n's message and hint",
        r.status == 200
        and "Workflow could not be started!" in r.content
        and "Activate it." in r.content,
        r.brief(),
    )


async def browser(t: Suite, scenario) -> None:
    async with t.browser() as b:
        for name, stream, expect, issue in (
            ("json", True, "Hello from n8n (json).", None),
            ("json", False, "Hello from n8n (json).", None),
            ("json-usage", False, "Hello with usage.", None),
            ("json-usage", True, "Hello with usage.", known.N8N_DICT_IN_STREAM),
            ("stream-ndjson", True, "Hello from n8n (ndjson stream).", None),
            ("json-tools", True, TOOL_HEADER, None),
        ):
            await scenario(name)
            c = await b.chat(MODEL, f"browser {name}", stream=stream)
            last_status = c.status_history[-1] if c.status_history else {}
            t.check(
                f"browser.{name}.{'stream' if stream else 'nonstream'}",
                f"browser path {name} stream={stream}: answer saved, status done",
                c.done and expect in c.content and last_status.get("done") is True,
                c.brief(),
                known=issue,
            )
            if name == "json-usage":
                t.check(
                    f"browser.{name}.{'stream' if stream else 'nonstream'}.usage",
                    f"browser path usage saved (stream={stream})",
                    (c.usage or {}).get("total_tokens") == 8,
                    f"usage={c.usage}",
                )
