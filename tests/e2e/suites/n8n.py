"""
n8n suite: pipelines/n8n/n8n.py against mocks/mock_n8n.py.

The webhook scenario is selected through the N8N_URL valve
(``.../webhook/<scenario>``, see mocks/mock_n8n.py).

Groups (``--only n8n.<group>``)
  api      API path: JSON answers, request payload, usage, intermediateSteps tool
           display (verbosity, truncation), <think> formatting, conversation
           history / INPUT_FIELD / RESPONSE_FIELD valves, plain text, NDJSON, SSE
           and OpenAI-style streams (plain lines, UTF-8 split across writes,
           braces in strings, SSE fields), a large object trickling in as
           small writes (server CPU), upstream error
  browser  browser path: saved answers, usage and final status (JSON, NDJSON,
           UTF-8, n8n error chunk, broken stream, webhook error), chat context
           sent to the workflow (chat turn vs. background tasks), Stop
  tasks    background title task (without and with a chat id)
"""

import asyncio
import os
import time
from typing import Optional

import httpx

from harness import Suite, short
from harness.config import ADMIN_EMAIL

GROUPS = ("api", "browser", "tasks")
FID = "n8n"
MODEL = FID  # n8n.py is a single pipe: the model id is the function id
PATH = "pipelines/n8n/n8n.py"
TOKEN = "n8n-secret"
CF_ID = "cf-client-id"
CF_SECRET = "cf-secret"
TOOL_HEADER = "Tool Calls (2 steps)"
SSE_ANSWER = "SSE part one. plain line between events\nSSE part two."
UTF8_ANSWER = "Größe naïve 日本 🙂"
BRACES_ANSWER = 'a } b { c } "q" {'
ERROR_CHUNK = "N8N Error: Tool node failed: quota exceeded"
WEBHOOK_ERROR = "N8N Error: Workflow could not be started!\n\nHint: Activate it."
LARGE_SIZE = 400_000  # mock_n8n.LARGE_FLAT_SIZE
# Server CPU seconds for the whole large-flat request (observed on v0.11.4-slim):
# ~0.4 s with a linear parser, ~5 s with one that rescans its buffer on every
# chunk (#182 as first pushed). CPU time grows with the load on the host (up to
# ~3x on a busy one), so the object is large enough that the quadratic cost stays
# well above the limit on an idle host and the linear one well below it on a busy
# one; with 200 KB the quadratic parser needed only 1.4 s on an idle host.
LARGE_MAX_CPU_S = 1.5
LARGE_MAX_S = 30.0  # the mock alone needs ~12 s
HEALTH_MAX_S = 1.0
STOP_AFTER_S = 3
STOP_WAIT_S = 5
TASK_WAIT_S = 15
# Every valve the suite changes, with the value it runs with (restored at the end).
VALVES = {
    "N8N_BEARER_TOKEN": TOKEN,
    "CF_ACCESS_CLIENT_ID": CF_ID,
    "CF_ACCESS_CLIENT_SECRET": CF_SECRET,
    "TOOL_DISPLAY_VERBOSITY": "detailed",
    "TOOL_INPUT_MAX_LENGTH": 500,
    "TOOL_OUTPUT_MAX_LENGTH": 500,
    "SEND_CONVERSATION_HISTORY": False,
    "INPUT_FIELD": "chatInput",
    "RESPONSE_FIELD": "output",
}
SECRET_VALVES = ("N8N_BEARER_TOKEN", "CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET")


def server_cpu_seconds() -> Optional[float]:
    """User + system CPU seconds of the Open WebUI server process (the driver
    runs in the same container), or None when it cannot be found."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                if b"open_webui.main" not in fh.read():
                    continue
            with open(f"/proc/{pid}/stat", encoding="ascii") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError, IndexError):
            continue
    return None


def visible_statuses(c) -> list:
    """Saved statuses without the hidden ones (e.g. the thinking indicator)."""
    return [s for s in c.status_history if not s.get("hidden")]


def last_status(c) -> dict:
    statuses = visible_statuses(c)
    return statuses[-1] if statuses else {}


def status_brief(c) -> str:
    last = last_status(c)
    return f"last_status={last.get('description')!r} done={last.get('done')}"


async def run(t: Suite) -> None:
    mock = t.mock("n8n")
    if not await t.install(FID, PATH, "N8N"):
        return
    valves = {"N8N_URL": f"{mock.url}/webhook/json", **VALVES}
    await t.owui.update_valves(FID, **valves)
    stored = await t.owui.get_valves(FID)
    t.check(
        "valves.encrypted",
        "N8N_BEARER_TOKEN / CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET stored "
        "encrypted",
        all(str(stored.get(k)).startswith("encrypted:") for k in SECRET_VALVES),
        " ".join(f"{k}={short(stored.get(k), 30)}" for k in SECRET_VALVES),
    )
    ids = await t.owui.model_ids(FID)
    t.check("models", "n8n pipe listed as model 'n8n'", MODEL in ids, f"ids={ids}")

    async def scenario(name: str, **changes) -> None:
        await t.owui.update_valves(FID, N8N_URL=f"{mock.url}/webhook/{name}", **changes)
        await mock.reset()

    if t.selected("api"):
        await api(t, mock, scenario)
    if t.selected("browser"):
        await browser(t, mock, scenario)
    if t.selected("tasks"):
        await tasks(t, mock, scenario)
    await t.owui.update_valves(FID, **valves)
    t.assert_no_secrets(TOKEN, CF_ID, CF_SECRET)
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
    body = req.get("body") or {}
    headers = req.get("headers") or {}
    t.check(
        "api.json.request",
        "webhook payload: chatInput / currentMessage, empty systemPrompt and "
        "messages, user fields, decrypted bearer token and Cloudflare headers",
        body.get("chatInput") == "Hello n8n"
        and body.get("currentMessage") == "Hello n8n"
        and body.get("systemPrompt") == ""
        and body.get("messages") == []
        and body.get("user_id") == t.owui.user.get("id")
        and body.get("user_email") == ADMIN_EMAIL
        and body.get("user_role") == "admin"
        and headers.get("authorization") == f"Bearer {TOKEN}"
        and headers.get("cf-access-client-id") == CF_ID
        and headers.get("cf-access-client-secret") == CF_SECRET,
        f"headers={headers} body={short(body, 300)}",
    )

    await scenario("json-usage")
    for stream in (False, True):
        r = await t.owui.chat(MODEL, "usage please", stream=stream)
        ok = (
            r.status == 200
            and r.content == "Hello with usage."
            and (r.usage or {}).get("total_tokens") == 8
        )
        if stream:  # a real stream: delta chunks and [DONE]
            ok = ok and r.done and r.chunks >= 2 and not r.message_content
        t.check(
            f"api.usage.{'stream' if stream else 'nonstream'}",
            f"usage from the webhook answer is reported (stream={stream}"
            + (", delta chunks and [DONE])" if stream else ")"),
            ok,
            r.brief() + (f" done={r.done} chunks={r.chunks}" if stream else ""),
        )

    await tools(t, scenario)

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

    await payload_valves(t, mock, scenario)

    await scenario("text")
    r = await t.owui.chat(MODEL, "plain please", stream=False)
    t.check(
        "api.text",
        "text/plain webhook reply is the answer",
        r.status == 200 and r.content == "Plain text answer from n8n.",
        r.brief(),
    )

    await streams(t, scenario)
    await large_flat(t, scenario)

    await scenario("error")
    mark = t.mark()
    r = await t.owui.chat(MODEL, "fail please", stream=False)
    await t.log.settle(0.5)
    t.expect_errors(mark, ("function_n8n:pipe", "N8N error: Status 500"))
    t.check(
        "api.error",
        "webhook HTTP 500 -> readable error with n8n's message and hint",
        r.status == 200 and r.content == WEBHOOK_ERROR,
        r.brief(),
    )


async def tools(t: Suite, scenario) -> None:
    """intermediateSteps tool display: list / dict form, verbosity, truncation."""
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

    await scenario("json-tools", TOOL_DISPLAY_VERBOSITY="minimal")
    minimal = await t.owui.chat(MODEL, "use tools", stream=False)
    await scenario("json-tools", TOOL_DISPLAY_VERBOSITY="compact")
    compact = await t.owui.chat(MODEL, "use tools", stream=False)
    t.check(
        "api.tools.verbosity",
        "TOOL_DISPLAY_VERBOSITY minimal (tool names) / compact (name -> result)",
        "1. Calculator\n2. Wikipedia" in minimal.content
        and "**Tool:**" not in minimal.content
        and "**1. Calculator** → 4" in compact.content
        and "**Tool:**" not in compact.content,
        f"minimal={short(minimal.content, 250)} compact={short(compact.content, 250)}",
    )

    await scenario(
        "json-tools",
        TOOL_DISPLAY_VERBOSITY="detailed",
        TOOL_INPUT_MAX_LENGTH=5,
        TOOL_OUTPUT_MAX_LENGTH=12,
    )
    r = await t.owui.chat(MODEL, "use tools", stream=False)
    t.check(
        "api.tools.truncation",
        "TOOL_INPUT_MAX_LENGTH / TOOL_OUTPUT_MAX_LENGTH truncate tool input and result",
        '{\n  "...' in r.content
        and '{\n  "title":...' in r.content
        and "Self-hosted" not in r.content,
        short(r.content, 500),
    )
    await t.owui.update_valves(
        FID,
        TOOL_DISPLAY_VERBOSITY=VALVES["TOOL_DISPLAY_VERBOSITY"],
        TOOL_INPUT_MAX_LENGTH=VALVES["TOOL_INPUT_MAX_LENGTH"],
        TOOL_OUTPUT_MAX_LENGTH=VALVES["TOOL_OUTPUT_MAX_LENGTH"],
    )


async def payload_valves(t: Suite, mock, scenario) -> None:
    """SEND_CONVERSATION_HISTORY + INPUT_FIELD, RESPONSE_FIELD."""
    await scenario("json", SEND_CONVERSATION_HISTORY=True, INPUT_FIELD="question")
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ans"},
        {"role": "user", "content": "second"},
    ]
    messages = [{"role": "system", "content": "Be brief.\nBe brief."}, *history]
    r = await t.owui.chat(MODEL, messages, stream=False)
    body = (await mock.last()).get("body") or {}
    t.check(
        "api.history",
        "SEND_CONVERSATION_HISTORY + INPUT_FIELD: history sent, system prompt "
        "deduplicated, input under the custom field",
        r.status == 200
        and body.get("question") == "second"
        and body.get("currentMessage") == "second"
        and body.get("systemPrompt") == "Be brief."
        and body.get("messages") == history
        and "chatInput" not in body,
        f"HTTP {r.status} body={short(body, 400)}",
    )
    await t.owui.update_valves(
        FID,
        SEND_CONVERSATION_HISTORY=VALVES["SEND_CONVERSATION_HISTORY"],
        INPUT_FIELD=VALVES["INPUT_FIELD"],
    )

    await scenario("json-custom", RESPONSE_FIELD="reply")
    r = await t.owui.chat(MODEL, "custom field", stream=False)
    t.check(
        "api.response-field",
        "RESPONSE_FIELD selects the answer field of the webhook reply",
        r.status == 200 and r.content == "Custom field answer.",
        r.brief(),
    )
    await t.owui.update_valves(FID, RESPONSE_FIELD=VALVES["RESPONSE_FIELD"])


async def streams(t: Suite, scenario) -> None:
    """Streamed webhook replies (API path, client stream=True unless noted)."""
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

    await scenario("stream-plain")
    r = await t.owui.chat(MODEL, "plain stream please", stream=True)
    t.check(
        "api.stream-plain",
        "plain-text stream kept line by line (last line without newline)",
        r.status == 200 and r.content == "Plain line one\nPlain line two\nTail",
        r.brief(),
    )

    for name in ("stream-sse-mixed", "stream-sse-coalesced"):
        await scenario(name)
        r = await t.owui.chat(MODEL, "sse please", stream=True)
        t.check(
            f"api.{name}",
            f"SSE stream ({name}): JSON events assembled",
            r.status == 200
            and "SSE part one." in r.content
            and "SSE part two." in r.content,
            r.brief(),
        )
        t.check(
            f"api.{name}.plain-line",
            f"SSE stream ({name}): the plain line between the events is kept",
            SSE_ANSWER in r.content,
            r.brief(),
        )
        t.check(
            f"api.{name}.control-lines",
            f"SSE stream ({name}): comments and [DONE] are not part of the answer",
            "[DONE]" not in r.content and "keep-alive" not in r.content,
            r.brief(),
        )

    await scenario("stream-sse-fields")
    r = await t.owui.chat(MODEL, "sse fields please", stream=True)
    t.check(
        "api.stream-sse-fields",
        "SSE event:/id:/retry: fields dropped, multi-line data: joined",
        r.status == 200 and r.content == "Event one. multi\nline",
        r.brief(),
    )

    await scenario("stream-openai")
    r = await t.owui.chat(MODEL, "openai please", stream=True)
    t.check(
        "api.stream-openai",
        "OpenAI-style chunks: deltas assembled, finish / usage chunks and [DONE] "
        "not in the answer",
        r.status == 200 and r.content == "OpenAI style.",
        r.brief(),
    )

    await scenario("stream-utf8-split")
    r = await t.owui.chat(MODEL, "utf8 please", stream=True)
    t.check(
        "api.stream-utf8-split",
        "UTF-8 characters split across network chunks arrive intact",
        r.status == 200 and r.content == UTF8_ANSWER,
        r.brief(),
    )

    await scenario("stream-braces")
    r = await t.owui.chat(MODEL, "braces please", stream=True)
    t.check(
        "api.stream-braces",
        "braces and quotes inside streamed JSON strings do not break the parser",
        r.status == 200 and r.content == BRACES_ANSWER,
        r.brief(),
    )


async def large_flat(t: Suite, scenario) -> None:
    """A flat 400 KB object trickling in as 1 KiB writes (30 ms apart): the
    answer is complete, and parsing it does not keep Open WebUI's event loop
    busy (a parser that rescans its buffer on every chunk burns several CPU
    seconds here; real TCP coalescing hides that from the wall time) and
    /health stays responsive meanwhile."""
    await scenario("stream-large-flat")
    latencies: list = []
    finished = asyncio.Event()

    async def probe_health() -> None:
        while not finished.is_set():
            started = time.monotonic()
            try:
                await t.owui.http.get("/health", timeout=30)
            except httpx.HTTPError:
                pass
            latencies.append(time.monotonic() - started)
            try:
                await asyncio.wait_for(finished.wait(), 0.1)
            except asyncio.TimeoutError:
                pass

    prober = asyncio.create_task(probe_health())
    cpu_before = server_cpu_seconds()
    started = time.monotonic()
    r = await t.owui.chat(MODEL, "large please", stream=True)
    elapsed = time.monotonic() - started
    cpu_after = server_cpu_seconds()
    finished.set()
    await prober
    cpu = (
        cpu_after - cpu_before
        if cpu_before is not None and cpu_after is not None
        else None
    )
    worst = max(latencies, default=0.0)
    t.check(
        "api.stream-large-flat",
        "flat 400 KB object in 1 KiB writes: answer complete, Open WebUI spends "
        f"< {LARGE_MAX_CPU_S:g} CPU s on it, /health answers in < "
        f"{HEALTH_MAX_S:g} s meanwhile",
        r.status == 200
        and r.content == "x" * LARGE_SIZE
        and cpu is not None
        and cpu < LARGE_MAX_CPU_S
        and elapsed < LARGE_MAX_S
        and worst < HEALTH_MAX_S,
        f"HTTP {r.status} content_len={len(r.content)} "
        f"content_ok={r.content == 'x' * LARGE_SIZE} elapsed={elapsed:.2f}s "
        f"server_cpu={'n/a' if cpu is None else f'{cpu:.2f}s'} "
        f"health_max={worst:.3f}s health_probes={len(latencies)} "
        f"errors={short(r.errors, 200)}",
    )


async def browser(t: Suite, mock, scenario) -> None:
    async with t.browser() as b:
        for name, stream, expect in (
            ("json", True, "Hello from n8n (json)."),
            ("json", False, "Hello from n8n (json)."),
            ("json-usage", False, "Hello with usage."),
            ("json-usage", True, "Hello with usage."),
            ("stream-ndjson", True, "Hello from n8n (ndjson stream)."),
            ("json-tools", True, TOOL_HEADER),
        ):
            await scenario(name)
            c = await b.chat(MODEL, f"browser {name}", stream=stream)
            last = last_status(c)
            sid = f"browser.{name}.{'stream' if stream else 'nonstream'}"
            t.check(
                sid,
                f"browser path {name} stream={stream}: answer saved, final status done",
                c.done and expect in c.content and last.get("done") is True,
                f"{c.brief()} {status_brief(c)}",
            )
            if name == "json-usage":
                t.check(
                    f"{sid}.usage",
                    f"browser path usage saved (stream={stream})",
                    (c.usage or {}).get("total_tokens") == 8,
                    f"usage={c.usage}",
                )
            if name == "stream-ndjson":
                t.check(
                    "browser.stream-ndjson.final-status",
                    "NDJSON stream: 'Streaming complete' is the last visible status "
                    "(hidden thinking status ignored)",
                    last.get("description") == "Streaming complete"
                    and last.get("done") is True,
                    f"{status_brief(c)} statuses={c.status_history}",
                )

        await scenario("stream-utf8-split")
        c = await b.chat(MODEL, "browser utf8", stream=True)
        t.check(
            "browser.stream-utf8-split",
            "browser path: UTF-8 split across network chunks saved intact",
            c.done and c.content == UTF8_ANSWER,
            c.brief(),
        )

        await scenario("stream-error-chunk")
        c = await b.chat(MODEL, "browser error chunk", stream=True)
        last = last_status(c)
        t.check(
            "browser.stream-error-chunk",
            "n8n error chunk: shown as 'N8N Error: ...', final status is that error "
            "(done)",
            ERROR_CHUNK in c.content
            and last.get("description") == ERROR_CHUNK
            and last.get("done") is True,
            f"{c.brief()} {status_brief(c)}",
        )

        await scenario("stream-midfail")
        mark = t.mark()
        c = await b.chat(MODEL, "browser midfail", stream=True)
        await t.log.settle(0.5)
        t.expect_errors(mark, ("function_n8n:pipe", "Streaming error:"))
        last = last_status(c)
        t.check(
            "browser.stream.midfail-status",
            "stream cut off mid-way: partial answer kept, final status "
            "'N8N streaming error: ...' (done)",
            "partial" in c.content
            and str(last.get("description")).startswith("N8N streaming error:")
            and last.get("done") is True,
            f"{c.brief()} {status_brief(c)}",
        )

        await scenario("error")
        mark = t.mark()
        c = await b.chat(MODEL, "browser error", stream=True)
        await t.log.settle(0.5)
        t.expect_errors(mark, ("function_n8n:pipe", "N8N error: Status 500"))
        last = last_status(c)
        t.check(
            "browser.error.stream",
            "webhook HTTP 500 in the browser: readable error saved, same text as "
            "the final status (done)",
            c.content.strip() == WEBHOOK_ERROR
            and last.get("description") == WEBHOOK_ERROR
            and last.get("done") is True,
            f"{c.brief()} {status_brief(c)}",
        )

        await context(t, b, mock, scenario)
        await stop(t, b, scenario)


async def context(t: Suite, b, mock, scenario) -> None:
    """Chat turns carry the chat context, background tasks do not (so workflows
    keyed on chat_id keep task prompts out of the chat memory)."""
    await scenario("json")
    c = await b.chat(
        MODEL,
        "context please",
        stream=True,
        background_tasks={
            "title_generation": True,
            "tags_generation": True,
            "follow_up_generation": True,
        },
        wait_title=True,
    )
    deadline = time.time() + TASK_WAIT_S
    entries = await mock.requests()
    while time.time() < deadline and sum(bool(e.get("task")) for e in entries) < 3:
        await asyncio.sleep(1)
        entries = await mock.requests()
    turns = [e.get("body") or {} for e in entries if not e.get("task")]
    tasks_sent = [e.get("body") or {} for e in entries if e.get("task")]
    turn = turns[0] if turns else {}
    t.check(
        "browser.context",
        "chat turn sends chat_id / message_id and the user fields, background "
        "tasks (title, ...) send no chat context",
        len(turns) == 1
        and turn.get("chat_id") == c.chat_id
        and turn.get("message_id") == c.message_id
        and turn.get("user_id") == t.owui.user.get("id")
        and turn.get("user_email") == ADMIN_EMAIL
        and turn.get("user_role") == "admin"
        and bool(tasks_sent)
        and all(
            "chat_id" in task
            and task.get("chat_id") is None
            and task.get("message_id") is None
            for task in tasks_sent
        )
        and c.title == "Mock Title",
        f"title={c.title!r} chat_id={c.chat_id} message_id={c.message_id} "
        f"turns={[(x.get('chat_id'), x.get('message_id'), x.get('user_role')) for x in turns]} "
        f"tasks={[(x.get('chat_id'), x.get('message_id')) for x in tasks_sent]}",
    )


async def stop(t: Suite, b, scenario) -> None:
    """Stop during a stream / while waiting for the reply: final 'Stopped'
    status, and the aiohttp session is closed."""
    for name, stream, kind in (
        ("slow-ndjson", True, "stream"),
        ("slow-json", False, "nonstream"),
    ):
        await scenario(name)
        mark = t.mark()
        c = await b.chat(
            MODEL,
            f"stop {name}",
            stream=stream,
            stop_after_s=STOP_AFTER_S,
            stop_wait=STOP_WAIT_S,
        )
        await t.log.settle(2)
        leaks = t.log.lines(mark, "Unclosed client session", "Unclosed connection")
        last = last_status(c)
        t.check(
            f"browser.stop.{kind}",
            f"Stop {'during the stream' if stream else 'while waiting for the reply'}"
            ": final status 'Stopped' (done), no unclosed session",
            bool(c.stopped)
            and all(status == 200 for status, _ in c.stopped)
            and last.get("description") == "Stopped"
            and last.get("done") is True
            and not leaks,
            f"{status_brief(c)} unclosed={len(leaks)} stopped={short(c.stopped, 120)} "
            f"{c.brief()}",
        )


async def tasks(t: Suite, mock, scenario) -> None:
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
    await mock.reset()
    status, answer, raw = await t.owui.title_task(
        MODEL, [{"role": "user", "content": "Hi"}], chat_id="explicit-chat-123"
    )
    body = (await mock.last()).get("body") or {}
    t.check(
        "tasks.title.explicit-chat",
        "title task for an explicit chat id: answered, sent to the workflow "
        "without chat_id / message_id",
        status == 200
        and answer
        and "Mock Title" in answer
        and "chat_id" in body
        and body.get("chat_id") is None
        and body.get("message_id") is None,
        f"HTTP {status} answer={short(answer)} chat_id={body.get('chat_id')!r} "
        f"message_id={body.get('message_id')!r}",
    )
