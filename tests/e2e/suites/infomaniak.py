"""
Infomaniak suite: pipelines/infomaniak/infomaniak.py against
mocks/mock_infomaniak.py.

Groups (``--only infomaniak.<group>``)
  models   /1/ai/models -> llm models only, NAME_PREFIX valve
  api      API path non-stream / stream (normal, coalesced, split SSE), errors
           (OpenAI-style message, Infomaniak description, log line),
           allow-listed body (extra client keys dropped)
  browser  browser path: saved answer, usage and status events (normal,
           coalesced, split SSE, no final newline, CRLF, broken stream, error),
           Stop during the stream and while waiting for the headers
  tasks    background title task
"""

from harness import Suite, short

GROUPS = ("models", "api", "browser", "tasks")
FID = "infomaniak"
PATH = "pipelines/infomaniak/infomaniak.py"
KEY = "ik-secret"
PRODUCT_ID = 12345
PREFIX = "Infomaniak: "  # NAME_PREFIX default
# Client keys Open WebUI passes through to the pipe but the allow-list must drop.
NOT_ALLOWED = {"user": "u-e2e", "foo_not_allowed": "x"}
NONSTREAM_ANSWER = "Hello from Infomaniak (non-stream)."
SENDING = "Sending request to Infomaniak AI..."
STREAMING = "Streaming response from Infomaniak AI..."
ERROR_400 = "Error: Mock: model not found"
ERROR_DESC = "Error: Mock description"
# The provoked upstream errors (main / PR #182 wording).
UPSTREAM_ERRORS = (
    ("function_infomaniak:pipe", "Error in Infomaniak AI request: 400"),
    ("function_infomaniak:pipe", "Infomaniak AI API error: HTTP 400"),
)
# The provoked broken stream: the pipe (PR #182) and Open WebUI log it.
MIDFAIL = "Response payload is not completed"
MIDFAIL_ERRORS = (
    ("function_infomaniak:stream_response", "Error while streaming Infomaniak AI"),
    ("open_webui.functions:stream_content", MIDFAIL),
    ("open_webui.utils.middleware:stream_body_handler", MIDFAIL),
)
STOP_AFTER_S = 3
STOP_WAIT_S = 5


def model(name: str) -> str:
    return f"{FID}.{name}"


def statuses(c) -> list:
    """Saved (description, done) pairs."""
    return [(s.get("description"), s.get("done")) for s in c.status_history]


def status_brief(c) -> str:
    return f"status={statuses(c)}"


async def run(t: Suite) -> None:
    mock = t.mock("infomaniak")
    if not await t.install(FID, PATH, "Infomaniak"):
        return
    await t.owui.update_valves(
        FID,
        INFOMANIAK_API_KEY=KEY,
        INFOMANIAK_PRODUCT_ID=PRODUCT_ID,
        INFOMANIAK_BASE_URL=mock.url,
        NAME_PREFIX=PREFIX,
    )
    stored = str((await t.owui.get_valves(FID)).get("INFOMANIAK_API_KEY"))
    t.check(
        "valves.encrypted",
        "INFOMANIAK_API_KEY stored encrypted",
        stored.startswith("encrypted:") and KEY not in stored,
        f"stored={short(stored, 40)}",
    )

    if t.selected("models"):
        await models(t)
    if t.selected("api"):
        await api(t, mock)
    if t.selected("browser"):
        await browser(t, mock)
    if t.selected("tasks"):
        status, answer, raw = await t.owui.title_task(
            model("mixtral"), [{"role": "user", "content": "Hi"}]
        )
        t.check(
            "tasks.title",
            "title task via /api/v1/tasks/title/completions",
            status == 200 and answer and "Mock Title" in answer,
            f"HTTP {status} answer={short(answer)} raw={short(raw, 200)}",
        )
    t.assert_no_secrets(KEY)
    t.scan_log()


async def models(t: Suite) -> None:
    listed = {m["id"]: m.get("name") for m in await t.owui.models()}
    wanted = [model(n) for n in ("mixtral", "llama3", "burst", "split")]
    t.check(
        "models",
        "llm models listed with prefix, stt model filtered out",
        all(w in listed for w in wanted)
        and model("whisper") not in listed
        and str(listed.get(model("mixtral"))).startswith(PREFIX),
        f"listed={[(k, v) for k, v in listed.items() if k.startswith(FID + '.')]}",
    )
    base = str(listed.get(model("mixtral")))[len(PREFIX) :]
    await t.owui.update_valves(FID, NAME_PREFIX="IK> ")
    listed = {m["id"]: m.get("name") for m in await t.owui.models()}
    name = listed.get(model("mixtral"))
    await t.owui.update_valves(FID, NAME_PREFIX=PREFIX)
    listed = {m["id"]: m.get("name") for m in await t.owui.models()}
    restored = listed.get(model("mixtral"))
    t.check(
        "models.name-prefix",
        "NAME_PREFIX valve changes the model names, and changing it back restores them",
        bool(base) and name == f"IK> {base}" and restored == f"{PREFIX}{base}",
        f"name={name!r} restored={restored!r}",
    )


async def api(t: Suite, mock) -> None:
    await mock.reset()
    r = await t.owui.chat(
        model("mixtral"), "Hello", stream=False, seed=7, **NOT_ALLOWED
    )
    req = await mock.last()
    body = req.get("body") or {}
    t.check(
        "api.nonstream",
        "API non-stream: answer and usage",
        r.status == 200
        and r.content == "Hello from Infomaniak (non-stream)."
        and (r.usage or {}).get("total_tokens") == 14,
        r.brief(),
    )
    t.check(
        "api.nonstream.request",
        "upstream: product id in URL, decrypted bearer key, model id stripped, "
        "extra client keys dropped",
        f"/2/ai/{PRODUCT_ID}/" in req.get("path", "")
        and req.get("headers", {}).get("authorization") == f"Bearer {KEY}"
        and body.get("model") == "mixtral"
        and not set(NOT_ALLOWED) & set(body)
        and body.get("seed") == 7,
        f"path={req.get('path')} auth={req.get('headers', {}).get('authorization')} "
        f"model={body.get('model')} sent extra={sorted(NOT_ALLOWED)} "
        f"upstream keys={sorted(body)}",
    )
    for name, answer in (
        ("mixtral", "Hello from Infomaniak (stream)."),
        ("burst", "Hello from Infomaniak (burst)."),
        ("split", "Hello from Infomaniak (split)."),
    ):
        await mock.reset()
        r = await t.owui.chat(model(name), "Hello", stream=True)
        req = await mock.last()
        t.check(
            f"api.stream.{name}",
            f"API stream ({name}): answer, usage, [DONE]",
            r.status == 200
            and r.content == answer
            and (r.usage or {}).get("total_tokens") == 13
            and r.done
            and ((req.get("body") or {}).get("stream_options") or {}).get(
                "include_usage"
            ),
            r.brief(),
        )

    error_logs = []
    for stream in (False, True):
        mark = t.mark()
        r = await t.owui.chat(model("bad-model"), "Hello", stream=stream)
        await t.log.settle(0.5)
        error_logs.append(
            [b for b in t.log.error_blocks(mark) if "function_infomaniak" in b]
        )
        t.expect_errors(mark, *UPSTREAM_ERRORS)
        t.check(
            f"api.error-400.{'stream' if stream else 'nonstream'}",
            f"upstream HTTP 400 -> the provider's error message (stream={stream})",
            r.status == 200 and r.content == ERROR_400,
            r.brief(),
        )
    t.check(
        "api.error-400.exact",
        "upstream HTTP 400 logged as one ERROR line with status and body, no "
        "traceback (per request)",
        all(
            len(blocks) == 1
            and "Infomaniak AI API error: HTTP 400" in blocks[0]
            and "Mock: model not found" in blocks[0]
            and "Traceback" not in blocks[0]
            for blocks in error_logs
        ),
        " || ".join(
            f"{len(blocks)} blocks: "
            + " | ".join(short(b.splitlines()[0], 160) for b in blocks)
            + (" (with Traceback)" if any("Traceback" in b for b in blocks) else "")
            for blocks in error_logs
        ),
    )
    for stream in (False, True):
        mark = t.mark()
        r = await t.owui.chat(model("bad-desc"), "Hello", stream=stream)
        await t.log.settle(0.5)
        t.expect_errors(mark, *UPSTREAM_ERRORS)
        t.check(
            f"api.error-desc.{'stream' if stream else 'nonstream'}",
            f"upstream HTTP 400 with an Infomaniak error.description -> that "
            f"description (stream={stream})",
            r.status == 200 and r.content == ERROR_DESC,
            r.brief(),
        )


async def browser(t: Suite, mock) -> None:
    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(model("mixtral"), "Hello", stream=False)
        t.check(
            "browser.nonstream",
            "browser path non-stream: answer and usage saved",
            c.done
            and c.content == NONSTREAM_ANSWER
            and (c.usage or {}).get("total_tokens") == 14,
            c.brief(),
        )
        t.check(
            "browser.status.nonstream",
            "browser path non-stream: status 'Sending ...' -> 'Request completed' "
            "(done)",
            statuses(c) == [(SENDING, False), ("Request completed", True)],
            f"{status_brief(c)} answer_ok={c.content == NONSTREAM_ANSWER}",
        )
        for name, answer in (
            ("mixtral", "Hello from Infomaniak (stream)."),
            ("burst", "Hello from Infomaniak (burst)."),
            ("split", "Hello from Infomaniak (split)."),
        ):
            await mock.reset()
            c = await b.chat(model(name), "Hello", stream=True)
            sent = ((await mock.last()).get("body") or {}).get("stream_options") or {}
            t.check(
                f"browser.stream.{name}",
                f"browser path stream ({name}): answer saved",
                c.done and c.content == answer,
                c.brief(),
            )
            # The detail tells usage lost in the stream (include_usage requested)
            # from usage the upstream never sent (include_usage missing).
            t.check(
                f"browser.stream.{name}.usage",
                f"browser path stream ({name}): usage saved",
                (c.usage or {}).get("total_tokens") == 13,
                f"usage={c.usage} include_usage_sent={sent.get('include_usage')} "
                + (f"answer_ok={c.content == answer}" if c.content else "content="),
            )
            if name == "mixtral":
                t.check(
                    "browser.status.stream",
                    "browser path stream: status 'Sending ...' -> 'Streaming "
                    "response ...' -> 'Streaming completed' (done)",
                    statuses(c)
                    == [
                        (SENDING, False),
                        (STREAMING, False),
                        ("Streaming completed", True),
                    ],
                    f"{status_brief(c)} answer_ok={c.content == answer}",
                )

        mark = t.mark()
        c = await b.chat(model("bad-model"), "Hello", stream=True)
        await t.log.settle(0.5)
        t.expect_errors(mark, *UPSTREAM_ERRORS)
        t.check(
            "browser.status.error",
            "browser path upstream error: status 'Sending ...' -> the error (done), "
            "error saved as the answer",
            statuses(c) == [(SENDING, False), (ERROR_400, True)]
            and c.content == ERROR_400,
            f"{status_brief(c)} answer_ok={c.content == ERROR_400}",
        )

        for name, answer in (("noeol", "No EOL."), ("crlf", "CR LF.")):
            c = await b.chat(model(name), "Hello", stream=True)
            t.check(
                f"browser.stream.{name}",
                f"browser path stream ({name}): answer and usage saved",
                c.done
                and c.content == answer
                and (c.usage or {}).get("total_tokens") == 13,
                c.brief(),
            )

        mark = t.mark()
        c = await b.chat(model("midfail"), "Hello", stream=True)
        await t.log.settle(0.5)
        t.expect_errors(mark, *MIDFAIL_ERRORS)
        last = c.status_history[-1] if c.status_history else {}
        t.check(
            "browser.stream.midfail",
            "browser path stream cut off: partial answer saved, final status "
            "'Error: ...' (done)",
            c.content.startswith("partial")
            and str(last.get("description")).startswith(f"Error: {MIDFAIL}")
            and last.get("done") is True,
            f"{status_brief(c)} {c.brief()}",
        )

        for name, kind, when in (
            ("slow", "stream", "during the stream"),
            ("slowheaders", "headers", "while waiting for the response headers"),
        ):
            c = await b.chat(
                model(name),
                "Hello",
                stream=True,
                stop_after_s=STOP_AFTER_S,
                stop_wait=STOP_WAIT_S,
            )
            last = c.status_history[-1] if c.status_history else {}
            t.check(
                f"browser.stop.{kind}",
                f"Stop {when}: final status 'Stopped' (done)",
                bool(c.stopped)
                and all(status == 200 for status, _ in c.stopped)
                and last.get("description") == "Stopped"
                and last.get("done") is True
                and (name != "slow" or c.content.startswith("t0 ")),
                f"{status_brief(c)} stopped={short(c.stopped, 120)} {c.brief()}",
            )
