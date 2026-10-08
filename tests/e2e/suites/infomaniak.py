"""
Infomaniak suite: pipelines/infomaniak/infomaniak.py against
mocks/mock_infomaniak.py.

Groups (``--only infomaniak.<group>``)
  models   /1/ai/models -> llm models only
  api      API path non-stream / stream (normal, coalesced, split SSE), errors
  browser  browser path: saved answer and usage (normal, coalesced, split SSE)
  tasks    background title task
"""

from harness import Suite, known, short

FID = "infomaniak"
PATH = "pipelines/infomaniak/infomaniak.py"
KEY = "ik-secret"
PRODUCT_ID = 12345


def model(name: str) -> str:
    return f"{FID}.{name}"


async def run(t: Suite) -> None:
    mock = t.mock("infomaniak")
    if not await t.install(FID, PATH, "Infomaniak"):
        return
    await t.owui.update_valves(
        FID,
        INFOMANIAK_API_KEY=KEY,
        INFOMANIAK_PRODUCT_ID=PRODUCT_ID,
        INFOMANIAK_BASE_URL=mock.url,
    )
    stored = str((await t.owui.get_valves(FID)).get("INFOMANIAK_API_KEY"))
    t.check(
        "valves.encrypted",
        "INFOMANIAK_API_KEY stored encrypted",
        stored.startswith("encrypted:") and KEY not in stored,
        f"stored={short(stored, 40)}",
    )

    if t.selected("models"):
        listed = {m["id"]: m.get("name") for m in await t.owui.models()}
        wanted = [model(n) for n in ("mixtral", "llama3", "burst", "split")]
        t.check(
            "models",
            "llm models listed with prefix, stt model filtered out",
            all(w in listed for w in wanted)
            and model("whisper") not in listed
            and str(listed.get(model("mixtral"))).startswith("Infomaniak: "),
            f"listed={[(k, v) for k, v in listed.items() if k.startswith(FID + '.')]}",
        )
    if t.selected("api"):
        await api(t, mock)
    if t.selected("browser"):
        await browser(t)
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
    t.scan_log()


async def api(t: Suite, mock) -> None:
    await mock.reset()
    r = await t.owui.chat(model("mixtral"), "Hello", stream=False)
    req = await mock.last()
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
        "upstream: product id in URL, decrypted bearer key, model id stripped",
        f"/2/ai/{PRODUCT_ID}/" in req.get("path", "")
        and req.get("headers", {}).get("authorization") == f"Bearer {KEY}"
        and (req.get("body") or {}).get("model") == "mixtral",
        f"path={req.get('path')} auth={req.get('headers', {}).get('authorization')} "
        f"model={(req.get('body') or {}).get('model')}",
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
    for stream in (False, True):
        mark = t.mark()
        r = await t.owui.chat(model("bad-model"), "Hello", stream=stream)
        await t.log.settle(0.5)
        t.expect_errors(mark)
        t.check(
            f"api.error-400.{'stream' if stream else 'nonstream'}",
            f"upstream HTTP 400 -> readable error (stream={stream})",
            r.status == 200 and "Mock: model not found" in r.content,
            r.brief(),
        )


async def browser(t: Suite) -> None:
    async with t.browser() as b:
        c = await b.chat(model("mixtral"), "Hello", stream=False)
        t.check(
            "browser.nonstream",
            "browser path non-stream: answer and usage saved",
            c.done
            and c.content == "Hello from Infomaniak (non-stream)."
            and (c.usage or {}).get("total_tokens") == 14,
            c.brief(),
        )
        for name, answer in (
            ("mixtral", "Hello from Infomaniak (stream)."),
            ("burst", "Hello from Infomaniak (burst)."),
            ("split", "Hello from Infomaniak (split)."),
        ):
            c = await b.chat(model(name), "Hello", stream=True)
            t.check(
                f"browser.stream.{name}",
                f"browser path stream ({name}): answer saved",
                c.done and c.content == answer,
                c.brief(),
                known=None if name == "mixtral" else known.INFOMANIAK_CHUNKING,
            )
            t.check(
                f"browser.stream.{name}.usage",
                f"browser path stream ({name}): usage saved",
                (c.usage or {}).get("total_tokens") == 13,
                f"usage={c.usage}",
                known=known.INFOMANIAK_CHUNKING,
            )
