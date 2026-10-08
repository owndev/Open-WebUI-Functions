"""
Gemini suite: pipelines/google/google_gemini.py against mocks/mock_gemini.py
(plus filters/google_search_tool.py for Search grounding).

Groups (``--only gemini.<group>``)
  models     model list, image / video indicators
  api        API path (no websocket): non-stream, stream
  thinking   thinking summaries are not replayed to the API (#176)
  browser    browser path: saved answer + usage, stream and non-stream
  tasks      background title task (direct call and automatic after a chat)
  image      image generation (forced non-stream, image saved to the chat)
  video      Veo long-running operation, video saved to the chat
  grounding  google_search_tool filter -> googleSearch tool, sources, citations
"""

from harness import Suite, known, short

GROUPS = (
    "models",
    "api",
    "thinking",
    "browser",
    "tasks",
    "image",
    "video",
    "grounding",
)
FID = "gemini"
PATH = "pipelines/google/google_gemini.py"
KEY = "mock-gemini-key"
TEXT = f"{FID}.gemini-2.5-flash"
IMAGE_PREVIEW = f"{FID}.gemini-3.1-flash-image-preview"
IMAGE_GA = f"{FID}.gemini-3.1-flash-image"
VEO = f"{FID}.veo-3.1-generate-preview"
EMBEDDING = f"{FID}.text-embedding-004"
SEARCH_FILTER = "google_search_tool"


def _generate(entry: dict) -> bool:
    return entry.get("action") in ("generateContent", "streamGenerateContent")


def _image_delivered(chat) -> bool:
    """Image attached as chat file or embedded as markdown image."""
    files = [f for f in chat.files if f.get("type") == "image" and f.get("url")]
    return bool(files) or "![" in chat.content


async def run(t: Suite) -> None:
    mock = t.mock("gemini")
    await mock.reset()
    if not await t.install(FID, PATH, "Google Gemini"):
        return
    await t.owui.update_valves(
        FID, GOOGLE_API_KEY=KEY, BASE_URL=mock.url + "/", VIDEO_POLL_INTERVAL=5
    )
    stored = (await t.owui.get_valves(FID)).get("GOOGLE_API_KEY", "")
    t.check(
        "valves.encrypted",
        "GOOGLE_API_KEY is stored encrypted",
        stored.startswith("encrypted:") and KEY not in stored,
        f"stored={short(stored, 40)}",
    )

    if t.selected("models"):
        await models(t)
    # API path first: no websocket session of this user may be open (B1).
    if t.selected("api"):
        await api(t, mock)
    if t.selected("thinking"):
        await thinking(t, mock)
    if t.selected("browser"):
        await browser(t, mock)
    if t.selected("tasks"):
        await tasks(t, mock)
    if t.selected("image"):
        await image(t, mock)
    if t.selected("video"):
        await video(t, mock)
    if t.selected("grounding"):
        await grounding(t, mock)
    t.scan_log()


async def models(t: Suite) -> None:
    listed = {m["id"]: m.get("name", "") for m in await t.owui.models()}
    wanted = {TEXT, IMAGE_PREVIEW, IMAGE_GA, VEO}
    t.check(
        "models",
        "models listed (gemini-*, veo-*; embedding models filtered out)",
        wanted <= set(listed) and EMBEDDING not in listed,
        f"listed={sorted(k for k in listed if k.startswith(FID + '.'))}",
    )
    t.check(
        "models.image-preview",
        "gemini-3.1-flash-image-preview is marked as image model",
        "🎨" in listed.get(IMAGE_PREVIEW, ""),
        f"name={listed.get(IMAGE_PREVIEW)!r}",
    )
    t.check(
        "models.image-ga",
        "gemini-3.1-flash-image is marked as image model",
        "🎨" in listed.get(IMAGE_GA, ""),
        f"name={listed.get(IMAGE_GA)!r}",
        known=known.GEMINI_172,
    )
    t.check(
        "models.video",
        "Veo model is marked as video model",
        "🎬" in listed.get(VEO, ""),
        f"name={listed.get(VEO)!r}",
    )


async def api(t: Suite, mock) -> None:
    await mock.reset()
    r = await t.owui.chat(TEXT, "Hello Gemini", stream=False)
    req = await mock.last(_generate)
    t.check(
        "api.nonstream",
        "API non-stream: answer with thinking wrapped in <details>",
        r.status == 200
        and "Hello from mock (non-stream)." in r.content
        and "<details>" in r.content
        and "Mock thinking." in r.content,
        r.brief(),
    )
    t.check(
        "api.nonstream.usage",
        "API non-stream: usage reported",
        (r.usage or {}).get("prompt_tokens") == 11,
        f"usage={r.usage}",
        known=known.GEMINI_NONSTREAM_USAGE,
    )
    t.check(
        "api.nonstream.request",
        "upstream got generateContent with the decrypted API key",
        req.get("action") == "generateContent"
        and req.get("headers", {}).get("x-goog-api-key") == KEY,
        f"action={req.get('action')} key={req.get('headers', {}).get('x-goog-api-key')}",
    )

    await mock.reset()
    mark = t.mark()
    r = await t.owui.chat(TEXT, "Hello Gemini", stream=True)
    req = await mock.last(_generate)
    await t.log.settle(0.5)
    t.check(
        "api.stream",
        "API stream without websocket session: answer streamed",
        r.status == 200
        and "Hello from mock (stream)." in r.content
        and "Error" not in r.content,
        r.brief() + f" upstream={req.get('action')}",
        known=known.GEMINI_B1,
        since=mark,
    )
    t.check(
        "api.stream.usage",
        "API stream: usage chunk forwarded",
        (r.usage or {}).get("prompt_tokens") == 9,
        f"usage={r.usage}",
        known=known.GEMINI_B1,
        since=mark,
    )


async def thinking(t: Suite, mock) -> None:
    previous = (
        "<details>\n<summary>Thought (3s)</summary>\n\n> SECRET-THOUGHT\n\n"
        "</details>First answer."
    )
    history = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": previous},
        {"role": "user", "content": "Second question"},
    ]
    await mock.reset()
    r = await t.owui.chat(TEXT, history, stream=False)
    req = await mock.last(_generate)
    replayed = [
        part.get("text", "")
        for content in (req.get("body") or {}).get("contents") or []
        if content.get("role") == "model"
        for part in content.get("parts") or []
    ]
    t.check(
        "thinking.strip",
        "thinking summary of earlier turns is not replayed to Gemini (#176)",
        r.status == 200
        and any("First answer." in p for p in replayed)
        and not any("SECRET-THOUGHT" in p for p in replayed),
        f"model parts sent upstream={replayed}",
    )


async def browser(t: Suite, mock) -> None:
    async with t.browser() as b:
        for stream, answer, tokens in (
            (True, "Hello from mock (stream).", 9),
            (False, "Hello from mock (non-stream).", 11),
        ):
            await mock.reset()
            c = await b.chat(TEXT, "Hello from the browser", stream=stream)
            t.check(
                f"browser.{'stream' if stream else 'nonstream'}",
                f"browser path stream={stream}: answer saved to the chat",
                c.done and answer in c.content and not c.error,
                c.brief(),
            )
            t.check(
                f"browser.{'stream' if stream else 'nonstream'}.usage",
                f"browser path stream={stream}: usage saved",
                (c.usage or {}).get("prompt_tokens") == tokens,
                f"usage={c.usage} events={c.event_types}",
                known=None if stream else known.GEMINI_NONSTREAM_USAGE,
            )


async def tasks(t: Suite, mock) -> None:
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]
    await mock.reset()
    status, answer, raw = await t.owui.title_task(TEXT, messages)
    t.check(
        "tasks.title",
        "title task via /api/v1/tasks/title/completions (no event emitter)",
        status == 200 and answer and "Mock Title" in answer,
        f"HTTP {status} answer={short(answer)} raw={short(raw, 200)}",
        known=known.GEMINI_B5,
    )
    mark = t.mark()
    async with t.browser() as b:
        c = await b.chat(
            TEXT,
            "Name a city",
            stream=True,
            background_tasks={"title_generation": True, "tags_generation": True},
            wait_title=True,
        )
    t.check(
        "tasks.auto-title",
        "chat title generated by the pipe after a browser-path chat",
        c.done and c.title == "Mock Title",
        f"title={c.title!r} {c.brief()}",
        known=known.GEMINI_B5,
        since=mark,  # the failing title task shows only in the server log
    )


async def image(t: Suite, mock) -> None:
    await mock.reset()
    r = await t.owui.chat(IMAGE_PREVIEW, "Draw a red pixel", stream=True)
    reqs = await mock.requests(_generate)
    body = (reqs[-1].get("body") or {}) if reqs else {}
    modalities = (body.get("generationConfig") or {}).get("responseModalities")
    t.check(
        "image.api",
        "image model via API (stream requested): forced non-stream, IMAGE modality",
        r.status == 200
        and [e.get("action") for e in reqs] == ["generateContent"]
        and modalities
        and "IMAGE" in modalities
        and "Here is your image." in r.content,
        r.brief()
        + f" upstream={[e.get('action') for e in reqs]} modalities={modalities}",
    )
    async with t.browser() as b:
        for model, sid, issue in (
            (IMAGE_PREVIEW, "image.browser-preview", None),
            (IMAGE_GA, "image.browser-ga", known.GEMINI_172),
        ):
            await mock.reset()
            c = await b.chat(model, "Draw a red pixel", stream=True)
            reqs = await mock.requests(_generate)
            image_files = [f for f in c.files if f.get("type") == "image"]
            fetched = None
            if image_files:
                fetched = await t.owui.file_status(image_files[0]["url"])
            t.check(
                sid,
                f"{model.split('.', 1)[1]}: generated image saved to the chat",
                c.done
                and _image_delivered(c)
                and (fetched is None or fetched[0] == 200),
                c.brief()
                + f" upstream={[e.get('action') for e in reqs]} file_get={fetched}",
                known=issue,
            )


async def video(t: Suite, mock) -> None:
    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(VEO, "A cat playing piano", stream=True, wait=180)
        reqs = await mock.requests()
        actions = [e.get("action") or e.get("path") for e in reqs]
        videos = [f for f in c.files if "video" in (f.get("content_type") or "")]
        t.check(
            "video.browser",
            "Veo: operation polled, video downloaded and saved to the chat",
            c.done
            and "predictLongRunning" in actions
            and any("/operations/" in str(a) for a in actions)
            and "download" in actions
            and (videos or "Generated Video" in c.content),
            c.brief() + f" upstream={actions}",
        )


async def grounding(t: Suite, mock) -> None:
    if not await t.install(
        SEARCH_FILTER,
        "filters/google_search_tool.py",
        "Google Search Tool",
        "grounding.load",
    ):
        return
    status = await t.owui.upsert_model(TEXT, "Gemini 2.5 Flash", [SEARCH_FILTER])
    filter_ids = await t.owui.model_filter_ids(TEXT)
    t.check(
        "grounding.attach",
        "google_search_tool attached to the model (visible after models refresh)",
        status == 200 and filter_ids == [SEARCH_FILTER],
        f"HTTP {status} filterIds={filter_ids}",
    )
    await mock.reset()
    r = await t.owui.chat(TEXT, "Search the web", features={"web_search": True})
    req = await mock.last(_generate)
    t.check(
        "grounding.api",
        "API path, features.web_search=true: googleSearch tool sent upstream",
        r.status == 200 and "googleSearch" in (req.get("tool_kinds") or []),
        r.brief() + f" tool_kinds={req.get('tool_kinds')}",
    )
    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(TEXT, "Search the web", features={"web_search": True})
        req = await mock.last(_generate)
    t.check(
        "grounding.browser",
        "browser path: googleSearch sent, grounding sources + [1] citation saved",
        c.done
        and "googleSearch" in (req.get("tool_kinds") or [])
        and any("example.com" in str(s) for s in c.sources)
        and "[1]" in c.content,
        c.brief() + f" tool_kinds={req.get('tool_kinds')}",
    )
    # detach again so later suites see the plain model
    await t.owui.upsert_model(TEXT, "Gemini 2.5 Flash", [])
