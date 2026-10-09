"""
Gemini suite: pipelines/google/google_gemini.py against mocks/mock_gemini.py
(plus filters/google_search_tool.py for Search grounding).

Groups (``--only gemini.<group>``)
  models       model list, image / video indicators
  api          API path (no websocket): non-stream, stream (thinking in <details>)
  thinking     summaries not replayed (#176), budget / level / include / strip valves
  browser      browser path: saved answer + usage, stream and non-stream
  tasks        title task (direct, automatic), task answers without <details>,
               tasks of a web_search chat without grounding tools
  image        image models via API and browser: forced non-stream, one file
  images       thought images (skipped, fallback, IMAGE_SAFETY), dedup, two final
               images, image link for API clients, image history, optimization
  nano         gemini-nano-banana-2.1: listed, forced non-stream, thinking level
  imgvalve     IMAGE_GENERATION_MODELS valve (+ MODEL_ADDITIONAL)
  imgconfig    ImageConfig valves, lite 1K only, 2.5 without, user valve, body
  imgtools     tools per image model with web_search (no native tools / urlContext,
               no Search for 2.5-flash-image and 3.1-flash-lite-image)
  nostream     GOOGLE_STREAMING_ENABLED=false with stream=true (#170)
  video        Veo: text + video saved, request shape, image-to-video
  grounding    google_search_tool filter -> googleSearch + urlContext, sources,
               citations; no grounding without web_search
  vertex       Vertex AI Search sources (retrievedContext.text)
  errors       upstream 400 / 500 (retry) / blocked prompt / SAFETY finish / image
               error status / streamed answer starting with "data:"
  retry        RETRY_COUNT for streams (503 on the first chunk, then 200)
  status       Stop during Veo polling leaves no running status
  valves       model cache vs. valve changes, safety, whitelist, additional,
               system prompt, user headers, API version, params, valve names
  concurrency  forwarded user headers belong to the requesting user
  streamimg    inline image from a model the pipe does not detect (stream path)

The browser path sends the web UI's default params, i.e. native function
calling with Open WebUI's built-in tools (functionDeclarations). The "images"
group and the image error status use ``function_calling=legacy`` so they test
the image handling itself; the tools sent to image models are checked by
"image", "nano" and "imgtools".
"""

import asyncio
import base64
import random
import re
import struct
import zlib

from harness import Suite, short

GROUPS = (
    "models",
    "api",
    "thinking",
    "browser",
    "tasks",
    "image",
    "images",
    "nano",
    "imgvalve",
    "imgconfig",
    "imgtools",
    "nostream",
    "video",
    "grounding",
    "vertex",
    "errors",
    "retry",
    "status",
    "valves",
    "concurrency",
    "streamimg",
)
FID = "gemini"
PATH = "pipelines/google/google_gemini.py"
KEY = "mock-gemini-key"
TEXT = f"{FID}.gemini-2.5-flash"
PRO3 = f"{FID}.gemini-3-pro-preview"
IMAGE_PREVIEW = f"{FID}.gemini-3.1-flash-image-preview"
IMAGE_GA = f"{FID}.gemini-3.1-flash-image"
IMAGE_LITE = f"{FID}.gemini-3.1-flash-lite-image"
IMAGE_25 = f"{FID}.gemini-2.5-flash-image"
NANO = f"{FID}.gemini-nano-banana-2.1"
FUTURE = f"{FID}.gemini-4-flash-image"  # only via MODEL_ADDITIONAL
IMAGEGEN = f"{FID}.gemini-2.0-flash-imagegen"  # not detected as image model
VEO = f"{FID}.veo-3.1-generate-preview"
EMBEDDING = f"{FID}.text-embedding-004"
SEARCH_FILTER = "google_search_tool"
# grounding chunk URI returned by mocks/mock_gemini.py
GROUNDING_URI = "https://example.com/a"
TITLE_JSON = '{"title": "Mock Title"}'
VIDEO_PROMPT = "A cat playing piano"
# Native function calling off: no built-in tools reach the image model.
LEGACY = {"function_calling": "legacy"}
# usage the pipe reports (total = prompt + completion) for the mock's answers
USAGE_TEXT = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
USAGE_STREAM = {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
USAGE_IMAGE = {"prompt_tokens": 12, "completion_tokens": 1290, "total_tokens": 1302}
SECOND_USER = "e2e-gemini-user@example.com"

# Valves the groups change; restored after every group. MODEL_CACHE_TTL=0 so
# model list valves (whitelist, additional) apply at once on every version.
DEFAULTS = dict(
    STREAMING_ENABLED=True,
    INCLUDE_THOUGHTS=True,
    STRIP_THINKING_FROM_HISTORY=True,
    THINKING_BUDGET=-1,
    THINKING_LEVEL="",
    IMAGE_GENERATION_ASPECT_RATIO="default",
    IMAGE_GENERATION_RESOLUTION="default",
    IMAGE_GENERATION_MODELS="",
    IMAGE_ENABLE_OPTIMIZATION=True,
    IMAGE_PNG_COMPRESSION_THRESHOLD_MB=0.5,
    IMAGE_HISTORY_MAX_REFERENCES=5,
    IMAGE_ADD_LABELS=True,
    IMAGE_DEDUP_HISTORY=True,
    IMAGE_HISTORY_FIRST=True,
    MODEL_ADDITIONAL="",
    MODEL_WHITELIST="",
    MODEL_CACHE_TTL=0,
    DEFAULT_SYSTEM_PROMPT="",
    USE_PERMISSIVE_SAFETY=False,
    ENABLE_FORWARD_USER_INFO_HEADERS=False,
    API_VERSION="v1alpha",
    RETRY_COUNT=2,
    VIDEO_GENERATION_NEGATIVE_PROMPT="",
)
# WARNING lines of the pipe that mean a lost event, image, video or answer part
# (the pipe swallows these failures, so they never show as ERROR).
WARNINGS = tuple(
    ("function_gemini", text)
    for text in (
        "Failed to emit",
        "upload failed",
        "Error processing content part",
        "Failed to access content parts",
        "Unexpected error configuring ThinkingConfig",
        "download failed",
        "could not obtain video bytes",
        "Polling error",
        "Skipping image (parse failure)",
    )
)
# Valves and UserValves of google_gemini.py 1.16.1 with their defaults (valve
# names are the public API; the harness container sets no GOOGLE_* variables,
# so the defaults are the os.getenv fallbacks)
VALVES_1_16_1 = {
    "BASE_URL": "https://generativelanguage.googleapis.com/",
    "GOOGLE_API_KEY": "",
    "API_VERSION": "v1alpha",
    "STREAMING_ENABLED": True,
    "INCLUDE_THOUGHTS": True,
    "STRIP_THINKING_FROM_HISTORY": True,
    "THINKING_BUDGET": -1,
    "THINKING_LEVEL": "",
    "USE_VERTEX_AI": False,
    "VERTEX_PROJECT": None,
    "VERTEX_LOCATION": "global",
    "VERTEX_AI_RAG_STORE": None,
    "USE_PERMISSIVE_SAFETY": False,
    "MODEL_CACHE_TTL": 600,
    "RETRY_COUNT": 2,
    "DEFAULT_SYSTEM_PROMPT": "",
    "ENABLE_FORWARD_USER_INFO_HEADERS": False,
    "MODEL_ADDITIONAL": "",
    "MODEL_WHITELIST": "",
    "USE_ENTERPRISE_WEB_SEARCH": False,
    "IMAGE_GENERATION_ASPECT_RATIO": "default",
    "IMAGE_GENERATION_RESOLUTION": "default",
    "IMAGE_MAX_SIZE_MB": 15.0,
    "IMAGE_MAX_DIMENSION": 2048,
    "IMAGE_COMPRESSION_QUALITY": 85,
    "IMAGE_ENABLE_OPTIMIZATION": True,
    "IMAGE_PNG_COMPRESSION_THRESHOLD_MB": 0.5,
    "IMAGE_HISTORY_MAX_REFERENCES": 5,
    "IMAGE_ADD_LABELS": True,
    "IMAGE_DEDUP_HISTORY": True,
    "IMAGE_HISTORY_FIRST": True,
    "VIDEO_GENERATION_ASPECT_RATIO": "default",
    "VIDEO_GENERATION_RESOLUTION": "default",
    "VIDEO_GENERATION_DURATION": "default",
    "VIDEO_GENERATION_NEGATIVE_PROMPT": "",
    "VIDEO_GENERATION_PERSON_GENERATION": "default",
    "VIDEO_GENERATION_ENHANCE_PROMPT": True,
    "VIDEO_POLL_INTERVAL": 10,
    "VIDEO_POLL_TIMEOUT": 600,
}
USER_VALVES_1_16_1 = {
    "IMAGE_GENERATION_ASPECT_RATIO": "default",
    "IMAGE_GENERATION_RESOLUTION": "default",
    "VIDEO_GENERATION_ASPECT_RATIO": "default",
    "VIDEO_GENERATION_RESOLUTION": "default",
    "VIDEO_GENERATION_DURATION": "default",
}


# ----------------------------------------------------------------- helpers
def _png(r: int, g: int, b: int, size: int = 1, noise: bool = False) -> str:
    """Base64 of an RGB PNG (same generator as mocks/mock_gemini.py for 1x1)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rng = random.Random(1)
    rows = b"".join(
        b"\x00"
        + (
            bytes(rng.randrange(256) for _ in range(size * 3))
            if noise
            else bytes((r, g, b)) * size
        )
        for _ in range(size)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


# images of the mock (mocks/mock_gemini.py) and of common.PNG_B64
THOUGHT_PNG_1 = _png(0, 0, 255)
THOUGHT_PNG_2 = _png(0, 255, 0)
FINAL_PNG = _png(255, 255, 0)
RED_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJ"
    "RU5ErkJggg=="
)
PNG_NAMES = {
    THOUGHT_PNG_1: "thought1",
    THOUGHT_PNG_2: "thought2",
    FINAL_PNG: "final",
    RED_PNG: "red",
}


def _gen(entry: dict) -> bool:
    return entry.get("action") in ("generateContent", "streamGenerateContent")


def _actions(entries: list) -> list:
    return [e.get("action") for e in entries]


def _camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(word[:1].upper() + word[1:] for word in rest)


def _norm(value):
    """camelCase every dict key (google-genai sends some nested keys in snake_case)."""
    if isinstance(value, dict):
        return {_camel(k): _norm(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_norm(v) for v in value]
    return value


def _body(entry: dict) -> dict:
    body = (entry or {}).get("body")
    return _norm(body) if isinstance(body, dict) else {}


def _gc(entry: dict) -> dict:
    return _body(entry).get("generationConfig") or {}


def _last_text(entry: dict) -> str:
    for content in reversed(_body(entry).get("contents") or []):
        if content.get("role", "user") == "user":
            return " ".join(p.get("text", "") for p in content.get("parts") or [])
    return ""


def _strings(value) -> list:
    """All string values nested anywhere in ``value`` (dicts and lists)."""
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return [value] if isinstance(value, str) else []


def _usage(usage) -> dict:
    """The usage fields the pipe reports (Open WebUI adds input/output_tokens)."""
    usage = usage or {}
    return {
        k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _status_closed(chat) -> bool:
    """No spinner left behind: every status action that was started (done=False)
    ends with an entry with done=True."""
    started, last = set(), {}
    for entry in chat.status_history:
        action = entry.get("action")
        if entry.get("done") is False:
            started.add(action)
        last[action] = entry
    return all(last[action].get("done") is True for action in started)


def _status_ok(chat) -> bool:
    """Statuses of a successful answer: all closed, none reports a failure or a
    stop (the pipe closes a status left running with 'failed' / 'stopped')."""
    return _status_closed(chat) and not _bad_statuses(chat)


def _bad_statuses(chat) -> list:
    return [
        d
        for d in chat.status_descriptions
        if d and ("fail" in d.lower() or "stopped" in d.lower())
    ]


def _status_report(chat) -> str:
    return (
        f"statuses_closed={_status_closed(chat)} failed_statuses={_bad_statuses(chat)}"
    )


def _last_status(chat, action: str) -> dict:
    entries = [s for s in chat.status_history if s.get("action") == action]
    return entries[-1] if entries else {}


def _image_url(content: str):
    return f"data:image/png;base64,{content}"


def _b64(data) -> str:
    """Standard base64 (google-genai sends inline data URL-safe encoded)."""
    return str(data or "").replace("-", "+").replace("_", "/")


async def _fetch_b64(t: Suite, url: str) -> str:
    """Base64 of an Open WebUI file (empty when it cannot be fetched)."""
    path = url[len(t.owui.base) :] if url.startswith(t.owui.base) else url
    r = await t.owui.request("GET", path)
    return base64.b64encode(r.content).decode() if r.status_code == 200 else ""


async def _image_files(t: Suite, chat, expected: list = ()) -> tuple:
    """(sorted names of the saved image files, all URLs are /api/v1/files/,
    report text) of a browser-path chat.

    The report carries ``image_files=<n>`` and ``interim_thought_images=<n>``
    (thought images attached although not expected, #181).
    """
    files = [f for f in chat.files if f.get("type") == "image"]
    names, urls_ok = [], True
    for entry in files:
        url = entry.get("url") or ""
        urls_ok = urls_ok and url.startswith("/api/v1/files/")
        data = await _fetch_b64(t, url) if url.startswith("/api/") else ""
        names.append(PNG_NAMES.get(data, "other" if data else "unreadable"))
    wanted = [PNG_NAMES[e] for e in expected]
    interim = sum(1 for n in names if n.startswith("thought") and n not in wanted)
    report = (
        f"image_files={len(files)} images={names} interim_thought_images={interim} "
        f"file_urls_ok={urls_ok}"
    )
    return sorted(names), urls_ok, report


async def _set(t: Suite, **valves) -> None:
    await t.owui.update_valves(FID, **valves)


async def _filter_ready(t: Suite) -> bool:
    info = await t.owui.function(SEARCH_FILTER)
    return bool(info and info.get("is_active"))


# --------------------------------------------------------------------- run
async def run(t: Suite) -> None:
    mock = t.mock("gemini")
    await mock.reset()
    if not await t.install(FID, PATH, "Google Gemini"):
        return
    t.fail_on_warnings(*WARNINGS)
    await _set(
        t,
        GOOGLE_API_KEY=KEY,
        BASE_URL=mock.url + "/",
        VIDEO_POLL_INTERVAL=5,
        **DEFAULTS,
    )
    stored = (await t.owui.get_valves(FID)).get("GOOGLE_API_KEY", "")
    t.check(
        "valves.encrypted",
        "GOOGLE_API_KEY is stored encrypted",
        stored.startswith("encrypted:") and KEY not in stored,
        f"stored={short(stored, 40)}",
    )
    if any(t.selected(g) for g in ("tasks", "imgtools", "grounding")):
        await t.install(
            SEARCH_FILTER,
            "filters/google_search_tool.py",
            "Google Search Tool",
            "grounding.load",
        )

    # API path first: no websocket session of this user may be open (the API
    # stream "Error during streaming" bug fixed in 1.17.0 only showed then).
    for group, scenario in (
        ("models", models),
        ("api", api),
        ("thinking", thinking),
        ("browser", browser),
        ("tasks", tasks),
        ("image", image),
        ("images", images),
        ("nano", nano),
        ("imgvalve", imgvalve),
        ("imgconfig", imgconfig),
        ("imgtools", imgtools),
        ("nostream", nostream),
        ("video", video),
        ("grounding", grounding),
        ("vertex", vertex),
        ("errors", errors),
        ("retry", retry),
        ("status", status),
        ("valves", valves),
        ("concurrency", concurrency),
        ("streamimg", streamimg),
    ):
        if t.selected(group):
            try:
                await scenario(t, mock)
            finally:
                await _set(t, **DEFAULTS)
    t.assert_no_secrets(KEY)
    t.scan_log()


# ------------------------------------------------------------------ models
async def models(t: Suite, mock) -> None:
    listed = {m["id"]: m.get("name", "") for m in await t.owui.models()}
    wanted = {TEXT, PRO3, IMAGE_PREVIEW, IMAGE_GA, IMAGE_LITE, IMAGE_25, NANO, VEO}
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
    )
    t.check(
        "models.video",
        "Veo model is marked as video model",
        "🎬" in listed.get(VEO, ""),
        f"name={listed.get(VEO)!r}",
    )


# --------------------------------------------------------------------- api
async def api(t: Suite, mock) -> None:
    await mock.reset()
    r = await t.owui.chat(TEXT, "Hello Gemini", stream=False)
    req = await mock.last(_gen)
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
        "API non-stream: usage reported (prompt / completion / total)",
        _usage(r.usage) == USAGE_TEXT,
        r.brief(),
    )
    t.check(
        "api.nonstream.request",
        "upstream got generateContent with the decrypted API key",
        req.get("action") == "generateContent"
        and req.get("headers", {}).get("x-goog-api-key") == KEY,
        f"action={req.get('action')} key={req.get('headers', {}).get('x-goog-api-key')}",
    )

    await mock.reset()
    r = await t.owui.chat(TEXT, "Hello Gemini", stream=True)
    req = await mock.last(_gen)
    await t.log.settle(0.5)
    t.check(
        "api.stream",
        "API stream without websocket session: answer streamed, thinking in <details>",
        r.status == 200
        and r.done
        and "Hello from mock (stream)." in r.content
        and "<details>" in r.content
        and "Mock pondering." in r.content
        and "Error" not in r.content
        and not r.errors,
        r.brief() + f" upstream={req.get('action')}",
    )
    t.check(
        "api.stream.usage",
        "API stream: usage chunk forwarded (prompt / completion / total)",
        _usage(r.usage) == USAGE_STREAM,
        r.brief(),
    )


# ---------------------------------------------------------------- thinking
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

    def replayed(entry: dict) -> list:
        return [
            part.get("text", "")
            for content in _body(entry).get("contents") or []
            if content.get("role") == "model"
            for part in content.get("parts") or []
        ]

    await mock.reset()
    r = await t.owui.chat(TEXT, history, stream=False)
    parts = replayed(await mock.last(_gen))
    t.check(
        "thinking.strip",
        "thinking summary of earlier turns is not replayed to Gemini (#176)",
        r.status == 200
        and any("First answer." in p for p in parts)
        and not any("SECRET-THOUGHT" in p for p in parts),
        f"model parts sent upstream={parts}",
    )

    await _set(t, THINKING_BUDGET=0)
    await mock.reset()
    await t.owui.chat(TEXT, "Hi", stream=False)
    think0 = _gc(await mock.last(_gen)).get("thinkingConfig") or {}
    await _set(t, THINKING_BUDGET=-1)
    await mock.reset()
    await t.owui.chat(TEXT, "Hi", stream=False, thinking_budget=1024)
    think1 = _gc(await mock.last(_gen)).get("thinkingConfig") or {}
    await mock.reset()
    await t.owui.chat(TEXT, "Hi", stream=False)
    think2 = _gc(await mock.last(_gen)).get("thinkingConfig") or {}
    t.check(
        "thinking.budget",
        "THINKING_BUDGET=0 -> 0; body thinking_budget=1024 -> 1024; default -> -1",
        think0.get("thinkingBudget") == 0
        and think1.get("thinkingBudget") == 1024
        and think2.get("thinkingBudget") == -1
        and "thinkingLevel" not in think0,
        f"budget0={think0} body1024={think1} default={think2}",
    )

    await mock.reset()
    await t.owui.chat(PRO3, "Hi", stream=False, reasoning_effort="medium")
    think_pro = _gc(await mock.last(_gen)).get("thinkingConfig") or {}
    await _set(t, THINKING_LEVEL="low")
    await mock.reset()
    await t.owui.chat(IMAGE_PREVIEW, "Draw", stream=False)
    think_img = _gc(await mock.last(_gen)).get("thinkingConfig") or {}
    t.check(
        "thinking.level",
        "gemini-3-pro reasoning_effort=medium -> HIGH; 3.1-flash-image "
        "THINKING_LEVEL=low -> MINIMAL; no thinkingBudget with a level",
        str(think_pro.get("thinkingLevel", "")).upper() == "HIGH"
        and "thinkingBudget" not in think_pro
        and str(think_img.get("thinkingLevel", "")).upper() == "MINIMAL"
        and "thinkingBudget" not in think_img,
        f"pro3={think_pro} image={think_img}",
    )

    await _set(t, THINKING_LEVEL="", INCLUDE_THOUGHTS=False)
    await mock.reset()
    r = await t.owui.chat(TEXT, "Hi", stream=False)
    think = _gc(await mock.last(_gen)).get("thinkingConfig") or {}
    t.check(
        "thinking.include-off",
        "INCLUDE_THOUGHTS=false: no includeThoughts upstream, no <details> in the answer",
        r.status == 200
        and not think.get("includeThoughts")
        and "<details>" not in r.content
        and "Hello from mock (non-stream)." in r.content,
        f"thinkingConfig={think} {r.brief()}",
    )

    await _set(t, INCLUDE_THOUGHTS=True, STRIP_THINKING_FROM_HISTORY=False)
    await mock.reset()
    await t.owui.chat(TEXT, history, stream=False)
    parts = replayed(await mock.last(_gen))
    t.check(
        "thinking.strip-off",
        "STRIP_THINKING_FROM_HISTORY=false replays the summary unchanged",
        any("SECRET-THOUGHT" in p for p in parts),
        f"model parts sent upstream={parts}",
    )


# ----------------------------------------------------------------- browser
async def browser(t: Suite, mock) -> None:
    async with t.browser() as b:
        for stream, answer, thought, usage in (
            (True, "Hello from mock (stream).", "Mock pondering.", USAGE_STREAM),
            (False, "Hello from mock (non-stream).", "Mock thinking.", USAGE_TEXT),
        ):
            await mock.reset()
            c = await b.chat(TEXT, "Hello from the browser", stream=stream)
            name = "stream" if stream else "nonstream"
            t.check(
                f"browser.{name}",
                f"browser path stream={stream}: answer + <details> thinking saved",
                c.done
                and answer in c.content
                and "<details>" in c.content
                and thought in c.content
                and not c.error,
                c.brief(),
            )
            t.check(
                f"browser.{name}.usage",
                f"browser path stream={stream}: usage saved (prompt / completion / total)",
                _usage(c.usage) == usage,
                c.brief() + f" events={c.event_types}",
            )


# ------------------------------------------------------------------- tasks
async def tasks(t: Suite, mock) -> None:
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]
    await mock.reset()
    status, answer, raw = await t.owui.title_task(TEXT, messages)
    detail = f"HTTP {status} answer={short(answer)} raw={short(raw, 200)}"
    t.check(
        "tasks.title",
        "title task via /api/v1/tasks/title/completions (no event emitter)",
        status == 200 and answer and "Mock Title" in answer,
        detail,
    )
    t.check(
        "tasks.no-details",
        "title task answer is the JSON only (no <details> thinking summary)",
        status == 200 and (answer or "").strip() == TITLE_JSON,
        detail,
    )
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
    )

    if not await _filter_ready(t):
        return
    await t.owui.upsert_model(TEXT, "Gemini 2.5 Flash", [SEARCH_FILTER])
    try:
        async with t.browser() as b:
            await mock.reset()
            c = await b.chat(
                TEXT,
                "Search the web",
                features={"web_search": True},
                background_tasks={"title_generation": True, "tags_generation": True},
                wait_title=True,
            )
            for _ in range(20):  # the tags task may still run
                reqs = await mock.requests(_gen)
                if sum("### Task:" in _last_text(e) for e in reqs) >= 2:
                    break
                await asyncio.sleep(0.5)
    finally:
        await t.owui.delete_model(TEXT)
    task_reqs = [e for e in reqs if "### Task:" in _last_text(e)]
    chat_reqs = [e for e in reqs if "### Task:" not in _last_text(e)]
    t.check(
        "tasks.no-grounding",
        "background tasks of a web_search chat are sent without grounding tools",
        c.done
        and c.title == "Mock Title"
        and chat_reqs
        and "googleSearch" in (chat_reqs[0].get("tool_kinds") or [])
        and len(task_reqs) >= 2
        and all(
            not {"googleSearch", "urlContext"} & set(e.get("tool_kinds") or [])
            for e in task_reqs
        ),
        f"title={c.title!r} chat_tools={[e.get('tool_kinds') for e in chat_reqs]} "
        f"task_tools={[e.get('tool_kinds') for e in task_reqs]}",
    )


# ------------------------------------------------------------------- image
async def image(t: Suite, mock) -> None:
    await mock.reset()
    r = await t.owui.chat(IMAGE_PREVIEW, "Draw a red pixel", stream=True)
    reqs = await mock.requests(_gen)
    modalities = _gc(reqs[-1]).get("responseModalities") if reqs else None
    t.check(
        "image.api",
        "image model via API (stream requested): forced non-stream, IMAGE modality",
        r.status == 200
        and _actions(reqs) == ["generateContent"]
        and modalities
        and "IMAGE" in modalities
        and "Here is your image." in r.content
        and not r.errors,
        r.brief() + f" upstream={_actions(reqs)} modalities={modalities}",
    )
    async with t.browser() as b:
        for model, sid in (
            (IMAGE_PREVIEW, "image.browser-preview"),
            (IMAGE_GA, "image.browser-ga"),
        ):
            await mock.reset()
            c = await b.chat(model, "Draw a red pixel", stream=True)
            reqs = await mock.requests(_gen)
            names, urls_ok, report = await _image_files(t, c, [FINAL_PNG])
            t.check(
                sid,
                f"{model.split('.', 1)[1]} (browser, built-in tools): forced "
                "non-stream, one image file with the final image, text, usage, "
                "statuses closed without failure",
                c.done
                and _actions(reqs) == ["generateContent"]
                and names == ["final"]
                and urls_ok
                and "Here is your image." in c.content
                and "![" not in c.content
                and _usage(c.usage) == USAGE_IMAGE
                and _status_ok(c),
                c.brief()
                + f" model={model.split('.', 1)[1]} upstream={_actions(reqs)} {report}"
                f" {_status_report(c)}",
            )


# ------------------------------------------------------------------ images
async def images(t: Suite, mock) -> None:
    cases = (
        ("thought-skip", "Draw a red pixel", [FINAL_PNG], "Here is your image."),
        ("thought-fallback", "only-thought-images please", [THOUGHT_PNG_2], None),
        ("safety-no-fallback", "image-safety please", [], None),
        ("dedup", "duplicate-image please", [FINAL_PNG], "Here is your image."),
        ("two-final", "two-final-images please", [FINAL_PNG, RED_PNG], "Here are"),
    )
    async with t.browser() as b:
        for name, prompt, expected, text in cases:
            await mock.reset()
            mark = t.mark()
            c = await b.chat(IMAGE_PREVIEW, prompt, stream=True, params=LEGACY)
            names, urls_ok, report = await _image_files(t, c, expected)
            await t.log.settle(0.5)
            fallback_logged = bool(
                t.log.lines(mark, "attaching the last thought image")
            )
            ok = (
                c.done
                and names == sorted(PNG_NAMES[e] for e in expected)
                and urls_ok
                and "![" not in c.content
                and (text is None or text in c.content)
                and _usage(c.usage) == USAGE_IMAGE
                and _status_ok(c)
            )
            if name == "thought-fallback":
                ok = ok and fallback_logged
            t.check(
                f"images.{name}",
                f"{prompt!r}: image files {[PNG_NAMES[e] for e in expected]} "
                "(no interim thought image), text, usage, statuses closed",
                ok,
                c.brief() + f" {report} {_status_report(c)}"
                f" fallback_logged={fallback_logged}",
            )

    await mock.reset()
    r = await t.owui.chat(IMAGE_PREVIEW, "Draw a red pixel", stream=False)
    links = re.findall(r"!\[[^\]]*\]\((/api/v1/files/[^)\s]+)\)", r.content)
    linked = [PNG_NAMES.get(await _fetch_b64(t, url), "other") for url in links]
    t.check(
        "images.api-delivery",
        "API client (no chat message): the answer links the uploaded final image",
        r.status == 200
        and "Here is your image." in r.content
        and linked == ["final"]
        and "data:image" not in r.content,
        r.brief() + f" answer_text={'Here is your image.' in r.content}"
        f" image_links={len(links)} linked={linked}",
    )

    await images_history(t, mock)
    await images_optimization(t, mock)


async def images_history(t: Suite, mock) -> None:
    """Images of earlier turns: limit, dedup, labels and order."""
    a, b, c = _png(10, 10, 10), _png(20, 20, 20), _png(30, 30, 30)
    labels = {a: "A", b: "B", c: "C"}

    def image(data: str) -> dict:
        return {"type": "image_url", "image_url": {"url": _image_url(data)}}

    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First"},
                image(a),
                image(a),
                image(c),
            ],
        },
        {"role": "assistant", "content": "Nice images."},
        {"role": "user", "content": [{"type": "text", "text": "Combine"}, image(b)]},
    ]

    async def sent() -> list:
        contents = _body(await mock.last(_gen)).get("contents") or []
        parts = contents[-1].get("parts") or [] if contents else []
        return [
            p["text"]
            if "text" in p
            else labels.get(_b64((p.get("inlineData") or {}).get("data")), "?")
            for p in parts
        ]

    await _set(t, IMAGE_HISTORY_MAX_REFERENCES=2)
    await mock.reset()
    r1 = await t.owui.chat(IMAGE_PREVIEW, history, stream=False)
    first = await sent()
    await _set(t, IMAGE_ADD_LABELS=False, IMAGE_HISTORY_FIRST=False)
    await mock.reset()
    r2 = await t.owui.chat(IMAGE_PREVIEW, history, stream=False)
    second = await sent()
    t.check(
        "images.history",
        "image history: duplicates dropped, IMAGE_HISTORY_MAX_REFERENCES=2, "
        "[Image N] labels, history first; IMAGE_ADD_LABELS / IMAGE_HISTORY_FIRST off",
        r1.status == 200
        and r2.status == 200
        and first == ["Combine", "[Image 1]", "A", "[Image 2]", "C"]
        and second == ["Combine", "B", "A"],
        f"labels+history_first={first} plain+current_first={second}",
    )


async def images_optimization(t: Suite, mock) -> None:
    """IMAGE_ENABLE_OPTIMIZATION: input PNGs above the threshold become JPEG."""
    noisy = _png(0, 0, 0, size=64, noise=True)
    message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": _image_url(noisy)}},
            ],
        }
    ]

    async def inline() -> list:
        out = []
        for content in _body(await mock.last(_gen)).get("contents") or []:
            for part in content.get("parts") or []:
                data = part.get("inlineData")
                if data:
                    out.append((data.get("mimeType"), _b64(data.get("data"))))
        return out

    await _set(t, IMAGE_PNG_COMPRESSION_THRESHOLD_MB=0.001)
    sent = {}
    for model in (TEXT, IMAGE_PREVIEW):
        await mock.reset()
        await t.owui.chat(model, message, stream=False)
        sent[model] = await inline()
    await _set(t, IMAGE_ENABLE_OPTIMIZATION=False)
    await mock.reset()
    await t.owui.chat(TEXT, message, stream=False)
    plain = await inline()

    def summary(items: list) -> list:
        return [(mime, len(data or "")) for mime, data in items]

    t.check(
        "images.optimization",
        "PNG above IMAGE_PNG_COMPRESSION_THRESHOLD_MB sent as smaller JPEG (text and "
        "image model); IMAGE_ENABLE_OPTIMIZATION=false sends it unchanged",
        all(
            len(items) == 1
            and items[0][0] == "image/jpeg"
            and len(items[0][1] or "") < len(noisy)
            for items in sent.values()
        )
        and plain == [("image/png", noisy)],
        f"png={len(noisy)} optimized={ {k.split('.', 1)[1]: summary(v) for k, v in sent.items()} } "
        f"disabled={summary(plain)}",
    )


# -------------------------------------------------------------------- nano
async def nano(t: Suite, mock) -> None:
    listed = {m["id"]: m.get("name", "") for m in await t.owui.models()}
    t.check(
        "nano.listed",
        "gemini-nano-banana-2.1 is marked as image model",
        "🎨" in listed.get(NANO, ""),
        f"name={listed.get(NANO)!r}",
    )
    await mock.reset()
    r = await t.owui.chat(NANO, "Draw a banana", stream=True, reasoning_effort="low")
    reqs = await mock.requests(_gen)
    gc = _gc(reqs[-1]) if reqs else {}
    think = gc.get("thinkingConfig") or {}
    detail = (
        r.brief() + f" model=gemini-nano-banana-2.1 upstream={_actions(reqs)}"
        f" modalities={gc.get('responseModalities')} thinking={think}"
    )
    t.check(
        "nano.request",
        "nano-banana via API (stream requested): forced non-stream, IMAGE modality, "
        "reasoning_effort=low -> thinkingLevel MEDIUM, no thinkingBudget",
        r.status == 200
        and _actions(reqs) == ["generateContent"]
        and "IMAGE" in (gc.get("responseModalities") or [])
        and str(think.get("thinkingLevel", "")).upper() == "MEDIUM"
        and "thinkingBudget" not in think
        and "Here is your image." in r.content,
        detail,
    )
    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(NANO, "Draw a banana", stream=True)
        reqs = await mock.requests(_gen)
    names, urls_ok, report = await _image_files(t, c, [FINAL_PNG])
    t.check(
        "nano.browser",
        "nano-banana browser path (built-in tools): forced non-stream, one image file",
        c.done
        and _actions(reqs) == ["generateContent"]
        and names == ["final"]
        and urls_ok
        and "Here is your image." in c.content,
        c.brief() + f" model=gemini-nano-banana-2.1 upstream={_actions(reqs)} {report}",
    )


# ---------------------------------------------------------------- imgvalve
async def imgvalve(t: Suite, mock) -> None:
    await _set(
        t,
        MODEL_ADDITIONAL="gemini-4-flash-image",
        IMAGE_GENERATION_ASPECT_RATIO="16:9",
        THINKING_LEVEL="high",
    )
    await mock.reset()
    r = await t.owui.chat(FUTURE, "Draw a cat", stream=True)
    reqs = await mock.requests(_gen)
    gc = _gc(reqs[-1]) if reqs else {}
    think = gc.get("thinkingConfig") or {}
    detail = (
        r.brief() + f" model=gemini-4-flash-image upstream={_actions(reqs)}"
        f" modalities={gc.get('responseModalities')}"
        f" imageConfig={gc.get('imageConfig')} thinking={think}"
    )
    t.check(
        "imgvalve.unlisted",
        "gemini-4-flash-image (MODEL_ADDITIONAL only): image model by its name, "
        "no imageConfig, thinkingBudget instead of a level",
        r.status == 200
        and _actions(reqs) == ["generateContent"]
        and "IMAGE" in (gc.get("responseModalities") or [])
        and "imageConfig" not in gc
        and "thinkingBudget" in think
        and "thinkingLevel" not in think,
        detail,
    )
    await _set(t, IMAGE_GENERATION_MODELS="gemini-4-flash-image")
    await mock.reset()
    r = await t.owui.chat(FUTURE, "Draw a cat", stream=True)
    reqs = await mock.requests(_gen)
    gc = _gc(reqs[-1]) if reqs else {}
    think = gc.get("thinkingConfig") or {}
    detail = (
        r.brief() + f" model=gemini-4-flash-image upstream={_actions(reqs)}"
        f" modalities={gc.get('responseModalities')}"
        f" imageConfig={gc.get('imageConfig')} thinking={think}"
    )
    t.check(
        "imgvalve.listed",
        "IMAGE_GENERATION_MODELS lists it: imageConfig 16:9 + thinkingLevel HIGH",
        r.status == 200
        and _actions(reqs) == ["generateContent"]
        and (gc.get("imageConfig") or {}).get("aspectRatio") == "16:9"
        and str(think.get("thinkingLevel", "")).upper() == "HIGH"
        and "thinkingBudget" not in think,
        detail,
    )


# --------------------------------------------------------------- imgconfig
async def imgconfig(t: Suite, mock) -> None:
    await _set(
        t, IMAGE_GENERATION_ASPECT_RATIO="16:9", IMAGE_GENERATION_RESOLUTION="4K"
    )
    for model, want in (
        (IMAGE_PREVIEW, {"aspectRatio": "16:9", "imageSize": "4K"}),
        (IMAGE_LITE, {"aspectRatio": "16:9"}),
        (IMAGE_25, None),
    ):
        name = model.split(".", 1)[1]
        await mock.reset()
        r = await t.owui.chat(model, "Draw", stream=False)
        req = await mock.last(_gen)
        got = _gc(req).get("imageConfig")
        t.check(
            f"imgconfig.{name}",
            f"{name}: imageConfig {want} (16:9 / 4K valves; lite only 1K)",
            r.status == 200
            and got == want
            and req.get("action") == "generateContent"
            and "IMAGE" in (_gc(req).get("responseModalities") or [])
            and "Here is your image." in r.content,
            r.brief() + f" model={name} imageConfig={got} action={req.get('action')}",
        )
    user_path = f"/api/v1/functions/id/{FID}/valves/user/update"
    status, _ = await t.owui.api(
        "POST", user_path, {"IMAGE_GENERATION_ASPECT_RATIO": "1:1"}
    )
    try:
        await mock.reset()
        r = await t.owui.chat(IMAGE_PREVIEW, "Draw", stream=False)
        got = _gc(await mock.last(_gen)).get("imageConfig") or {}
    finally:
        await t.owui.api(
            "POST", user_path, {"IMAGE_GENERATION_ASPECT_RATIO": "default"}
        )
    t.check(
        "imgconfig.user-valve",
        "user valve IMAGE_GENERATION_ASPECT_RATIO=1:1 overrides the admin valve",
        status == 200 and r.status == 200 and got.get("aspectRatio") == "1:1",
        f"user valves HTTP {status} imageConfig={got} {r.brief()}",
    )
    await mock.reset()
    r = await t.owui.chat(IMAGE_PREVIEW, "Draw", stream=False, aspect_ratio="9:16")
    got = _gc(await mock.last(_gen)).get("imageConfig") or {}
    t.check(
        "imgconfig.body",
        "request body aspect_ratio=9:16 overrides the valves",
        r.status == 200 and got.get("aspectRatio") == "9:16",
        f"imageConfig={got} {r.brief()}",
    )


# ---------------------------------------------------------------- imgtools
async def imgtools(t: Suite, mock) -> None:
    if not await _filter_ready(t):
        return
    for model, want in (
        (IMAGE_GA, ["googleSearch"]),
        (IMAGE_25, []),
        (IMAGE_LITE, []),
        (TEXT, ["googleSearch", "urlContext"]),
    ):
        name = model.split(".", 1)[1]
        await t.owui.upsert_model(model, name, [SEARCH_FILTER])
        try:
            async with t.browser() as b:
                await mock.reset()
                c = await b.chat(model, "Search the web", features={"web_search": True})
                req = await mock.last(_gen)
        finally:
            await t.owui.delete_model(model)
        sent = req.get("tool_kinds") or []
        kinds = [k for k in sent if k != "functionDeclarations"]
        image_model = model != TEXT
        names, _, report = await _image_files(t, c, [FINAL_PNG])
        t.check(
            f"imgtools.{name}",
            f"{name} + web_search (built-in tools): grounding tools {want}"
            + (", no native tools, image saved" if image_model else ""),
            c.done
            and not c.error
            and kinds == want
            and (
                not image_model
                or ("functionDeclarations" not in sent and names == ["final"])
            ),
            c.brief() + f" tool_kinds={sent} {report}",
        )


# ---------------------------------------------------------------- nostream
async def nostream(t: Suite, mock) -> None:
    await _set(t, STREAMING_ENABLED=False)
    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(
            TEXT,
            "Hello without streaming",
            stream=True,
            background_tasks={"follow_up_generation": True},
        )
        reqs = await mock.requests(_gen)
        await asyncio.sleep(3)  # follow-ups are saved into the same message
        c = await b.reload(c)
    t.check(
        "nostream.browser",
        "STREAMING_ENABLED=false, browser stream=true: answer + usage saved (#170)",
        c.done
        and "Hello from mock (non-stream)." in c.content
        and _usage(c.usage) == USAGE_TEXT
        and reqs
        and reqs[0].get("action") == "generateContent",
        c.brief() + f" upstream={_actions(reqs)}",
    )
    await mock.reset()
    r = await t.owui.chat(TEXT, "Hello", stream=True)
    t.check(
        "nostream.api",
        "STREAMING_ENABLED=false, API stream=true: SSE answer, usage, [DONE]",
        r.status == 200
        and "Hello from mock (non-stream)." in r.content
        and _usage(r.usage) == USAGE_TEXT
        and r.done
        and not r.errors,
        r.brief() + f" done={r.done}",
    )


# ------------------------------------------------------------------- video
async def video(t: Suite, mock) -> None:
    await _set(t, VIDEO_GENERATION_NEGATIVE_PROMPT="blurry")
    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(VEO, VIDEO_PROMPT, stream=True, wait=180)
        reqs = await mock.requests()
    actions = [e.get("action") or e.get("path") for e in reqs]
    videos = [f for f in c.files if "video" in (f.get("content_type") or "")]
    started = [e for e in reqs if e.get("action") == "predictLongRunning"]
    body = _body(started[0]) if started else {}
    instance = (body.get("instances") or [{}])[0]
    params = body.get("parameters") or {}
    last = _last_status(c, "video_generation")
    t.check(
        "video.browser",
        "Veo browser path: operation polled, video downloaded, one video file and "
        "the answer text saved, final status done",
        c.done
        and "predictLongRunning" in actions
        and any("/operations/" in str(a) for a in actions)
        and "download" in actions
        and len(videos) == 1
        and "Generated video attached." in c.content
        and last.get("done") is True
        and _status_ok(c),
        c.brief() + f" saved_text={c.content[:60]!r} videos={len(videos)}"
        f" upstream={actions} last_status={last} {_status_report(c)}",
    )
    t.check(
        "video.request",
        "Veo request: prompt in instances[0], VIDEO_GENERATION_NEGATIVE_PROMPT as "
        "parameters.negativePrompt",
        instance.get("prompt") == VIDEO_PROMPT
        and params.get("negativePrompt") == "blurry",
        f"instance={short(instance)} parameters={short(params)}",
    )

    await mock.reset()
    r = await t.owui.chat(
        VEO,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Animate this"},
                    {"type": "image_url", "image_url": {"url": _image_url(FINAL_PNG)}},
                ],
            }
        ],
        stream=False,
        timeout=180,
    )
    started = await mock.requests(lambda e: e.get("action") == "predictLongRunning")
    instance = (_body(started[0]).get("instances") or [{}])[0] if started else {}
    picture = instance.get("image") or {}
    t.check(
        "video.image-to-video",
        "Veo image-to-video (API): the attached image is sent as instances[0].image",
        r.status == 200
        and instance.get("prompt") == "Animate this"
        and _b64(picture.get("bytesBase64Encoded") or picture.get("imageBytes"))
        == FINAL_PNG
        and "video" in r.content.lower(),
        r.brief() + f" image_keys={list(picture)}",
    )


# --------------------------------------------------------------- grounding
async def grounding(t: Suite, mock) -> None:
    if not await _filter_ready(t):
        return
    status = await t.owui.upsert_model(TEXT, "Gemini 2.5 Flash", [SEARCH_FILTER])
    try:
        filter_ids = await t.owui.model_filter_ids(TEXT)
        t.check(
            "grounding.attach",
            "google_search_tool attached to the model (visible after models refresh)",
            status == 200 and filter_ids == [SEARCH_FILTER],
            f"HTTP {status} filterIds={filter_ids}",
        )
        await mock.reset()
        r = await t.owui.chat(TEXT, "Search the web", features={"web_search": True})
        req = await mock.last(_gen)
        t.check(
            "grounding.api",
            "API path, features.web_search=true: googleSearch + urlContext sent, "
            "[1] citation in the answer",
            r.status == 200
            and req.get("tool_kinds") == ["googleSearch", "urlContext"]
            and "[1]" in r.content
            and "Error" not in r.content,
            r.brief() + f" tool_kinds={req.get('tool_kinds')}",
        )
        async with t.browser() as b:
            await mock.reset()
            c = await b.chat(TEXT, "Search the web", features={"web_search": True})
            req = await mock.last(_gen)
            t.check(
                "grounding.browser",
                "browser path: googleSearch sent, grounding sources + [1] citation saved",
                c.done
                and "googleSearch" in (req.get("tool_kinds") or [])
                and any(s == GROUNDING_URI for s in _strings(c.sources))
                and "[1]" in c.content,
                c.brief() + f" tool_kinds={req.get('tool_kinds')}",
            )
            await mock.reset()
            c = await b.chat(TEXT, "Search the web", features={"web_search": False})
            req = await mock.last(_gen)
        sent = req.get("tool_kinds") or []
        t.check(
            "grounding.negative",
            "filter attached but web_search off: no googleSearch / urlContext, no "
            "sources, no citation",
            c.done
            and req
            and not {"googleSearch", "urlContext"} & set(sent)
            and not c.sources
            and "[1]" not in c.content
            and "Hello from mock (stream)." in c.content,
            c.brief() + f" tool_kinds={sent}",
        )
    finally:
        # delete the override, so later runs and suites see the pipe's model
        await t.owui.delete_model(TEXT)


# ------------------------------------------------------------------ vertex
async def vertex(t: Suite, mock) -> None:
    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(TEXT, "vertex-context please", stream=True)
    found = [
        s
        for s in c.sources
        if (s.get("source") or {}).get("type") == "vertex_ai_search"
    ]
    source = (found[0].get("source") or {}) if found else {}
    document = found[0].get("document") if found else None
    t.check(
        "vertex.retrieved-context",
        "Vertex AI Search source (retrievedContext): title, gs:// URI and chunk text "
        "saved as source document",
        c.done
        and len(found) == 1
        and source.get("name") == "Vertex Doc"
        and source.get("uri") == "gs://e2e-bucket/doc.pdf"
        and document == ["chunk body"],
        c.brief() + f" vertex_sources={len(found)} vertex_document={document!r}"
        f" source={short(source)}",
    )


# ------------------------------------------------------------------ errors
async def errors(t: Suite, mock) -> None:
    mark = t.mark()
    await mock.reset()
    r = await t.owui.chat(TEXT, "force-400 now", stream=False)
    calls = len(await mock.requests(_gen))
    t.check(
        "errors.400",
        "upstream 400: error text answered, not retried",
        r.status == 200
        and "400" in r.content
        and "force-400" in r.content
        and calls == 1,
        r.brief() + f" upstream_calls={calls}",
    )
    await mock.reset()
    r = await t.owui.chat(TEXT, "force-500-once now", stream=False)
    calls = len(await mock.requests(_gen))
    t.check(
        "errors.retry",
        "upstream 500 once: retried (RETRY_COUNT=2) and answered",
        r.status == 200 and "Hello from mock (non-stream)." in r.content and calls == 2,
        r.brief() + f" upstream_calls={calls}",
    )
    await mock.reset()
    r = await t.owui.chat(TEXT, "prompt-blocked now", stream=False)
    t.check(
        "errors.prompt-blocked",
        "blocked prompt: '[Blocked due to Prompt Safety: SAFETY]'",
        r.status == 200 and "[Blocked due to Prompt Safety: SAFETY]" in r.content,
        r.brief(),
    )
    await mock.reset()
    r = await t.owui.chat(TEXT, "finish-safety now", stream=False)
    t.check(
        "errors.safety",
        "finishReason SAFETY: '[Blocked by safety settings (HARM_CATEGORY_HARASSMENT)]'",
        r.status == 200
        and "[Blocked by safety settings (HARM_CATEGORY_HARASSMENT)]" in r.content,
        r.brief(),
    )

    await _set(t, RETRY_COUNT=0)
    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(IMAGE_PREVIEW, "force-500 image", stream=True, params=LEGACY)
    last = _last_status(c, "image_processing")
    t.check(
        "errors.image-status",
        "image request failing upstream: final image status 'Image request failed' "
        "(done), error text saved",
        c.done
        and "500" in c.content
        and last.get("done") is True
        and last.get("description") == "Image request failed"
        and _status_closed(c),
        c.brief() + f" image_status={last}",
    )

    await _set(t, RETRY_COUNT=2, INCLUDE_THOUGHTS=False)
    await mock.reset()
    r = await t.owui.chat(TEXT, "data-prefix please", stream=True)
    detail = r.brief() + f" done={r.done}"
    t.check(
        "errors.data-prefix-stream",
        "streamed answer starting with 'data:' reaches the API client unchanged",
        r.status == 200
        and r.content == "data: starts like SSE."
        and r.done
        and not r.errors,
        detail,
    )
    await t.log.settle(1)
    t.expect_errors(
        mark,
        ("function_gemini", "mock: force-400"),
        ("function_gemini", "mock: force-500"),
    )


# ------------------------------------------------------------------- retry
async def retry(t: Suite, mock) -> None:
    mark = t.mark()
    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(TEXT, "force-503-once stream", stream=True)
        reqs = await mock.requests(_gen)
    await t.log.settle(0.5)
    # the 503 is provoked; whether it was retried is the check below
    t.expect_errors(mark, ("function_gemini", "mock: force-503-once"))
    t.check(
        "retry.stream",
        "stream with 503 on the first chunk: retried (RETRY_COUNT=2), answer streamed",
        c.done
        and _actions(reqs) == ["streamGenerateContent", "streamGenerateContent"]
        and [e.get("status") for e in reqs] == [503, 200]
        and "Hello from mock (stream)." in c.content
        and "Error" not in c.content,
        f"upstream_calls={len(reqs)} statuses={[e.get('status') for e in reqs]} "
        + c.brief(),
    )


# ------------------------------------------------------------------ status
async def status(t: Suite, mock) -> None:
    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(
            VEO, "slow-video: a dog surfing", stream=True, stop_after_s=8, stop_wait=10
        )
        await asyncio.sleep(3)
        c = await b.reload(c)
    stop_ok = bool(c.stopped) and all(code == 200 for code, _ in c.stopped)
    last = _last_status(c, "video_generation")
    polled = len(await mock.requests(lambda e: "/operations/" in e.get("path", "")))
    t.check(
        "status.cancel-terminal",
        "Stop during Veo polling: the video_generation status ends with done=true",
        stop_ok and polled >= 1 and last.get("done") is True,
        f"stop_ok={stop_ok} last_video_status_done={last.get('done')} "
        f"last_video_status={last} polls={polled} stopped={short(c.stopped)} "
        + c.brief(),
    )


# ------------------------------------------------------------------ valves
async def valves(t: Suite, mock) -> None:
    async def ids() -> list:
        return sorted(
            m["id"] for m in await t.owui.models() if m["id"].startswith(FID + ".")
        )

    # The model list cache must follow valve changes (MODEL_CACHE_TTL 600 s).
    await _set(t, MODEL_CACHE_TTL=600)
    before = await ids()
    await _set(t, MODEL_WHITELIST="gemini-2.5-flash")
    after = await ids()
    t.check(
        "valves.model-cache",
        "MODEL_WHITELIST change shows in the next model list refresh "
        "(MODEL_CACHE_TTL=600)",
        len(before) > 1 and after == [TEXT],
        f"before={len(before)} models after={after} stale_list={after == before}",
    )
    await _set(t, MODEL_CACHE_TTL=0, MODEL_WHITELIST="")

    await _set(t, USE_PERMISSIVE_SAFETY=True)
    await mock.reset()
    await t.owui.chat(TEXT, "Hi", stream=False)
    safety = _body(await mock.last(_gen)).get("safetySettings") or []
    t.check(
        "valves.safety",
        "USE_PERMISSIVE_SAFETY: 4 safetySettings with BLOCK_NONE",
        len(safety) == 4 and all(s.get("threshold") == "BLOCK_NONE" for s in safety),
        f"safetySettings={safety}",
    )
    await _set(t, USE_PERMISSIVE_SAFETY=False, MODEL_WHITELIST="gemini-2.5-flash")
    listed = await ids()
    t.check(
        "valves.whitelist",
        "MODEL_WHITELIST limits the model list",
        listed == [TEXT],
        f"ids={listed}",
    )
    await _set(t, MODEL_WHITELIST="", MODEL_ADDITIONAL="gemini-extra-model")
    listed = await ids()
    t.check(
        "valves.additional",
        "MODEL_ADDITIONAL adds a model",
        f"{FID}.gemini-extra-model" in listed,
        f"ids={listed}",
    )

    def system(entry: dict) -> str:
        instruction = _body(entry).get("systemInstruction") or {}
        return " ".join(p.get("text", "") for p in instruction.get("parts") or [])

    await _set(t, MODEL_ADDITIONAL="", DEFAULT_SYSTEM_PROMPT="Be brief.")
    await mock.reset()
    await t.owui.chat(TEXT, "Hi", stream=False)
    alone = system(await mock.last(_gen))
    await mock.reset()
    await t.owui.chat(
        TEXT,
        [
            {"role": "system", "content": "Use French."},
            {"role": "user", "content": "Hi"},
        ],
        stream=False,
    )
    combined = system(await mock.last(_gen))
    t.check(
        "valves.system-prompt",
        "DEFAULT_SYSTEM_PROMPT used alone and prepended to a user system prompt",
        alone == "Be brief." and combined == "Be brief.\n\nUse French.",
        f"alone={alone!r} combined={combined!r}",
    )

    await _set(t, DEFAULT_SYSTEM_PROMPT="", ENABLE_FORWARD_USER_INFO_HEADERS=True)
    await mock.reset()
    await t.owui.chat(TEXT, "Hi", stream=False)
    headers = (await mock.last(_gen)).get("headers") or {}
    t.check(
        "valves.user-headers",
        "ENABLE_FORWARD_USER_INFO_HEADERS: X-OpenWebUI-User-Id/-Email sent",
        headers.get("x-openwebui-user-email") == "admin@example.com"
        and headers.get("x-openwebui-user-id") == t.owui.user.get("id"),
        f"headers={headers}",
    )
    await _set(t, ENABLE_FORWARD_USER_INFO_HEADERS=False)
    await mock.reset()
    await t.owui.chat(TEXT, "Hi", stream=False)
    first = await mock.last(_gen)
    await _set(t, API_VERSION="v1beta")
    await mock.reset()
    await t.owui.chat(TEXT, "Hi", stream=False)
    path = (await mock.last(_gen)).get("path", "")
    t.check(
        "valves.api-version",
        "API_VERSION default v1alpha / v1beta in the path; no user headers when off",
        first.get("path", "").startswith("/v1alpha/")
        and path.startswith("/v1beta/")
        and "x-openwebui-user-id" not in (first.get("headers") or {}),
        f"default_path={first.get('path')} v1beta_path={path} "
        f"headers_off={first.get('headers')}",
    )
    await _set(t, API_VERSION="v1alpha")
    await mock.reset()
    await t.owui.chat(
        TEXT, "Hi", stream=False, temperature=0.3, max_tokens=50, top_p=0.9
    )
    gc = _gc(await mock.last(_gen))
    t.check(
        "valves.params",
        "temperature / max_tokens / top_p mapped into generationConfig",
        gc.get("temperature") == 0.3
        and gc.get("maxOutputTokens") == 50
        and gc.get("topP") == 0.9,
        f"generationConfig={short(gc, 300)}",
    )

    problems, counts = [], []
    for user, snapshot in ((False, VALVES_1_16_1), (True, USER_VALVES_1_16_1)):
        props = (await t.owui.valves_spec(FID, user)).get("properties") or {}
        counts.append(len(props))
        kind = "UserValves" if user else "Valves"
        for name, default in snapshot.items():
            if name not in props:
                problems.append(f"{kind}.{name} missing")
            elif (props[name] or {}).get("default") != default:
                problems.append(
                    f"{kind}.{name} default {props[name].get('default')!r} != {default!r}"
                )
    t.check(
        "valves.names",
        "every Valves / UserValves name of 1.16.1 still exists with its default "
        "(public API)",
        not problems,
        f"valves={counts[0]} user_valves={counts[1]} problems={problems}",
    )


# ------------------------------------------------------------- concurrency
async def concurrency(t: Suite, mock) -> None:
    await _set(t, ENABLE_FORWARD_USER_INFO_HEADERS=True)
    other = await t.owui.create_user("E2E Gemini User", SECOND_USER, role="admin")
    try:
        rounds = []
        for _ in range(3):
            await mock.reset()
            r = await other.chat(TEXT, "Hello from user two", stream=False)
            user_chat = await mock.last(_gen)
            await t.owui.models(refresh=True)
            listing = await mock.last(lambda e: e.get("path", "").endswith("/models"))
            await mock.reset()
            await t.owui.chat(TEXT, "Hello from the admin", stream=False)
            admin_chat = await mock.last(_gen)
            rounds.append(
                (
                    r.status,
                    (user_chat.get("headers") or {}).get("x-openwebui-user-email"),
                    (listing.get("headers") or {}).get("x-openwebui-user-email"),
                    bool(listing),
                    (admin_chat.get("headers") or {}).get("x-openwebui-user-email"),
                )
            )
    finally:
        await other.close()
    leaked = next((lst for _, _, lst, _, _ in rounds if lst), None)
    t.check(
        "concurrency.user-headers",
        "forwarded user headers belong to the requesting user: user two's chat, "
        "the admin's model list refresh (no user), the admin's chat",
        all(
            status == 200
            and user == SECOND_USER
            and listed
            and listing_user is None
            and admin == "admin@example.com"
            for status, user, listing_user, listed, admin in rounds
        ),
        f"model_list_user={leaked} rounds(status,user_chat,model_list,listed,"
        f"admin_chat)={rounds}",
    )


# --------------------------------------------------------------- streamimg
async def streamimg(t: Suite, mock) -> None:
    """A model the pipe does not detect as image model streams an image."""
    await _set(t, MODEL_ADDITIONAL="gemini-2.0-flash-imagegen")
    async with t.browser() as b:
        await mock.reset()
        c = await b.chat(IMAGEGEN, "Draw", stream=True)
        reqs = await mock.requests(_gen)
    names, urls_ok, report = await _image_files(t, c, [FINAL_PNG])
    t.check(
        "streamimg.browser",
        "undetected model streaming an inline image: final image uploaded and "
        "attached once (thought image skipped)",
        c.done
        and _actions(reqs) == ["streamGenerateContent"]
        and names == ["final"]
        and urls_ok
        and "Here is your image." in c.content
        and "![" not in c.content,
        c.brief() + f" upstream={_actions(reqs)} {report}",
    )
