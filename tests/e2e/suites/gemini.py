"""
Gemini suite: pipelines/google/google_gemini.py against mocks/mock_gemini.py
(plus filters/google_search_tool.py for Search grounding).

Groups (``--only gemini.<group>``)
  models       model list, image / video indicators
  api          API path (no websocket): non-stream, stream (thinking in a
               <details type="reasoning" done="true" duration="N"> block)
  thinking     summaries not replayed (#176, old and 1.19.0 shapes), budget / level /
               include / strip valves; browser: live done="false" block, duration
               up to the answer, Stop while thinking
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
               citations, search statuses; no grounding without web_search
  vertex       Vertex AI Search sources (retrievedContext.text)
  errors       upstream 400 / 500 (retry) / blocked prompt / SAFETY finish / image
               error status / streamed answer starting with "data:"
  retry        RETRY_COUNT for streams (503 on the first chunk, then 200)
  status       Stop during Veo polling leaves no running status
  valves       model cache vs. valve changes, safety, whitelist, additional,
               system prompt, user headers, API version, params, valve names
  concurrency  forwarded user headers belong to the requesting user
  streamimg    inline image from a model the pipe does not detect (stream path)
  imgedit      image editing across turns (#194): generated images and uploads of
               a saved chat sent with the edit request (in order, each once),
               follow-up tasks without them, guided regeneration,
               IMAGE_HISTORY_MAX_REFERENCES keeps the current and the newest
               images, older saved forms (data: URL files, markdown links),
               temporary chats and database errors use the request, files of
               another user are read only for an admin (also on the Veo path)
  toolsapi    native tool calling, API path: client tools -> tool_calls (stream,
               non-stream, finish_reason), text answers, streaming valve off,
               malformed / unexpected calls, continuation with signatures and
               without, odd client histories, tool_choice, name mapping,
               default_api. prefix, schema clean-up, synthetic ids
  tools        native tool calling, browser path (Open WebUI's tool loop):
               built-in, parallel, rounds, thinking, workspace, image result,
               OpenAPI, MCP and direct tools, approval, follow-up turns, unknown
               tool, malformed call, grounding with tools (and the next turn),
               task / legacy / no built-in tools, streaming valve off
  owuitools    Open WebUI's built-in image generation / editing (image engine
               "gemini" on the mock) and code interpreter tools called by a
               Gemini text model

The browser path sends the web UI's default params, i.e. native function
calling with Open WebUI's built-in tools (functionDeclarations). The "images"
group and the image error status use ``function_calling=legacy`` so they test
the image handling itself; the tools sent to image models are checked by
"image", "nano" and "imgtools". The tool groups live in suites/_gemini_tools.py
(the mock answers its ``MOCKTOOLS:`` directive with function calls).
"""

import asyncio
import base64
import json
import random
import re
import struct
import time
import uuid
import zlib

from harness import BrowserSession, Suite, short
from harness.browser import FRONTEND_FEATURES
from harness.known import version_tuple

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
    "imgedit",
    "toolsapi",
    "tools",
    "owuitools",
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
# WARNING lines of the pipe that mean a lost event, image, video, answer part or
# tool data (the pipe swallows these failures, so they never show as ERROR); a
# scenario that provokes one on purpose uses t.expect_warnings.
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
        # native tool calling (1.18.0)
        "Skipping duplicate tool declaration",
        "Dropping unmatched function call",
        "Dropping unmatched tool result",
        "Could not restore stored model content",
        "Invalid tool call arguments",
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


# The pipe's thinking block (1.19.0): Open WebUI's native reasoning <details>,
# done="true" with the duration, or done="false" (live) without it
BLOCK_RE = re.compile(
    r'<details type="reasoning" done="(true|false)"(?: duration="(\d+)")?>\n'
    r"<summary>Thought \((\d+)s\)</summary>\n\n(.*?)\n\n</details>",
    re.DOTALL,
)


def _block(content) -> dict:
    """The thinking block at the start of ``content``: ``done``, ``duration``
    (None without the attribute), ``summary_s`` (N of "Thought (Ns)"),
    ``thoughts`` and the text after it (``rest``); {} without such a block."""
    m = BLOCK_RE.match(content if isinstance(content, str) else "")
    if not m:
        return {}
    return {
        "done": m.group(1),
        "duration": int(m.group(2)) if m.group(2) is not None else None,
        "summary_s": int(m.group(3)),
        "thoughts": m.group(4),
        "rest": content[m.end() :],
    }


def _block_ok(content, thought: str, answer: str) -> bool:
    """``content`` is a finished thinking block holding ``thought`` (duration
    attribute = the summary's seconds) followed by the answer ``answer``."""
    b = _block(content)
    return (
        b.get("done") == "true"
        and b.get("duration") is not None
        and b["duration"] == b["summary_s"]
        and thought in b["thoughts"]
        and answer in b["rest"]
    )


def _block_report(content) -> str:
    b = _block(content)
    if not b:
        return f"block=None head={short(content, 80)}"
    return (
        f"block=done:{b['done']},duration:{b['duration']},summary:{b['summary_s']}s "
        f"rest={short(b['rest'], 60)}"
    )


def _live_view(events: list) -> list:
    """Replay the message's content events the way the web UI does (replace /
    chat:message set the content, chat:message:delta / message append to it):
    one ``(event type, content afterwards)`` per content event."""
    content, view = "", []
    for event in events:
        kind, data = event.get("type"), event.get("data") or {}
        if kind in ("replace", "chat:message"):
            content = data.get("content") or ""
        elif kind in ("chat:message:delta", "message"):
            content += data.get("content") or ""
        else:
            continue
        view.append((kind, content))
    return view


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
    groups = ("tasks", "imgtools", "grounding", "tools", "toolsapi")
    if any(t.selected(g) for g in groups):
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
        ("imgedit", imgedit),
        ("toolsapi", toolsapi_group),
        ("tools", tools_group),
        ("owuitools", owuitools_group),
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
        'API non-stream: answer with thinking in a <details type="reasoning" '
        'done="true" duration="N"> block (summary "Thought (Ns)")',
        r.status == 200
        and _block_ok(r.content, "Mock thinking.", "Hello from mock (non-stream)."),
        r.brief() + " " + _block_report(r.content),
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
        "API stream without websocket session: answer streamed, thinking in the "
        '<details type="reasoning" done="true" duration="N"> block',
        r.status == 200
        and r.done
        and _block_ok(r.content, "Mock pondering.", "Hello from mock (stream).")
        and "Error" not in r.content
        and not r.errors,
        r.brief() + f" upstream={req.get('action')} " + _block_report(r.content),
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
        "thinking summary of earlier turns is not replayed to Gemini (#176): the "
        "plain <details> block of chats saved before 1.19.0",
        r.status == 200
        and any("First answer." in p for p in parts)
        and not any("SECRET-THOUGHT" in p for p in parts),
        f"model parts sent upstream={parts}",
    )

    # The 1.19.0 shapes: a real answer of the pipe (done="true") and the live
    # block (done="false") that an answer stopped while thinking keeps
    await mock.reset()
    real = await t.owui.chat(TEXT, "Hello Gemini", stream=True)
    stopped = (
        '<details type="reasoning" done="false">\n<summary>Thought (2s)</summary>'
        "\n\n> STOPPED-THOUGHT\n\n</details>Partial answer."
    )
    native_history = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": real.content},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": stopped},
        {"role": "user", "content": "Third question"},
    ]
    await mock.reset()
    r = await t.owui.chat(TEXT, native_history, stream=False)
    parts = replayed(await mock.last(_gen))
    t.check(
        "thinking.strip-reasoning",
        'the <details type="reasoning"> blocks of earlier turns (the pipe\'s own '
        'done="true" block, a stopped done="false" block) are not replayed',
        r.status == 200
        and _block_ok(real.content, "Mock pondering.", "Hello from mock (stream).")
        and any("Hello from mock (stream)." in p for p in parts)
        and any("Partial answer." in p for p in parts)
        and not any("Mock pondering." in p or "STOPPED-THOUGHT" in p for p in parts),
        f"model parts sent upstream={parts} " + _block_report(real.content),
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
        and "<details" not in r.content
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

    await _set(t, STRIP_THINKING_FROM_HISTORY=True)
    async with t.browser() as b:
        await thinking_live(t, mock, b)
        await thinking_stop(t, mock, b, replayed)
        await thinking_tool_round(t, mock, b)


PACED_THOUGHTS = [f"Paced thought {i}." for i in range(1, 11)]  # mock_gemini.PACED


async def thinking_live(t: Suite, mock, b) -> None:
    """Browser stream: the live thinking block (paced-thinking: ten thoughts 0.15 s
    apart, then the answer in three parts 1.2 s apart)."""
    await mock.reset()
    c = await b.chat(TEXT, "paced-thinking please", stream=True)
    view = _live_view(c.events)
    thinking_statuses = [
        s
        for s in [e.get("data") or {} for e in c.events if e.get("type") == "status"]
        + c.status_history
        if s.get("action") == "thinking"
    ]
    kinds = [k for k, _ in view]
    first_delta = (
        kinds.index("chat:message:delta") if "chat:message:delta" in kinds else 0
    )
    live = [_block(x) for k, x in view[:first_delta] if k == "replace"]
    while_thinking = [x for x in live if x.get("done") == "false"]
    switch = live[-1] if live else {}
    # content the web UI shows once every event before the final replace arrived
    shown = view[-3][1] if len(view) >= 3 else ""
    report = (
        f"content_events={len(view)} live={len(while_thinking)} "
        f"live_rest={[x.get('rest') for x in while_thinking][:3]} "
        f"live_duration={[x.get('duration') for x in while_thinking][:3]} "
        f"switch={short(switch.get('done'))}/{short(switch.get('rest'))} "
        f"thinking_statuses={len(thinking_statuses)} shown_is_saved={shown == c.content} "
        + _block_report(c.content)
    )
    t.check(
        "thinking.live",
        "browser stream: while thinking, throttled replace events with a <details "
        'type="reasoning" done="false"> block (no thinking status); the first answer '
        'part switches it to done="true" with all thoughts, then deltas; the result '
        "is the saved answer",
        c.done
        and not c.error
        and bool(view)
        and _block(view[0][1]).get("done") == "false"
        and 2 <= len(while_thinking) < len(PACED_THOUGHTS)
        and all(
            x.get("duration") is None and x.get("rest") == "" for x in while_thinking
        )
        and "Paced thought 1." in while_thinking[0].get("thoughts", "")
        and len(live) == len(while_thinking) + 1
        and switch.get("done") == "true"
        and all(text in switch.get("thoughts", "") for text in PACED_THOUGHTS)
        and switch.get("rest") == "Paced "
        and kinds[first_delta:]
        == ["chat:message:delta"] * 2 + ["replace", "chat:message"]
        and shown == c.content
        and _block_ok(c.content, PACED_THOUGHTS[-1], "Paced answer done.")
        and not thinking_statuses,
        report,
    )
    duration = _block(c.content).get("duration")
    t.check(
        "thinking.duration",
        "streamed thinking lasts from the first thought to the first answer part "
        "(1.5 s), not to the end of the stream (3.9 s)",
        c.done and duration is not None and 1 <= duration <= 2,
        f"duration={duration} " + _block_report(c.content),
    )


async def thinking_stop(t: Suite, mock, b, replayed) -> None:
    """Stop while Gemini thinks (slow-thinking: twelve thoughts 0.5 s apart): the
    saved done="false" block is left out of the next turn's history."""
    await mock.reset()
    c = await b.chat(
        TEXT, "slow-thinking please", stream=True, stop_after_s=3.5, stop_wait=10
    )
    await asyncio.sleep(2)
    c = await b.reload(c)
    saved = c.message.get("content") or ""
    block = _block(saved)
    stop_ok = bool(c.stopped) and all(code == 200 for code, _ in c.stopped)
    await mock.reset()
    follow = await b.chat(
        TEXT, "Next question", stream=True, chat_id=c.chat_id, parent_id=c.message_id
    )
    reqs = [e for e in await mock.requests(_gen) if "### Task:" not in _last_text(e)]
    parts = replayed(reqs[0]) if reqs else []
    t.check(
        "thinking.stop",
        'Stop while thinking: the saved content is the done="false" block with the '
        "thoughts so far (no answer, no thinking status); the next turn sends "
        "neither the block nor its thoughts",
        stop_ok
        and c.done
        and block.get("done") == "false"
        and "Slow thought 1." in block.get("thoughts", "")
        and block.get("rest") == ""
        and "Slow answer." not in saved
        and not any(s.get("action") == "thinking" for s in c.status_history)
        and follow.done
        and bool(reqs)
        and "Next question" in _last_text(reqs[0])
        and not any("Slow thought" in p for p in parts),
        f"stop_ok={stop_ok} done={c.done} {_block_report(saved)} "
        f"thoughts={short(block.get('thoughts'), 60)} statuses={c.status_descriptions} "
        f"next_turn_model_parts={parts} follow={follow.brief()}",
    )


async def thinking_tool_round(t: Suite, mock, b) -> None:
    """A browser turn whose first round ends with a tool call (the mock's
    MOCKTOOLS directive, Open WebUI's built-in get_current_timestamp): the live
    block of that round is cleared, Open WebUI's output items take over."""
    await mock.reset()
    rounds = [[{"name": "get_current_timestamp", "args": {}}]]
    c = await b.chat(PRO3, "MOCKTOOLS:" + json.dumps({"rounds": rounds}), stream=True)
    if c.task_ids and not c.done:  # keep its tool loop away from later scenarios
        await b.stop(c)
    view = _live_view(c.events)
    first = _block(view[0][1]) if view else {}
    reasoning = [text for text, _ in c.reasoning_items]
    saved = c.message.get("content") or ""
    t.check(
        "thinking.tool-round",
        'tool turn: the first round shows the live done="false" block, then '
        "clears it (replace with empty content) when it ends with a tool call; the "
        "saved turn has the thoughts in a reasoning item and no <details> block",
        c.done
        and [k for k, _ in view] == ["replace", "replace"]
        and first.get("done") == "false"
        and "Mock pondering." in first.get("thoughts", "")
        and view[-1][1] == ""
        and any("Mock pondering." in text for text in reasoning)
        and c.output_text.startswith("MOCK-FINAL get_current_timestamp=")
        and "<details" not in saved
        and "<details" not in c.output_text,
        f"content_events={[(k, short(x, 50)) for k, x in view]} "
        f"reasoning={[short(x, 30) for x in reasoning]} saved={short(saved, 80)} "
        f"output_text={short(c.output_text, 80)}",
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
                f"browser path stream={stream}: answer + thinking block "
                '(<details type="reasoning" done="true" duration="N">) saved',
                c.done and _block_ok(c.content, thought, answer) and not c.error,
                c.brief() + " " + _block_report(c.content),
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
                "non-stream, one image file with the final image, text after the "
                'thinking block (type="reasoning", done="true"), usage, statuses '
                "closed without failure",
                c.done
                and _actions(reqs) == ["generateContent"]
                and names == ["final"]
                and urls_ok
                and _block_ok(c.content, "Mock image thinking.", "Here is your image.")
                and "![" not in c.content
                and _usage(c.usage) == USAGE_IMAGE
                and _status_ok(c),
                c.brief()
                + f" model={model.split('.', 1)[1]} upstream={_actions(reqs)} {report}"
                f" {_status_report(c)} {_block_report(c.content)}",
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
    await _set(
        t,
        IMAGE_HISTORY_MAX_REFERENCES=5,
        IMAGE_ADD_LABELS=True,
        IMAGE_HISTORY_FIRST=True,
    )
    await mock.reset()
    r3 = await t.owui.chat(IMAGE_PREVIEW, history, stream=False)
    third = await sent()
    t.check(
        "images.history",
        "image history: IMAGE_HISTORY_MAX_REFERENCES=2 keeps the current image "
        "and the newest history image, [Image N] labels, history first; "
        "IMAGE_ADD_LABELS / IMAGE_HISTORY_FIRST off; limit 5: duplicates dropped",
        r1.status == 200
        and r2.status == 200
        and r3.status == 200
        and first == ["Combine", "[Image 1]", "C", "[Image 2]", "B"]
        and second == ["Combine", "B", "C"]
        and third == ["Combine", "[Image 1]", "A", "[Image 2]", "C", "[Image 3]", "B"],
        f"limit2,labels,history_first={first} plain,current_first={second} "
        f"limit5={third}",
    )

    # More images in the current message than the limit: the first ones only
    d = _png(40, 40, 40)
    labels[d] = "D"
    many = [
        {"role": "user", "content": [{"type": "text", "text": "First"}, image(d)]},
        {"role": "assistant", "content": "Nice image."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Three"},
                image(a),
                image(b),
                image(c),
            ],
        },
    ]
    await _set(t, IMAGE_HISTORY_MAX_REFERENCES=2)
    await mock.reset()
    r4 = await t.owui.chat(IMAGE_PREVIEW, many, stream=False)
    fourth = await sent()
    t.check(
        "images.history-current",
        "IMAGE_HISTORY_MAX_REFERENCES=2 with 3 images in the current message: "
        "the first 2 are sent, no history image",
        r4.status == 200 and fourth == ["Three", "[Image 1]", "A", "[Image 2]", "B"],
        f"HTTP {r4.status} sent={fourth}",
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
            queries = _last_status(c, "web_search_queries_generated")
            sites = _last_status(c, "web_search")
            t.check(
                "grounding.status",
                "search statuses in Open WebUI's own localized shape: "
                "web_search_queries_generated with the queries, then web_search "
                "'Searched {{count}} sites' with items (title, link) and no urls; "
                "the last status stays visible",
                c.done
                and queries.get("queries") == ["mock search query"]
                and queries.get("done") is True
                and sites.get("description") == "Searched {{count}} sites"
                and sites.get("items")
                == [{"title": "Example A", "link": GROUNDING_URI}]
                and "urls" not in sites
                and sites.get("done") is True
                and bool(c.status_history)
                and c.status_history[-1] == sites
                and not any(s.get("hidden") for s in c.status_history),
                f"statuses={short(c.status_history, 400)}",
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


# ----------------------------------------------------------------- imgedit
# Image editing across turns (#194): Open WebUI drops the files of assistant
# messages before it calls a pipe, and generated images are attached there only,
# so the pipe reads the image history of a saved chat from the chat itself.
IMGEDIT_USER = "e2e-imgedit-user@example.com"
UPLOAD_PNG_1 = _png(40, 80, 120)
UPLOAD_PNG_2 = _png(120, 80, 40)
FOREIGN_PNG = _png(200, 0, 200)
SAVED_PNG = _png(10, 120, 60)  # a data: URL image file of a saved user message
LINKED_PNG = _png(60, 10, 120)  # a data: URL markdown image of a saved answer
# final images of the mock's "final-image-<n>" trigger, one per turn
GEN_PNGS = {n: _png(255, 255, n) for n in (1, 2, 3)}
EDIT_NAMES = {
    **PNG_NAMES,
    UPLOAD_PNG_1: "upload1",
    UPLOAD_PNG_2: "upload2",
    FOREIGN_PNG: "foreign",
    SAVED_PNG: "saved",
    LINKED_PNG: "linked",
    **{png: f"gen{n}" for n, png in GEN_PNGS.items()},
}
# Test-only filter for imgedit.db-error: after its inlet, the next chat lookup
# for the marked chat fails once. Open WebUI loads the chat before the inlet
# filters run, so the lookup that fails is the pipe's.
DB_FAULT_FILTER_ID = "e2e_db_fault"
DB_FAULT_FILTER = '''"""
title: E2E DB Fault
author: owndev
version: 0.1.0
license: Apache License 2.0
description: Test-only filter for tests/e2e (gemini imgedit.db-error). When the last user message contains "db-fault", the next Chats.get_messages_map_by_chat_id call for that chat raises RuntimeError("e2e db fault"), once.
"""

from typing import Optional


class Filter:
    async def inlet(self, body: dict, __metadata__: Optional[dict] = None) -> dict:
        chat_id = (__metadata__ or {}).get("chat_id")
        messages = body.get("messages") or []
        content = messages[-1].get("content") if messages else ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text") or "" for p in content if isinstance(p, dict)
            )
        if not chat_id or "db-fault" not in str(content):
            return body
        from open_webui.models.chats import Chats

        original = type(Chats).get_messages_map_by_chat_id

        async def fail_once(id, *args, **kwargs):
            if id != chat_id:
                return await original(Chats, id, *args, **kwargs)
            vars(Chats).pop("get_messages_map_by_chat_id", None)
            raise RuntimeError("e2e db fault")

        Chats.get_messages_map_by_chat_id = fail_once
        return body
'''


async def _edit_request(mock) -> dict:
    """What the last generate request (background tasks left out) sent: the
    number of contents, the parts of the last content (texts, images by name,
    "?" for an unknown image) and the image names alone (Open WebUI puts an
    <attached_files> text in front of a user message with an upload)."""
    reqs = [e for e in await mock.requests(_gen) if "### Task:" not in _last_text(e)]
    contents = (_body(reqs[-1]).get("contents") or []) if reqs else []
    parts = (contents[-1].get("parts") or []) if contents else []
    sent, images = [], []
    for part in parts:
        if "text" in part:
            sent.append(part["text"])
            continue
        data = _b64((part.get("inlineData") or {}).get("data"))
        images.append(EDIT_NAMES.get(data, "?"))
        sent.append(images[-1])
    return {"contents": len(contents), "sent": sent, "images": images}


def _edit_report(req: dict) -> str:
    return f"contents={req['contents']} sent={[short(s, 60) for s in req['sent']]}"


async def _upload_png(owui, data: str, name: str) -> dict:
    """Upload a PNG and return the file item the web UI saves with the user
    message (type "file", content type image/png, the file id as url)."""
    raw = base64.b64decode(data)
    status, record = await owui.upload_file(name, raw, "image/png")
    file_id = record.get("id", "") if isinstance(record, dict) else ""
    return {
        "type": "file",
        "id": file_id if status == 200 else "",
        "url": file_id if status == 200 else "",
        "name": name,
        "status": "uploaded",
        "size": len(raw),
        "content_type": "image/png",
    }


async def _saved_message(t: Suite, chat_id: str, message_id: str) -> dict:
    """A message of a saved chat as Open WebUI stores it."""
    saved = (await t.owui.get_chat(chat_id)).get("chat") or {}
    return ((saved.get("history") or {}).get("messages") or {}).get(message_id) or {}


async def _grant_read(t: Suite, model: str, name: str) -> int:
    """Let every user read a pipe model (a non-admin user only reaches a pipe
    model with a read grant); remove it with ``t.owui.delete_model``."""
    status, _ = await t.owui.api(
        "POST",
        "/api/v1/models/model/access/update",
        {
            "id": model,
            "name": name,
            "access_grants": [
                {"principal_type": "user", "principal_id": "*", "permission": "read"}
            ],
        },
    )
    return status


async def _video_image(mock) -> str:
    """Name of the image the last Veo request sent as instances[0].image
    ("" without one)."""
    started = await mock.requests(lambda e: e.get("action") == "predictLongRunning")
    instance = (_body(started[-1]).get("instances") or [{}])[0] if started else {}
    picture = instance.get("image") or {}
    data = _b64(picture.get("bytesBase64Encoded") or picture.get("imageBytes"))
    return EDIT_NAMES.get(data, "?") if data else ""


async def _temporary_chat(
    b: BrowserSession, model: str, messages: list, wait: float = 120
) -> tuple:
    """A turn of a temporary chat like the web UI: chat id
    "temporary:<socket id>", the history in the request's ``messages``. Open
    WebUI saves nothing, so the answer comes from the socket events.
    Returns (HTTP status, data of the final chat:completion event or {})."""
    assistant_id = str(uuid.uuid4())
    text = messages[-1].get("content")
    body = {
        "model": model,
        "stream": True,
        "messages": messages,
        "session_id": b.sid,
        "chat_id": f"temporary:{b.sid}",
        "id": assistant_id,
        "parent_id": None,
        "user_message": {
            "id": str(uuid.uuid4()),
            "parentId": None,
            "childrenIds": [assistant_id],
            "role": "user",
            "content": text if isinstance(text, str) else "",
            "timestamp": int(time.time()),
            "models": [model],
        },
        "features": dict(FRONTEND_FEATURES),
        "params": dict(LEGACY),
        "background_tasks": {},
    }
    status, _ = await b.owui.api("POST", "/api/chat/completions", body)
    deadline = time.time() + wait
    while status == 200 and time.time() < deadline:
        for event in list(b.events):
            data = event.get("data") or {}
            done = data.get("data") or {}
            if (
                event.get("message_id") == assistant_id
                and data.get("type") == "chat:completion"
                and done.get("done")
            ):
                return status, done
        await asyncio.sleep(0.5)
    return status, {}


def _completion_text(data: dict) -> str:
    """Answer text of a chat:completion event (content, else the output text)."""
    if data.get("content"):
        return str(data["content"])
    return "".join(
        part.get("text", "")
        for item in data.get("output") or []
        if isinstance(item, dict) and item.get("type") == "message"
        for part in item.get("content") or []
        if isinstance(part, dict) and part.get("type") == "output_text"
    )


async def imgedit(t: Suite, mock) -> None:
    """The issue's flow on both models: draw, then edit in the same chat."""
    for model in (IMAGE_GA, NANO):
        short_id = model.split(".", 1)[1]
        async with t.browser() as b:
            await mock.reset()
            c1 = await b.chat(model, "Draw a certificate for Alice", params=LEGACY)
            await mock.reset()
            c2 = await b.chat(
                model,
                "Change the name to Bob",
                params=LEGACY,
                chat_id=c1.chat_id,
                parent_id=c1.message_id,
            )
            req = await _edit_request(mock)
        names, urls_ok, report = await _image_files(t, c2, [FINAL_PNG])
        t.check(
            f"imgedit.{short_id}",
            f"{short_id}, browser follow-up: the image of turn 1 (attached as a "
            "file only) is sent with the edit request as [Image 1] after the "
            "prompt, the edit gets its own image file",
            c1.done
            and c2.done
            and req["contents"] == 1
            and req["sent"] == ["Change the name to Bob", "[Image 1]", "final"]
            and names == ["final"]
            and urls_ok,
            f"turn1_done={c1.done} turn1_files={len(c1.files)} {c2.brief()} "
            f"{_edit_report(req)} {report}",
        )
    await imgedit_task(t, mock)
    await imgedit_upload(t, mock)
    await imgedit_guided(t, mock)
    await imgedit_dedup(t, mock)
    await imgedit_saved(t, mock)
    await imgedit_temporary(t, mock)
    await imgedit_db_error(t, mock)
    await imgedit_foreign(t, mock)


async def imgedit_task(t: Suite, mock) -> None:
    """Open WebUI copies the chat's metadata (chat id, user message id) into its
    background tasks, and the task model is the chat's model by default."""
    async with t.browser() as b:
        await mock.reset()
        c1 = await b.chat(IMAGE_GA, "Draw a certificate", params=LEGACY)
        await mock.reset()
        c2 = await b.chat(
            IMAGE_GA,
            "Change the name to Bob",
            params=LEGACY,
            chat_id=c1.chat_id,
            parent_id=c1.message_id,
            background_tasks={"follow_up_generation": True},
        )
        task_reqs = []
        for _ in range(40):  # the follow-up task runs after the answer
            task_reqs = [
                e for e in await mock.requests(_gen) if "### Task:" in _last_text(e)
            ]
            if task_reqs:
                break
            await asyncio.sleep(0.5)
        req = await _edit_request(mock)
    task_images = [
        sum(
            "inlineData" in part
            for content in _body(e).get("contents") or []
            for part in content.get("parts") or []
        )
        for e in task_reqs
    ]
    t.check(
        "imgedit.task",
        "follow-up task of an image model in a saved chat: sent without the "
        "chat's images (the edit request has them)",
        c1.done
        and c2.done
        and req["sent"] == ["Change the name to Bob", "[Image 1]", "final"]
        and task_reqs
        and not any(task_images),
        f"edit: {_edit_report(req)} task_requests={len(task_reqs)} "
        f"task_images={task_images} turn2: {c2.brief()}",
    )


async def imgedit_upload(t: Suite, mock) -> None:
    """An upload between generated turns, guided regeneration, the image limit,
    in the web UI's default mode (Open WebUI puts an <attached_files> text in
    front of a user message with an upload). Each turn generates another image
    (the mock's final-image-<n> trigger), so their order shows."""
    logo = await _upload_png(t.owui, UPLOAD_PNG_1, "logo.png")
    photo = await _upload_png(t.owui, UPLOAD_PNG_2, "photo.png")
    async with t.browser() as b:
        await mock.reset()
        c1 = await b.chat(IMAGE_GA, "Draw a certificate final-image-1")
        turn = {"chat_id": c1.chat_id}
        await mock.reset()
        c2 = await b.chat(
            IMAGE_GA,
            "Put this logo on it final-image-2",
            parent_id=c1.message_id,
            user_files=[logo],
            **turn,
        )
        r2 = await _edit_request(mock)
        await mock.reset()
        c3 = await b.chat(
            IMAGE_GA,
            "Make the border red final-image-3",
            parent_id=c2.message_id,
            **turn,
        )
        r3 = await _edit_request(mock)
        # Guided regeneration of turn 2: the web UI sends the saved user message
        # again, Open WebUI appends the guidance as a new user message.
        user2 = await _saved_message(t, c1.chat_id, c2.message.get("parentId"))
        guidance = "Make the logo bigger"
        await mock.reset()
        g = await b.chat(
            IMAGE_GA,
            guidance,
            parent_id=c1.message_id,
            extra_body={"user_message": user2, "regeneration_prompt": guidance},
            **turn,
        )
        rg = await _edit_request(mock)
        # Turn 4 twice (siblings), with and without an upload
        await _set(t, IMAGE_HISTORY_MAX_REFERENCES=1)
        try:
            await mock.reset()
            c4 = await b.chat(
                IMAGE_GA,
                "Use this photo instead",
                parent_id=c3.message_id,
                user_files=[photo],
                **turn,
            )
            r4 = await _edit_request(mock)
            await mock.reset()
            c5 = await b.chat(
                IMAGE_GA, "Make it brighter", parent_id=c3.message_id, **turn
            )
            r5 = await _edit_request(mock)
        finally:
            await _set(
                t,
                IMAGE_HISTORY_MAX_REFERENCES=DEFAULTS["IMAGE_HISTORY_MAX_REFERENCES"],
            )
    uploads = f"uploads={bool(logo['id'])},{bool(photo['id'])}"
    t.check(
        "imgedit.upload",
        "browser (default mode), an upload in turn 2: turn 2 sends the generated "
        "image of turn 1 and the upload once each, turn 3 the images of turns "
        "1 and 2 in their order from the saved chat",
        c1.done
        and c2.done
        and c3.done
        and r2["images"] == ["gen1", "upload1"]
        and "Put this logo on it" in (r2["sent"] or [""])[0]
        and r3["sent"]
        == [
            "Make the border red final-image-3",
            "[Image 1]",
            "gen1",
            "[Image 2]",
            "upload1",
            "[Image 3]",
            "gen2",
        ],
        f"{uploads} turn2: {_edit_report(r2)} turn3_done={c3.done} turn3: "
        f"{_edit_report(r3)}",
    )
    t.check(
        "imgedit.guided",
        "guided regeneration of turn 2 (the guidance is a new user message): the "
        "generated image of turn 1 and the upload of turn 2 are sent",
        g.done
        and rg["sent"] == [guidance, "[Image 1]", "gen1", "[Image 2]", "upload1"],
        f"{uploads} saved_user_message={bool(user2)} done={g.done} {_edit_report(rg)}",
    )
    t.check(
        "imgedit.limit",
        "IMAGE_HISTORY_MAX_REFERENCES=1: the upload of the edit message is kept, "
        "the history dropped; without an upload the image generated in the "
        "latest turn is kept",
        c4.done
        and c5.done
        and r4["images"] == ["upload2"]
        and r5["sent"] == ["Make it brighter", "[Image 1]", "gen3"],
        f"{uploads} with_upload: {_edit_report(r4)} without: {_edit_report(r5)}",
    )


async def imgedit_guided(t: Suite, mock) -> None:
    """Guided regeneration of a user message whose saved text is empty (only an
    upload) or contained in the guidance."""
    logo = await _upload_png(t.owui, UPLOAD_PNG_1, "logo.png")
    for label, text, guidance, title in (
        ("image-only", "", "Make it a watercolor", "an image-only user message"),
        ("text-in-guidance", "Bob", "Change Bob to Alice", "a text in the guidance"),
    ):
        async with t.browser() as b:
            await mock.reset()
            c1 = await b.chat(IMAGE_GA, text, user_files=[logo])
            user1 = await _saved_message(t, c1.chat_id, c1.message.get("parentId"))
            await mock.reset()
            g = await b.chat(
                IMAGE_GA,
                guidance,
                chat_id=c1.chat_id,
                extra_body={"user_message": user1, "regeneration_prompt": guidance},
            )
            rg = await _edit_request(mock)
        t.check(
            f"imgedit.guided-{label}",
            f"guided regeneration of {title} ({text!r} with an upload, guidance "
            f"{guidance!r}): the upload is sent",
            c1.done and g.done and rg["sent"] == [guidance, "[Image 1]", "upload1"],
            f"upload={bool(logo['id'])} turn1_done={c1.done} "
            f"saved_user_message={bool(user1)} done={g.done} {_edit_report(rg)}",
        )


async def imgedit_dedup(t: Suite, mock) -> None:
    """An image of an earlier turn attached again (like a downloaded generated
    image): sent once, and counted at its newest place."""
    logo = await _upload_png(t.owui, UPLOAD_PNG_1, "logo.png")
    photo = await _upload_png(t.owui, UPLOAD_PNG_2, "photo.png")
    again = await _upload_png(t.owui, UPLOAD_PNG_1, "logo-again.png")  # new file
    async with t.browser() as b:
        await mock.reset()
        c1 = await b.chat(
            IMAGE_GA,
            "Draw with this logo final-image-1",
            params=LEGACY,
            user_files=[logo],
        )
        turn = {"params": LEGACY, "chat_id": c1.chat_id}
        c2 = await b.chat(
            IMAGE_GA,
            "Add this photo final-image-2",
            parent_id=c1.message_id,
            user_files=[photo],
            **turn,
        )
        await mock.reset()
        c3 = await b.chat(
            IMAGE_GA,
            "Put the logo back final-image-3",
            parent_id=c2.message_id,
            user_files=[again],
            **turn,
        )
        r3 = await _edit_request(mock)
        await _set(t, IMAGE_HISTORY_MAX_REFERENCES=2)
        try:
            await mock.reset()
            c4 = await b.chat(
                IMAGE_GA, "Make it brighter", parent_id=c3.message_id, **turn
            )
            r4 = await _edit_request(mock)
        finally:
            await _set(
                t,
                IMAGE_HISTORY_MAX_REFERENCES=DEFAULTS["IMAGE_HISTORY_MAX_REFERENCES"],
            )
        await _set(t, IMAGE_DEDUP_HISTORY=False)
        try:  # default mode: the prompt gets an <attached_files> text in front
            await mock.reset()
            c5 = await b.chat(
                IMAGE_GA,
                "Use the photo again",
                chat_id=c1.chat_id,
                parent_id=c3.message_id,
                user_files=[photo],
            )
            r5 = await _edit_request(mock)
        finally:
            await _set(t, IMAGE_DEDUP_HISTORY=DEFAULTS["IMAGE_DEDUP_HISTORY"])
    uploads = f"uploads={bool(logo['id'])},{bool(photo['id'])},{bool(again['id'])}"
    t.check(
        "imgedit.dedup-current",
        "the logo of turn 1 attached again in turn 3: sent once, as the current "
        "image, after the other images of turns 1 and 2",
        c1.done
        and c2.done
        and c3.done
        and r3["sent"]
        == [
            "Put the logo back final-image-3",
            "[Image 1]",
            "gen1",
            "[Image 2]",
            "upload2",
            "[Image 3]",
            "gen2",
            "[Image 4]",
            "upload1",
        ],
        f"{uploads} done={c1.done},{c2.done},{c3.done} turn3: {_edit_report(r3)}",
    )
    t.check(
        "imgedit.dedup-newest",
        "IMAGE_HISTORY_MAX_REFERENCES=2: the logo counts at its newest place "
        "(turn 3) and is kept with the image of turn 3",
        c4.done
        and r4["sent"]
        == ["Make it brighter", "[Image 1]", "upload1", "[Image 2]", "gen3"],
        f"{uploads} done={c4.done} turn4: {_edit_report(r4)}",
    )
    t.check(
        "imgedit.no-dedup",
        "IMAGE_DEDUP_HISTORY=false, default mode, the photo of turn 2 attached "
        "again: sent from turn 2 and as the current image, the 4 newest earlier "
        "images kept (the saved edit message is not history)",
        c5.done
        and "Use the photo again" in (r5["sent"] or [""])[0]
        and r5["sent"][1:]
        == [
            "[Image 1]",
            "upload2",
            "[Image 2]",
            "gen2",
            "[Image 3]",
            "upload1",
            "[Image 4]",
            "gen3",
            "[Image 5]",
            "upload2",
        ],
        f"{uploads} done={c5.done} {_edit_report(r5)}",
    )


async def imgedit_saved(t: Suite, mock) -> None:
    """A saved chat with image forms the web UI of today does not create (made
    through the chats API): a data: URL image file and a text file in a user
    message, markdown image links (an Open WebUI file as pipeline versions
    before 1.15.2 wrote it, a data: URL as after a failed upload) in an answer,
    next to the image file of today's answers (Open WebUI never passes it)."""
    linked = await _upload_png(t.owui, UPLOAD_PNG_2, "linked.png")
    generated = await _upload_png(t.owui, GEN_PNGS[1], "generated.png")
    notes_status, notes = await t.owui.upload_file(
        "notes.txt", b"meeting notes", "text/plain"
    )
    notes_id = notes.get("id", "") if isinstance(notes, dict) else ""
    user_id, answer_id, now = str(uuid.uuid4()), str(uuid.uuid4()), int(time.time())
    user1 = {
        "id": user_id,
        "parentId": None,
        "childrenIds": [answer_id],
        "role": "user",
        "content": "Draw a certificate like this",
        "timestamp": now,
        "models": [IMAGE_GA],
        "files": [
            {"type": "image", "url": _image_url(SAVED_PNG)},
            {
                "type": "file",
                "id": notes_id,
                "url": notes_id,
                "name": "notes.txt",
                "status": "uploaded",
                "size": 13,
                "content_type": "text/plain",
            },
        ],
    }
    answer1 = {
        "id": answer_id,
        "parentId": user_id,
        "childrenIds": [],
        "role": "assistant",
        "content": "Here is your image.\n\n"
        f"![Generated Image](/api/v1/files/{linked['id']}/content)\n\n"
        f"![Generated Image]({_image_url(LINKED_PNG)})",
        "files": [{"type": "image", "url": f"/api/v1/files/{generated['id']}/content"}],
        "model": IMAGE_GA,
        "done": True,
        "timestamp": now,
    }
    status, chat = await t.owui.api(
        "POST",
        "/api/v1/chats/new",
        {
            "chat": {
                "title": "E2E saved image chat",
                "models": [IMAGE_GA],
                "history": {
                    "currentId": answer_id,
                    "messages": {user_id: user1, answer_id: answer1},
                },
                "messages": [user1, answer1],
            }
        },
    )
    chat_id = chat.get("id") if isinstance(chat, dict) else None
    c2, req = None, {"contents": 0, "sent": [], "images": []}
    if status == 200 and chat_id:
        async with t.browser() as b:
            await mock.reset()
            c2 = await b.chat(
                IMAGE_GA,
                "Change the name to Bob",
                params=LEGACY,
                chat_id=chat_id,
                parent_id=answer_id,
            )
            req = await _edit_request(mock)
    t.check(
        "imgedit.saved",
        "saved chat: the data: URL image file of a user message, the markdown "
        "image links of an answer (an Open WebUI file, a data: URL) and its image "
        "file are sent in their order, the text file is not",
        bool(linked["id"])
        and bool(generated["id"])
        and notes_status == 200
        and getattr(c2, "done", False)
        and req["sent"]
        == [
            "Change the name to Bob",
            "[Image 1]",
            "saved",
            "[Image 2]",
            "upload2",
            "[Image 3]",
            "linked",
            "[Image 4]",
            "gen1",
        ],
        f"chat: HTTP {status} notes: HTTP {notes_status} "
        f"done={getattr(c2, 'done', None)} {_edit_report(req)}",
    )


async def imgedit_temporary(t: Suite, mock) -> None:
    """A temporary chat is not saved: the image history comes from the request,
    which the web UI builds without the files of assistant messages."""
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Draw a certificate like this"},
                {"type": "image_url", "image_url": {"url": _image_url(UPLOAD_PNG_1)}},
            ],
        },
        {"role": "assistant", "content": "Here is your image."},
        {"role": "user", "content": "Change the name to Bob"},
    ]
    mark = t.mark()
    async with t.browser() as b:
        await mock.reset()
        status, done = await _temporary_chat(b, IMAGE_GA, history)
        req = await _edit_request(mock)
    await t.log.settle(0.5)
    chat_lookups = t.log.lines(mark, "for image history")
    answer = _completion_text(done)
    t.check(
        "imgedit.temporary",
        "temporary chat (not saved): the image history comes from the request "
        "(the earlier upload is sent; the generated image is not, the web UI "
        "sends no assistant files)",
        status == 200
        and "Here is your image." in answer
        and req["sent"] == ["Change the name to Bob", "[Image 1]", "upload1"]
        and not chat_lookups,
        f"HTTP {status} done={bool(done)} answer={short(answer)} "
        f"{_edit_report(req)} chat_lookups={chat_lookups}",
    )


async def imgedit_db_error(t: Suite, mock) -> None:
    """A database error while the pipe loads the saved chat (injected by the
    test-only filter DB_FAULT_FILTER): the image history comes from the
    request, which has the upload but not the generated image of turn 1."""
    logo = await _upload_png(t.owui, UPLOAD_PNG_1, "logo.png")
    status, _ = await t.owui.install_function(
        DB_FAULT_FILTER_ID, "E2E DB Fault", DB_FAULT_FILTER
    )
    active = await t.owui.set_active(DB_FAULT_FILTER_ID, True)
    attached = await t.owui.upsert_model(
        IMAGE_GA, "Gemini 3.1 Flash Image", [DB_FAULT_FILTER_ID]
    )
    c1 = c2 = None
    req = {"contents": 0, "sent": [], "images": []}
    mark = t.mark()
    try:
        async with t.browser() as b:
            await mock.reset()
            c1 = await b.chat(IMAGE_GA, "Draw a certificate", params=LEGACY)
            await mock.reset()
            c2 = await b.chat(
                IMAGE_GA,
                "Change the name to Bob db-fault",
                params=LEGACY,
                chat_id=c1.chat_id,
                parent_id=c1.message_id,
                user_files=[logo],
            )
            req = await _edit_request(mock)
    finally:
        await t.owui.delete_model(IMAGE_GA)
        await t.owui.api("DELETE", f"/api/v1/functions/id/{DB_FAULT_FILTER_ID}/delete")
    await t.log.settle(0.5)
    failed = t.log.lines(mark, "for image history")
    t.check(
        "imgedit.db-error",
        "database error while the pipe loads the saved chat: logged, the answer "
        "completes and the request's images are sent (the upload)",
        status == 200
        and active is True
        and attached == 200
        and getattr(c1, "done", False)
        and getattr(c2, "done", False)
        and "Here is your image." in (c2.content or "")
        and req["sent"] == ["Change the name to Bob db-fault", "[Image 1]", "upload1"]
        and any("e2e db fault" in line for line in failed),
        f"filter: HTTP {status} active={active} model: HTTP {attached} "
        f"{_edit_report(req)} log={[short(line, 120) for line in failed[:1]]} "
        f"turn2: {c2.brief() if c2 else '-'}",
    )


async def imgedit_foreign(t: Suite, mock) -> None:
    """Files of another user are never read: a message can name any file id.
    An admin may read them (also when continuing the user's chat, which Open
    WebUI allows up to 0.11)."""
    setup_error, grants, denied, video_denied, id_logged = "", [], 0, 0, []
    c1 = c2 = c3 = ru = ra = vo = vf = None
    empty = {"contents": 0, "sent": [], "images": []}
    r1 = r2 = r3 = user_req = admin_req = empty
    own_image = foreign_image = ""
    other = None
    owui_version: tuple = ()
    try:
        owui_version = version_tuple(await t.owui.version())
        other = await t.owui.create_user("E2E Image Edit User", IMGEDIT_USER)
        grants = [
            await _grant_read(t, IMAGE_GA, "Gemini 3.1 Flash Image"),
            await _grant_read(t, VEO, "Veo 3.1 Generate Preview"),
        ]
        await t.owui.models(refresh=True)
        foreign = await _upload_png(t.owui, FOREIGN_PNG, "foreign.png")  # admin's
        url = f"/api/v1/files/{foreign['id']}/content"
        async with BrowserSession(other) as b:
            await mock.reset()
            c1 = await b.chat(
                IMAGE_GA,
                "Draw a certificate like this",
                params=LEGACY,
                user_files=[
                    {
                        "type": "image",
                        "url": url,
                        "name": "foreign.png",
                        "content_type": "image/png",
                    }
                ],
            )
            r1 = await _edit_request(mock)
            await mock.reset()
            c2 = await b.chat(
                IMAGE_GA,
                "Change the name to Bob",
                params=LEGACY,
                chat_id=c1.chat_id,
                parent_id=c1.message_id,
            )
            r2 = await _edit_request(mock)
        async with t.browser() as b:  # the admin continues the user's chat
            await mock.reset()
            c3 = await b.chat(
                IMAGE_GA,
                "Make it blue",
                params=LEGACY,
                chat_id=c1.chat_id,
                parent_id=c2.message_id,
            )
            r3 = await _edit_request(mock)
        history = [
            {"role": "user", "content": "Draw a certificate"},
            {
                "role": "assistant",
                "content": f"Here is your image.\n\n![Generated Image]({url})",
            },
            {"role": "user", "content": "Change the name to Bob"},
        ]
        mark = t.mark()
        await mock.reset()
        ru = await other.chat(IMAGE_GA, history, stream=False)
        user_req = await _edit_request(mock)
        await t.log.settle(0.5)
        denied = len(t.log.lines(mark, "does not belong to the requesting user"))
        id_logged = [
            line
            for line in t.log.lines(mark, foreign["id"])
            if "function_gemini" in line
        ]
        await mock.reset()
        ra = await t.owui.chat(IMAGE_GA, history, stream=False)
        admin_req = await _edit_request(mock)
        # Veo image-to-video: markdown links to the user's own upload and to the
        # admin's file
        own = await _upload_png(other, UPLOAD_PNG_1, "frame.png")
        await mock.reset()
        vo = await other.chat(
            VEO,
            f"Animate this ![frame](/api/v1/files/{own['id']}/content)",
            stream=False,
            timeout=180,
        )
        own_image = await _video_image(mock)
        mark = t.mark()
        await mock.reset()
        vf = await other.chat(
            VEO, f"Animate this ![frame]({url})", stream=False, timeout=180
        )
        foreign_image = await _video_image(mock)
        await t.log.settle(0.5)
        video_denied = len(t.log.lines(mark, "does not belong to the requesting user"))
    except Exception as exc:  # noqa: BLE001 - reported in the checks
        setup_error = f"setup failed: {exc!r} "
    finally:
        if other is not None:
            await other.close()
        await t.owui.delete_model(IMAGE_GA)
        await t.owui.delete_model(VEO)
    t.check(
        "imgedit.foreign",
        "browser, a non-admin user with an image file of another user in the "
        "chat: turn 1 does not send it as the current image, turn 2 sends only "
        "the user's own generated image",
        not setup_error
        and grants == [200, 200]
        and c1.done
        and c2.done
        and r1["images"] == []
        and "Draw a certificate like this" in (r1["sent"] or [""])[0]
        and r2["sent"] == ["Change the name to Bob", "[Image 1]", "final"],
        f"{setup_error}grants={grants} turn1_done={getattr(c1, 'done', None)} "
        f"turn1: {_edit_report(r1)} turn2_done={getattr(c2, 'done', None)} "
        f"turn2: {_edit_report(r2)}",
    )
    # Open WebUI 0.12 gives an admin read access only to another user's chat:
    # chat_completion (main.py) asks Chats.get_accessible_chat_by_id(chat_id,
    # user, permission="write"), which needs the owner, a chat shared with
    # share_mode "continue" or a shared folder (models/chats.py), and answers
    # 404, so the pipe is never called. Up to 0.11 it allowed the owner or any
    # admin (Chats.is_chat_owner(...) or user.role == "admin").
    admin_refused = owui_version >= (0, 12)
    t.check(
        "imgedit.admin",
        (
            "browser, an admin continues the user's chat: Open WebUI 0.12 "
            "refuses it (HTTP 404, an admin may only read another user's chat), "
            "the pipe is not called"
            if admin_refused
            else "browser, an admin continues the user's chat: the chat is read, "
            "the admin's file of turn 1 and the user's generated image are sent"
        ),
        not setup_error
        and (
            getattr(c3, "http_status", None) == 404 and r3["contents"] == 0
            if admin_refused
            else getattr(c3, "done", False)
            and r3["sent"]
            == ["Make it blue", "[Image 1]", "foreign", "[Image 2]", "final"]
        ),
        f"{setup_error}owui={'.'.join(map(str, owui_version)) or '?'} "
        f"turn3: HTTP {getattr(c3, 'http_status', None)} "
        f"done={getattr(c3, 'done', None)} {_edit_report(r3)}",
    )
    t.check(
        "imgedit.api-foreign",
        "API path: a markdown link to another user's file in the history is not "
        "read for a non-admin user (logged without the file id), an admin's "
        "request reads it",
        not setup_error
        and getattr(ru, "status", None) == 200
        and getattr(ra, "status", None) == 200
        and user_req["sent"] == ["Change the name to Bob"]
        and admin_req["sent"] == ["Change the name to Bob", "[Image 1]", "foreign"]
        and denied > 0
        and not id_logged,
        f"{setup_error}user: HTTP {getattr(ru, 'status', None)} "
        f"{_edit_report(user_req)} admin: HTTP {getattr(ra, 'status', None)} "
        f"{_edit_report(admin_req)} denied_logged={denied} file_id_logged={id_logged}",
    )
    t.check(
        "imgedit.video-file",
        "Veo image-to-video, API path, non-admin user: a markdown link to the "
        "user's own upload is sent as instances[0].image",
        not setup_error
        and getattr(vo, "status", None) == 200
        and own_image == "upload1",
        f"{setup_error}HTTP {getattr(vo, 'status', None)} image={own_image!r}",
    )
    t.check(
        "imgedit.video-foreign",
        "Veo image-to-video, API path, non-admin user: a markdown link to another "
        "user's file is not read (logged), the video is made from the text",
        not setup_error
        and getattr(vf, "status", None) == 200
        and foreign_image == ""
        and video_denied > 0,
        f"{setup_error}HTTP {getattr(vf, 'status', None)} image={foreign_image!r} "
        f"denied_logged={video_denied}",
    )


# ------------------------------------------------------ native tool calling
# The scenarios live in suites/_gemini_tools.py (imported here, after this
# module is complete, because it uses the helpers above).
async def toolsapi_group(t: Suite, mock) -> None:
    from . import _gemini_tools

    await _gemini_tools.toolsapi(t, mock)


async def tools_group(t: Suite, mock) -> None:
    from . import _gemini_tools

    await _gemini_tools.tools(t, mock)


# Open WebUI's built-in image and code interpreter tools: suites/_gemini_owuitools.py
async def owuitools_group(t: Suite, mock) -> None:
    from . import _gemini_owuitools

    await _gemini_owuitools.owuitools(t, mock)
