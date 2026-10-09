"""
Shared helpers for the provider mocks.

Every mock is an aiohttp application that

- records each upstream request (method, path, query, selected headers, JSON body),
- exposes the record via ``GET /__requests`` and clears it via ``POST /__reset``,
- answers Open WebUI background-task prompts (title / tags / follow-ups) with the
  JSON those tasks expect, so title generation through a pipe can be asserted.

The mocks only listen on 127.0.0.1 inside the Open WebUI container; see
``serve_all.py`` for the ports.
"""

import asyncio
import json
import os
import time
from typing import Any, Optional

from aiohttp import web

# Headers worth recording (secrets are mock values, so recording them is fine and
# lets scenarios assert that encrypted valves are decrypted before use).
RECORDED_HEADERS = (
    "authorization",
    "api-key",
    "x-goog-api-key",
    "x-ms-model-mesh-model-name",
    "content-type",
    "cf-access-client-id",
    "cf-access-client-secret",
    "x-openwebui-user-id",
    "x-openwebui-user-email",
    "x-openwebui-chat-id",
)

REQUESTS_KEY = web.AppKey("requests", list)

TASK_TITLE = "Mock Title"
TASK_TAGS = ["General", "Testing"]
TASK_FOLLOW_UPS = ["Tell me more", "Why?"]

# 1x1 PNG (red pixel), used as "generated image" by the Gemini mock.
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJ"
    "RU5ErkJggg=="
)


def fault_status() -> Optional[int]:
    """HTTP status every provider route answers with (``E2E_MOCK_FAULT``,
    e.g. ``500``), or None for normal behaviour."""
    value = os.environ.get("E2E_MOCK_FAULT", "").strip()
    return int(value) if value.isdigit() and 400 <= int(value) <= 599 else None


@web.middleware
async def _fault_middleware(request: web.Request, handler) -> web.StreamResponse:
    """Fault injection: every provider route (all but ``/__*`` control routes)
    is recorded and answered with ``E2E_MOCK_FAULT``."""
    status = fault_status()
    if status is None or request.path.startswith("/__"):
        return await handler(request)
    await record(request, fault=status)
    return web.json_response(
        {
            "error": {
                "code": status,
                "message": f"e2e mock fault injection (E2E_MOCK_FAULT={status})",
                "status": "E2E_MOCK_FAULT",
            }
        },
        status=status,
    )


def new_app() -> web.Application:
    """Create an application with the request recorder and control routes.

    Control routes: ``GET /__requests`` (record as JSON list), ``POST /__reset``
    (clear the record), ``POST /__shutdown`` (exit the mock process).

    Fault injection: with ``E2E_MOCK_FAULT=<4xx|5xx>`` in the mock process's
    environment every provider route answers with that HTTP status (used to
    check that KNOWN results do not hide unrelated failures).
    """
    app = web.Application(
        client_max_size=64 * 1024 * 1024, middlewares=[_fault_middleware]
    )
    app[REQUESTS_KEY] = []
    app.router.add_get("/__requests", _list_requests)
    app.router.add_post("/__reset", _reset_requests)
    app.router.add_post("/__shutdown", _shutdown)
    return app


async def _list_requests(request: web.Request) -> web.Response:
    return web.json_response(request.app[REQUESTS_KEY])


async def _reset_requests(request: web.Request) -> web.Response:
    request.app[REQUESTS_KEY].clear()
    return web.json_response({"ok": True})


async def _shutdown(request: web.Request) -> web.Response:
    asyncio.get_running_loop().call_later(0.2, os._exit, 0)
    return web.json_response({"ok": True})


async def record(request: web.Request, **extra: Any) -> Any:
    """Read the request body, append an entry to the record and return the body.

    The body is returned parsed as JSON when possible, otherwise as text.
    """
    raw = await request.read()
    try:
        body: Any = json.loads(raw) if raw else None
    except ValueError:
        body = raw[:4000].decode("utf-8", "replace")
    entry = {
        "t": time.time(),
        "method": request.method,
        "path": request.path,
        "query": dict(request.query),
        "headers": {
            k.lower(): v
            for k, v in request.headers.items()
            if k.lower() in RECORDED_HEADERS
        },
        "body": body,
        **extra,
    }
    request.app[REQUESTS_KEY].append(entry)
    request["e2e_entry"] = entry
    return body


def annotate(request: web.Request, **fields: Any) -> None:
    """Add fields to the record entry of ``request`` (after ``record()``)."""
    request["e2e_entry"].update(fields)


def task_answer(text: str) -> Optional[str]:
    """Return the JSON answer for an Open WebUI background-task prompt.

    Open WebUI's task templates start with ``### Task:``; returns ``None`` for
    ordinary chat messages.
    """
    if not isinstance(text, str) or "### Task:" not in text:
        return None
    # The first line after "### Task:" names the task.
    task_line = (text.split("### Task:", 1)[1].strip().splitlines() or [""])[0]
    task_line = task_line.lower()
    if "follow-up" in task_line or "follow up" in task_line:
        return json.dumps({"follow_ups": TASK_FOLLOW_UPS})
    if "tags" in task_line:
        return json.dumps({"tags": TASK_TAGS})
    if "title" in task_line:
        return json.dumps({"title": TASK_TITLE})
    return json.dumps({"result": "mock task answer"})


def last_user_text(messages: Any) -> str:
    """Text of the last user message of an OpenAI-style ``messages`` list."""
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content or "")
    return ""
