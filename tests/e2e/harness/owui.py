"""
Async REST client for the Open WebUI endpoints the scenarios need.

Gotchas encoded here (Open WebUI 0.11.x):
- the first account created on a fresh volume is the admin (``signup``)
- ``POST /api/v1/functions/id/{id}/valves/update`` REPLACES all valves: unsent
  valves fall back to their defaults. ``update_valves`` therefore merges the
  current values with the changes and always sends the full set
- ``/api/models`` is cached; pass ``refresh=true`` after creating models or
  changing valves that influence the model list (``replace_valves`` and
  ``upsert_model`` do that, so a later ``--reuse`` run never sees a stale list)
- "API path" = ``POST /api/chat/completions`` without ``chat_id``; Open WebUI
  answers directly (JSON or SSE) and runs outlet filters, but nothing is saved
- a streamed answer only counts ``choices[].delta.content``: Open WebUI's
  middleware and OpenAI clients ignore ``choices[].message`` in stream chunks
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import quote

import httpx

from .config import ADMIN_EMAIL, ADMIN_NAME, ADMIN_PASSWORD, OWUI_URL
from .results import short

# Error recorded when a stream chunk carries a full chat.completion message.
STREAM_MESSAGE_ERROR = (
    "stream answered with chat.completion (choices[].message.content in a stream "
    "chunk; Open WebUI and OpenAI clients ignore it)"
)
# Plaintext secrets shorter than this are not tracked (too likely to occur in
# unrelated log text).
MIN_SECRET_LENGTH = 6


@dataclass
class ChatResult:
    """Outcome of an API-path chat completion."""

    status: int
    stream: bool
    content: str = ""
    usage: Optional[dict] = None
    errors: list = field(default_factory=list)
    done: bool = False  # SSE terminated by "data: [DONE]"
    chunks: int = 0
    raw: str = ""
    json: Any = None
    # stream only: choices[].message.content of chat.completion chunks (not part
    # of ``content``, see STREAM_MESSAGE_ERROR)
    message_content: str = ""
    # stream only: SSE events after the first "data: [DONE]" (OpenAI-style
    # clients stop reading there) and the delta content among them (also part
    # of ``content``)
    after_done: int = 0
    after_done_content: str = ""

    def brief(self) -> str:
        return (
            f"HTTP {self.status} stream={self.stream} content={short(self.content)} "
            f"usage={self.usage} errors={short(self.errors, 200)}"
        )


def parse_sse(text: str) -> dict:
    """Parse an OpenAI-style SSE body into content / usage / errors / [DONE].

    Only ``choices[].delta.content`` counts as content. A chunk with a full
    ``choices[].message`` (a chat.completion dict answered to a stream request)
    goes to ``message_content`` and adds ``STREAM_MESSAGE_ERROR`` to ``errors``.
    Events after the first ``[DONE]`` are still parsed, and also counted in
    ``after_done`` (their delta content in ``after_done_content``): an
    OpenAI-style client stops reading at ``[DONE]`` and never sees them.
    """
    out = {
        "content": "",
        "usage": None,
        "errors": [],
        "done": False,
        "chunks": 0,
        "message_content": "",
        "after_done": 0,
        "after_done_content": "",
    }
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if out["done"]:
            out["after_done"] += 1
        if payload == "[DONE]":
            out["done"] = True
            continue
        try:
            data = json.loads(payload)
        except ValueError:
            out["errors"].append(f"unparsable SSE line: {payload[:120]}")
            continue
        out["chunks"] += 1
        if isinstance(data, dict) and data.get("error"):
            out["errors"].append(data["error"])
        for choice in (data.get("choices") if isinstance(data, dict) else None) or []:
            delta_text = (choice.get("delta") or {}).get("content") or ""
            out["content"] += delta_text
            if out["done"]:
                out["after_done_content"] += delta_text
            message = choice.get("message")
            if isinstance(message, dict):
                out["message_content"] += message.get("content") or ""
                if STREAM_MESSAGE_ERROR not in out["errors"]:
                    out["errors"].append(STREAM_MESSAGE_ERROR)
        if isinstance(data, dict) and data.get("usage"):
            out["usage"] = data["usage"]
    return out


def completion_text(data: Any) -> Optional[str]:
    """``choices[0].message.content`` of a chat.completion dict, else None."""
    if isinstance(data, dict) and data.get("choices"):
        return (data["choices"][0].get("message") or {}).get("content")
    return None


class OWUI:
    def __init__(self, base: str = OWUI_URL):
        self.base = base.rstrip("/")
        self.http = httpx.AsyncClient(
            base_url=self.base, timeout=httpx.Timeout(300.0, connect=10.0)
        )
        self.token: Optional[str] = None
        self.user: dict = {}
        # Plaintext values written to password valves: value -> "<fid>.<valve>"
        # (``Suite.scan_log`` fails when one of them shows up in the log).
        self.secrets: dict = {}
        self._password_valves: dict = {}

    async def close(self) -> None:
        await self.http.aclose()

    # ------------------------------------------------------------------ basics
    async def wait_healthy(self, timeout: float = 300) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if (await self.http.get("/health", timeout=5)).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
        return False

    async def login(
        self,
        email: str = ADMIN_EMAIL,
        password: str = ADMIN_PASSWORD,
        name: str = ADMIN_NAME,
    ) -> dict:
        """Sign in (default: the e2e admin), signing up first on a fresh volume."""
        creds = {"email": email, "password": password}
        r = await self.http.post("/api/v1/auths/signin", json=creds)
        if r.status_code != 200:
            r = await self.http.post(
                "/api/v1/auths/signup", json={**creds, "name": name}
            )
        r.raise_for_status()
        self.user = r.json()
        self.token = self.user["token"]
        return self.user

    async def create_user(
        self,
        name: str = "E2E User",
        email: str = "user@example.com",
        password: str = ADMIN_PASSWORD,
        role: str = "user",
    ) -> "OWUI":
        """Add a user (admin API; kept when it already exists) and return a
        second client signed in as that user. Close it with ``close()``."""
        status, data = await self.api(
            "POST",
            "/api/v1/auths/add",
            {"name": name, "email": email, "password": password, "role": role},
        )
        other = OWUI(self.base)
        r = await other.http.post(
            "/api/v1/auths/signin", json={"email": email, "password": password}
        )
        if r.status_code != 200:  # add failed for another reason than "exists"
            await other.close()
            raise RuntimeError(
                f"create_user {email}: auths/add HTTP {status} {short(data)}, "
                f"signin HTTP {r.status_code}"
            )
        other.user = r.json()
        other.token = other.user["token"]
        other.secrets = self.secrets  # one registry for every client
        return other

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def request(
        self, method: str, path: str, body: Any = None, timeout: float = 300
    ) -> httpx.Response:
        return await self.http.request(
            method, path, json=body, headers=self.headers, timeout=timeout
        )

    async def api(
        self, method: str, path: str, body: Any = None, timeout: float = 300
    ) -> tuple[int, Any]:
        """Request and decode JSON (falls back to text)."""
        r = await self.request(method, path, body, timeout)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, r.text

    async def version(self) -> str:
        _, data = await self.api("GET", "/api/version")
        return data.get("version", "?") if isinstance(data, dict) else "?"

    # --------------------------------------------------------------- functions
    async def function(self, fid: str) -> Optional[dict]:
        status, data = await self.api("GET", f"/api/v1/functions/id/{fid}")
        return data if status == 200 and isinstance(data, dict) else None

    async def install_function(
        self, fid: str, name: str, content: str, description: str = "e2e"
    ) -> tuple[int, Any]:
        """Create the function, or update its code when it already exists.

        Creating a function with ``requirements:`` pip-installs them first
        (google-genai takes ~30 s), hence the long timeout.
        """
        form = {
            "id": fid,
            "name": name,
            "content": content,
            "meta": {"description": description},
        }
        self._password_valves.pop(fid, None)  # the valves may have changed
        if await self.function(fid):
            return await self.api(
                "POST", f"/api/v1/functions/id/{fid}/update", form, timeout=900
            )
        return await self.api("POST", "/api/v1/functions/create", form, timeout=900)

    async def set_active(self, fid: str, active: bool = True) -> Optional[bool]:
        info = await self.function(fid)
        if info is not None and bool(info.get("is_active")) != active:
            _, info = await self.api("POST", f"/api/v1/functions/id/{fid}/toggle")
        return info.get("is_active") if isinstance(info, dict) else None

    async def set_global(self, fid: str, is_global: bool = True) -> Optional[bool]:
        """Global filters apply to every model (``/toggle/global``)."""
        info = await self.function(fid)
        if info is not None and bool(info.get("is_global")) != is_global:
            _, info = await self.api(
                "POST", f"/api/v1/functions/id/{fid}/toggle/global"
            )
        return info.get("is_global") if isinstance(info, dict) else None

    async def get_valves(self, fid: str) -> dict:
        _, data = await self.api("GET", f"/api/v1/functions/id/{fid}/valves")
        return data if isinstance(data, dict) else {}

    async def valves_spec(self, fid: str, user: bool = False) -> dict:
        """JSON schema of the function's Valves (``user=True``: UserValves)."""
        path = f"/api/v1/functions/id/{fid}/valves/{'user/' if user else ''}spec"
        _, data = await self.api("GET", path)
        return data if isinstance(data, dict) else {}

    async def valve_names(self, fid: str, user: bool = False) -> list:
        """Sorted valve names (Valves, or UserValves with ``user=True``)."""
        return sorted((await self.valves_spec(fid, user)).get("properties") or {})

    async def _remember_secrets(self, fid: str, valves: dict) -> None:
        """Track plaintext values written to password-typed valves."""
        if fid not in self._password_valves:
            props = (await self.valves_spec(fid)).get("properties") or {}
            self._password_valves[fid] = {
                name
                for name, spec in props.items()
                if ((spec or {}).get("input") or {}).get("type") == "password"
            }
        for name in self._password_valves[fid]:
            value = valves.get(name)
            if (
                isinstance(value, str)
                and len(value) >= MIN_SECRET_LENGTH
                and not value.startswith("encrypted:")
            ):
                self.secrets[value] = f"{fid}.{name}"

    async def replace_valves(self, fid: str, valves: dict) -> dict:
        """Store exactly ``valves`` (every valve not sent reverts to its default).

        Valves can change what ``pipes()`` returns (e.g. ``AZURE_AI_MODEL``), so
        the cached model list is refreshed afterwards; otherwise chat requests
        answer "Model not found" until something else refreshes it. Plaintext
        values of password valves are remembered in ``secrets``.
        """
        await self._remember_secrets(fid, valves)
        status, data = await self.api(
            "POST", f"/api/v1/functions/id/{fid}/valves/update", valves
        )
        if status != 200:
            raise RuntimeError(f"valves/update {fid} failed: HTTP {status} {data}")
        await self.models(refresh=True)
        return data

    async def update_valves(self, fid: str, **changes: Any) -> dict:
        """Change some valves, keeping the others (sends the merged full set)."""
        return await self.replace_valves(fid, {**await self.get_valves(fid), **changes})

    # ------------------------------------------------------------------ models
    async def models(self, refresh: bool = True) -> list:
        path = "/api/models?refresh=true" if refresh else "/api/models"
        _, data = await self.api("GET", path, timeout=120)
        return data.get("data", []) if isinstance(data, dict) else []

    async def model_ids(self, prefix: str = "") -> list:
        return [m["id"] for m in await self.models() if m["id"].startswith(prefix)]

    async def upsert_model(
        self,
        model_id: str,
        name: str,
        filter_ids: Optional[list] = None,
        capabilities: Optional[dict] = None,
    ) -> int:
        """Create/update the workspace model that overrides a pipe model.

        This is how filters are attached per model (``meta.filterIds``). The
        model cache is refreshed afterwards so chat completions see the change.
        """
        form = {
            "id": model_id,
            "base_model_id": None,
            "name": name,
            "params": {},
            "meta": {
                "description": "e2e",
                "filterIds": list(filter_ids or []),
                "capabilities": capabilities or {},
            },
        }
        status, existing = await self.api(
            "GET", f"/api/v1/models/model?id={quote(model_id)}"
        )
        if status == 200 and isinstance(existing, dict) and existing.get("id"):
            status, _ = await self.api("POST", "/api/v1/models/model/update", form)
        else:
            status, _ = await self.api("POST", "/api/v1/models/create", form)
        await self.models(refresh=True)
        return status

    async def delete_model(self, model_id: str) -> bool:
        """Delete a workspace model (override); False when there is none.

        The pipe model it overrode shows up again (cache refreshed).
        """
        status, data = await self.api(
            "POST", "/api/v1/models/model/delete", {"id": model_id}
        )
        await self.models(refresh=True)
        return status == 200 and data is True

    async def model_filter_ids(self, model_id: str) -> Optional[list]:
        for model in await self.models(refresh=True):
            if model["id"] == model_id:
                return ((model.get("info") or {}).get("meta") or {}).get("filterIds")
        return None

    # -------------------------------------------------------------------- chat
    async def chat(
        self,
        model: str,
        messages: Any,
        stream: bool = False,
        timeout: float = 300,
        **extra: Any,
    ) -> ChatResult:
        """API-path chat completion (what an OpenAI-compatible client does).

        ``messages`` may be a plain string (one user message).
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"model": model, "messages": messages, "stream": stream, **extra}
        r = await self.request("POST", "/api/chat/completions", body, timeout)
        result = ChatResult(status=r.status_code, stream=stream, raw=r.text)
        if stream and "text/event-stream" in r.headers.get("content-type", ""):
            parsed = parse_sse(r.text)
            result.content = parsed["content"]
            result.usage = parsed["usage"]
            result.errors = parsed["errors"]
            result.done = parsed["done"]
            result.chunks = parsed["chunks"]
            result.message_content = parsed["message_content"]
            result.after_done = parsed["after_done"]
            result.after_done_content = parsed["after_done_content"]
            return result
        try:
            result.json = r.json()
        except ValueError:
            result.content = r.text
            if r.status_code != 200:
                result.errors.append(r.text[:300])
            return result
        if isinstance(result.json, dict):
            result.content = completion_text(result.json) or ""
            result.usage = result.json.get("usage")
            if r.status_code != 200 or result.json.get("detail"):
                result.errors.append(result.json.get("detail") or result.json)
        return result

    async def title_task(
        self, model: str, messages: list, chat_id: Optional[str] = None
    ) -> tuple[int, Optional[str], Any]:
        """Run the title background task directly (no event emitter).

        Returns (HTTP status, answer text, raw JSON).
        """
        body = {"model": model, "messages": messages}
        if chat_id:
            body["chat_id"] = chat_id
        status, data = await self.api(
            "POST", "/api/v1/tasks/title/completions", body, timeout=300
        )
        return status, completion_text(data), data

    async def get_chat(self, chat_id: str) -> dict:
        _, data = await self.api("GET", f"/api/v1/chats/{chat_id}")
        return data if isinstance(data, dict) else {}

    async def stop_task(self, task_id: str) -> tuple[int, Any]:
        """Stop a running chat task (the web UI's Stop button)."""
        return await self.api("POST", f"/api/tasks/stop/{quote(task_id)}")

    async def file_status(self, url: str) -> tuple[int, str]:
        """(HTTP status, content type) of an Open WebUI file URL."""
        path = url[len(self.base) :] if url.startswith(self.base) else url
        r = await self.request("GET", path)
        return r.status_code, r.headers.get("content-type", "")
