"""
"Browser path": chat the way the Open WebUI frontend does.

The frontend keeps a socket.io connection open and sends ``session_id``,
``chat_id``/``parent_id``, the assistant message ``id`` and the ``user_message``
with every completion request. Open WebUI then runs the pipe as a background
task, hands it a real ``__event_emitter__`` (status, source, files, ... events
are pushed to the socket AND persisted into the chat), and returns only
``{status, task_ids, chat_id}``. The answer, usage, sources and statusHistory are
read back from the saved chat.

Only this path exercises event emitters and persisted content; the API path
(``OWUI.chat``) does not save anything.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .owui import OWUI
from .results import short

DEFAULT_TITLE = "New Chat"
# The web UI always sends these feature flags (toggles in the message input).
FRONTEND_FEATURES = {
    "image_generation": False,
    "code_interpreter": False,
    "web_search": False,
    "memory": False,
}


@dataclass
class BrowserChat:
    """A browser-path completion and what Open WebUI saved for it."""

    http_status: int
    response: Any
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    message: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    title: Optional[str] = None

    @property
    def done(self) -> bool:
        return bool(self.message.get("done"))

    @property
    def content(self) -> str:
        if self.message.get("content"):
            return self.message["content"]
        parts = []  # newer Open WebUI keeps the answer in "output" items as well
        for item in self.message.get("output") or []:
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text":
                        parts.append(part.get("text", ""))
        return "".join(parts)

    @property
    def usage(self) -> Optional[dict]:
        return self.message.get("usage")

    @property
    def sources(self) -> list:
        return self.message.get("sources") or []

    @property
    def source_names(self) -> list:
        return [(s.get("source") or {}).get("name") for s in self.sources]

    @property
    def status_history(self) -> list:
        return self.message.get("statusHistory") or []

    @property
    def status_descriptions(self) -> list:
        return [s.get("description") for s in self.status_history]

    @property
    def files(self) -> list:
        return self.message.get("files") or []

    @property
    def error(self) -> Any:
        return self.message.get("error")

    @property
    def event_types(self) -> list:
        return [e.get("type") for e in self.events]

    def brief(self) -> str:
        return (
            f"HTTP {self.http_status} done={self.done} content={short(self.content)} "
            f"usage={self.usage} status={self.status_descriptions} "
            f"sources={self.source_names} files={len(self.files)} "
            f"error={short(self.error) if self.error else None}"
        )


class BrowserSession:
    """socket.io connection + saved-chat completions, like a browser tab.

    Use as ``async with BrowserSession(owui) as browser: ...``. Close it before
    running API-path scenarios that must not see a websocket session (some
    behaviour differs when the user has a connected socket).
    """

    def __init__(self, owui: OWUI):
        self.owui = owui
        self.sio = None
        self.events: list = []

    async def __aenter__(self) -> "BrowserSession":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def connect(self) -> None:
        import socketio  # python-socketio ships with the Open WebUI image

        self.sio = socketio.AsyncClient(reconnection=False)

        @self.sio.on("events")
        async def _on_events(data):
            self.events.append(data)

        await self.sio.connect(
            self.owui.base,
            socketio_path="/ws/socket.io",
            auth={"token": self.owui.token},
            transports=["websocket"],
            wait_timeout=15,
        )
        await asyncio.sleep(0.3)

    async def close(self) -> None:
        if self.sio is not None:
            await self.sio.disconnect()
            self.sio = None
            await asyncio.sleep(0.5)

    @property
    def sid(self) -> Optional[str]:
        return self.sio.sid if self.sio else None

    async def chat(
        self,
        model: str,
        text: str,
        stream: bool = True,
        features: Optional[dict] = None,
        history: Optional[list] = None,
        chat_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        background_tasks: Optional[dict] = None,
        params: Optional[dict] = None,
        wait: float = 120,
        wait_title: bool = False,
    ) -> BrowserChat:
        """Send one user message and wait until the saved answer is ``done``.

        ``background_tasks`` e.g. ``{"title_generation": True}`` lets Open WebUI
        run its background tasks after the answer (they call the pipe again with
        ``__task__`` set); ``wait_title`` waits for the generated chat title.
        """
        assistant_id, user_id = str(uuid.uuid4()), str(uuid.uuid4())
        user_message = {
            "id": user_id,
            "parentId": parent_id,
            "childrenIds": [assistant_id],
            "role": "user",
            "content": text,
            "timestamp": int(time.time()),
            "models": [model],
        }
        body = {
            "model": model,
            "stream": stream,
            "messages": list(history or []) + [{"role": "user", "content": text}],
            "session_id": self.sid,
            "id": assistant_id,
            "parent_id": parent_id,
            "user_message": user_message,
            "features": {**FRONTEND_FEATURES, **(features or {})},
            "params": params or {},
            "background_tasks": background_tasks or {},
        }
        if chat_id:
            body["chat_id"] = chat_id
        status, data = await self.owui.api("POST", "/api/chat/completions", body)
        result = BrowserChat(status, data, message_id=assistant_id)
        if status != 200 or not isinstance(data, dict) or not data.get("chat_id"):
            return result
        result.chat_id = data["chat_id"]

        deadline = time.time() + wait
        while time.time() < deadline:
            await asyncio.sleep(0.5)
            if (await self._load(result)).get("done"):
                break
        # Outlet filters and late events are persisted right after "done".
        await asyncio.sleep(1.5)
        if wait_title:
            title_deadline = time.time() + 60
            while time.time() < title_deadline and result.title in (
                None,
                DEFAULT_TITLE,
            ):
                await asyncio.sleep(1)
                await self._load(result)
        return await self.reload(result)

    async def reload(self, result: BrowserChat) -> BrowserChat:
        """Re-read the saved message and its socket events (e.g. after
        background tasks that run once the answer is done)."""
        await self._load(result)
        result.events = [
            (e.get("data") or {})
            for e in self.events
            if e.get("message_id") == result.message_id
        ]
        return result

    async def _load(self, result: BrowserChat) -> dict:
        saved = await self.owui.get_chat(result.chat_id)
        chat = saved.get("chat") or {}
        result.title = saved.get("title") or chat.get("title")
        messages = (chat.get("history") or {}).get("messages") or {}
        result.message = messages.get(result.message_id) or {}
        return result.message
