"""
"Browser path": chat the way the Open WebUI frontend does.

The frontend keeps a socket.io connection open and sends ``session_id``,
``chat_id``/``parent_id``, the assistant message ``id`` (or ``message_ids``, one
per model, for a multi-model chat) and the ``user_message`` with every
completion request. Open WebUI then runs the pipe as a background task per
model, hands it a real ``__event_emitter__`` (status, source, files, ... events
are pushed to the socket AND persisted into the chat), and returns only
``{status, task_ids, chat_id}``. The answer, usage, sources and statusHistory are
read back from the saved chat. The Stop button is ``POST /api/tasks/stop/<id>``
for those task ids.

Only this path exercises event emitters and persisted content; the API path
(``OWUI.chat``) does not save anything.

Native tool calling: Open WebUI runs the tools of a pipe that answers with
``tool_calls`` (built-in, workspace, MCP and OpenAPI tools on the server; direct
tools in the browser, through an ``execute:tool`` socket call that the browser
acknowledges with the result) and saves the turn as ``output`` items
(``function_call``, ``function_call_output``, ``reasoning``, ``message``).
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .owui import OWUI
from .results import short

DEFAULT_TITLE = "New Chat"
# Answer to an execute:tool call (direct tools) when the scenario set none.
NO_DIRECT_ANSWER = {"error": "no direct tool answer configured"}
# Seconds between socket.io "heartbeat" events (the web UI sends them too).
HEARTBEAT_INTERVAL = 20
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
    model: Optional[str] = None
    task_ids: list = field(default_factory=list)
    # multi-model chat (``chat(models=...)``): one BrowserChat per model, in
    # request order; the chat itself is answers[0]
    answers: list = field(default_factory=list)
    # responses of POST /api/tasks/stop/<id> (``stop_after_s`` / ``stop()``)
    stopped: list = field(default_factory=list)

    @property
    def done(self) -> bool:
        return bool(self.message.get("done"))

    @property
    def content(self) -> str:
        """Saved content; when it is empty (tool turns: Open WebUI's final save
        writes only ``output``) the text of the output's message items."""
        if self.message.get("content"):
            return self.message["content"]
        return self.output_text

    @property
    def output(self) -> list:
        """Output items Open WebUI saved (function calls, reasoning, messages)."""
        return [i for i in self.message.get("output") or [] if isinstance(i, dict)]

    @property
    def output_text(self) -> str:
        """Text of the output's message items, joined."""
        parts = []
        for item in self.output:
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        parts.append(part.get("text", ""))
        return "".join(parts)

    @property
    def function_calls(self) -> list:
        """``[(name, status, call_id, arguments)]`` of the function_call items."""
        return [
            (i.get("name"), i.get("status"), i.get("call_id"), i.get("arguments"))
            for i in self.output
            if i.get("type") == "function_call"
        ]

    @property
    def function_outputs(self) -> dict:
        """``{call_id: text}`` of the function_call_output items."""
        outputs = {}
        for item in self.output:
            if item.get("type") != "function_call_output":
                continue
            value = item.get("output")
            if isinstance(value, list):
                value = "".join(p.get("text", "") for p in value if isinstance(p, dict))
            outputs[item.get("call_id")] = value if isinstance(value, str) else ""
        return outputs

    @property
    def reasoning_items(self) -> list:
        """``[(content text, reasoning_details)]`` of the reasoning items."""
        return [
            (
                "".join(
                    p.get("text", "")
                    for p in i.get("content") or []
                    if isinstance(p, dict)
                ),
                i.get("reasoning_details") or [],
            )
            for i in self.output
            if i.get("type") == "reasoning"
        ]

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

    def answer(self, model: str) -> Optional["BrowserChat"]:
        """The (first) answer of ``model`` in a multi-model chat."""
        return next((a for a in self.answers if a.model == model), None)

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
        # execute:tool calls Open WebUI made to this "browser" (direct tools) and
        # the answer it acknowledges them with: a callable(call data) -> ack,
        # e.g. ``lambda d: [{"value": 1}, {}]`` ([result, headers]; a plain dict
        # is sent as JSON; None makes Open WebUI report unparsable arguments).
        self.execute_calls: list = []
        self.direct_tool_answer = lambda data: NO_DIRECT_ANSWER
        self._heartbeat: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "BrowserSession":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def connect(self) -> None:
        import socketio  # python-socketio ships with the Open WebUI image

        # handle_sigint=False: engine.io would replace the driver's SIGINT
        # handler on the event loop with one that cancels every task and stops
        # the loop, so an interrupted run was not reported as interrupted.
        self.sio = socketio.AsyncClient(reconnection=False, handle_sigint=False)

        @self.sio.on("events")
        async def _on_events(data):
            self.events.append(data)
            inner = data.get("data") if isinstance(data, dict) else None
            if isinstance(inner, dict) and inner.get("type") == "execute:tool":
                # Open WebUI waits for this ack (sio.call): the tool result
                self.execute_calls.append(data)
                return self.direct_tool_answer(inner.get("data") or {})
            return None

        await self.sio.connect(
            self.owui.base,
            socketio_path="/ws/socket.io",
            auth={"token": self.owui.token},
            transports=["websocket"],
            wait_timeout=15,
        )
        self._heartbeat = asyncio.create_task(self._beat(self.sio))
        await asyncio.sleep(0.3)

    @staticmethod
    async def _beat(sio) -> None:
        """Send ``heartbeat`` like the web UI: Open WebUI reaps a session that
        sent none for 120 s, and server -> client calls (direct tools,
        ``__event_call__``) to it then fail with "Client session disconnected."."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                await sio.emit("heartbeat", {})
            except Exception:  # noqa: BLE001 - disconnected: nothing to keep alive
                return

    async def close(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            self._heartbeat = None
        if self.sio is not None:
            await self.sio.disconnect()
            self.sio = None
            await asyncio.sleep(0.5)

    @property
    def sid(self) -> Optional[str]:
        """The Socket.IO namespace sid (what the web UI sends as ``socket.id``).

        ``AsyncClient.sid`` is the engine.io sid, which Open WebUI does not know:
        every server -> client call (direct tools, ``__event_call__``) then
        answered "Client session disconnected."
        """
        return self.sio.get_sid() if self.sio else None

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
        models: Optional[list] = None,
        stop_after_s: Optional[float] = None,
        stop_wait: float = 15,
        tool_ids: Optional[list] = None,
        tool_servers: Optional[list] = None,
        extra_body: Optional[dict] = None,
        until=None,
        user_files: Optional[list] = None,
    ) -> BrowserChat:
        """Send one user message and wait until the saved answer is ``done``.

        ``background_tasks`` e.g. ``{"title_generation": True}`` lets Open WebUI
        run its background tasks after the answer (they call the pipe again with
        ``__task__`` set); ``wait_title`` waits for the generated chat title.

        ``models``: answer with several models side by side (the web UI's
        multi-model chat): the request carries ``message_ids`` with one
        assistant message id per model (``model`` is put first when it is not
        ``models[0]``). The result is the first model's answer; ``answers``
        holds every model's answer and the call waits until all are done.

        ``stop_after_s``: press Stop that many seconds after sending
        (``POST /api/tasks/stop/<task id>`` for every task); then wait up to
        ``stop_wait`` seconds for ``done`` (a stopped answer may never be done).

        ``tool_ids`` (workspace tools, ``server:<id>``, ``server:mcp:<id>``) and
        ``tool_servers`` (direct tool servers, answered through
        ``direct_tool_answer``) go into the request as the web UI sends them;
        ``extra_body`` is merged into the request body last. ``until`` is a
        predicate on the saved message that also ends the wait (e.g. a tool call
        waiting for approval: ``status == "pending"``).

        ``user_files``: the file items of the user message (saved with it, e.g.
        an upload from ``owui.upload_file``). Like the web UI, image files are
        not sent as the request's ``files``: Open WebUI turns the image files
        of the saved user messages into image_url parts.
        """
        if models:
            ids = list(models) if models[0] == model else [model, *models]
        else:
            ids = [model]
        assistant_ids = [str(uuid.uuid4()) for _ in ids]
        user_id = str(uuid.uuid4())
        user_message = {
            "id": user_id,
            "parentId": parent_id,
            "childrenIds": assistant_ids,
            "role": "user",
            "content": text,
            "timestamp": int(time.time()),
            "models": ids,
        }
        if user_files is not None:
            user_message["files"] = list(user_files)
        body = {
            "model": ids[0],
            "stream": stream,
            "messages": list(history or []) + [{"role": "user", "content": text}],
            "session_id": self.sid,
            "parent_id": parent_id,
            "user_message": user_message,
            "features": {**FRONTEND_FEATURES, **(features or {})},
            "params": params or {},
            "background_tasks": background_tasks or {},
        }
        if models:
            body["message_ids"] = [
                {"model_id": m, "message_id": a, "modelIdx": i}
                for i, (m, a) in enumerate(zip(ids, assistant_ids))
            ]
        else:
            body["id"] = assistant_ids[0]
        if chat_id:
            body["chat_id"] = chat_id
        if tool_ids is not None:
            body["tool_ids"] = list(tool_ids)
        if tool_servers is not None:
            body["tool_servers"] = list(tool_servers)
        body.update(extra_body or {})
        status, data = await self.owui.api("POST", "/api/chat/completions", body)
        answers = [
            BrowserChat(status, data, message_id=a, model=m)
            for m, a in zip(ids, assistant_ids)
        ]
        result = answers[0]
        result.answers = answers
        if status != 200 or not isinstance(data, dict) or not data.get("chat_id"):
            return result
        for answer in answers:
            answer.chat_id = data["chat_id"]
            answer.task_ids = list(data.get("task_ids") or [])

        if stop_after_s is not None:
            await asyncio.sleep(stop_after_s)
            await self.stop(result)
            wait = min(wait, stop_wait)
        await self._wait(answers, wait, until)
        if wait_title:
            title_deadline = time.time() + 60
            while time.time() < title_deadline and result.title in (
                None,
                DEFAULT_TITLE,
            ):
                await asyncio.sleep(1)
                await self._load(result)
        for answer in answers[1:]:
            await self.reload(answer)
        return await self.reload(result)

    async def wait(
        self, result: BrowserChat, wait: float = 120, until=None
    ) -> BrowserChat:
        """Wait until the saved answer is ``done`` (or ``until(message)`` holds),
        e.g. after resolving a tool call that waited for approval."""
        await self._wait([result], wait, until)
        return await self.reload(result)

    async def _wait(self, answers: list, wait: float, until=None) -> None:
        deadline = time.time() + wait
        while time.time() < deadline:
            await asyncio.sleep(0.5)
            messages = [await self._load(a) for a in answers]
            if all(m.get("done") for m in messages):
                break
            if until is not None and any(until(m) for m in messages):
                break
        # Outlet filters and late events are persisted right after "done".
        await asyncio.sleep(1.5)

    async def stop(self, result: BrowserChat) -> list:
        """Press Stop: ``POST /api/tasks/stop/<id>`` for every task of the chat
        request. Returns and records ``(HTTP status, body)`` per task."""
        result.stopped = [await self.owui.stop_task(tid) for tid in result.task_ids]
        return result.stopped

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
