"""
Native tool calling scenarios of the gemini suite: groups ``tools`` (browser
path: Open WebUI's tool loop runs the tools) and ``toolsapi`` (API path: the
client gets the tool calls). Run by suites/gemini.py; the leading underscore
keeps this module out of the suite discovery.

The Gemini mock answers the ``MOCKTOOLS:`` directive (mocks/mock_gemini.py) with
function calls and records what the pipe sent upstream; the tool servers are
mocks/mock_tools.py (OpenAPI on 9111, MCP on 9112) and probe/workspace_tool.py
(a workspace Python tool).

Conventions (docs/testing.md, "Native tool calling"):
- every group starts with a preflight: a plain API request to the text model
  must answer "Hello from mock"; every detail starts with ``mock_ok=<bool>``, the
  proof that the provider works, which the evidence of a known bug of these
  groups must include (with ``E2E_MOCK_FAULT`` it is False, so nothing is
  reported as KNOWN)
- stable detail tokens: ``http= done= calls=[name:status] outputs=<n>
  rd=[format:id] upstream=[answers] declared_n= safe_names= sig_echoed=[id:sig]
  fr=[id:name:keys] kinds=[role:parts] tool_kinds=[...] include_flag=
  server_echoed= finish=[...] openai_finish= usage=`` and ``final=`` (last)
"""

import base64
import copy
import hashlib
import json
import re
from typing import Optional

import httpx

from harness import Suite, short
from harness.config import ADMIN_EMAIL, MCP_PORT, MOCK_HOST, MOCK_PORTS
from harness.config import WORKSPACE_TOOL_FILE

from .gemini import (
    GROUNDING_URI,
    PRO3,
    SEARCH_FILTER,
    TEXT,
    USAGE_STREAM,
    _body,
    _filter_ready,
    _gen,
    _last_text,
    _set,
    _status_closed,
    _strings,
    _usage,
)

PRO3_ID = PRO3.split(".", 1)[1]  # the model id the mock sees
WS_TOOL = "e2e_ws_tool"
OPENAPI_ID = "e2etools"
MCP_ID = "e2emcp"
TOOLS_URL = f"http://{MOCK_HOST}:{MOCK_PORTS['tools']}"
MCP_URL = f"http://{MOCK_HOST}:{MCP_PORT}/mcp"
LOOKUP = "lookup.v2"  # operationId of mocks/mock_tools.py that needs mapping
ASK = {"tool_approval_mode": "ask"}
LEGACY = {"function_calling": "legacy"}
CALL_0 = "mock-call-0-0"
# The mock's source of a tool round with Google Search (final answers: GROUNDING_URI)
TOOL_ROUND_URI = "https://example.com/tool-round"
TIMESTAMP = {"name": "get_current_timestamp", "args": {}}
CALC = {"name": "calculate_timestamp", "args": {"days_ago": 1}}
REJECTED = "Error: tool call rejected by user."
NOT_FOUND = 'Error: Tool "no_such_tool" not found.'
# Server log texts of the AFC era (google_gemini.py 1.17.0): none may show up
LOG_FORBIDDEN = (
    "'callable'",
    "__signature__",
    "Duplicate function declaration",
    "AFC is enabled",
)
# Open WebUI's direct tool server (the browser answers execute:tool)
DIRECT_SERVER = {
    "url": "http://direct.local:9999",
    "openapi": {"info": {"title": "Direct"}},
    "info": {"title": "Direct", "version": "1"},
    "specs": [
        {
            "name": "direct_lookup",
            "description": "Look something up on the client side",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        }
    ],
}
# Usage Open WebUI saves for two tool rounds (USAGE_TOOL of the mock: 21 / 5 /
# 26) and a streamed final answer (9 / 4 / 13): input / output / total summed,
# prompt / completion of the last round (F7 of the spec).
USAGE_ROUNDS = {
    "input_tokens": 21 + 21 + 9,
    "output_tokens": 5 + 5 + 4,
    "total_tokens": 26 + 26 + 13,
    "prompt_tokens": 9,
    "completion_tokens": 4,
}
# The pipe's name mapping (spec section 3.4, copied into mocks/mock_gemini.py too)
_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


def gemini_function_name(name: str) -> str:
    if _SAFE_NAME.fullmatch(name):
        return name
    base = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    if not re.match(r"[A-Za-z_]", base[:1] or "0"):
        base = "_" + base
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base[:55]}_{digest}"


def issued(ref: str, model: str = PRO3_ID) -> str:
    """The thought signature the mock issues for a call (id, or name w/o id)."""
    return base64.b64encode(f"mock-sig|{model}|{ref}".encode()).decode()


def rd_item(call_id: str, index: int = 0) -> dict:
    """reasoning_details item the pipe returns for a signed function call."""
    return {
        "type": "reasoning.encrypted",
        "format": "google-gemini-v1",
        "id": call_id,
        "index": index,
        "data": issued(call_id),
    }


def directive(rounds: list, **options) -> str:
    """User text that makes the mock answer with function calls."""
    return "Use the tools. MOCKTOOLS:" + json.dumps({"rounds": rounds, **options})


def client_tool(name: str, parameters: Optional[dict] = None, desc: str = "") -> dict:
    """An OpenAI tool spec as API clients send it."""
    function = {"name": name, "description": desc or f"client function {name}"}
    if parameters is not None:
        function["parameters"] = parameters
    return {"type": "function", "function": function}


CLIENT_FN = client_tool(
    "client_fn",
    {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
)


def _json(text) -> dict:
    try:
        value = json.loads(text) if isinstance(text, str) else text
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _answers(reqs: list) -> str:
    """``[fc,final]``: what the mock answered, per generate request."""
    kinds = []
    for entry in reqs:
        if entry.get("fault"):
            kinds.append(f"http-{entry['fault']}")
        else:
            kinds.append(str(entry.get("answer") or "?"))
    return "[" + ",".join(kinds) + "]"


def _declared(entry: dict) -> list:
    return list(entry.get("declared") or [])


def _safe(names: list) -> bool:
    return all(isinstance(n, str) and _SAFE_NAME.fullmatch(n) for n in names) and len(
        set(names)
    ) == len(names)


def _fcs(entry: dict) -> list:
    return [(x.get("id"), x.get("name"), x.get("sig")) for x in entry.get("fc") or []]


def _frs(entry: dict) -> list:
    return [
        (x.get("id"), x.get("name"), sorted((x.get("response") or {}).keys()))
        for x in entry.get("fr") or []
    ]


def _req_tokens(entry: dict) -> str:
    """kinds / sig_echoed / fr tokens of one recorded request."""
    sigs = ",".join(f"{i}:{s}" for i, _, s in _fcs(entry))
    frs = ",".join(f"{i}:{n}:{'+'.join(k)}" for i, n, k in _frs(entry))
    return (
        f"kinds=[{','.join(entry.get('kinds') or [])}] sig_echoed=[{sigs}] fr=[{frs}]"
    )


def _kinds(entry: dict) -> str:
    return "[" + ",".join(entry.get("tool_kinds") or []) + "]"


def _item_text(item: dict) -> str:
    return "".join(
        p.get("text", "") for p in item.get("content") or [] if isinstance(p, dict)
    )


def _pending(message: dict) -> bool:
    return any(
        i.get("type") == "function_call" and i.get("status") == "pending"
        for i in message.get("output") or []
        if isinstance(i, dict)
    )


def _calls(chat) -> list:
    return [(n, s, i) for n, s, i, _ in chat.function_calls]


def _rd(chat) -> list:
    return [d for _, details in chat.reasoning_items for d in details]


def _final(chat) -> str:
    return (chat.output_text or chat.content or "").strip()


def _no_details(chat) -> bool:
    return "<details" not in chat.output_text and "<details" not in (
        chat.message.get("content") or ""
    )


def _bdetail(chat, reqs: list, extra: str = "") -> str:
    """Stable tokens of a browser-path tool turn."""
    calls = ",".join(f"{n}:{s}" for n, s, _ in _calls(chat))
    rd = ",".join(f"{d.get('format')}:{d.get('id')}" for d in _rd(chat))
    names = _declared(reqs[0]) if reqs else []
    error = short(chat.error, 120) if chat.error else None
    return (
        f"http={chat.http_status} done={chat.done} error={error} calls=[{calls}] "
        f"outputs={len(chat.function_outputs)} rd=[{rd}] upstream={_answers(reqs)} "
        f"declared_n={len(names)} safe_names={_safe(names)} {extra} "
        f"final={short(_final(chat), 200)}"
    )


def _tc(call: dict) -> tuple:
    """(index, id, name, arguments) of a stream-merged or a message tool call."""
    function = call.get("function") or {}
    return (
        call.get("index"),
        call.get("id"),
        call.get("name") or function.get("name"),
        call.get("arguments") or function.get("arguments"),
    )


def _adetail(r, reqs: list, extra: str = "") -> str:
    """Stable tokens of an API-path answer."""
    calls = ",".join(
        f"{i}:{c}:" + str(n).replace("\n", "\\n")  # one line per detail
        for i, c, n, _ in (_tc(x) for x in r.tool_calls)
    )
    rd = ",".join(f"{d.get('format')}:{d.get('id')}" for d in r.reasoning_details)
    names = _declared(reqs[0]) if reqs else []
    return (
        f"http={r.status} tool_calls=[{calls}] rd=[{rd}] "
        f"finish=[{','.join(r.finish_reasons)}] openai_finish={r.openai_finish_reason} "
        f"done_last={r.done_last} usage={_usage(r.usage) if r.usage else None} "
        f"upstream={_answers(reqs)} declared_n={len(names)} {extra} "
        f"content={short(r.content, 200)}"
    )


class Ctx:
    """One group: the suite, the Gemini mock and the preflight result."""

    def __init__(self, t: Suite, mock, mock_ok: bool):
        self.t = t
        self.mock = mock
        self.mock_ok = mock_ok
        self.answered = 0  # generate requests the mock answered with HTTP 200

    def check(self, sid, title, ok, detail) -> bool:
        return self.t.check(sid, title, ok, f"mock_ok={self.mock_ok} {detail}")

    async def chat(self, b, *args, **kwargs):
        """b.chat(...); a chat that is neither done nor waiting for approval is
        stopped, so its tool loop cannot reach the mock during later scenarios."""
        c = await b.chat(*args, **kwargs)
        if c.task_ids and not c.done and not _pending(c.message):
            await b.stop(c)
        return c

    async def requests(self) -> list:
        reqs = await self.mock.requests(_gen)
        self.answered += sum(1 for e in reqs if e.get("status") == 200)
        return reqs


async def preflight(t: Suite, mock) -> Ctx:
    """A plain API request must reach the mock (mock_ok of every detail)."""
    await mock.reset()
    r = await t.owui.chat(TEXT, "Hello preflight", stream=False)
    return Ctx(t, mock, r.status == 200 and "Hello from mock" in r.content)


# ===================================================================== tools
async def tools(t: Suite, mock) -> None:
    ctx = await preflight(t, mock)
    mark = t.mark()
    tool_mock = t.mock("tools")
    with open(WORKSPACE_TOOL_FILE, encoding="utf-8") as fh:
        ws_source = fh.read()
    previous_servers = None
    try:
        ws = await t.owui.create_tool(WS_TOOL, "E2E Workspace Tool", ws_source)
        previous_servers = await t.owui.set_tool_servers(_connections())
        async with t.browser() as b:
            first = await builtin(ctx, b, stream=True)
            await builtin(ctx, b, stream=False)
            await multiturn(ctx, b, first)
            await parallel(ctx, b)
            await rounds(ctx, b)
            await text_before(ctx, b)
            await thinking(ctx, b)
            await workspace(ctx, b, ws)
            await image_result(ctx, b)
            await openapi(ctx, b, tool_mock)
            await mcp(ctx, b)
            await direct(ctx, b)
            await approval(ctx, b)
            await unknown(ctx, b)
            await malformed(ctx, b)
            await grounding(ctx, b)
            await task(ctx, b)
            await legacy(ctx, b)
            await nobuiltin(ctx, b)
            await nostream(ctx, b)
    finally:
        await t.owui.delete_tool(WS_TOOL)
        if previous_servers is not None:
            await t.owui.set_tool_servers(previous_servers)
    text = t.log.since(mark)
    found = [s for s in LOG_FORBIDDEN if s in text]
    ctx.check(
        "tools.log",
        "server log of the tool turns: no 'callable', __signature__, Duplicate "
        "function declaration or AFC is enabled",
        not found,
        f"answered={ctx.answered} found={found}",
    )


def _connections() -> list:
    return [
        {
            "url": TOOLS_URL,
            "path": "openapi.json",
            "type": "openapi",
            "auth_type": "none",
            "key": "",
            "config": {"enable": True},
            "info": {"id": OPENAPI_ID, "name": "E2E Tools", "description": "e2e"},
        },
        {
            "url": MCP_URL,
            "path": "",
            "type": "mcp",
            "auth_type": "none",
            "key": "",
            "config": {"enable": True},
            "info": {"id": MCP_ID, "name": "E2E MCP", "description": "e2e"},
        },
    ]


async def builtin(ctx: Ctx, b, stream: bool):
    """One built-in tool call through Open WebUI's loop (no AFC)."""
    await ctx.mock.reset()
    c = await ctx.chat(b, PRO3, directive([[TIMESTAMP]]), stream=stream)
    reqs = await ctx.requests()
    first, second = (reqs + [{}, {}])[:2]
    names = _declared(first)
    actions = [e.get("action") for e in reqs]
    saved = (
        c.done
        and not c.error
        and _calls(c) == [("get_current_timestamp", "completed", CALL_0)]
        and "current_timestamp" in c.function_outputs.get(CALL_0, "")
        and _rd(c) == [rd_item(CALL_0)]
        and _final(c).startswith("MOCK-FINAL get_current_timestamp=")
        and _no_details(c)
        and _status_closed(c)
    )
    upstream = (
        _answers(reqs) == "[fc,final]"
        and {"get_current_timestamp", "calculate_timestamp"} <= set(names)
        and _safe(names)
        and "tool_function" not in names
        and second.get("kinds") == ["user:text", "model:fc", "user:fr"]
        and _fcs(second) == [(CALL_0, "get_current_timestamp", "issued")]
        and _frs(second) == [(CALL_0, "get_current_timestamp", ["output"])]
    )
    if not stream:
        upstream = upstream and actions == ["generateContent", "streamGenerateContent"]
        # the thoughts of the non-streamed tool round (mock: "Mock thinking.")
        saved = saved and any("Mock thinking." in x for x, _ in c.reasoning_items)
    name = "builtin" if stream else "builtin-nonstream"
    ctx.check(
        f"tools.{name}",
        f"built-in tool, browser stream={stream}: Open WebUI runs it, function call "
        "+ output + signed reasoning item saved, 2 upstream requests (no AFC), "
        "follow-up with FC(sig) + FR"
        + ("" if stream else "; first non-stream, continuation streamed"),
        saved and upstream,
        _bdetail(
            c,
            reqs,
            f"{_req_tokens(second)} actions={actions} no_details={_no_details(c)} "
            f"statuses_closed={_status_closed(c)} "
            f"reasoning={[(short(x, 40), len(d)) for x, d in c.reasoning_items]}",
        ),
    )
    return c


async def multiturn(ctx: Ctx, b, first) -> None:
    """Later turns replay the turn-1 call with its signature (stripped by Open
    WebUI while another model answers)."""
    if not first.chat_id:
        return
    parent = first.message_id
    for sid, model, want in (
        ("multiturn", PRO3, "issued"),
        ("multiturn-other-model", TEXT, "none"),
        ("multiturn-back", PRO3, "issued"),
    ):
        await ctx.mock.reset()
        c = await ctx.chat(
            b, model, f"Next turn ({sid}).", chat_id=first.chat_id, parent_id=parent
        )
        reqs = await ctx.requests()
        req = reqs[-1] if reqs else {}
        old = [x for x in req.get("fc") or [] if x.get("id") == CALL_0]
        old_fr = [x for x in req.get("fr") or [] if x.get("id") == CALL_0]
        ctx.check(
            f"tools.{sid}",
            f"follow-up turn on {model.split('.', 1)[1]}: the turn-1 call is "
            f"replayed with its response, signature {want}",
            c.done
            and req.get("status") == 200
            and len(old) == 1
            and old[0].get("sig") == want
            and len(old_fr) == 1
            and not req.get("orphan_fr"),
            _bdetail(
                c,
                reqs,
                f"old_fc={[(x.get('id'), x.get('sig')) for x in old]} "
                f"old_fr={len(old_fr)} orphan_fr={req.get('orphan_fr')} "
                f"{_req_tokens(req)}",
            ),
        )
        parent = c.message_id


async def parallel(ctx: Ctx, b) -> None:
    for split in (False, True):
        await ctx.mock.reset()
        c = await ctx.chat(b, PRO3, directive([[TIMESTAMP, CALC]], split=split))
        reqs = await ctx.requests()
        second = reqs[1] if len(reqs) > 1 else {}
        ids = ["mock-call-0-0", "mock-call-0-1"]
        user_fr = sorted({x.get("i") for x in second.get("fr") or []})
        final = _final(c)
        ctx.check(
            "tools.parallel" + ("-split" if split else ""),
            f"two calls in one round (split={split}): both run, follow-up has one "
            "model content with 2 FC (sig on the first only) and one user content "
            "with 2 FR",
            c.done
            and [(i, s) for _, s, i in _calls(c)]
            == [(ids[0], "completed"), (ids[1], "completed")]
            and set(c.function_outputs) >= set(ids)
            and _fcs(second)
            == [
                (ids[0], "get_current_timestamp", "issued"),
                (ids[1], "calculate_timestamp", "none"),
            ]
            and len({x.get("i") for x in second.get("fc") or []}) == 1
            and len(user_fr) == 1
            # the responses in the order of the calls (without ids, e.g. on
            # Vertex, the order is all that pairs them)
            and [(i, n) for i, n, _ in _frs(second)]
            == [(ids[0], "get_current_timestamp"), (ids[1], "calculate_timestamp")]
            and "get_current_timestamp=" in final
            and "calculate_timestamp=" in final,
            _bdetail(c, reqs, _req_tokens(second)),
        )


async def rounds(ctx: Ctx, b) -> None:
    await ctx.mock.reset()
    c = await ctx.chat(b, PRO3, directive([[TIMESTAMP], [CALC]]))
    reqs = await ctx.requests()
    third = reqs[2] if len(reqs) > 2 else {}
    usage = {k: (c.usage or {}).get(k) for k in USAGE_ROUNDS}
    ctx.check(
        "tools.rounds",
        "two sequential tool rounds: 3 upstream requests, each replayed model "
        "content keeps its own signature, both rounds saved, usage summed",
        c.done
        and _answers(reqs) == "[fc,fc,final]"
        and _fcs(third)
        == [
            ("mock-call-0-0", "get_current_timestamp", "issued"),
            ("mock-call-1-0", "calculate_timestamp", "issued"),
        ]
        and len({x.get("i") for x in third.get("fc") or []}) == 2
        and _calls(c)
        == [
            ("get_current_timestamp", "completed", "mock-call-0-0"),
            ("calculate_timestamp", "completed", "mock-call-1-0"),
        ]
        and usage == USAGE_ROUNDS,
        _bdetail(c, reqs, f"{_req_tokens(third)} usage={usage}"),
    )

    await _set(ctx.t, INCLUDE_THOUGHTS=False)
    try:
        await ctx.mock.reset()
        c = await ctx.chat(b, TEXT, directive([[TIMESTAMP], [CALC]]))
        reqs = await ctx.requests()
    finally:
        await _set(ctx.t, INCLUDE_THOUGHTS=True)
    third = reqs[2] if len(reqs) > 2 else {}
    contents = sorted({x.get("i") for x in third.get("fc") or []})
    final = _final(c)
    ctx.check(
        "tools.rounds-nosig",
        "two rounds on gemini-2.5-flash without thoughts (no signatures): the turn "
        "completes, the final answer echoes both results, no placeholder signature "
        "(Gemini 2.x does not check them)",
        c.done
        and "get_current_timestamp=" in final
        and "calculate_timestamp=" in final
        and len(_fcs(third)) == 2
        and all(s == "none" for _, _, s in _fcs(third)),
        # informational: Open WebUI merges two rounds without text or reasoning in
        # between into one assistant message (fc_contents=1)
        _bdetail(c, reqs, f"fc_contents={len(contents)} {_req_tokens(third)}"),
    )


async def text_before(ctx: Ctx, b) -> None:
    await ctx.mock.reset()
    c = await ctx.chat(b, PRO3, directive([[TIMESTAMP]], text_before="Let me check."))
    reqs = await ctx.requests()
    second = reqs[1] if len(reqs) > 1 else {}
    message_at = next(
        (
            k
            for k, i in enumerate(c.output)
            if i.get("type") == "message" and "Let me check." in _item_text(i)
        ),
        None,
    )
    call_at = next(
        (k for k, i in enumerate(c.output) if i.get("type") == "function_call"), None
    )
    model_parts = [
        ("fc" if "functionCall" in p else "text", p.get("text"))
        for content in _body(second).get("contents") or []
        if content.get("role") == "model"
        for p in content.get("parts") or []
    ]
    ctx.check(
        "tools.text-before",
        "text before the call: saved before the function call, replayed as text "
        "part followed by FC(sig)",
        c.done
        and message_at is not None
        and call_at is not None
        and message_at < call_at
        and model_parts == [("text", "Let me check."), ("fc", None)]
        and _fcs(second) == [(CALL_0, "get_current_timestamp", "issued")],
        _bdetail(
            c,
            reqs,
            f"message_at={message_at} call_at={call_at} model_parts={model_parts} "
            f"{_req_tokens(second)}",
        ),
    )


async def thinking(ctx: Ctx, b) -> None:
    await ctx.mock.reset()
    c = await ctx.chat(b, PRO3, directive([[TIMESTAMP]]))
    reqs = await ctx.requests()
    items = c.reasoning_items
    pondering = [text for text, _ in items if "Mock pondering." in text]
    ctx.check(
        "tools.thinking",
        "INCLUDE_THOUGHTS=true: the tool round's reasoning item has the thought text "
        "and the signature, the continuation has its own reasoning item",
        c.done
        and bool(items)
        and "Mock pondering." in items[0][0]
        and items[0][1] == [rd_item(CALL_0)]
        and len(pondering) >= 2
        and _no_details(c),
        _bdetail(c, reqs, f"reasoning={[(short(x, 40), len(d)) for x, d in items]}"),
    )
    await _set(ctx.t, INCLUDE_THOUGHTS=False)
    try:
        await ctx.mock.reset()
        c = await ctx.chat(b, PRO3, directive([[TIMESTAMP]]))
        reqs = await ctx.requests()
    finally:
        await _set(ctx.t, INCLUDE_THOUGHTS=True)
    items = c.reasoning_items
    ctx.check(
        "tools.thinking-off",
        "INCLUDE_THOUGHTS=false: the reasoning item has no text but the signature, "
        "no <details> anywhere",
        c.done
        and bool(items)
        and items[0] == ("", [rd_item(CALL_0)])
        and _no_details(c)
        and "Mock pondering." not in json.dumps(c.output),
        _bdetail(c, reqs, f"reasoning={[(short(x, 40), len(d)) for x, d in items]}"),
    )


async def workspace(ctx: Ctx, b, ws: tuple) -> None:
    await ctx.mock.reset()
    calls = [
        {"name": "add_numbers", "args": {"a": "4", "b": 5.0, "extra": 1}},
        {"name": "whoami", "args": {"label": "hi"}},
    ]
    c = await ctx.chat(b, PRO3, directive([calls]), tool_ids=[WS_TOOL])
    reqs = await ctx.requests()
    names = _declared(reqs[0]) if reqs else []
    added = _json(c.function_outputs.get("mock-call-0-0"))
    who = _json(c.function_outputs.get("mock-call-0-1"))
    responses = short(
        [x.get("response") for x in (reqs[-1] if reqs else {}).get("fr") or []], 300
    )
    ctx.check(
        "tools.workspace",
        "workspace Python tool: arguments coerced to the annotations, __user__ / "
        "__metadata__ / __event_emitter__ injected, its status saved",
        c.done
        and added.get("types") == ["int", "float", "str"]
        and added.get("sum") == 9
        and who.get("user_email") == ADMIN_EMAIL
        and who.get("chat_id") == c.chat_id
        and who.get("has_event_emitter") is True
        and "whoami hi" in c.status_descriptions
        and {"add_numbers", "whoami"} <= set(names),
        _bdetail(
            c,
            reqs,
            f"create_tool={ws[0]} add={added} whoami={short(who, 160)} "
            f"statuses={c.status_descriptions} responses={responses}",
        ),
    )


async def image_result(ctx: Ctx, b) -> None:
    """A tool result with an image: Open WebUI passes the image on in a user
    message after the tool results (F13 of the spec); the request is still a
    later round of the turn."""
    await ctx.mock.reset()
    c = await ctx.chat(
        b, PRO3, directive([[{"name": "make_image", "args": {}}]]), tool_ids=[WS_TOOL]
    )
    reqs = await ctx.requests()
    second = reqs[1] if len(reqs) > 1 else {}
    ctx.check(
        "tools.image-result",
        "tool result with an image: the continuation sends FC(sig), FR and Open "
        "WebUI's image message (text + image) and is answered as a later round of "
        "the turn (no <details> block)",
        c.done
        and not c.error
        and _calls(c) == [("make_image", "completed", CALL_0)]
        and _answers(reqs) == "[fc,final]"
        and second.get("kinds")
        == ["user:text", "model:fc", "user:fr", "user:text+inline"]
        and _fcs(second) == [(CALL_0, "make_image", "issued")]
        and _final(c).startswith("MOCK-FINAL make_image=")
        and _no_details(c),
        _bdetail(c, reqs, f"no_details={_no_details(c)} {_req_tokens(second)}"),
    )


async def openapi(ctx: Ctx, b, tool_mock) -> None:
    await ctx.mock.reset()
    await tool_mock.reset()
    calls = [
        {"name": "get_weather", "args": {"city": "Bern"}},
        {
            "name": "convert_units",
            "args": {"value": 10, "unit_from": "in", "unit_to": "cm"},
        },
        {"name": LOOKUP, "args": {"key": "k1"}},
    ]
    c = await ctx.chat(b, PRO3, directive([calls]), tool_ids=[f"server:{OPENAPI_ID}"])
    reqs = await ctx.requests()
    log = await tool_mock.requests()
    weather = [e for e in log if e.get("path") == "/weather"]
    convert = [e for e in log if e.get("path") == "/convert"]
    lookup = [e for e in log if e.get("path") == "/lookup"]
    names = _declared(reqs[0]) if reqs else []
    outputs = c.function_outputs
    mapped = gemini_function_name(LOOKUP)
    ctx.check(
        "tools.openapi",
        "OpenAPI tool server: GET query and POST body reach the server, results "
        f"in the function responses; operationId {LOOKUP} declared as {mapped} and "
        "called by its own name",
        c.done
        and [e.get("query", {}).get("city") for e in weather] == ["Bern"]
        and [e.get("body") for e in convert]
        == [{"value": 10, "unit_from": "in", "unit_to": "cm"}]
        and [e.get("query", {}).get("key") for e in lookup] == ["k1"]
        and _json(outputs.get("mock-call-0-0")).get("temp_c") == 21.5
        and _json(outputs.get("mock-call-0-1")).get("result") == 25.4
        and _json(outputs.get("mock-call-0-2")).get("found") is True
        and _safe(names)
        and mapped in names
        and (LOOKUP, "completed", "mock-call-0-2") in _calls(c),
        _bdetail(
            c,
            reqs,
            f"server_log={[e.get('path') for e in log]} mapped_declared="
            f"{mapped in names}",
        ),
    )


async def mcp(ctx: Ctx, b) -> None:
    try:
        async with httpx.AsyncClient(timeout=5) as http:
            mcp_up = (await http.get(MCP_URL)).status_code > 0
    except httpx.HTTPError:
        mcp_up = False
    await ctx.mock.reset()
    name = f"{MCP_ID}_mcp_echo"
    c = await ctx.chat(
        b,
        PRO3,
        directive([[{"name": name, "args": {"text": "yo", "times": 3}}]]),
        tool_ids=[f"server:mcp:{MCP_ID}"],
    )
    reqs = await ctx.requests()
    ctx.check(
        "tools.mcp",
        f"MCP tool server: {name} runs, its text result is the function response",
        c.done
        and c.function_outputs.get(CALL_0) == "yo yo yo"
        and "yo yo yo" in _final(c),
        _bdetail(
            c, reqs, f"mcp_mock_up={mcp_up} {_req_tokens(reqs[-1] if reqs else {})}"
        ),
    )


async def direct(ctx: Ctx, b) -> None:
    b.direct_tool_answer = lambda data: [
        {"value": f"client:{(data.get('params') or {}).get('key')}"},
        {"content-type": "application/json"},
    ]
    before = len(b.execute_calls)
    try:
        await ctx.mock.reset()
        c = await ctx.chat(
            b,
            PRO3,
            directive([[{"name": "direct_lookup", "args": {"key": "k1"}}]]),
            tool_servers=[copy.deepcopy(DIRECT_SERVER)],
        )
        reqs = await ctx.requests()
    finally:
        b.direct_tool_answer = lambda data: {"error": "no direct tool answer"}
    executed = len(b.execute_calls) - before
    ctx.check(
        "tools.direct",
        "direct tool (browser tool server): Open WebUI calls the browser "
        "(execute:tool), its answer is the function response",
        c.done
        and executed == 1
        and _json(c.function_outputs.get(CALL_0)) == {"value": "client:k1"}
        and "client:k1" in _final(c),
        _bdetail(c, reqs, f"execute_calls={executed}"),
    )


async def _ask(ctx: Ctx, b, text: str, action: str, first_only: bool = True):
    """Ask mode: send, wait for the pause, resolve the pending call(s)."""
    await ctx.mock.reset()
    c = await ctx.chat(b, PRO3, text, params=ASK, until=_pending, wait=60)
    paused = _pending(c.message) and not c.done
    resolved = []
    for _ in range(2):
        pending = [i for _, s, i in _calls(c) if s == "pending"]
        if not pending:
            break
        status, _ = await ctx.t.owui.resolve_tool_call(
            c.chat_id, c.message_id, pending[0], action
        )
        resolved.append((pending[0], status))

        def other_pending(message: dict, call_id=pending[0]) -> bool:
            return any(
                i.get("type") == "function_call"
                and i.get("status") == "pending"
                and i.get("call_id") != call_id
                for i in message.get("output") or []
                if isinstance(i, dict)
            )

        c = await b.wait(c, wait=60, until=None if first_only else other_pending)
        if first_only:
            break
    if c.task_ids and not c.done:
        await b.stop(c)
    return c, paused, resolved


async def approval(ctx: Ctx, b) -> None:
    previous = await ctx.t.owui.set_chat_config(ENABLE_TOOL_PERMISSIONS=True)
    try:
        c, paused, resolved = await _ask(ctx, b, directive([[CALC]]), "approve")
        reqs = await ctx.requests()
        follow = reqs[-1] if reqs else {}
        final = _final(c)
        ctx.check(
            "tools.approval",
            "tool approval (ask): the call waits as pending, after approve it runs; "
            "the continuation has FC(sig) + FR and the answer echoes the result",
            paused
            and [s for _, s in resolved] == [200]
            and c.done
            and "calculate_timestamp=" in final
            and "calculated_timestamp" in final
            and _fcs(follow) == [(CALL_0, "calculate_timestamp", "issued")]
            and _frs(follow) == [(CALL_0, "calculate_timestamp", ["output"])],
            _bdetail(
                c, reqs, f"paused={paused} resolved={resolved} {_req_tokens(follow)}"
            ),
        )
        # Open WebUI 0.11.4 drops an approved call from the saved history
        await ctx.mock.reset()
        c2 = await ctx.chat(
            b,
            PRO3,
            "Thanks, and now?",
            chat_id=c.chat_id,
            parent_id=c.message_id,
            params=ASK,
        )
        reqs = await ctx.requests()
        req = reqs[-1] if reqs else {}
        ctx.check(
            "tools.multiturn-approved",
            "follow-up after an approved call: a valid request (no orphan FC or FR)",
            c2.done
            and req.get("status") == 200
            and not req.get("orphan_fr")
            and _answers(reqs) == "[text]",
            _bdetail(c2, reqs, f"orphan_fr={req.get('orphan_fr')} {_req_tokens(req)}"),
        )

        c, paused, resolved = await _ask(ctx, b, directive([[TIMESTAMP]]), "reject")
        reqs = await ctx.requests()
        follow = reqs[-1] if reqs else {}
        responses = [x.get("response") for x in follow.get("fr") or []]
        ctx.check(
            "tools.approval-reject",
            "tool approval (ask), rejected: the function response is the error and "
            "the answer echoes it",
            paused
            and c.done
            and responses == [{"error": REJECTED}]
            and REJECTED in _final(c),
            _bdetail(
                c, reqs, f"paused={paused} resolved={resolved} responses={responses}"
            ),
        )

        c, paused, resolved = await _ask(
            ctx, b, directive([[TIMESTAMP, CALC]]), "approve", first_only=False
        )
        reqs = await ctx.requests()
        # informational: Open WebUI 0.11.4 asks for the first call only and does
        # not run the second one (calls= / resolved= of the detail)
        ctx.check(
            "tools.approval-parallel",
            "tool approval (ask) with two calls, approve: the turn pauses and "
            "completes",
            paused
            and c.done
            and all(e.get("status") == 200 for e in reqs)
            and _final(c).startswith("MOCK-FINAL"),
            _bdetail(c, reqs, f"paused={paused} resolved={resolved}"),
        )
    finally:
        await ctx.t.owui.set_chat_config(**previous)


async def unknown(ctx: Ctx, b) -> None:
    await ctx.mock.reset()
    c = await ctx.chat(
        b,
        PRO3,
        directive(
            [[{"name": "no_such_tool", "args": {"x": 1}}]], allow_undeclared=True
        ),
    )
    reqs = await ctx.requests()
    second = reqs[1] if len(reqs) > 1 else {}
    responses = [x.get("response") for x in second.get("fr") or []]
    ctx.check(
        "tools.unknown",
        "call of an undeclared tool: Open WebUI answers 'not found', the function "
        "response carries it as error, the turn completes",
        c.done
        and responses == [{"error": NOT_FOUND}]
        and _final(c).startswith("MOCK-FINAL"),
        _bdetail(c, reqs, f"responses={responses}"),
    )


async def malformed(ctx: Ctx, b) -> None:
    await ctx.mock.reset()
    c = await ctx.chat(b, PRO3, directive([[TIMESTAMP]], malformed=True))
    reqs = await ctx.requests()
    ctx.check(
        "tools.malformed",
        "finishReason MALFORMED_FUNCTION_CALL: an error text that names it, the "
        "turn completes",
        c.done and "MALFORMED_FUNCTION_CALL" in c.content,
        _bdetail(c, reqs, f"content={short(c.content, 160)}"),
    )


async def grounding(ctx: Ctx, b) -> None:
    t = ctx.t
    if not await _filter_ready(t):
        return
    await t.owui.upsert_model(PRO3, "Gemini 3 Pro Preview", [SEARCH_FILTER])
    try:
        await ctx.mock.reset()
        c = await ctx.chat(
            b,
            PRO3,
            directive([[TIMESTAMP]], server_side=True),
            features={"web_search": True},
        )
        reqs = await ctx.requests()
        # next turn with web search switched off: no server-side parts upstream
        await ctx.mock.reset()
        c2 = None
        if c.chat_id:
            c2 = await ctx.chat(
                b,
                PRO3,
                "Next turn without web search.",
                chat_id=c.chat_id,
                parent_id=c.message_id,
            )
        reqs2 = await ctx.requests()
    finally:
        await t.owui.delete_model(PRO3)
    first, second = (reqs + [{}, {}])[:2]
    sources = _strings(c.sources)
    formats = [d.get("format") for d in _rd(c)]
    ctx.check(
        "tools.grounding3",
        "Gemini 3 + web_search: googleSearch, urlContext and functions in one "
        "request with includeServerSideToolInvocations; the round keeps its "
        "signature and its stored content; the server-side parts are echoed back "
        "with the call; the sources of both rounds saved, no inline [n] marker in "
        "the answer after the call (it was already streamed)",
        c.done
        and {"googleSearch", "urlContext", "functionDeclarations"}
        <= set(first.get("tool_kinds") or [])
        and first.get("include_flag") is True
        and formats == ["google-gemini-v1", "google-gemini-v1-content"]
        and second.get("server_echoed") is True
        and _fcs(second) == [(CALL_0, "get_current_timestamp", "issued")]
        and GROUNDING_URI in sources
        and TOOL_ROUND_URI in sources
        and "[1]" not in _final(c),
        _bdetail(
            c,
            reqs,
            f"tool_kinds={_kinds(first)} include_flag={first.get('include_flag')} "
            f"server_echoed={second.get('server_echoed')} sources="
            f"{GROUNDING_URI in sources}/{TOOL_ROUND_URI in sources} "
            f"{_req_tokens(second)}",
        ),
    )
    # The stored content (with the server-side toolCall / toolResponse) is only
    # replayed in its own turn; a later request may have no Search grounding.
    req = reqs2[-1] if reqs2 else {}
    kinds = req.get("kinds") or []
    old = [(x.get("id"), x.get("sig")) for x in req.get("fc") or []]
    old_fr = len([x for x in req.get("fr") or [] if x.get("id") == CALL_0])
    ctx.check(
        "tools.grounding3-next-turn",
        "next turn without web search after a grounding + tool round: the old "
        "round is replayed as function call (with its signature) and response, "
        "without the server-side parts",
        c2 is not None
        and c2.done
        and req.get("status") == 200
        and not req.get("include_flag")
        and kinds[:3] == ["user:text", "model:fc", "user:fr"]
        and not any("toolCall" in k or "toolResponse" in k for k in kinds)
        and _fcs(req) == [(CALL_0, "get_current_timestamp", "issued")],
        _bdetail(c2, reqs2, f"old_fc={old} old_fr={old_fr} {_req_tokens(req)}")
        if c2
        else "no chat",
    )

    await t.owui.upsert_model(TEXT, "Gemini 2.5 Flash", [SEARCH_FILTER])
    try:
        await ctx.mock.reset()
        c = await ctx.chat(
            b, TEXT, "Search the web (tools available).", features={"web_search": True}
        )
        reqs = await ctx.requests()
    finally:
        await t.owui.delete_model(TEXT)
    first = reqs[0] if reqs else {}
    ctx.check(
        "tools.grounding25",
        "Gemini 2.5 + web_search: grounding wins, no function declarations",
        c.done and first.get("tool_kinds") == ["googleSearch", "urlContext"],
        _bdetail(
            c,
            reqs,
            f"tool_kinds={_kinds(first)} include_flag={first.get('include_flag')}",
        ),
    )


async def task(ctx: Ctx, b) -> None:
    await ctx.mock.reset()
    c = await ctx.chat(
        b,
        PRO3,
        directive([[TIMESTAMP]]),
        background_tasks={"title_generation": True},
        wait_title=True,
    )
    reqs = await ctx.requests()
    tasks = [e for e in reqs if "### Task:" in _last_text(e)]
    ctx.check(
        "tools.task",
        "title task of a tool turn: no function declarations, no toolConfig",
        c.done
        and bool(tasks)
        and all(
            "functionDeclarations" not in (e.get("tool_kinds") or [])
            and not e.get("tool_config")
            for e in tasks
        ),
        _bdetail(
            c,
            reqs,
            f"title={c.title!r} task_tool_kinds={[e.get('tool_kinds') for e in tasks]} "
            f"task_tool_config={[e.get('tool_config') for e in tasks]}",
        ),
    )


async def legacy(ctx: Ctx, b) -> None:
    await ctx.mock.reset()
    c = await ctx.chat(
        b,
        PRO3,
        "Add 1 and 2 with the workspace tool.",
        params=LEGACY,
        tool_ids=[WS_TOOL],
    )
    reqs = await ctx.requests()
    kinds = [e.get("tool_kinds") or [] for e in reqs]
    ctx.check(
        "tools.legacy",
        "function_calling=legacy: no request declares functions, the answer is saved",
        c.done
        and bool(reqs)
        and all("functionDeclarations" not in k for k in kinds)
        and "Hello from mock" in c.content,
        _bdetail(c, reqs, f"tool_kinds={kinds}"),
    )


async def nobuiltin(ctx: Ctx, b) -> None:
    t = ctx.t
    await t.owui.upsert_model(
        PRO3, "Gemini 3 Pro Preview", capabilities={"builtin_tools": False}
    )
    try:
        await ctx.mock.reset()
        c = await ctx.chat(b, PRO3, "Hello without built-in tools.")
        reqs = await ctx.requests()
    finally:
        await t.owui.delete_model(PRO3)
    first = reqs[0] if reqs else {}
    raw_tools = (first.get("body") or {}).get("tools") or []
    ctx.check(
        "tools.nobuiltin",
        "model capability builtin_tools=false and no tool_ids: no function "
        "declarations and no empty Tool",
        c.done
        and bool(reqs)
        and "functionDeclarations" not in (first.get("tool_kinds") or [])
        and all(isinstance(x, dict) and x for x in raw_tools),
        _bdetail(c, reqs, f"tools={short(raw_tools, 120)}"),
    )


async def nostream(ctx: Ctx, b) -> None:
    """GOOGLE_STREAMING_ENABLED=false, browser stream=true: Gemini is called
    without streaming, the turn is saved like a streamed one."""
    await _set(ctx.t, STREAMING_ENABLED=False)
    try:
        await ctx.mock.reset()
        c = await ctx.chat(b, PRO3, directive([[TIMESTAMP]]), stream=True)
        reqs = await ctx.requests()
    finally:
        await _set(ctx.t, STREAMING_ENABLED=True)
    actions = [e.get("action") for e in reqs]
    ctx.check(
        "tools.nostream",
        "GOOGLE_STREAMING_ENABLED=false, browser stream=true: two non-streamed "
        "requests, the call runs, the signed reasoning item and the final answer "
        "are saved, no <details> block",
        c.done
        and not c.error
        and _calls(c) == [("get_current_timestamp", "completed", CALL_0)]
        and _rd(c) == [rd_item(CALL_0)]
        and _final(c).startswith("MOCK-FINAL get_current_timestamp=")
        and _no_details(c)
        and actions == ["generateContent", "generateContent"],
        _bdetail(c, reqs, f"actions={actions} no_details={_no_details(c)}"),
    )


# ================================================================== toolsapi
async def toolsapi(t: Suite, mock) -> None:
    ctx = await preflight(t, mock)
    await api_stream(ctx)
    await api_stream_text(ctx)
    await api_nonstream(ctx)
    await api_nostream(ctx)
    await api_malformed(ctx)
    await api_continuation(ctx)
    await api_continuation_psf(ctx)
    await api_continuation_nosig(ctx)
    await api_history_edge(ctx)
    await api_odd_shapes(ctx)
    await api_tool_choice(ctx)
    await api_tool_choice_grounding(ctx)
    await api_names(ctx)
    await api_default_api(ctx)
    await api_schema(ctx)
    await api_noid(ctx)
    await api_unchanged(ctx)


async def api_stream(ctx: Ctx) -> None:
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(
        PRO3,
        directive([[{"name": "client_fn", "args": {"x": 1}}]]),
        stream=True,
        tools=[CLIENT_FN],
    )
    reqs = await ctx.requests()
    calls = [_tc(x) for x in r.tool_calls]
    ctx.check(
        "toolsapi.stream",
        "API stream with client tools: tool_calls + reasoning_details chunks, last "
        "finish_reason tool_calls (also for the openai SDK), [DONE] last, usage; "
        "thinking only in the <details> block (no reasoning_content)",
        r.status == 200
        and bool(r.reasoning_details)
        and all(
            set(d) == {"type", "format", "id", "index", "data"}
            for d in r.reasoning_details
        )
        and rd_item(CALL_0) in r.reasoning_details
        and len(calls) == 1
        and calls[0][:3] == (0, CALL_0, "client_fn")
        and _json(calls[0][3]) == {"x": 1}
        and r.finish_reasons[-1:] == ["tool_calls"]
        and r.done_last
        and bool(r.usage)
        and r.openai_finish_reason == "tool_calls"
        and "<details" in r.content
        and not r.reasoning_content,
        _adetail(r, reqs, f"reasoning_content={short(r.reasoning_content, 40)}"),
    )


async def api_stream_text(ctx: Ctx) -> None:
    """API stream with client tools that the model answers with text (P3)."""
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(TEXT, "Hello Gemini", stream=True, tools=[CLIENT_FN])
    reqs = await ctx.requests()
    ctx.check(
        "toolsapi.stream-text",
        "API stream with client tools answered with text: the answer, last "
        "finish_reason stop (also for the openai SDK), [DONE] last, usage, no "
        "tool_calls",
        r.status == 200
        and "Hello from mock (stream)." in r.content
        and not r.tool_calls
        and r.finish_reasons[-1:] == ["stop"]
        and r.openai_finish_reason == "stop"
        and r.done_last
        and bool(r.usage),
        _adetail(r, reqs),
    )


async def api_nostream(ctx: Ctx) -> None:
    """GOOGLE_STREAMING_ENABLED=false: an API stream still gets SSE."""
    await _set(ctx.t, STREAMING_ENABLED=False)
    try:
        await ctx.mock.reset()
        r = await ctx.t.owui.chat(
            PRO3,
            directive([[{"name": "client_fn", "args": {"x": 1}}]]),
            stream=True,
            tools=[CLIENT_FN],
        )
        reqs = await ctx.requests()
    finally:
        await _set(ctx.t, STREAMING_ENABLED=True)
    calls = [_tc(x) for x in r.tool_calls]
    actions = [e.get("action") for e in reqs]
    ctx.check(
        "toolsapi.nostream",
        "GOOGLE_STREAMING_ENABLED=false, API stream with client tools: one "
        "non-streamed request, answered as SSE with tool_calls and finish_reason "
        "tool_calls",
        r.status == 200
        and [(c, n) for _, c, n, _ in calls] == [(CALL_0, "client_fn")]
        and r.finish_reasons[-1:] == ["tool_calls"]
        and r.openai_finish_reason == "tool_calls"
        and r.done_last
        and actions == ["generateContent"],
        _adetail(r, reqs, f"actions={actions}"),
    )


async def api_malformed(ctx: Ctx) -> None:
    """MALFORMED_FUNCTION_CALL / UNEXPECTED_TOOL_CALL, streamed and not."""
    seen, ok, statuses, answers = [], True, set(), []
    for finish in ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"):
        for stream in (False, True):
            await ctx.mock.reset()
            r = await ctx.t.owui.chat(
                PRO3,
                directive(
                    [[{"name": "client_fn", "args": {"x": 1}}]], malformed=finish
                ),
                stream=stream,
                tools=[CLIENT_FN],
            )
            reqs = await ctx.requests()
            statuses.add(r.status)
            answers.append(_answers(reqs).strip("[]"))
            got = (
                r.status == 200
                and _answers(reqs) == "[malformed]"
                and finish in (r.content or "")
            )
            seen.append(f"{finish}:{stream}:{got}:{short(r.content, 60)}")
            ok = ok and got
    http = ",".join(str(s) for s in sorted(statuses))
    ctx.check(
        "toolsapi.malformed",
        "finishReason MALFORMED_FUNCTION_CALL or UNEXPECTED_TOOL_CALL without a "
        "call: an error text that names it, streamed and non-streamed",
        ok,
        f"http={http} upstream=[{','.join(answers)}] seen={seen}",
    )


async def api_nonstream(ctx: Ctx) -> None:
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(
        PRO3,
        directive([[{"name": "client_fn", "args": {"x": 1}}]]),
        stream=False,
        tools=[CLIENT_FN],
    )
    reqs = await ctx.requests()
    calls = [_tc(x) for x in r.tool_calls]
    ctx.check(
        "toolsapi.nonstream",
        "API non-stream with client tools: chat.completion with message.tool_calls, "
        "reasoning_details, finish_reason tool_calls and usage",
        r.status == 200
        and [(c, n) for _, c, n, _ in calls] == [(CALL_0, "client_fn")]
        and _json(calls[0][3]) == {"x": 1}
        and rd_item(CALL_0) in r.reasoning_details
        and r.finish_reasons == ["tool_calls"]
        and bool(r.usage),
        _adetail(r, reqs),
    )


def _history(tool_call_id, sig: bool = True, content: str = '{"y": 2}') -> list:
    """An API client's tool loop: the call it got, the result it computed."""
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": "client_fn", "arguments": '{"x": 1}'},
            }
        ],
    }
    if sig:
        assistant["reasoning_details"] = [rd_item(tool_call_id)]
    return [
        assistant,
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "client_fn",
            "content": content,
        },
    ]


async def api_continuation(ctx: Ctx) -> None:
    text = directive([[{"name": "client_fn", "args": {"x": 1}}]])
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(
        PRO3,
        [{"role": "user", "content": text}, *_history(CALL_0)],
        stream=False,
        tools=[CLIENT_FN],
    )
    reqs = await ctx.requests()
    req = reqs[-1] if reqs else {}
    responses = [x.get("response") for x in req.get("fr") or []]
    ctx.check(
        "toolsapi.continuation",
        "API continuation (assistant tool_calls + reasoning_details, tool message): "
        "FC with the signature, FR with id / name / output, the final answer",
        r.status == 200
        and _fcs(req) == [(CALL_0, "client_fn", "issued")]
        and [(i, n) for i, n, _ in _frs(req)] == [(CALL_0, "client_fn")]
        and responses == [{"output": '{"y": 2}'}]
        and "MOCK-FINAL client_fn=" in r.content,
        _adetail(r, reqs, _req_tokens(req)),
    )


async def api_continuation_psf(ctx: Ctx) -> None:
    """reasoning_details under provider_specific_fields (LiteLLM-style clients)."""
    text = directive([[{"name": "client_fn", "args": {"x": 1}}]])
    history = _history(CALL_0)
    history[0]["provider_specific_fields"] = {
        "reasoning_details": history[0].pop("reasoning_details")
    }
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(
        PRO3,
        [{"role": "user", "content": text}, *history],
        stream=False,
        tools=[CLIENT_FN],
    )
    reqs = await ctx.requests()
    req = reqs[-1] if reqs else {}
    ctx.check(
        "toolsapi.continuation-psf",
        "API continuation with reasoning_details under provider_specific_fields: "
        "the function call carries the issued signature",
        r.status == 200
        and req.get("status") == 200
        and _fcs(req) == [(CALL_0, "client_fn", "issued")]
        and "MOCK-FINAL client_fn=" in r.content,
        _adetail(r, reqs, _req_tokens(req)),
    )


async def api_continuation_nosig(ctx: Ctx) -> None:
    text = directive([[{"name": "client_fn", "args": {"x": 1}}]])
    messages = [
        {"role": "user", "content": "An older turn."},
        *_history("old-call-1", sig=False, content='{"y": 0}'),
        {"role": "assistant", "content": "Old answer."},
        {"role": "user", "content": text},
        *_history(CALL_0, sig=False),
    ]
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(PRO3, messages, stream=False, tools=[CLIENT_FN])
    reqs = await ctx.requests()
    req = reqs[-1] if reqs else {}
    ctx.check(
        "toolsapi.continuation-nosig",
        "API continuation without reasoning_details on Gemini 3: the current turn's "
        "call gets skip_thought_signature_validator, the older turn's call none",
        r.status == 200
        and req.get("status") == 200
        and _fcs(req)
        == [("old-call-1", "client_fn", "none"), (CALL_0, "client_fn", "skip")],
        _adetail(r, reqs, _req_tokens(req)),
    )


# The pipe's thinking summary as API clients echo it in the assistant content
THOUGHT_SUMMARY = (
    "<details>\n<summary>Thought (0s)</summary>\n\n> old thinking\n\n</details>"
)


async def _api_warned(ctx: Ctx, messages: list, warning: str):
    """API request with a client history; (result, requests, warning logged)."""
    mark = ctx.t.mark()
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(PRO3, messages, stream=False, tools=[CLIENT_FN])
    reqs = await ctx.requests()
    await ctx.t.log.settle(0.5)
    signature = ("function_gemini", warning)
    warned = bool(ctx.t.log.warnings(mark, signature))
    ctx.t.expect_warnings(mark, signature)
    return r, reqs, warned


async def api_history_edge(ctx: Ctx) -> None:
    """An older turn as API clients send it: the thinking summary before the
    call, empty (non-JSON) arguments, a tool result in content parts."""
    text = directive([[{"name": "client_fn", "args": {"x": 1}}]])
    old_call = {
        "id": "old-call-1",
        "type": "function",
        "function": {"name": "client_fn", "arguments": ""},
    }
    parts = [
        {"type": "text", "text": "part one "},
        {"type": "text", "text": "part two"},
    ]
    messages = [
        {"role": "user", "content": "An older turn."},
        {"role": "assistant", "content": THOUGHT_SUMMARY, "tool_calls": [old_call]},
        {"role": "tool", "tool_call_id": "old-call-1", "content": parts},
        {"role": "assistant", "content": "Old answer."},
        {"role": "user", "content": text},
        *_history(CALL_0),
    ]
    r, reqs, warned = await _api_warned(ctx, messages, "Invalid tool call arguments")
    req = reqs[-1] if reqs else {}
    old_fc = [x.get("args") for x in req.get("fc") or [] if x.get("id") == "old-call-1"]
    old_fr = [
        x.get("response") for x in req.get("fr") or [] if x.get("id") == "old-call-1"
    ]
    ctx.check(
        "toolsapi.history-edge",
        "older turn with the thinking summary before the call (stripped), empty "
        "arguments (sent as {} with a WARNING) and a tool result in content parts "
        "(their text joined)",
        r.status == 200
        and req.get("status") == 200
        and old_fc == [{}]
        and old_fr == [{"output": "part one part two"}]
        and warned
        and req.get("kinds")
        == [
            "user:text",
            "model:fc",
            "user:fr",
            "model:text",
            "user:text",
            "model:fc",
            "user:fr",
        ]
        and "MOCK-FINAL" in r.content,
        _adetail(
            r,
            reqs,
            f"warned={warned} old_fc={old_fc} old_fr={old_fr} {_req_tokens(req)}",
        ),
    )


async def api_odd_shapes(ctx: Ctx) -> None:
    """Tool calls that break the OpenAI format (some clients and proxies send
    them): a numeric id and a "function" that is not an object. The request
    still works; the call without a function object gets a placeholder name."""
    text = directive([[{"name": "client_fn", "args": {"x": 1}}]])
    messages = [
        {"role": "user", "content": "An older turn."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": 7,
                    "type": "function",
                    "function": {"name": "client_fn", "arguments": '{"x": 7}'},
                },
                {"id": "odd-call", "type": "function", "function": "client_fn"},
            ],
        },
        {"role": "tool", "tool_call_id": 7, "content": "seven"},
        {"role": "tool", "tool_call_id": "odd-call", "content": "odd"},
        {"role": "assistant", "content": "Old answer."},
        {"role": "user", "content": text},
        *_history(CALL_0),
    ]
    r, reqs, warned = await _api_warned(ctx, messages, "Invalid tool call arguments")
    req = reqs[-1] if reqs else {}
    old = [(i, n) for i, n, _ in _fcs(req)][:2]
    old_fr = [(i, n) for i, n, _ in _frs(req)][:2]
    ctx.check(
        "toolsapi.odd-shapes",
        "older turn with a numeric tool call id and a tool call whose function is "
        "not an object: the request works (id as text, the odd call with a "
        "placeholder name and a WARNING)",
        r.status == 200
        and req.get("status") == 200
        and old == [("7", "client_fn"), ("odd-call", gemini_function_name(""))]
        and old_fr == old
        and warned
        and "MOCK-FINAL client_fn=" in r.content,
        _adetail(r, reqs, f"warned={warned} {_req_tokens(req)}"),
    )


async def api_tool_choice(ctx: Ctx) -> None:
    named = {"type": "function", "function": {"name": "client_fn"}}
    undeclared = {"type": "function", "function": {"name": "nope"}}
    cases = (
        ("none", "NONE", None),
        ("required", "ANY", None),
        (named, "ANY", ["client_fn"]),
        (undeclared, None, None),  # ignored: Gemini rejects undeclared names
        (None, None, None),
    )
    seen, ok, reqs_all = [], True, []
    for choice, mode, allowed in cases:
        extra = {"tool_choice": choice} if choice is not None else {}
        await ctx.mock.reset()
        r = await ctx.t.owui.chat(
            PRO3, "Pick a tool.", stream=False, tools=[CLIENT_FN], **extra
        )
        reqs = await ctx.requests()
        req = reqs[-1] if reqs else {}
        reqs_all += reqs
        got_mode = str(req.get("fc_mode")).upper() if req.get("fc_mode") else None
        seen.append((got_mode, req.get("allowed_names"), len(_declared(req))))
        ok = ok and (
            r.status == 200
            and got_mode == mode
            and (req.get("allowed_names") or None) == allowed
            and _declared(req) == ["client_fn"]
        )
    modes = ",".join(f"{m}:{a}" if a else str(m) for m, a, _ in seen)
    ctx.check(
        "toolsapi.tool-choice",
        "tool_choice none / required / named function / undeclared function / "
        "absent -> functionCallingConfig NONE / ANY / ANY + allowed names / none / "
        "none",
        ok,
        f"http=200 upstream={_answers(reqs_all)} "
        f"declared_n=[{','.join(str(n) for _, _, n in seen)}] modes=[{modes}]",
    )


async def api_tool_choice_grounding(ctx: Ctx) -> None:
    """tool_choice with Search grounding on Gemini 3 (tool combination)."""
    t = ctx.t
    if not await _filter_ready(t):
        return
    await t.owui.upsert_model(PRO3, "Gemini 3 Pro Preview", [SEARCH_FILTER])
    try:
        await ctx.mock.reset()
        r = await t.owui.chat(
            PRO3,
            "Pick a tool.",
            stream=False,
            tools=[CLIENT_FN],
            tool_choice="required",
            features={"web_search": True},
        )
        reqs = await ctx.requests()
    finally:
        await t.owui.delete_model(PRO3)
    req = reqs[0] if reqs else {}
    mode = str(req.get("fc_mode")).upper() if req.get("fc_mode") else None
    ctx.check(
        "toolsapi.tool-choice-grounding",
        "tool_choice required with web_search on Gemini 3: googleSearch and the "
        "functions with includeServerSideToolInvocations and mode ANY",
        r.status == 200
        and {"googleSearch", "functionDeclarations"} <= set(req.get("tool_kinds") or [])
        and req.get("include_flag") is True
        and mode == "ANY",
        _adetail(
            r,
            reqs,
            f"tool_kinds={_kinds(req)} include_flag={req.get('include_flag')} "
            f"mode={mode}",
        ),
    )


async def api_names(ctx: Ctx) -> None:
    long_name = "x" * 80
    # "trailing_nl\n": a valid name plus a newline (e.g. a YAML block scalar
    # operationId); "$" alone would let it through unchanged
    names = ["ns:tool", "a.b c", long_name, "trailing_nl\n"]
    tools = [
        CLIENT_FN,
        *(client_tool(n) for n in names),
        client_tool("client_fn", desc="dup"),
    ]
    mark = ctx.t.mark()
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(
        PRO3,
        directive([[{"name": n, "args": {}} for n in names]]),
        stream=False,
        tools=tools,
    )
    reqs = await ctx.requests()
    await ctx.t.log.settle(0.5)
    signature = ("function_gemini", "Skipping duplicate tool declaration")
    warned = bool(ctx.t.log.warnings(mark, signature))
    ctx.t.expect_warnings(mark, signature)
    declared = _declared(reqs[0]) if reqs else []
    want = ["client_fn", *(gemini_function_name(n) for n in names)]
    returned = [n for _, _, n, _ in (_tc(x) for x in r.tool_calls)]
    ctx.check(
        "toolsapi.names",
        "tool names Gemini does not accept are declared with the mapped names "
        "(unique), the duplicate once with a WARNING, tool_calls carry the original "
        "names",
        r.status == 200
        and sorted(declared) == sorted(want)
        and _safe(declared)
        and warned
        and returned == names,
        _adetail(r, reqs, f"declared={declared} warned={warned} returned={returned}"),
    )


async def api_default_api(ctx: Ctx) -> None:
    """Gemini sometimes prefixes a call with default_api. (spec section 3.4)."""
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(
        PRO3,
        directive(
            [[{"name": "default_api.client_fn", "args": {"x": 1}}]],
            allow_undeclared=True,
        ),
        stream=False,
        tools=[CLIENT_FN],
    )
    reqs = await ctx.requests()
    names = [n for _, _, n, _ in (_tc(x) for x in r.tool_calls)]
    ctx.check(
        "toolsapi.default-api",
        "a function call named default_api.client_fn is returned as client_fn",
        r.status == 200 and _answers(reqs) == "[fc]" and names == ["client_fn"],
        _adetail(r, reqs, f"names={names}"),
    )


SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "x-vendor": {"a": 1},
    "properties": {
        "count": {
            "type": "integer",
            "minimum": 0,
            "exclusiveMinimum": True,
            "x-hint": "h",
        },
        "pair": {"type": "array", "items": [{"type": "string"}, {"type": "number"}]},
        "item": {"$ref": "#/$defs/Item"},
        "maybe": {"type": ["string", "null"]},
        # property names that are also keywords to drop: they stay
        "examples": {"type": "string"},
        "x-trace": {"type": "string"},
        # a numeric exclusiveMinimum stays (only the boolean form is dropped)
        "ratio": {"type": "number", "exclusiveMinimum": 0},
    },
    "required": ["count", "examples", "ghost"],
    "$defs": {
        "Item": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    },
}


async def api_schema(ctx: Ctx) -> None:
    tools = [
        client_tool("schema_fn", SCHEMA),
        client_tool("noparam_fn", {"type": "object", "properties": {}}),
        client_tool("notype_fn", {"properties": {"a": {"type": "string"}}}),
    ]
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(PRO3, "Look at the schemas.", stream=False, tools=tools)
    reqs = await ctx.requests()
    decl = (reqs[0] if reqs else {}).get("decl") or {}
    schema = (decl.get("schema_fn") or {}).get("schema") or {}
    props = schema.get("properties") or {}
    count, pair = props.get("count") or {}, props.get("pair") or {}
    noparam = decl.get("noparam_fn") or {}
    notype = (decl.get("notype_fn") or {}).get("schema") or {}
    checks = {
        "no_$schema": "$schema" not in schema,
        "no_x-": "x-vendor" not in schema and "x-hint" not in count,
        "bool_exclusive": "exclusiveMinimum" not in count and count.get("minimum") == 0,
        "prefixItems": isinstance(pair.get("prefixItems"), list)
        and "items" not in pair,
        "required": schema.get("required") == ["count", "examples"],
        "keyword_names": "examples" in props and "x-trace" in props,
        "numeric_exclusive": (props.get("ratio") or {}).get("exclusiveMinimum") == 0,
        "$ref": (props.get("item") or {}).get("$ref") == "#/$defs/Item"
        and "Item" in (schema.get("$defs") or {}),
        "null_type": (props.get("maybe") or {}).get("type") == ["string", "null"],
        "noparam": bool(noparam)
        and noparam.get("schema") is None
        and noparam.get("parameters") is None
        and set(noparam.get("keys") or []) <= {"name", "description"},
        "root_type": notype.get("type") == "object",
    }
    failed = [k for k, v in checks.items() if not v]
    ctx.check(
        "toolsapi.schema",
        "client tool schemas are sent as parametersJsonSchema, cleaned up as "
        "specified; a tool without parameters sends none",
        r.status == 200 and not failed,
        _adetail(r, reqs, f"failed={failed} schema={short(schema, 200)}"),
    )


async def api_noid(ctx: Ctx) -> None:
    text = directive([[{"name": "client_fn", "args": {"x": 1}}]], noid=True)
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(TEXT, text, stream=False, tools=[CLIENT_FN])
    reqs = await ctx.requests()
    message = ((r.json or {}).get("choices") or [{}])[0].get("message") or {}
    ids = [c for _, c, _, _ in (_tc(x) for x in r.tool_calls)]
    echo = [
        {"role": "user", "content": text},
        {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls") or [],
            **(
                {"reasoning_details": message["reasoning_details"]}
                if message.get("reasoning_details")
                else {}
            ),
        },
        *[
            {
                "role": "tool",
                "tool_call_id": i,
                "name": "client_fn",
                "content": '{"y": 2}',
            }
            for i in ids
        ],
    ]
    await ctx.mock.reset()
    r2 = await ctx.t.owui.chat(TEXT, echo, stream=False, tools=[CLIENT_FN])
    reqs2 = await ctx.requests()
    req = reqs2[-1] if reqs2 else {}
    ctx.check(
        "toolsapi.noid",
        "function calls without id get synthetic owui_ ids; echoed back, they are "
        "not sent upstream",
        r.status == 200
        and len(ids) == 1
        and str(ids[0]).startswith("owui_")
        and r2.status == 200
        # no placeholder signature: gemini-2.5-flash does not check signatures
        and _fcs(req) == [(None, "client_fn", "none")]
        and [(i, n) for i, n, _ in _frs(req)] == [(None, "client_fn")]
        and req.get("synthetic_ids_upstream") is False
        and "MOCK-FINAL" in r2.content,
        _adetail(
            r,
            reqs,
            f"ids={ids} continuation={_req_tokens(req)} "
            f"synthetic_ids_upstream={req.get('synthetic_ids_upstream')} "
            f"answer2={short(r2.content, 80)}",
        ),
    )


async def api_unchanged(ctx: Ctx) -> None:
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(TEXT, "Hello Gemini", stream=True)
    reqs = await ctx.requests()
    ctx.check(
        "toolsapi.unchanged",
        "API stream without client tools: the unchanged answer (thinking in "
        "<details>, usage, finish_reason stop, no tool_calls)",
        r.status == 200
        and r.done
        and "Hello from mock (stream)." in r.content
        and "<details>" in r.content
        and _usage(r.usage) == USAGE_STREAM
        and not r.tool_calls
        and r.openai_finish_reason == "stop",
        _adetail(r, reqs),
    )
