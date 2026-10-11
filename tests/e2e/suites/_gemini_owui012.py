"""
Open WebUI 0.12 features around the Gemini pipe's native tool calling: group
``owui012`` of the gemini suite (run by suites/gemini.py; the leading
underscore keeps this module out of the suite discovery).

- context compaction (Admin Settings -> Interface): from 0.12 Open WebUI also
  compacts between tool rounds (``compaction``, ``compaction-followup``); the
  summary comes from the chat's model, i.e. the pipe with ``__task__`` =
  ``context_compaction``, and a failed or empty summary must not become the
  conversation summary (``summary-error``, ``summary-empty`` between tool
  rounds, ``saved-summary`` before a reply, which Open WebUI 0.11 does too)
- Tool Search (experimental, 0.12): deferred tools are left out of
  body["tools"]; ``search_tools`` returns their definitions and the model calls
  them by name (``toolsearch``, ``toolsearch-mapped``, ``toolsearch-followup``)
- model controls (0.12): refused for pipe models (``controls``); the custom
  parameters a control would set reach the pipe as Advanced Params of a
  workspace model and of the chat (``custom-params``)
- Stream Chat Response per chat (honoured by the 0.12 web UI) and per model:
  a tool turn with ``stream=false`` (``stream-chat``, ``stream-model``)

Checks of a feature that the tested Open WebUI does not have pass with
"feature not in Open WebUI <version>" in their title and check what that
version does instead. Every detail starts with ``mock_ok=`` (the preflight of
the tool groups) and ``owui=<version>``.
"""

import json

from harness import Suite, short
from harness.known import version_tuple

from ._gemini_tools import (
    CALC,
    LOOKUP,
    OPENAPI_ID,
    TIMESTAMP,
    Ctx,
    _answers,
    _bdetail,
    _calls,
    _connections,
    _declared,
    _fcs,
    _final,
    _frs,
    _req_tokens,
    directive,
    gemini_function_name,
    preflight,
)
from .gemini import PRO3, TEXT, _body, _gc, _gen, _last_text

CALC2 = {"name": "calculate_timestamp", "args": {"days_ago": 2}}
SUMMARY_HEADER = "[CONVERSATION SUMMARY]"
# Open WebUI's fallback summary when the model gave none (_generate_summary):
# "- <role>: <content[:500]>" per compacted message
FALLBACK_USER = "- user: Use the tools."
# Chat settings the group changes (restored at its end)
COMPACTION = dict(
    ENABLE_CONTEXT_COMPACTION=True,
    CONTEXT_COMPACTION_MODEL="",
    CONTEXT_COMPACTION_TOKEN_THRESHOLD=1,
    CONTEXT_COMPACTION_TOKEN_CAP=1,
)
TOOL_SEARCH = dict(
    ENABLE_TOOL_SEARCH=True,
    TOOL_SEARCH_DEFER_THRESHOLD=0,
    TOOL_SEARCH_ALWAYS_LOADED=["get_current_timestamp"],
    TOOL_SEARCH_DEFER_BUILTIN_TOOLS=True,
)
OFF = dict(ENABLE_CONTEXT_COMPACTION=False, ENABLE_TOOL_SEARCH=False)
# Workspace models of the group (deleted at its end)
PARAMS_MODEL = "e2e-g012-params"
PARAMS_MODEL_25 = "e2e-g012-params25"
NOSTREAM_MODEL = "e2e-g012-nostream"
CONTROLS_MODEL = "e2e-g012-controls"
# A Thinking control as the 0.12 model editor saves it
THINKING_CONTROL = {
    "thinking": {
        "label": "Thinking",
        "default": "high",
        "options": {
            "low": {"label": "Low", "params": {"reasoning_effort": "low"}},
            "high": {"label": "High", "params": {"reasoning_effort": "high"}},
        },
    }
}


def search(query: str) -> dict:
    """A call of Open WebUI's search_tools (Tool Search)."""
    return {"name": "search_tools", "args": {"query": query}}


def _system(entry: dict) -> str:
    """Text of a request's system instruction."""
    instruction = _body(entry).get("systemInstruction")
    if isinstance(instruction, str):
        return instruction
    parts = instruction.get("parts") if isinstance(instruction, dict) else None
    return "".join(
        p["text"]
        for p in parts or []
        if isinstance(p, dict) and isinstance(p.get("text"), str)
    )


def _summary(entry: dict) -> str:
    """The conversation summary in a request's system instruction ('' if none)."""
    text = _system(entry)
    return text.split(SUMMARY_HEADER, 1)[1].strip() if SUMMARY_HEADER in text else ""


def _split(reqs: list) -> tuple:
    """(turn requests, summary requests) of a recorded tool turn."""
    summaries = [e for e in reqs if e.get("answer") == "summary"]
    failed = [
        e
        for e in reqs
        if str(e.get("answer", "")).startswith("http-")
        and "Summarize the conversation history" in _last_text(e)
    ]
    turns = [e for e in reqs if e not in summaries and e not in failed]
    return turns, summaries + failed


def _events(chat, kind: str) -> list:
    return [
        (e.get("data") or {}).get("description")
        for e in chat.events
        if isinstance(e, dict) and e.get("type") == kind
    ]


def _thinking(entry: dict) -> dict:
    return _gc(entry).get("thinkingConfig") or {}


async def _upsert(t: Suite, model_id: str, base: str, params: dict) -> tuple:
    """Create (or update) a workspace model on a pipe model; (status, body)."""
    form = {
        "id": model_id,
        "base_model_id": base,
        "name": model_id,
        "params": params,
        "meta": {"description": "e2e owui012", "capabilities": {}},
    }
    status, existing = await t.owui.api("GET", f"/api/v1/models/model?id={model_id}")
    if status == 200 and isinstance(existing, dict) and existing.get("id"):
        result = await t.owui.api("POST", "/api/v1/models/model/update", form)
    else:
        result = await t.owui.api("POST", "/api/v1/models/create", form)
    await t.owui.models(refresh=True)
    return result


# ==================================================================== group
async def owui012(t: Suite, mock) -> None:
    ctx = await preflight(t, mock)
    version = await t.owui.version()
    v012 = version_tuple(version) >= (0, 12)
    config = await t.owui.chat_config()
    tool_search = "ENABLE_TOOL_SEARCH" in config
    restore = {
        key: config[key] for key in {**COMPACTION, **TOOL_SEARCH} if key in config
    }
    ctx.version = version
    previous_servers = None
    try:
        async with t.browser() as b:
            await t.owui.set_chat_config(**COMPACTION)
            await compaction(ctx, b, v012)
            await compaction_followup(ctx, b, v012)
            await summary_failed(ctx, b, v012, "summary-error")
            await summary_failed(ctx, b, v012, "summary-empty")
            await saved_summary(ctx, b)
            await t.owui.set_chat_config(**OFF)
            previous_servers = await t.owui.set_tool_servers(_connections()[:1])
            await t.owui.set_chat_config(**TOOL_SEARCH)
            await toolsearch(ctx, b, tool_search)
            await t.owui.set_chat_config(**OFF)
            await controls(ctx, v012)
            await custom_params(ctx, b)
            await stream_chat(ctx, b)
            await stream_model(ctx, b)
    finally:
        await t.owui.set_chat_config(**restore)
        if previous_servers is not None:
            await t.owui.set_tool_servers(previous_servers)
        for model_id in (PARAMS_MODEL, PARAMS_MODEL_25, NOSTREAM_MODEL, CONTROLS_MODEL):
            await t.owui.delete_model(model_id)


def _check(ctx: Ctx, sid: str, title: str, ok, detail: str) -> bool:
    return ctx.check(
        f"owui012.{sid}", title, ok, f"owui={getattr(ctx, 'version', '?')} {detail}"
    )


def _absent(ctx: Ctx, feature: str) -> str:
    return f"feature not in Open WebUI {getattr(ctx, 'version', '?')}: {feature}"


# =============================================================== compaction
async def compaction(ctx: Ctx, b, v012: bool) -> None:
    """Three tool rounds with compaction on and a threshold of 1 token: from
    0.12 Open WebUI compacts before the 3rd and 4th request of the turn."""
    await ctx.mock.reset()
    c = await ctx.chat(b, PRO3, directive([[TIMESTAMP], [CALC], [CALC2]]))
    reqs = await ctx.requests()
    turns, summaries = _split(reqs)
    r2, r3 = (turns + [{}, {}, {}, {}])[2:4]
    statuses = [e.get("status") for e in reqs]
    saved = (
        c.done
        and not c.error
        and [s for _, s, _ in _calls(c)] == ["completed"] * 3
        and len(c.function_outputs) == 3
    )
    if v012:
        ok = (
            saved
            and statuses == [200] * len(reqs)
            and _answers(turns) == "[fc,fc,fc,final]"
            and [e.get("answer") for e in summaries] == ["summary", "summary"]
            and all(not _declared(e) and not e.get("tool_kinds") for e in summaries)
            and r2.get("kinds") == ["user:text", "model:fc", "user:fr"]
            and _fcs(r2) == [("mock-call-1-0", "calculate_timestamp", "issued")]
            and [i for i, _, _ in _frs(r2)] == ["mock-call-1-0"]
            and not r2.get("orphan_fr")
            and _summary(r2) == "MOCK-SUMMARY of mock-call-0-0"
            and r3.get("kinds") == ["user:text", "model:fc", "user:fr"]
            and _fcs(r3) == [("mock-call-2-0", "calculate_timestamp", "issued")]
            and _summary(r3) == "MOCK-SUMMARY of mock-call-1-0"
            # the final answer echoes the responses Gemini got: the last round's
            and _final(c).startswith("MOCK-FINAL calculate_timestamp=")
            and _final(c).count("_timestamp=") == 1
        )
        title = (
            "compaction between tool rounds (3 rounds, threshold 1): the summary "
            "request goes through the pipe without tools and answers a plain "
            "summary; after it the pipe gets the summary in the system prompt, the "
            "user message and the last round only, and sends a valid request "
            "(function call with its signature + its response, no orphan)"
        )
    else:
        ok = (
            saved
            and _answers(turns) == "[fc,fc,fc,final]"
            and not summaries
            and len(r3.get("fc") or []) == 3
            and not _summary(r3)
        )
        title = _absent(
            ctx,
            "no compaction between tool rounds (only before a reply); the 3-round "
            "turn with compaction on keeps every round",
        )
    _check(
        ctx,
        "compaction",
        title,
        ok,
        _bdetail(
            c,
            turns,
            f"statuses={statuses} summaries={[e.get('answer') for e in summaries]} "
            f"summary_r2={short(_summary(r2), 80)!r} summary_r3="
            f"{short(_summary(r3), 80)!r} r2: {_req_tokens(r2)} r3: "
            f"{_req_tokens(r3)} events={_events(c, 'context_compaction')}",
        ),
    )


async def compaction_followup(ctx: Ctx, b, v012: bool) -> None:
    """A tool turn after a turn with two rounds: the compaction boundary falls
    into the earlier turn, Open WebUI puts the current user message before the
    kept rounds of that turn. The request must stay valid."""
    await ctx.t.owui.set_chat_config(ENABLE_CONTEXT_COMPACTION=False)
    try:
        await ctx.mock.reset()
        first = await ctx.chat(b, PRO3, directive([[TIMESTAMP], [CALC]]))
    finally:
        await ctx.t.owui.set_chat_config(**COMPACTION)
    if not first.chat_id:
        _check(ctx, "compaction-followup", "follow-up turn", False, "turn 1 failed")
        return
    await ctx.mock.reset()
    text = directive([[CALC2], [TIMESTAMP]])
    c = await ctx.chat(b, PRO3, text, chat_id=first.chat_id, parent_id=first.message_id)
    reqs = await ctx.requests()
    turns, summaries = _split(reqs)
    after = turns[1] if len(turns) > 1 else {}
    kinds = after.get("kinds") or []
    statuses = [e.get("status") for e in reqs]
    first_sigs = [
        s for x in after.get("fc") or [] if x.get("first") for s in [x["sig"]]
    ]
    valid = (
        c.done
        and not c.error
        and statuses == [200] * len(reqs)
        and not after.get("orphan_fr")
        and len(after.get("fc") or []) == len(after.get("fr") or [])
        and all(sig == "issued" for sig in first_sigs)
    )
    if v012:
        ok = (
            valid and len(summaries) >= 1 and _summary(after).startswith("MOCK-SUMMARY")
        )
        title = (
            "compaction in a follow-up tool turn whose boundary falls into the "
            "earlier turn: the request after it is valid (every call with its "
            "response, first calls signed); kinds= shows the order Open WebUI sends"
        )
    else:
        ok = valid and not summaries and not _summary(after)
        title = _absent(ctx, "no compaction between tool rounds; the turn completes")
    _check(
        ctx,
        "compaction-followup",
        title,
        ok,
        _bdetail(
            c,
            turns,
            f"statuses={statuses} summaries={len(summaries)} prompt_first="
            f"{kinds[:1] == ['user:text'] and kinds.count('user:text') == 1} "
            f"summary={short(_summary(after), 80)!r} {_req_tokens(after)}",
        ),
    )


async def summary_failed(ctx: Ctx, b, v012: bool, trigger: str) -> None:
    """The summary request between tool rounds fails (HTTP 400) or returns no
    text (thinking only, MAX_TOKENS): the conversation summary must not be the
    pipe's error or placeholder text, Open WebUI then uses its own summary of
    the compacted messages."""
    mark = ctx.t.mark()
    await ctx.mock.reset()
    word = "summary-400" if trigger == "summary-error" else trigger
    c = await ctx.chat(b, PRO3, directive([[TIMESTAMP], [CALC]]) + f" {word}")
    reqs = await ctx.requests()
    if trigger == "summary-error":
        ctx.t.expect_errors(mark, ("function_gemini", "mock: summary-400"))
    turns, summaries = _split(reqs)
    r2 = turns[2] if len(turns) > 2 else {}
    summary = _summary(r2)
    expected = "http-400" if trigger == "summary-error" else "summary"
    leaked = [
        text
        for text in (
            "Error",
            "INVALID_ARGUMENT",
            "mock: summary-400",
            "[No content generated",
            "Summary thinking.",
        )
        if text in summary
    ]
    if v012:
        ok = (
            c.done
            and not c.error
            and _answers(turns) == "[fc,fc,final]"
            and [e.get("answer") for e in summaries] == [expected]
            and summary.startswith(FALLBACK_USER)
            and not leaked
        )
        what = "fails (HTTP 400)" if trigger == "summary-error" else "has no text"
        title = (
            f"compaction between tool rounds, the summary request {what}: the "
            "next request carries Open WebUI's own summary of the compacted "
            "messages, not the pipe's error or placeholder text"
        )
    else:
        ok = c.done and not summaries and not summary
        title = _absent(ctx, "no compaction between tool rounds")
    _check(
        ctx,
        trigger,
        title,
        ok,
        _bdetail(
            c,
            turns,
            f"summaries={[e.get('answer') for e in summaries]} leaked={leaked} "
            f"summary={short(summary, 160)!r}",
        ),
    )


async def saved_summary(ctx: Ctx, b) -> None:
    """Compaction before a reply (Open WebUI 0.11 and 0.12): the summary is
    saved with the chat and replaces the compacted turns in every later
    request. A failed summary request must not save the pipe's error text."""
    mark = ctx.t.mark()
    turns, chat_id, parent = [], None, None
    for i, text in enumerate(
        ("First turn.", "Second turn.", "Third turn summary-400.")
    ):
        if i == 2:
            await ctx.mock.reset()
        c = await ctx.chat(b, TEXT, text, chat_id=chat_id, parent_id=parent)
        turns.append(c)
        chat_id, parent = c.chat_id, c.message_id
        if not c.done:
            break
    reqs = await ctx.requests()
    ctx.t.expect_errors(mark, ("function_gemini", "mock: summary-400"))
    main, summaries = _split(reqs)
    saved = await ctx.t.owui.get_chat(chat_id) if chat_id else {}
    messages = ((saved.get("chat") or {}).get("history") or {}).get("messages") or {}
    stored = [
        m.get("contextSummary")
        for m in messages.values()
        if m.get("role") == "user" and m.get("contextSummary")
    ]
    sent = _summary(main[-1]) if main else ""
    leaked = [
        text
        for text in ("Error", "INVALID_ARGUMENT", "mock: summary-400")
        if any(text in s for s in stored + [sent])
    ]
    ok = (
        len(turns) == 3
        and all(c.done for c in turns)
        and [e.get("answer") for e in summaries] == ["http-400"]
        and len(stored) == 1
        and stored[0].startswith("- user: First turn.")
        and sent == stored[0]
        and not leaked
    )
    _check(
        ctx,
        "saved-summary",
        "compaction before a reply (third turn), the summary request fails "
        "(HTTP 400): the summary Open WebUI saves with the chat and sends is its "
        "own summary of the compacted turns, not the pipe's error text",
        ok,
        f"done={[c.done for c in turns]} summaries="
        f"{[e.get('answer') for e in summaries]} stored={[short(s, 120) for s in stored]} "
        f"sent={short(sent, 120)!r} leaked={leaked} upstream={_answers(main)}",
    )


# ============================================================== tool search
async def toolsearch(ctx: Ctx, b, available: bool) -> None:
    """Tool Search with every tool deferred except get_current_timestamp."""
    await ctx.mock.reset()
    c = await ctx.chat(b, PRO3, directive([[search("calculate_timestamp")], [CALC]]))
    reqs = await ctx.requests()
    r0, r1 = (reqs + [{}, {}])[:2]
    declared0, declared1 = _declared(r0), _declared(r1)
    manifest = "<available_tools>" in _system(r0)
    found = [x for x in r1.get("fr") or [] if x.get("name") == "search_tools"]
    if available:
        ok = (
            c.done
            and not c.error
            and {"search_tools", "get_current_timestamp"} <= set(declared0)
            and "calculate_timestamp" not in declared0
            and manifest
            and "<name>calculate_timestamp</name>" in _system(r0)
            and "calculate_timestamp" in declared1
            and len(found) == 1
            and "calculate_timestamp" in json.dumps(found[0].get("response"))
            and _answers(reqs) == "[fc,fc,final]"
            and _calls(c)
            == [
                ("search_tools", "completed", "mock-call-0-0"),
                ("calculate_timestamp", "completed", "mock-call-1-0"),
            ]
            and "calculate_timestamp=" in _final(c)
        )
        title = (
            "Tool Search: search_tools and the always loaded tool declared, the "
            "deferred calculate_timestamp only listed in <available_tools>; after "
            "search_tools returned it, it is declared in the next round, called and "
            "run by Open WebUI"
        )
    else:
        ok = c.done and "search_tools" not in declared0 and not manifest
        title = _absent(
            ctx, "Tool Search: search_tools is not declared, no <available_tools> list"
        )
    _check(
        ctx,
        "toolsearch",
        title,
        ok,
        _bdetail(
            c,
            reqs,
            f"manifest={manifest} declared0_n={len(declared0)} "
            f"search_declared={'search_tools' in declared0} "
            f"found_declared={'calculate_timestamp' in declared1} {_req_tokens(r1)}",
        ),
    )
    await toolsearch_followup(ctx, b, c, available)
    await toolsearch_mapped(ctx, b, available)


async def toolsearch_followup(ctx: Ctx, b, first, available: bool) -> None:
    """The next turn calls the tool found in turn 1 without searching again
    (without Tool Search the tool is declared anyway)."""
    await ctx.mock.reset()
    c = await ctx.chat(
        b,
        PRO3,
        directive([[CALC2]]),
        chat_id=first.chat_id,
        parent_id=first.message_id,
    )
    reqs = await ctx.requests()
    r0 = reqs[0] if reqs else {}
    _check(
        ctx,
        "toolsearch-followup",
        "Tool Search, next turn: the tool search_tools returned in turn 1 is "
        "declared again (its result is in the history), the direct call runs"
        if available
        else _absent(ctx, "Tool Search: every tool is declared, the direct call runs"),
        bool(first.chat_id)
        and c.done
        and not c.error
        and "calculate_timestamp" in _declared(r0)
        and _answers(reqs) == "[fc,final]"
        and _calls(c) == [("calculate_timestamp", "completed", "mock-call-0-0")],
        _bdetail(c, reqs, _req_tokens(reqs[-1] if reqs else {})),
    )


async def toolsearch_mapped(ctx: Ctx, b, available: bool) -> None:
    """A deferred OpenAPI tool whose name Gemini does not accept (lookup.v2)."""
    tool_mock = ctx.t.mock("tools")
    await ctx.mock.reset()
    await tool_mock.reset()
    call = {"name": LOOKUP, "args": {"key": "k2"}}
    c = await ctx.chat(
        b,
        PRO3,
        directive([[search(LOOKUP)], [call]] if available else [[call]]),
        tool_ids=[f"server:{OPENAPI_ID}"],
    )
    reqs = await ctx.requests()
    log = await tool_mock.requests()
    lookup = [e.get("query", {}).get("key") for e in log if e.get("path") == "/lookup"]
    mapped = gemini_function_name(LOOKUP)
    r0, r1 = (reqs + [{}, {}])[:2]
    if available:
        ok = mapped not in _declared(r0) and mapped in _declared(r1)
        call_id = "mock-call-1-0"
        title = (
            f"Tool Search, a deferred OpenAPI tool ({LOOKUP}): declared as {mapped} "
            "after search_tools found it, called under its own name, the tool "
            "server gets the call"
        )
    else:
        ok = mapped in _declared(r0)
        call_id = "mock-call-0-0"
        title = _absent(
            ctx,
            f"Tool Search: the OpenAPI tool {LOOKUP} is declared as {mapped} in "
            "the first request and called",
        )
    _check(
        ctx,
        "toolsearch-mapped",
        title,
        ok
        and c.done
        and not c.error
        and lookup == ["k2"]
        and (LOOKUP, "completed", call_id) in _calls(c),
        _bdetail(
            c,
            reqs,
            f"lookup={lookup} mapped_declared=[{mapped in _declared(r0)},"
            f"{mapped in _declared(r1)}]",
        ),
    )


# ======================================================== model controls etc.
async def controls(ctx: Ctx, v012: bool) -> None:
    """Model controls (0.12) on a workspace model of a Gemini pipe model."""
    status, data = await _upsert(
        ctx.t, CONTROLS_MODEL, PRO3, {"model_controls": THINKING_CONTROL}
    )
    detail = json.dumps(data)[:200] if not isinstance(data, str) else data[:200]
    if v012:
        ok = status == 400 and "server-managed provider model" in detail
        title = (
            "model controls on a workspace model of a Gemini pipe model: Open "
            "WebUI 0.12 refuses them (HTTP 400), so no control reaches the pipe"
        )
    else:
        ok = status == 200
        title = _absent(ctx, "model controls (model_controls is an ordinary param)")
    await ctx.t.owui.delete_model(CONTROLS_MODEL)
    _check(ctx, "controls", title, ok, f"http={status} answer={detail}")


async def custom_params(ctx: Ctx, b) -> None:
    """The custom parameters a Thinking control would set, as Advanced Params
    of a workspace model and of the chat (Controls)."""
    s1, _ = await _upsert(
        ctx.t,
        PARAMS_MODEL,
        PRO3,
        {"custom_params": {"reasoning_effort": "low", "include_thoughts": "false"}},
    )
    s2, _ = await _upsert(
        ctx.t, PARAMS_MODEL_25, TEXT, {"custom_params": {"thinking_budget": "1024"}}
    )
    await ctx.mock.reset()
    r1 = await ctx.t.owui.chat(PARAMS_MODEL, "Hi", stream=False)
    think1 = _thinking(await ctx.mock.last(_gen))
    await ctx.mock.reset()
    r2 = await ctx.t.owui.chat(PARAMS_MODEL_25, "Hi", stream=False)
    think2 = _thinking(await ctx.mock.last(_gen))
    await ctx.mock.reset()
    c = await ctx.chat(
        b, PARAMS_MODEL, "Hi", params={"custom_params": {"reasoning_effort": "high"}}
    )
    think3 = _thinking(await ctx.mock.last(_gen))
    # informational: reasoning_effort on Gemini 2.5 (not mapped to a budget)
    await ctx.mock.reset()
    await ctx.t.owui.chat(TEXT, "Hi", stream=False, reasoning_effort="low")
    think4 = _thinking(await ctx.mock.last(_gen))

    def level(think: dict) -> str:
        return str(think.get("thinkingLevel", "")).upper()

    _check(
        ctx,
        "custom-params",
        "custom params of a workspace model reach the pipe (reasoning_effort=low "
        "-> thinkingLevel LOW, include_thoughts=false; 2.5: thinking_budget=1024), "
        "the chat's custom params win (reasoning_effort=high -> HIGH)",
        (s1, s2) == (200, 200)
        and r1.status == 200
        and level(think1) == "LOW"
        and think1.get("includeThoughts") is False
        and r2.status == 200
        and think2.get("thinkingBudget") == 1024
        and c.done
        and level(think3) == "HIGH"
        and think3.get("includeThoughts") is False,
        f"models=[{s1},{s2}] model_params={think1} model_params_25={think2} "
        f"chat_params={think3} chat_done={c.done} "
        f"reasoning_effort_25={think4}",
    )


async def stream_chat(ctx: Ctx, b) -> None:
    """Stream Chat Response off in the chat's Controls (honoured by the 0.12
    web UI): stream=false and params.stream_response=false."""
    await ctx.mock.reset()
    c = await ctx.chat(
        b,
        PRO3,
        directive([[TIMESTAMP], [CALC]]),
        stream=False,
        params={"stream_response": False},
    )
    reqs = await ctx.requests()
    actions = [e.get("action") for e in reqs]
    _check(
        ctx,
        "stream-chat",
        "Stream Chat Response off for the chat (stream=false), two tool rounds: "
        "the first request is not streamed, Open WebUI runs both tools, the turn "
        "is saved as done with the final answer",
        c.done
        and not c.error
        and actions
        == ["generateContent", "streamGenerateContent", "streamGenerateContent"]
        and [s for _, s, _ in _calls(c)] == ["completed", "completed"]
        and "calculate_timestamp=" in _final(c),
        _bdetail(c, reqs, f"actions={actions}"),
    )


async def stream_model(ctx: Ctx, b) -> None:
    """Stream Chat Response off in the model's Advanced Params: Open WebUI sets
    stream=false itself (the browser asks for a stream)."""
    status, _ = await _upsert(ctx.t, NOSTREAM_MODEL, PRO3, {"stream_response": False})
    await ctx.mock.reset()
    c = await ctx.chat(b, NOSTREAM_MODEL, directive([[TIMESTAMP]]), stream=True)
    reqs = await ctx.requests()
    actions = [e.get("action") for e in reqs]
    _check(
        ctx,
        "stream-model",
        "Stream Chat Response off for the model (Open WebUI sends stream=false), "
        "a tool round: the call runs, the turn is saved as done",
        status == 200
        and c.done
        and not c.error
        and actions == ["generateContent", "streamGenerateContent"]
        and _calls(c) == [("get_current_timestamp", "completed", "mock-call-0-0")]
        and _final(c).startswith("MOCK-FINAL get_current_timestamp="),
        _bdetail(c, reqs, f"model={status} actions={actions}"),
    )
