"""
Real-API smoke test of the Gemini pipe's native tool calling: one scenario per
risk R1-R15 of the 1.18.0 spec (section 7), the server-side behaviour the
mock-based harness cannot confirm.

Started by tests/e2e/smoke/gemini-tools.sh inside a fresh Open WebUI container
(docs/testing.md, "Real-API smoke test (manual, needs a key)"). It installs
google_gemini.py and the google_search_tool filter, stores GOOGLE_API_KEY in the
encrypted valve and points the pipe's BASE_URL at a recording pass-through proxy
(smoke/proxy.py) in front of generativelanguage.googleapis.com, so the requests
reach the real API and the proxy records what went up and what came back. With
``--dry-run-mock`` the proxy forwards to the e2e Gemini mock instead (the same
redirection the harness uses), which checks the runner itself without a key.
The tools are deterministic: Open WebUI's built-in tools, the e2e workspace
tool (probe/workspace_tool.py), the OpenAPI and MCP tool mocks and client tools.

Each scenario tells the model exactly which tool to call; a scenario whose model
did not do what the prompt asked (no call, one call instead of two) or that met
a temporary upstream error is repeated up to ``--retries`` times. Results:
PASS / FAIL / SKIP per risk with the evidence (upstream HTTP status, finish
reasons, function calls and whether their thought signatures came back), plus
a server log scan and a secret scan. The key is never printed or written:
stdout / stderr and every output file are redacted, and finding it anywhere is
a FAIL.

Outputs in --out: summary.md, smoke.json, upstream.jsonl (the proxy record).
Exit code: 0 = no FAIL, 1 = at least one FAIL, 2 = setup error (install,
preflight), 128 + signal after SIGINT / SIGTERM (partial results written).
"""

import argparse
import asyncio
import contextlib
import json
import os
import re
import signal
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from importlib import metadata
from typing import Optional

from harness import OWUI, BrowserSession, Results, ServerLog, SetupError, Suite
from harness import Mock, short
from harness.config import MCP_PORT, MOCK_HOST, MOCK_PORTS, WORKSPACE_TOOL_FILE
from harness.known import staged_version
from proxy import GENERATE, RecordingProxy
from suites.gemini import WARNINGS as PIPE_WARNINGS

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
FID = "gemini"
VERTEX_FID = "gemini_vertex"
GEMINI_PATH = "pipelines/google/google_gemini.py"
FILTER_PATH = "filters/google_search_tool.py"
SEARCH_FILTER = "google_search_tool"
WS_TOOL = "smoke_ws_tool"
OPENAPI_ID = "smoketools"
MCP_ID = "smokemcp"
PROXY_PORT = 9120
THINKING_BUDGET = 512  # tokens, Gemini 2.5 (Gemini 3: THINKING_LEVEL low)
REAL_API = "https://generativelanguage.googleapis.com"
MOCK_API = f"http://{MOCK_HOST}:{MOCK_PORTS['gemini']}"
TRANSIENT = (429, 500, 502, 503, 504)
MALFORMED = ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL")
# Open WebUI 0.11.4's built-in tools (spec F15): all declared on the browser path
BUILTINS = (
    "get_current_timestamp",
    "calculate_timestamp",
    "ask_user",
    "list_knowledge_bases",
    "search_knowledge_bases",
    "query_knowledge_bases",
    "grep_knowledge_files",
    "search_knowledge_files",
    "query_knowledge_files",
    "view_knowledge_file",
    "search_chats",
    "view_chat",
    "search_notes",
    "view_note",
    "write_note",
    "replace_note_content",
    "create_tasks",
    "update_task",
    "create_automation",
    "update_automation",
    "list_automations",
    "toggle_automation",
    "delete_automation",
    "search_calendar_events",
    "create_calendar_event",
    "update_calendar_event",
    "delete_calendar_event",
)
# Generate requests a scenario needs when the model does what it is asked
# (printed as the estimate; R9 goes to Vertex, not through the proxy).
ESTIMATE = {
    "R1": 4,
    "R2": 2,
    "R3": 5,
    "R4": 3,
    "R5": 2,
    "R6": 2,
    "R7": 3,
    "R8": 5,
    "R10": 1,
    "R12": 2,
    "R14": 2,
    "R15": 2,
}
RISKS = {
    "R1": "built-in, workspace, OpenAPI, MCP and typical client schemas accepted "
    "(parametersJsonSchema, no 400)",
    "R2": "a tool without parameters is declared without a schema and called",
    "R3": "Gemini 3 tool combination: Search + a function call in one round, "
    "continued; next turn without web search; tool_choice required / none",
    "R4": "API continuation without reasoning_details (placeholder signature on "
    "Gemini 3, none on 2.x)",
    "R5": "parallel calls: only the first signed, replayed as FC1(sig), FC2, FR1, FR2",
    "R6": "image tool result: function response followed by Open WebUI's image "
    "message (two user contents)",
    "R7": "function-call history in a request without declarations (legacy turn "
    "after a tool turn)",
    "R8": "Gemini 2.5 thinking + tools over several turns (signatures of non-call "
    "parts dropped)",
    "R9": "Vertex AI: grounding wins, synthetic ids, 64-character names",
    "R10": "Gemini 2.5 + web search + tools: grounding wins, no 400",
    "R11": "google-genai upgrade when the function is saved (note only)",
    "R12": "streamed tool turn: complete functionCall parts, final text not empty",
    "R13": "frequency of MALFORMED_FUNCTION_CALL / UNEXPECTED_TOOL_CALL and the "
    "default_api. prefix",
    "R14": "tool approval with parallel calls: the continuation is accepted",
    "R15": "INCLUDE_THOUGHTS=false: Gemini 3 still returns thought signatures",
}
ORDER = [f"R{n}" for n in range(1, 16)]

TIMESTAMP = {"name": "get_current_timestamp", "args": {}}
CALC = {"name": "calculate_timestamp", "args": {"days_ago": 1}}
TS_PROMPT = (
    "Call the get_current_timestamp tool now. It takes no arguments. After you "
    "have its result, reply with one short sentence that states the timestamp."
)
PARALLEL_PROMPT = (
    "Make two tool calls at once, in parallel, in your first step: "
    "get_current_timestamp (no arguments) and calculate_timestamp with "
    "days_ago=1. Do not answer before both results are back. Then reply with one "
    "short sentence that states both results."
)
CALC_PROMPT = (
    "Now call the calculate_timestamp tool with days_ago=1 and reply with the "
    "resulting date in one short sentence."
)
DONE_PROMPT = "Thanks. Reply with just the word DONE."
READY_PROMPT = "Reply with exactly the word READY. Do not call any tool."


# ------------------------------------------------------------------ secrets
class Redactor:
    """Replaces every secret with *** (stdout, stderr, output files) and
    counts the replacements: any count above zero is a FAIL."""

    def __init__(self, secrets: list):
        self.secrets = [s for s in secrets if isinstance(s, str) and len(s) >= 6]
        self.count = 0

    def __call__(self, text: str) -> str:
        for secret in self.secrets:
            if secret in text:
                self.count += text.count(secret)
                text = text.replace(secret, "***")
        return text


class RedactingStream:
    def __init__(self, stream, redactor: Redactor):
        self._stream = stream
        self._redact = redactor

    def write(self, text: str) -> int:
        return self._stream.write(self._redact(text))

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


# ------------------------------------------------------------------ helpers
def client_tool(name: str, parameters: Optional[dict] = None, desc: str = "") -> dict:
    function = {"name": name, "description": desc or f"Client function {name}."}
    if parameters is not None:
        function["parameters"] = parameters
    return {"type": "function", "function": function}


LOOKUP_CODE = client_tool(
    "lookup_code",
    {
        "type": "object",
        "properties": {"code": {"type": "integer", "description": "the code"}},
        "required": ["code"],
    },
    "Look up the meaning of a numeric code.",
)
GET_WEATHER = client_tool(
    "get_weather",
    {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "city name"}},
        "required": ["city"],
    },
    "Current weather for a city.",
)
# Schemas as Open WebUI builds them for workspace tools (pydantic: title,
# default, anyOf + null), OpenAPI servers (format, integer enum) and MCP servers
# ($defs, $ref, additionalProperties false), plus kept keywords (oneOf, const,
# type arrays) and a tool without parameters.
TYPICAL_TOOLS = [
    client_tool(
        "create_event",
        {
            "type": "object",
            "title": "create_event_args",
            "properties": {
                "title": {"type": "string", "title": "Title"},
                "start": {"type": "string", "format": "date-time", "title": "Start"},
                "minutes": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": None,
                    "title": "Minutes",
                },
                "tags": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "default": None,
                    "title": "Tags",
                },
                "note": {"type": ["string", "null"], "default": None},
            },
            "required": ["title", "start"],
        },
        "Create a calendar event.",
    ),
    client_tool(
        "forecast",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "city name"},
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "default": "metric",
                },
                "days": {"type": "integer", "format": "int32", "enum": [1, 3, 7]},
                "date": {"type": "string", "format": "date"},
            },
            "required": ["city"],
        },
        "Weather forecast (OpenAPI style parameters).",
    ),
    client_tool(
        "submit_order",
        {
            "type": "object",
            "title": "submit_orderArguments",
            "properties": {
                "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
                "mode": {"$ref": "#/$defs/Mode"},
                "kind": {"oneOf": [{"const": "gift"}, {"const": "normal"}]},
            },
            "required": ["items"],
            "additionalProperties": False,
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "qty": {"type": "integer", "minimum": 1},
                    },
                    "required": ["name", "qty"],
                    "additionalProperties": False,
                },
                "Mode": {"type": "string", "enum": ["fast", "slow"]},
            },
        },
        "Submit an order (MCP style parameters).",
    ),
    client_tool("ping", {"type": "object", "properties": {}}, "Ping the server."),
]


def gen(records: list) -> list:
    return [r for r in records if r.get("action") in GENERATE]


def forwarded(records: list) -> list:
    """Generate requests that reached the upstream (not blocked by the budget)."""
    return [r for r in gen(records) if not r.get("blocked")]


def upstream_failures(records: list) -> list:
    """Failed requests other than generate ones (the model listing), grouped:
    ``GET /v1beta/models http=400 x4: API key not valid. ...``."""
    groups = Counter(
        (
            r.get("method"),
            r.get("path"),
            r.get("status"),
            resp(r).get("error") or r.get("error") or "no error message",
        )
        for r in records
        if r.get("action") not in GENERATE and (r.get("status") or 0) >= 400
    )
    return [
        f"{method} {path} http={status}{f' x{n}' if n > 1 else ''}: {short(error, 200)}"
        for (method, path, status, error), n in groups.items()
    ]


def http(records: list) -> str:
    return "[" + ",".join(str(r.get("status")) for r in records) + "]"


def all_ok(records: list) -> bool:
    return bool(records) and all(r.get("status") == 200 for r in records)


def req(record: Optional[dict]) -> dict:
    return (record or {}).get("req") or {}


def resp(record: Optional[dict]) -> dict:
    return (record or {}).get("resp") or {}


def calls_of(records: list) -> list:
    """Function calls Gemini answered with, over all records: (name, sig)."""
    return [
        (c.get("name"), c.get("sig")) for r in records for c in resp(r).get("fc") or []
    ]


def last_finish(record: dict) -> Optional[str]:
    finish = resp(record).get("finish") or []
    return finish[-1] if finish else None


def upstream_line(r: dict) -> str:
    """One proxy record as an evidence line."""
    q, a = req(r), resp(r)
    mode = "stream" if r.get("action") == "streamGenerateContent" else "json"
    fc = ",".join(f"{c['name']}{'(sig)' if c['sig'] else ''}" for c in a.get("fc", []))
    replay = ",".join(f"{c['name']}:{c['sig']}" for c in q.get("fc", []))
    bits = [f"#{r.get('n')} {r.get('model')} {mode} http={r.get('status')}"]
    if a.get("error") or r.get("error"):
        bits.append(f"error={short(a.get('error') or r.get('error'), 220)}")
    bits.append(f"finish={','.join(a.get('finish') or []) or None}")
    if fc:
        bits.append(f"fc=[{fc}] fc_chunks={a.get('fc_chunks')}")
    if a.get("server"):
        bits.append(f"server_parts={a['server']}")
    if a.get("thought_sigs") or a.get("text_sigs"):
        bits.append(
            f"thought_sigs={a.get('thought_sigs')} text_sigs={a.get('text_sigs')}"
        )
    bits.append(f"parts={a.get('parts')} text={a.get('text_len')}")
    bits.append(
        f"| decl={q.get('declared_n')} tools={q.get('tool_kinds')} "
        f"include={q.get('include_flag')} mode={q.get('fc_mode')}"
    )
    if replay:
        bits.append(f"replay=[{replay}] fr={len(q.get('fr') or [])}")
    if q.get("server_replayed"):
        bits.append(f"server_replayed={q['server_replayed']}")
    bits.append(f"kinds={q.get('kinds')}")
    return " ".join(bits)


def final(c) -> str:
    return (c.output_text or c.content or "").strip()


def pipe_error(text: str) -> bool:
    return (
        text.startswith("Error")
        or "An error occurred" in text
        or "Error during streaming" in text
    )


def pending(message: dict) -> bool:
    return any(
        i.get("type") == "function_call" and i.get("status") == "pending"
        for i in message.get("output") or []
        if isinstance(i, dict)
    )


def chat_line(c) -> str:
    calls = ",".join(f"{n}:{s}" for n, s, _, _ in c.function_calls)
    details = sum(len(d) for _, d in c.reasoning_items)
    error = short(c.error, 160) if c.error else None
    return (
        f"done={c.done} error={error} calls=[{calls}] outputs={len(c.function_outputs)}"
        f" reasoning_items={len(c.reasoning_items)} reasoning_details={details}"
        f" final={short(final(c), 120)}"
    )


def chat_ok(c) -> bool:
    return c.done and not c.error and bool(final(c)) and not pipe_error(final(c))


def api_line(r) -> str:
    calls = ",".join(
        str((x.get("function") or {}).get("name") or x.get("name"))
        for x in r.tool_calls
    )
    return (
        f"http={r.status} tool_calls=[{calls}] finish={r.finish_reasons} "
        f"content={short(r.content, 120)}"
    )


@dataclass
class Outcome:
    status: str
    evidence: list = field(default_factory=list)
    retry: str = ""  # why another attempt may help ("" = it would not)


@dataclass
class Report:
    risk: str
    title: str
    status: str
    evidence: list
    attempts: int
    requests: int
    notes: list


# ------------------------------------------------------------------- runner
class Smoke:
    def __init__(self, args: argparse.Namespace, key: str, redact: Redactor):
        self.args = args
        self.key = key
        self.redact = redact
        self.dry = args.dry_run_mock
        self.m3, self.m25 = args.model, args.model_25
        self.only = {r.strip().upper() for r in args.only.split(",") if r.strip()}
        # hard cap of forwarded generate requests: default 2x the estimated maximum
        self.max_requests = args.max_requests or 2 * self.estimate()[1]
        self.proxy = RecordingProxy(
            MOCK_API if self.dry else REAL_API,
            PROXY_PORT,
            max_requests=self.max_requests,
        )
        self.owui = OWUI()
        self.results = Results(verbose=False)
        self.log = ServerLog()
        self.t = Suite("setup", self.owui, self.results, self.log)
        self.b: Optional[BrowserSession] = None
        self.reports: dict = {}
        self.meta: dict = {}
        self.latest_listed = False
        self.started = time.time()
        self.signal = ""
        self.setup_error = ""
        self.driver_error = ""
        self.timed_out = False
        self.final_attempt = True
        self.budget_skipped: list = []  # risks not run: the budget was used up

    # ---------------------------------------------------------------- output
    def say(self, text: str = "") -> None:
        print(text, flush=True)

    def selected(self, risk: str) -> bool:
        return not self.only or risk in self.only

    def ask(self, text: str, rounds: Optional[list] = None, **options) -> str:
        """The prompt; in dry-run mode plus the mock's MOCKTOOLS directive."""
        if self.dry and rounds is not None:
            return f"{text} MOCKTOOLS:" + json.dumps({"rounds": rounds, **options})
        return text

    def model(self, name: str, fid: str = FID) -> str:
        return f"{fid}.{name}"

    # ----------------------------------------------------------------- setup
    async def setup(self) -> None:
        await self.proxy.start()
        if not await self.owui.wait_healthy():
            raise SetupError("Open WebUI is not healthy")
        for name in ("gemini", "tools"):  # the dry run's Gemini mock, tool mocks
            mock = Mock(name)
            ready = await mock.wait_ready()
            await mock.close()
            if not ready:
                raise SetupError(f"mock {name} is not reachable")
        await self.owui.login()
        self.meta["owui_version"] = await self.owui.version()
        self.meta["pipe_version"] = staged_version(GEMINI_PATH)
        self.say(
            f"Open WebUI {self.meta['owui_version']}, google_gemini.py "
            f"{self.meta['pipe_version']}, "
            + ("DRY RUN against the e2e Gemini mock" if self.dry else f"{REAL_API}")
        )
        if not await self.t.install(FID, GEMINI_PATH, "Google Gemini"):
            raise SetupError(f"{GEMINI_PATH} did not load (see setup.load)")
        self.t.fail_on_warnings(*PIPE_WARNINGS)
        try:
            self.meta["google_genai"] = metadata.version("google-genai")
        except metadata.PackageNotFoundError:
            self.meta["google_genai"] = "?"
        valves = dict(
            GOOGLE_API_KEY=self.key,
            BASE_URL=self.proxy.url + "/",
            INCLUDE_THOUGHTS=True,
            THINKING_LEVEL="low",
            # Gemini 2.5 counts thinking tokens against maxOutputTokens
            THINKING_BUDGET=THINKING_BUDGET,
        )
        if self.args.api_version:
            valves["API_VERSION"] = self.args.api_version
        await self.owui.update_valves(FID, **valves)
        stored = (await self.owui.get_valves(FID)).get("GOOGLE_API_KEY", "")
        if not stored.startswith("encrypted:"):
            raise SetupError("GOOGLE_API_KEY is not stored encrypted")
        if not await self.t.install(
            SEARCH_FILTER, FILTER_PATH, "Google Search Tool", "filter-load"
        ):
            raise SetupError(f"{FILTER_PATH} did not load (see setup.filter-load)")
        with open(WORKSPACE_TOOL_FILE, encoding="utf-8") as fh:
            status, data = await self.owui.create_tool(
                WS_TOOL, "Smoke Workspace Tool", fh.read()
            )
        if status != 200:
            raise SetupError(f"workspace tool: HTTP {status} {short(data)}")
        await self.owui.set_tool_servers(self.connections())
        await self.ensure_models()
        await self.preflight()
        self.b = BrowserSession(self.owui)
        await self.b.connect()

    def connections(self) -> list:
        return [
            {
                "url": f"http://{MOCK_HOST}:{MOCK_PORTS['tools']}",
                "path": "openapi.json",
                "type": "openapi",
                "auth_type": "none",
                "key": "",
                "config": {"enable": True},
                "info": {"id": OPENAPI_ID, "name": "Smoke Tools", "description": "e2e"},
            },
            {
                "url": f"http://{MOCK_HOST}:{MCP_PORT}/mcp",
                "path": "",
                "type": "mcp",
                "auth_type": "none",
                "key": "",
                "config": {"enable": True},
                "info": {"id": MCP_ID, "name": "Smoke MCP", "description": "e2e"},
            },
        ]

    async def ensure_models(self, fid: str = FID) -> None:
        """Models the API does not list (yet) are added with MODEL_ADDITIONAL."""
        ids = await self.owui.model_ids(f"{fid}.")
        missing = [m for m in (self.m3, self.m25) if self.model(m, fid) not in ids]
        if missing:
            self.say(f"{fid}: not listed {missing}, added with MODEL_ADDITIONAL")
            await self.owui.update_valves(fid, MODEL_ADDITIONAL=",".join(missing))
            ids = await self.owui.model_ids(f"{fid}.")
        still = [m for m in (self.m3, self.m25) if self.model(m, fid) not in ids]
        if still:
            # name the cause: a bad key fails the model listing (HTTP 400 "API
            # key not valid"), and the pipe then lists no model at all
            failures = upstream_failures(self.proxy.records)
            cause = (
                "; failing upstream requests: " + " || ".join(failures[-3:])
                if failures
                else "; no failing upstream request recorded (see server.log)"
            )
            raise SetupError(f"{fid}: models {still} missing from Open WebUI{cause}")
        if fid == FID:
            latest = self.args.model_latest
            self.latest_listed = bool(latest) and self.model(latest) in ids
            self.meta["models_added"] = missing

    async def preflight(self) -> None:
        """One plain request per model: the key, the models and the proxy work."""
        for name in (self.m25, self.m3):
            mark = self.proxy.mark()
            r = await self.owui.chat(
                self.model(name),
                "Reply with the single word OK.",
                stream=False,
                max_tokens=self.args.max_tokens,
            )
            records = gen(self.proxy.since(mark))
            if r.status != 200 or not all_ok(records) or not r.content.strip():
                lines = [upstream_line(x) for x in records] or ["no upstream request"]
                raise SetupError(
                    f"preflight {name}: {api_line(r)}; upstream: " + " || ".join(lines)
                )
            self.say(f"preflight {name}: {api_line(r)}")

    # ------------------------------------------------------------- scenarios
    async def bchat(
        self,
        model: str,
        text: str,
        stream: bool = True,
        params: Optional[dict] = None,
        **kwargs,
    ):
        """Browser-path chat; one that is neither done nor waiting for approval
        is stopped, so its tool loop cannot reach the API later."""
        params = {"max_tokens": self.args.max_tokens, **(params or {})}
        c = await self.b.chat(
            model, text, stream=stream, params=params, wait=self.args.wait, **kwargs
        )
        if c.task_ids and not c.done and not pending(c.message):
            await self.b.stop(c)
        return c

    def budget_note(self) -> str:
        return (
            f"request budget used up: {self.max_requests} generate requests "
            f"forwarded (--max-requests), {self.proxy.blocked} more answered "
            "HTTP 429 by the proxy without forwarding"
        )

    async def risk(self, risk: str, fn) -> None:
        if not self.selected(risk):
            return
        self.say(f"\n--- {risk}: {RISKS[risk]}")
        if self.proxy.exhausted and risk != "R11":  # R11 makes no request
            self.budget_skipped.append(risk)
            note = f"not run: {self.budget_note()}"
            self.reports[risk] = Report(risk, RISKS[risk], SKIP, [note], 0, 0, [])
            self.say(f"[{SKIP}] {risk}  {RISKS[risk]}\n       {note}")
            return
        attempts, notes, requests = 0, [], 0
        while True:
            attempts += 1
            self.final_attempt = attempts > self.args.retries
            mark = self.proxy.mark()
            try:
                out = await fn()
            except Exception as exc:  # noqa: BLE001 - a FAIL with the traceback
                lines = traceback.format_exc().strip().splitlines()
                out = Outcome(FAIL, [f"runner error: {exc!r}", *lines[-4:]])
            records = gen(self.proxy.since(mark))
            requests += len(forwarded(records))
            if out.status == FAIL and not out.retry:
                if any(r.get("status") in TRANSIENT for r in records):
                    out.retry = "temporary upstream error"
            out.evidence += ["upstream " + upstream_line(r) for r in records]
            if self.proxy.exhausted:  # another attempt would only be blocked
                if any(r.get("blocked") for r in records):
                    notes.append(self.budget_note())
                elif out.status == FAIL and out.retry:
                    notes.append(f"no retry ({out.retry}): {self.budget_note()}")
                out.retry = ""
                break
            if out.status != FAIL or not out.retry or attempts > self.args.retries:
                break
            notes.append(f"attempt {attempts}: {out.retry}")
            self.say(f"    attempt {attempts} inconclusive ({out.retry}), retrying")
        if out.status == FAIL and out.retry:
            notes.append(f"gave up after {attempts} attempts: {out.retry}")
        report = Report(
            risk, RISKS[risk], out.status, out.evidence, attempts, requests, notes
        )
        self.reports[risk] = report
        self.say(
            f"[{out.status}] {risk}  {RISKS[risk]} ({requests} requests, "
            f"{attempts} attempt{'s' if attempts > 1 else ''})"
        )
        for line in notes + out.evidence:
            self.say(f"       {line}")

    async def r1(self) -> Outcome:
        ev, ok = [], True
        tool_ids = [WS_TOOL, f"server:{OPENAPI_ID}", f"server:mcp:{MCP_ID}"]
        for name in (self.m3, self.m25):
            mark = self.proxy.mark()
            c = await self.bchat(self.model(name), READY_PROMPT, tool_ids=tool_ids)
            records = gen(self.proxy.since(mark))
            declared = req(records[0] if records else None).get("declared") or []
            builtins = [n for n in BUILTINS if n in declared]
            groups = {
                "workspace": [
                    n for n in declared if n in ("add_numbers", "whoami", "make_image")
                ],
                "openapi": [
                    n for n in declared if n in ("get_weather", "convert_units")
                ]
                + [n for n in declared if str(n).startswith("lookup_v2")],
                "mcp": [n for n in declared if str(n).startswith(f"{MCP_ID}_")],
            }
            # Open WebUI versions differ in their built-in tools: missing ones are
            # listed, the risk is whether the API accepts what is declared
            good = all_ok(records) and chat_ok(c) and all(groups.values())
            ok = ok and good and bool(builtins)
            missing = [n for n in BUILTINS if n not in declared]
            ev.append(
                f"browser {name}: http={http(records)} declared_n={len(declared)} "
                f"builtins={len(builtins)}/{len(BUILTINS)} missing={missing} "
                + " ".join(f"{k}={len(v)}" for k, v in groups.items())
                + f" {chat_line(c)}"
            )
        for name in (self.m3, self.m25):
            mark = self.proxy.mark()
            r = await self.owui.chat(
                self.model(name),
                READY_PROMPT,
                stream=False,
                tools=TYPICAL_TOOLS,
                max_tokens=self.args.max_tokens,
            )
            records = gen(self.proxy.since(mark))
            q = req(records[0] if records else None)
            good = r.status == 200 and all_ok(records) and q.get("declared_n") == 4
            ok = ok and good
            ev.append(
                f"api {name} typical schemas: http={http(records)} "
                f"declared={q.get('declared')} noparams={q.get('noparams')} "
                f"{api_line(r)}"
            )
        return Outcome(PASS if ok else FAIL, ev)

    async def tool_turn(self, name: str, stream: bool, fid: str = FID, **kw) -> tuple:
        """A browser turn that asks for get_current_timestamp: (chat, records,
        whether Gemini called it)."""
        mark = self.proxy.mark()
        c = await self.bchat(
            self.model(name, fid), self.ask(TS_PROMPT, [[TIMESTAMP]]), stream, **kw
        )
        records = gen(self.proxy.since(mark))
        called = any(n == "get_current_timestamp" for n, _ in calls_of(records))
        if fid != FID:  # Vertex: not through the proxy
            called = any(n == "get_current_timestamp" for n, *_ in c.function_calls)
        return c, records, called

    async def r2(self) -> Outcome:
        c, records, called = await self.tool_turn(self.m3, stream=False)
        noparams = "get_current_timestamp" in req(records[0] if records else None).get(
            "noparams", []
        )
        ev = [
            f"browser {self.m3} stream=False: declared_without_schema={noparams} "
            f"called={called} http={http(records)} {chat_line(c)}"
        ]
        if not called and all_ok(records):
            return Outcome(FAIL, ev, "Gemini did not call get_current_timestamp")
        ok = noparams and called and all_ok(records) and chat_ok(c)
        return Outcome(PASS if ok else FAIL, ev)

    async def r12(self) -> Outcome:
        c, records, called = await self.tool_turn(self.m3, stream=True)
        first = resp(records[0] if records else None)
        partial = any(resp(r).get("partial_fc") for r in records)
        ev = [
            f"browser {self.m3} stream=True: called={called} fc_chunks="
            f"{first.get('fc_chunks')} partial_fc={partial} http={http(records)} "
            f"final_len={len(final(c))} {chat_line(c)}"
        ]
        if not called and all_ok(records):
            return Outcome(FAIL, ev, "Gemini did not call get_current_timestamp")
        ok = called and not partial and all_ok(records) and chat_ok(c)
        return Outcome(PASS if ok else FAIL, ev)

    async def r15(self) -> Outcome:
        await self.owui.update_valves(FID, INCLUDE_THOUGHTS=False)
        try:
            c, records, called = await self.tool_turn(self.m3, stream=True)
        finally:
            await self.owui.update_valves(FID, INCLUDE_THOUGHTS=True)
        sigs = [s for n, s in calls_of(records) if n == "get_current_timestamp"]
        details = sum(len(d) for _, d in c.reasoning_items)
        include = req(records[0] if records else None).get("thinking")
        ev = [
            f"browser {self.m3} INCLUDE_THOUGHTS=false: thinking_config={include} "
            f"called={called} signatures={sigs} saved_reasoning_details={details} "
            f"http={http(records)} {chat_line(c)}"
        ]
        if not called and all_ok(records):
            return Outcome(FAIL, ev, "Gemini did not call get_current_timestamp")
        ok = called and bool(sigs) and sigs[0] and details > 0 and all_ok(records)
        return Outcome(PASS if ok and chat_ok(c) else FAIL, ev)

    async def r5(self) -> Outcome:
        mark = self.proxy.mark()
        c = await self.bchat(
            self.model(self.m3), self.ask(PARALLEL_PROMPT, [[TIMESTAMP, CALC]])
        )
        records = gen(self.proxy.since(mark))
        first = resp(records[0] if records else None)
        produced = [(x["name"], x["sig"]) for x in first.get("fc") or []]
        follow = req(records[1] if len(records) > 1 else None)
        replay = [(x["name"], x["sig"]) for x in follow.get("fc") or []]
        contents = {x["i"] for x in follow.get("fc") or []}
        fr_contents = {x["i"] for x in follow.get("fr") or []}
        ev = [
            f"browser {self.m3}: produced={produced} replayed={replay} "
            f"fc_contents={len(contents)} fr={len(follow.get('fr') or [])} "
            f"fr_contents={len(fr_contents)} http={http(records)} {chat_line(c)}"
        ]
        if len(produced) < 2 and all_ok(records):
            return Outcome(FAIL, ev, "Gemini made fewer than 2 calls in one round")
        ok = (
            len(replay) >= 2
            and len(contents) == 1
            and len(fr_contents) == 1
            and len(follow.get("fr") or []) == len(replay)
            and all_ok(records)
            and chat_ok(c)
        )
        return Outcome(PASS if ok else FAIL, ev)

    async def r6(self) -> Outcome:
        mark = self.proxy.mark()
        c = await self.bchat(
            self.model(self.m3),
            self.ask(
                "Call the make_image tool with color red. Then name the color of "
                "the image you received in one word.",
                [[{"name": "make_image", "args": {"color": "red"}}]],
            ),
            tool_ids=[WS_TOOL],
        )
        records = gen(self.proxy.since(mark))
        called = any(n == "make_image" for n, _ in calls_of(records))
        shape = False
        for r in records[1:]:
            kinds = req(r).get("kinds") or []
            for i, kind in enumerate(kinds[:-1]):
                after = kinds[i + 1]
                if kind.startswith("user:fr") and after.startswith("user:"):
                    shape = shape or "inline" in after
        ev = [
            f"browser {self.m3}: called={called} fr_then_image_message={shape} "
            f"http={http(records)} {chat_line(c)}"
        ]
        if not called and all_ok(records):
            return Outcome(FAIL, ev, "Gemini did not call make_image")
        ok = called and shape and all_ok(records) and chat_ok(c)
        return Outcome(PASS if ok else FAIL, ev)

    async def r7(self) -> Outcome:
        c, records, called = await self.tool_turn(self.m3, stream=True)
        ev = [f"turn 1 (native): called={called} http={http(records)} {chat_line(c)}"]
        if not called:
            retry = (
                "Gemini did not call get_current_timestamp" if all_ok(records) else ""
            )
            return Outcome(FAIL, ev, retry)
        mark = self.proxy.mark()
        c2 = await self.bchat(
            self.model(self.m3),
            DONE_PROMPT,
            params={"function_calling": "legacy"},
            chat_id=c.chat_id,
            parent_id=c.message_id,
        )
        records2 = gen(self.proxy.since(mark))
        main = req(records2[-1] if records2 else None)
        kinds = main.get("kinds") or []
        history_fc = any(k.startswith("model:") and "fc" in k for k in kinds)
        declared = sum(req(r).get("declared_n") or 0 for r in records2)
        ev.append(
            f"turn 2 (legacy): declared_n={declared} history_fc={history_fc} "
            f"http={http(records2)} {chat_line(c2)}"
        )
        if not history_fc:
            ev.append("note: Open WebUI sent no function-call parts in this history")
        ok = declared == 0 and all_ok(records2) and chat_ok(c2) and chat_ok(c)
        return Outcome(PASS if ok else FAIL, ev)

    async def r8(self) -> Outcome:
        name = self.model(self.m25)
        c, records, called = await self.tool_turn(self.m25, stream=True)
        ev = [
            f"turn 1: called={called} thought_sigs="
            f"{sum(resp(r).get('thought_sigs') or 0 for r in records)} "
            f"fc_sigs={[s for _, s in calls_of(records)]} http={http(records)} "
            f"{chat_line(c)}"
        ]
        if not called:
            retry = (
                "Gemini did not call get_current_timestamp" if all_ok(records) else ""
            )
            return Outcome(FAIL, ev, retry)
        ok = all_ok(records) and chat_ok(c)
        parent = c
        for turn, text, rounds in ((2, CALC_PROMPT, [[CALC]]), (3, DONE_PROMPT, None)):
            mark = self.proxy.mark()
            c = await self.bchat(
                name,
                self.ask(text, rounds),
                chat_id=parent.chat_id,
                parent_id=parent.message_id,
            )
            recs = gen(self.proxy.since(mark))
            replay = [x["sig"] for x in req(recs[0] if recs else None).get("fc") or []]
            ev.append(
                f"turn {turn}: replayed_fc_sigs={replay} thought_sigs="
                f"{sum(resp(r).get('thought_sigs') or 0 for r in recs)} "
                f"calls={calls_of(recs)} http={http(recs)} {chat_line(c)}"
            )
            ok = ok and all_ok(recs) and chat_ok(c)
            parent = c
        return Outcome(PASS if ok else FAIL, ev)

    async def r3(self) -> Outcome:
        ev = []
        m3 = self.model(self.m3)
        await self.owui.upsert_model(m3, f"{self.m3} (smoke)", [SEARCH_FILTER])
        try:
            ok, server, retry = await self._r3_browser(m3, ev)
            if retry:
                return Outcome(FAIL, ev, retry)
            ok = await self._r3_api(m3, ev) and ok
        finally:
            await self.owui.delete_model(m3)
        if not ok:
            return Outcome(FAIL, ev)
        if not server:
            note = (
                "Gemini did not search in the tool round, so the stored-content "
                "replay (server-side parts sent back) was not exercised"
            )
            if not self.final_attempt:
                return Outcome(FAIL, ev, note)
            ev.append(f"note: {note}")
        return Outcome(PASS, ev)

    async def _r3_browser(self, m3: str, ev: list) -> tuple:
        """a) Search + call in one round, continued; b) next turn without web
        search. Returns (ok, server-side parts in the round, retry reason)."""
        mark = self.proxy.mark()
        c = await self.bchat(
            m3,
            self.ask(
                "In your first step do both at the same time: search the web with "
                "Google Search for today's top world news headline, and call the "
                "get_current_timestamp tool. Then answer in two short sentences: "
                "the headline and the timestamp.",
                [[TIMESTAMP]],
                server_side=True,
            ),
            features={"web_search": True},
        )
        records = gen(self.proxy.since(mark))
        q1 = req(records[0] if records else None)
        round_ = next((r for r in records if resp(r).get("fc")), None)
        server = resp(round_).get("server") or 0
        later = records[records.index(round_) + 1 :] if round_ else []
        replayed = req(later[0] if later else None).get("server_replayed") or 0
        combined = {"googleSearch", "functionDeclarations"} <= set(
            q1.get("tool_kinds") or []
        )
        ev.append(
            f"a) browser web_search + tools: combined={combined} include_flag="
            f"{q1.get('include_flag')} called={round_ is not None} "
            f"server_parts_in_round={server} server_parts_replayed={replayed} "
            f"http={http(records)} {chat_line(c)}"
        )
        if round_ is None:
            retry = (
                "Gemini did not call get_current_timestamp" if all_ok(records) else ""
            )
            return False, server, retry
        ok = combined and q1.get("include_flag") is True
        ok = ok and all_ok(records) and chat_ok(c) and replayed >= server
        mark = self.proxy.mark()
        c2 = await self.bchat(
            m3, DONE_PROMPT, chat_id=c.chat_id, parent_id=c.message_id
        )
        recs = gen(self.proxy.since(mark))
        q2 = req(recs[0] if recs else None)
        kinds = q2.get("kinds") or []
        srv = any("toolCall" in k or "toolResponse" in k for k in kinds)
        ev.append(
            f"b) next turn, web search off: include_flag={q2.get('include_flag')} "
            f"history_fc={len(q2.get('fc') or [])} server_parts_sent={srv} "
            f"http={http(recs)} {chat_line(c2)}"
        )
        return ok and all_ok(recs) and chat_ok(c2) and not srv, server, ""

    async def _r3_api(self, m3: str, ev: list) -> bool:
        """c) / d) tool_choice required / none together with web search."""
        ok = True
        for label, choice, want in (("c", "required", "ANY"), ("d", "none", "NONE")):
            if choice == "required":
                text = self.ask(
                    "Use the lookup_code tool with code 7, then explain it briefly.",
                    [[{"name": "lookup_code", "args": {"code": 7}}]],
                )
            else:
                text = "What is 2 + 2? Answer with the number only."
            mark = self.proxy.mark()
            r = await self.owui.chat(
                m3,
                text,
                stream=False,
                tools=[LOOKUP_CODE],
                tool_choice=choice,
                features={"web_search": True},
                max_tokens=self.args.max_tokens,
            )
            recs = gen(self.proxy.since(mark))
            q = req(recs[0] if recs else None)
            mode = str(q.get("fc_mode") or "").upper() or None
            calls = bool(r.tool_calls)
            ok = (
                ok
                and r.status == 200
                and all_ok(recs)
                and mode == want
                and q.get("include_flag") is True
                and calls == (want == "ANY")
            )
            ev.append(
                f"{label}) api tool_choice={choice} + web search: mode={mode} "
                f"include_flag={q.get('include_flag')} {api_line(r)}"
            )
        return ok

    async def r10(self) -> Outcome:
        m25 = self.model(self.m25)
        await self.owui.upsert_model(m25, f"{self.m25} (smoke)", [SEARCH_FILTER])
        try:
            mark = self.proxy.mark()
            c = await self.bchat(
                m25,
                "Search the web with Google Search for the current weather in Bern "
                "and answer in one short sentence.",
                features={"web_search": True},
            )
            records = gen(self.proxy.since(mark))
        finally:
            await self.owui.delete_model(m25)
        kinds = req(records[0] if records else None).get("tool_kinds") or []
        ok = (
            "googleSearch" in kinds
            and "functionDeclarations" not in kinds
            and all_ok(records)
            and chat_ok(c)
        )
        ev = [
            f"browser {self.m25}: tool_kinds={kinds} http={http(records)} "
            + chat_line(c)
        ]
        return Outcome(PASS if ok else FAIL, ev)

    async def r4(self) -> Outcome:
        ev, ok = [], True
        models = [self.m3, self.m25]
        if self.args.model_latest:
            if self.latest_listed:
                models.append(self.args.model_latest)
            else:
                ev.append(f"{self.args.model_latest}: not listed by the API, skipped")
        call = {
            "id": "smoke-r4-call",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Bern"}'},
        }
        messages = [
            {
                "role": "user",
                "content": "What is the weather in Bern? Use get_weather.",
            },
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {
                "role": "tool",
                "tool_call_id": "smoke-r4-call",
                "name": "get_weather",
                "content": '{"city": "Bern", "temp_c": 21.5, "sky": "sunny"}',
            },
        ]
        for name in models:
            mark = self.proxy.mark()
            r = await self.owui.chat(
                self.model(name),
                messages,
                stream=False,
                tools=[GET_WEATHER],
                max_tokens=self.args.max_tokens,
            )
            records = gen(self.proxy.since(mark))
            sigs = [
                x["sig"] for x in req(records[-1] if records else None).get("fc", [])
            ]
            good = r.status == 200 and all_ok(records) and bool(r.content.strip())
            good = good and not pipe_error(r.content)
            ok = ok and good
            ev.append(
                f"{name}: sent_signature={sigs} finish="
                f"{last_finish(records[-1]) if records else None} {api_line(r)}"
            )
        return Outcome(PASS if ok else FAIL, ev)

    async def r14(self) -> Outcome:
        previous = await self.owui.set_chat_config(ENABLE_TOOL_PERMISSIONS=True)
        resolved = []
        try:
            mark = self.proxy.mark()
            c = await self.bchat(
                self.model(self.m3),
                self.ask(PARALLEL_PROMPT, [[TIMESTAMP, CALC]]),
                params={"tool_approval_mode": "ask"},
                until=pending,
            )
            paused = pending(c.message) and not c.done
            for _ in range(4):
                ids = [i for _, s, i, _ in c.function_calls if s == "pending"]
                if not ids:
                    break
                status, _ = await self.owui.resolve_tool_call(
                    c.chat_id, c.message_id, ids[0], "approve"
                )
                resolved.append(status)

                def other(message: dict, done_id=ids[0]) -> bool:
                    return any(
                        i.get("type") == "function_call"
                        and i.get("status") == "pending"
                        and i.get("call_id") != done_id
                        for i in message.get("output") or []
                        if isinstance(i, dict)
                    )

                c = await self.b.wait(c, wait=self.args.wait, until=other)
            if c.task_ids and not c.done:
                await self.b.stop(c)
        finally:
            await self.owui.set_chat_config(**previous)
        records = gen(self.proxy.since(mark))
        produced = len(resp(records[0] if records else None).get("fc") or [])
        cont = records[1:]
        replayed = [x["sig"] for x in req(cont[0] if cont else None).get("fc") or []]
        ev = [
            f"browser {self.m3} ask mode: produced={produced} paused={paused} "
            f"approved={resolved} replayed_fc={replayed} continuation_http="
            f"{http(cont)} {chat_line(c)}"
        ]
        if produced < 2 and all_ok(records):
            return Outcome(FAIL, ev, "Gemini made fewer than 2 calls in one round")
        ok = paused and bool(cont) and all_ok(records) and c.done
        ok = ok and not pipe_error(final(c))
        return Outcome(PASS if ok else FAIL, ev)

    async def r13(self) -> Outcome:
        ev, ok = [], True
        for n in range(self.args.turns):
            stream = n % 2 == 1
            mark = self.proxy.mark()
            options = {"malformed": True} if n == 0 else {}
            r = await self.owui.chat(
                self.model(self.m3),
                self.ask(
                    f"Call the lookup_code tool with code {n + 1}. Do not answer "
                    "before calling it.",
                    [[{"name": "lookup_code", "args": {"code": n + 1}}]],
                    **options,
                ),
                stream=stream,
                tools=[LOOKUP_CODE],
                max_tokens=self.args.max_tokens,
            )
            records = gen(self.proxy.since(mark))
            finish = last_finish(records[-1]) if records else None
            handled = finish not in MALFORMED or finish in (r.content or "")
            ok = ok and r.status == 200 and handled
            ev.append(
                f"api turn {n + 1} stream={stream}: upstream_finish={finish} {api_line(r)}"
            )
        responses = [r for r in gen(self.proxy.records) if r.get("resp")]
        finishes = Counter(last_finish(r) or "none" for r in responses)
        names = [
            c.get("name") or "" for r in responses for c in resp(r).get("fc") or []
        ]
        prefixed = [n for n in names if n.startswith("default_api")]
        bad = sum(finishes.get(f, 0) for f in MALFORMED)
        ev.insert(
            0,
            f"whole run: responses={len(responses)} finish={dict(finishes)} "
            f"malformed_or_unexpected={bad}/{len(responses)} function_calls="
            f"{len(names)} default_api_prefixed={len(prefixed)}",
        )
        return Outcome(PASS if ok else FAIL, ev)

    async def r11(self) -> Outcome:
        version = str(self.meta.get("google_genai"))
        numbers = tuple(int(n) for n in re.findall(r"\d+", version)[:3])
        if numbers and numbers < (1, 68, 0):
            return Outcome(
                FAIL,
                [
                    f"google-genai {version} after saving google_gemini.py: the "
                    "requirement google-genai>=1.68.0 was not installed"
                ],
            )
        return Outcome(
            SKIP,
            [
                f"note only: Open WebUI {self.meta.get('owui_version')} "
                f"({os.environ.get('E2E_IMAGE', '?')}), google-genai "
                f"{self.meta.get('google_genai')} after saving google_gemini.py "
                "(requirements google-genai>=1.68.0). Open WebUI 0.9.0-0.11.3 "
                "bundle 1.66.0: run this test with --image v0.11.3-slim to see the "
                "upgrade on save and the in-process reload work (every other risk "
                "then runs on the upgraded SDK)."
            ],
        )

    # ----------------------------------------------------------------- Vertex
    async def r9(self) -> Outcome:
        project = self.args.vertex_project
        if self.args.vertex_adc and not project:
            return Outcome(
                SKIP,
                [
                    "Vertex credentials without a project: pass --vertex-project ID "
                    "(the file has no project_id)"
                ],
            )
        if not project:
            return Outcome(
                SKIP,
                [
                    "no Vertex AI configuration: pass --vertex-credentials FILE "
                    "(service account or ADC JSON) and --vertex-project ID (or "
                    "set GOOGLE_APPLICATION_CREDENTIALS / GOOGLE_CLOUD_PROJECT) "
                    "to run R9"
                ],
            )
        ev, ok = [], True
        mark_log = self.log.mark()
        if not await self.owui.function(VERTEX_FID):
            if not await self.t.install(
                VERTEX_FID, GEMINI_PATH, "Google Gemini (Vertex)", "vertex-load"
            ):
                return Outcome(FAIL, ["the Vertex copy of the pipe did not load"])
            await self.owui.update_valves(
                VERTEX_FID,
                USE_VERTEX_AI=True,
                VERTEX_PROJECT=project,
                VERTEX_LOCATION=self.args.vertex_location,
                VERTEX_AI_RAG_STORE="",
                INCLUDE_THOUGHTS=True,
                THINKING_LEVEL="low",
            )
            await self.ensure_models(VERTEX_FID)
        c, _, called = await self.tool_turn(self.m3, stream=True, fid=VERTEX_FID)
        ids = [i for _, _, i, _ in c.function_calls]
        ev.append(
            f"a) tool turn {self.m3}: called={called} call_ids={ids} {chat_line(c)}"
        )
        if not called and chat_ok(c):
            return Outcome(FAIL, ev, "Gemini did not call get_current_timestamp")
        ok = ok and called and chat_ok(c)
        long_name = "lookup_" + "x" * 57  # 64 characters, the Vertex limit
        r = await self.owui.chat(
            self.model(self.m3, VERTEX_FID),
            READY_PROMPT,
            stream=False,
            tools=[client_tool(long_name)],
            max_tokens=self.args.max_tokens,
        )
        ok = ok and r.status == 200 and bool(r.content) and not pipe_error(r.content)
        ev.append(f"b) api, a {len(long_name)}-character tool name: {api_line(r)}")
        vm3 = self.model(self.m3, VERTEX_FID)
        await self.owui.upsert_model(vm3, f"{self.m3} (Vertex smoke)", [SEARCH_FILTER])
        try:
            c = await self.bchat(
                vm3,
                "Search the web for today's top headline and answer in one sentence.",
                features={"web_search": True},
            )
        finally:
            await self.owui.delete_model(vm3)
        ok = ok and chat_ok(c)
        ev.append(f"c) web search + tools (grounding wins): {chat_line(c)}")
        if self.args.vertex_rag_store:
            await self.owui.update_valves(
                VERTEX_FID, VERTEX_AI_RAG_STORE=self.args.vertex_rag_store
            )
            try:
                c = await self.bchat(
                    vm3, "What do the documents say? Answer in one short sentence."
                )
            finally:
                await self.owui.update_valves(VERTEX_FID, VERTEX_AI_RAG_STORE="")
            ok = ok and chat_ok(c)
            ev.append(f"d) Vertex AI Search data store + tools: {chat_line(c)}")
        else:
            ev.append("d) no --vertex-rag-store: Vertex AI Search not tried")
        errors = self.log.errors(mark_log)
        ev.append(f"server log errors: {errors[:3]}")
        return Outcome(PASS if ok and not errors else FAIL, ev)

    # ------------------------------------------------------------------- run
    def estimate(self) -> tuple:
        planned = 2 + self.args.turns * self.selected("R13")
        planned += sum(n for r, n in ESTIMATE.items() if self.selected(r))
        return planned, planned * (1 + self.args.retries)

    def plan(self) -> None:
        planned, most = self.estimate()
        cap = "--max-requests" if self.args.max_requests else "2x the estimated maximum"
        self.say(
            f"estimate: about {planned} generate requests upstream, about {most} "
            f"if every scenario used its --retries {self.args.retries}; hard cap "
            f"{self.max_requests} ({cap}): beyond it the proxy answers HTTP 429 "
            f"without forwarding. max_tokens {self.args.max_tokens} per request, "
            f"THINKING_LEVEL=low (Gemini 3), THINKING_BUDGET={THINKING_BUDGET} "
            "(Gemini 2.5)"
        )

    async def run(self) -> None:
        steps = (
            ("R11", self.r11),
            ("R1", self.r1),
            ("R2", self.r2),
            ("R12", self.r12),
            ("R5", self.r5),
            ("R6", self.r6),
            ("R15", self.r15),
            ("R7", self.r7),
            ("R8", self.r8),
            ("R3", self.r3),
            ("R10", self.r10),
            ("R4", self.r4),
            ("R14", self.r14),
            ("R9", self.r9),
            ("R13", self.r13),  # last: counts the finish reasons of the whole run
        )
        for risk, fn in steps:
            await self.risk(risk, fn)

    async def execute(self) -> None:
        self.plan()
        await self.setup()
        await self.run()

    # ---------------------------------------------------------------- finish
    def scans(self) -> None:
        """Server log scan and secret scan (rows LOG and SECRETS)."""
        if self.log.available:
            ok = self.t.scan_log()
            detail = self.results.items[-1].detail if self.results.items else ""
            self.reports["LOG"] = Report(
                "LOG",
                "no unexpected ERROR / Traceback / tool-data WARNING in the server log",
                PASS if ok else FAIL,
                [detail],
                1,
                0,
                [],
            )
        # the key plus the secret values of the Vertex credentials file
        secrets = self.redact.secrets
        leaks = self.t.secret_leaks(*secrets)
        mocks = os.path.join(self.args.out, "mocks.txt")
        with contextlib.suppress(OSError), open(mocks, encoding="utf-8") as fh:
            text = fh.read()
            if self.key in text:
                leaks.append("mocks.txt contains the key")
            if any(s in text for s in secrets if s != self.key):
                leaks.append("mocks.txt contains a Vertex credential")
        if self.redact.count:
            leaks.append(
                f"a secret was redacted from {self.redact.count} output text(s)"
            )
        vertex = len(secrets) - (self.key in secrets)
        what = (
            f"the key and the {vertex} Vertex secret values appear"
            if vertex
            else "the key appears"
        )
        self.reports["SECRETS"] = Report(
            "SECRETS",
            f"{what} nowhere: server log, mocks output, runner output",
            FAIL if leaks else PASS,
            leaks or [f"no secret found ({len(secrets)} values checked)"],
            1,
            0,
            [],
        )

    def exit_code(self) -> int:
        if self.signal:
            return 128 + signal.Signals[self.signal].value
        if self.setup_error or self.driver_error:
            return 2
        return 1 if any(r.status == FAIL for r in self.reports.values()) else 0

    def budget_hit(self) -> str:
        """Why the run hit the request budget ("" when it did not)."""
        if not (self.proxy.blocked or self.budget_skipped):
            return ""
        skipped = ",".join(self.budget_skipped) or "none"
        return f"{self.budget_note()}; risks not run: {skipped}"

    def write(self) -> None:
        upstream = forwarded(self.proxy.records)
        generate = gen(self.proxy.records)
        usage = Counter()
        for r in upstream:
            for k, v in (resp(r).get("usage") or {}).items():
                usage[k] += v or 0
        counts = Counter(r.status for r in self.reports.values())
        meta = {
            **self.meta,
            "mode": "dry-run (e2e Gemini mock)" if self.dry else "real API",
            "image": os.environ.get("E2E_IMAGE", "?"),
            "models": {
                "gemini_3": self.m3,
                "gemini_2_5": self.m25,
                "latest": self.args.model_latest,
            },
            "retries": self.args.retries,
            "max_tokens": self.args.max_tokens,
            "only": sorted(self.only),
            "requests": {
                "generate": len(upstream),
                "blocked": self.proxy.blocked,
                "other": len(self.proxy.records) - len(generate),
                "estimate": self.estimate()[0],
                "max": self.max_requests,
            },
            "tokens": dict(usage),
            "duration_s": round(time.time() - self.started),
            "counts": dict(counts),
        }
        for name, value in (
            ("interrupted", self.signal),
            ("setup_error", self.setup_error),
            ("driver_error", self.driver_error[-2000:]),
            ("timed_out", self.timed_out),
            ("budget_hit", self.budget_hit()),
        ):
            if value:
                meta[name] = value
        keys = ORDER + ["TIMEOUT", "BUDGET", "LOG", "SECRETS"]
        rows = [self.reports[k] for k in keys if k in self.reports]
        os.makedirs(self.args.out, exist_ok=True)
        data = {"meta": meta, "results": [asdict(r) for r in rows]}
        self._dump("smoke.json", json.dumps(data, indent=1))
        self._dump(
            "upstream.jsonl", "".join(json.dumps(r) + "\n" for r in self.proxy.records)
        )
        self._dump("summary.md", self.markdown(meta, rows))
        self.say(
            f"\nSMOKE SUMMARY ({meta['mode']}): {counts.get(PASS, 0)} PASS, "
            f"{counts.get(FAIL, 0)} FAIL, {counts.get(SKIP, 0)} SKIP; "
            f"{len(upstream)} generate requests upstream (estimate about "
            f"{meta['requests']['estimate']}, cap {self.max_requests}"
            + (f", {self.proxy.blocked} blocked" if self.proxy.blocked else "")
            + f"), tokens {dict(usage)}, {meta['duration_s']}s"
        )
        if meta.get("budget_hit"):
            self.say(f"BUDGET HIT: {meta['budget_hit']}")

    def _dump(self, name: str, text: str) -> None:
        with open(os.path.join(self.args.out, name), "w", encoding="utf-8") as fh:
            fh.write(self.redact(text))

    def markdown(self, meta: dict, rows: list) -> str:
        lines = [
            "# Gemini native tool calling: real-API smoke test",
            "",
            f"- mode: {meta['mode']}; image {meta['image']}, Open WebUI "
            f"{meta.get('owui_version', '?')}, google_gemini.py "
            f"{meta.get('pipe_version', '?')}, google-genai "
            f"{meta.get('google_genai', '?')}",
            f"- models: {meta['models']}",
            f"- upstream requests: {meta['requests']} (retries {meta['retries']}, "
            f"max_tokens {meta['max_tokens']}); tokens {meta['tokens']}",
            f"- result: {meta['counts']} in {meta['duration_s']}s",
        ]
        names = (
            "interrupted",
            "setup_error",
            "driver_error",
            "timed_out",
            "budget_hit",
        )
        for name in names:
            if meta.get(name):
                lines.append(f"- **{name}**: {short(meta[name], 600)}")
        lines += [
            "",
            "| Risk | Status | What | Attempts | Requests |",
            "|---|---|---|---:|---:|",
        ]
        for r in rows:
            lines.append(
                f"| {r.risk} | {r.status} | {r.title} | {r.attempts} | {r.requests} |"
            )
        for r in rows:
            lines += ["", f"## {r.risk} {r.status}: {r.title}", "", "```text"]
            lines += [*r.notes, *r.evidence, "```"]
        return "\n".join(lines) + "\n"

    async def finish(self) -> None:
        if self.timed_out:
            self.reports["TIMEOUT"] = Report(
                "TIMEOUT", "the run finished in --timeout", FAIL, [], 1, 0, []
            )
        if self.budget_hit():
            self.reports["BUDGET"] = Report(
                "BUDGET",
                f"the run stayed within the request budget ({self.max_requests} "
                "generate requests, --max-requests)",
                FAIL,
                [self.budget_hit()],
                1,
                0,
                [],
            )
        if self.dry:
            mock = Mock("gemini")
            with contextlib.suppress(Exception):
                received = await mock.requests(lambda e: e.get("action") in GENERATE)
                self.meta["mock_generate_requests"] = len(received)
            await mock.close()
        with contextlib.suppress(Exception):
            self.scans()
        self.write()
        if self.b is not None:
            with contextlib.suppress(Exception):
                await self.b.close()
        with contextlib.suppress(Exception):
            await self.owui.close()
        with contextlib.suppress(Exception):
            await self.proxy.close()


# --------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default="/e2e/out")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--model-25", default="gemini-2.5-flash")
    parser.add_argument("--model-latest", default="gemini-flash-latest")
    parser.add_argument("--only", default="", help="comma separated risks (R2,R5)")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--turns", type=int, default=4, help="R13 tool turns")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="hard cap of generate requests forwarded upstream (default: 2x the "
        "estimated maximum)",
    )
    parser.add_argument("--wait", type=float, default=180, help="per browser turn")
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument("--api-version", default="")
    parser.add_argument("--dry-run-mock", action="store_true")
    parser.add_argument("--vertex-project", default="")
    parser.add_argument("--vertex-location", default="global")
    parser.add_argument("--vertex-rag-store", default="")
    parser.add_argument("--vertex-adc", default="", help="ADC file in the container")
    args = parser.parse_args()
    unknown = [
        r for r in args.only.upper().split(",") if r.strip() and r.strip() not in ORDER
    ]
    if unknown:
        parser.error(f"--only: unknown risks {unknown} (R1 ... R15)")
    if args.retries < 0 or args.turns < 0 or args.max_tokens < 1:
        parser.error("--retries / --turns must be >= 0, --max-tokens >= 1")
    if args.max_requests < 0:
        parser.error("--max-requests must be >= 1 (0 = the default)")
    return args


VERTEX_SECRET_KEYS = ("private_key", "private_key_id", "client_secret", "refresh_token")


def vertex_secrets(args: argparse.Namespace) -> list:
    """Secret values of the Vertex credentials file, scanned and redacted like
    the key; its project_id is the default project. One line each: the PEM
    private key gives its base64 lines (a log line holds it JSON-escaped, so the
    whole value would never match), the other values themselves; values under
    16 characters and the BEGIN / END lines are left out (gemini-tools.sh scans
    the same values on the host)."""
    if not args.vertex_adc:
        return []
    try:
        with open(args.vertex_adc, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SetupError(f"Vertex credentials unreadable: {type(exc).__name__}")
    if not args.vertex_project:
        args.vertex_project = str(
            data.get("project_id") or data.get("quota_project_id") or ""
        )
    lines = []
    for key in VERTEX_SECRET_KEYS:
        if isinstance(data.get(key), str):
            lines += data[key].splitlines()
    return [s for s in lines if len(s) >= 16 and not s.startswith("-----")]


async def main() -> int:
    args = parse_args()
    key = os.environ.get("GOOGLE_API_KEY", "")
    if len(key) < 6:
        print(
            "error: GOOGLE_API_KEY is not set in the runner's environment",
            file=sys.stderr,
        )
        return 2
    try:
        extra = vertex_secrets(args)
    except SetupError as exc:
        print(f"setup error: {exc}", file=sys.stderr)
        return 2
    redact = Redactor([key, *extra])
    sys.stdout = RedactingStream(sys.stdout, redact)
    sys.stderr = RedactingStream(sys.stderr, redact)
    smoke = Smoke(args, key, redact)
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()

    def on_signal(sig: signal.Signals) -> None:
        if not smoke.signal:
            smoke.signal = sig.name
            print(f"\n{sig.name}: stopping the smoke test", file=sys.stderr, flush=True)
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, on_signal, sig)
    try:
        async with asyncio.timeout(args.timeout):
            await smoke.execute()
    except TimeoutError:
        smoke.timed_out = True
        print(
            f"time budget of {args.timeout:.0f}s used up", file=sys.stderr, flush=True
        )
    except SetupError as exc:
        smoke.setup_error = str(exc)
        print(f"setup error: {exc}", file=sys.stderr, flush=True)
    except asyncio.CancelledError:
        if not smoke.signal:
            smoke.driver_error = traceback.format_exc()
    except Exception:  # noqa: BLE001 - runner bug: reported, exit code 2
        smoke.driver_error = traceback.format_exc()
        print(smoke.driver_error, file=sys.stderr, flush=True)
    finally:
        await smoke.finish()
    return smoke.exit_code()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
