"""
Filters suite: filters/{google_search_tool,vertex_ai_search_tool,time_token_tracker}.py
in front of the probe pipe (tests/e2e/probe/probe_pipe.py), which reports what
reached the pipe (messages, __metadata__ features/params, ...), so filter -> pipe
coupling is checked without a provider. The tracker's Azure Log Analytics send
goes to mocks/mock_la.py: this suite maps <workspace>.ods.opinsights.azure.com to
127.0.0.1 in /etc/hosts, installs a throw-away test CA into the container's
system store and starts the mock (HTTPS on 127.0.0.1:443, control on :9105).
Token counts are compared with tiktoken in the driver (same encodings, cached in
TIKTOKEN_CACHE_DIR before the server needs them).

Groups (``--only filters.<group>``)
  model        filters attached per model (meta.filterIds): feature -> metadata
               mapping (API and browser path), API request without ``features``,
               tracker outlet on the API path (exact counts in its log line),
               browser-path tracker status (format, values, done), background
               task (no event emitter)
  global       the same filters switched to global
  spec         SEND_TO_LOG_ANALYTICS env parsing, encrypted shared key, valve names
  la           Log Analytics records: signature, headers, payload and exact token
               counts on the API and browser path, special tokens, multi-turn
               averages, the "exactly two messages" rule, send switched off, HTTP
               errors, slow / hanging endpoint, estimate marker
  valves       compact status (CALCULATE_ALL_MESSAGES / SHOW_* off)
  correlation  inlet/outlet correlation when Open WebUI rewrites the last user
               message after the inlet, concurrent identical requests
  encoding     model-specific tiktoken encoding (gpt-4o -> o200k_base)
  offline      tiktoken download hangs: estimates, one load at a time, retry
               window, server not blocked. Needs a fresh container (no --reuse):
               tiktoken keeps loaded encodings per process
  multimodel   multi-model chat: google_search_tool and the features dict the
               models share
  search       google_search_tool pass-through (features {} / null / without
               web_search), other feature keys kept, no per-user permission check
               (documented limitation)
  vertex       per-request data store, store only with the feature, features null
"""

import asyncio
import base64
import datetime
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote

import httpx

from harness import Suite, short
from harness.config import VERTEX_RAG_STORE

GROUPS = (
    "model",
    "global",
    "spec",
    "la",
    "valves",
    "correlation",
    "encoding",
    "offline",
    "multimodel",
    "search",
    "vertex",
)
# Groups that need the Log Analytics mock (and a warm tracker).
LA_GROUPS = ("la", "valves", "correlation", "encoding", "offline", "multimodel")
TRACKER_GROUPS = ("model", "global", *LA_GROUPS)

PROBE_FID = "e2e_probe"
PROBE_MODEL = f"{PROBE_FID}.echo"
SEARCH = "google_search_tool"
VERTEX = "vertex_ai_search_tool"
TRACKER = "time_token_tracker"
FILTERS = {
    SEARCH: "filters/google_search_tool.py",
    VERTEX: "filters/vertex_ai_search_tool.py",
    TRACKER: "filters/time_token_tracker.py",
}
# Workspace models on top of the probe pipe (removed again at the end).
GPT4O_MODEL = "gpt-4o-e2e"  # tiktoken: o200k_base
DAVINCI_MODEL = "text-davinci-003"  # tiktoken: p50k_base (offline group only)
SEARCH_MODEL = "e2e-probe-search"  # google_search_tool only
VERTEX_MODEL = "e2e-probe-vertex"  # vertex_ai_search_tool only
MULTI_SEARCH_MODEL = "e2e-probe-multi-search"  # google_search_tool + tracker
MULTI_PLAIN_MODEL = "e2e-probe-multi-plain"  # tracker only
WORKSPACE_MODELS = (
    GPT4O_MODEL,
    DAVINCI_MODEL,
    SEARCH_MODEL,
    VERTEX_MODEL,
    MULTI_SEARCH_MODEL,
    MULTI_PLAIN_MODEL,
)

# time_token_tracker valves (public API: may grow, never shrink)
TRACKER_VALVES = (
    "priority",
    "CALCULATE_ALL_MESSAGES",
    "SHOW_AVERAGE_TOKENS",
    "SHOW_RESPONSE_TIME",
    "SHOW_TOKEN_COUNT",
    "SHOW_TOKENS_PER_SECOND",
    "SEND_TO_LOG_ANALYTICS",
    "LOG_ANALYTICS_WORKSPACE_ID",
    "LOG_ANALYTICS_SHARED_KEY",
    "LOG_ANALYTICS_LOG_TYPE",
)

# Log Analytics mock (mocks/mock_la.py)
WORKSPACE = "e2ews"
LA_HOST = f"{WORKSPACE}.ods.opinsights.azure.com"
LA_KEY = base64.b64encode(b"e2e-log-analytics-shared-key-0123456789").decode()
LOG_TYPE = "E2E_Metrics"
LA_CTL = "http://127.0.0.1:9105"
LA_DIR = "/tmp/e2e-la"
LA_OUT = "/e2e/out/mock_la.txt"
LA_SETUP = r"""
set -e
mkdir -p "$1" && cd "$1"
if [ ! -f srv.crt ]; then
  openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.crt -days 2 \
    -subj "/CN=E2E Log Analytics CA" 2>/dev/null
  openssl req -newkey rsa:2048 -nodes -keyout srv.key -out srv.csr -subj "/CN=$2" \
    2>/dev/null
  printf "subjectAltName=DNS:%s\n" "$2" > ext.cnf
  openssl x509 -req -in srv.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out srv.crt -days 2 -extfile ext.cnf 2>/dev/null
  cp ca.crt /usr/local/share/ca-certificates/e2e-la-ca.crt
  update-ca-certificates >/dev/null 2>&1
fi
"""
LA_HOSTS_LINE = f"127.0.0.1 {LA_HOST}"
RECORD_KEYS = {
    "timestamp",
    "chatId",
    "messageId",
    "model",
    "userId",
    "responseTime",
    "requestTokens",
    "responseTokens",
    "tokensPerSecond",
}
AVG_KEYS = {"avgRequestTokens", "avgResponseTokens"}
OPTIONAL_KEYS = {"tokensEstimated"}  # PR #184 (2.6.2), see la.estimate-marker
SLOW_SECONDS = 6
HANG_SECONDS = 20  # longer than the tracker's own 10 s timeout

# tiktoken's encoding host; the offline group points it at the mock's tarpit
TIKTOKEN_HOST = "openaipublic.blob.core.windows.net"
TARPIT_LINE = f"127.0.0.3 {TIKTOKEN_HOST}"
TARPIT_SECONDS = 8  # mocks/mock_la.py

STORE2 = (
    "projects/e2e/locations/global/collections/default_collection/dataStores/"
    "per-request"
)
SYSTEM_PROMPT = "You are the E2E probe; repeat what reached you."

TRACKER_WARNINGS = (
    ("time_token_tracker", "No inlet data found"),
    ("time_token_tracker", "Could not emit status event"),
)

_INLET = re.compile(
    r"Inlet complete: key=([0-9a-f]+), model=([^,]+), request_tokens=(\d+), "
    r"counted_messages=(\d+)(?:, tokens_estimated=(True|False))?"
)
_OUTLET = re.compile(
    r"Outlet complete: key=([0-9a-f]+), model=([^,]+), response_time=([\d.]+)s, "
    r"req_tokens=(\d+), resp_tokens=(\d+), tokens_per_sec=([\d.]+)"
    r"(?:, tokens_estimated=(True|False))?"
)
_STATUS_FULL = re.compile(
    r"^(\d+\.\d{2})s \| Req: (\d+) \(Ø (\d+\.\d{2})\) \| "
    r"Resp: (\d+) \(Ø (\d+\.\d{2})\) \| (\d+\.\d{2}) T/s$"
)
_STATUS_NO_TOKENS = re.compile(r"^\d+\.\d{2}s \| \d+\.\d{2} T/s$")
MISSING = "<missing>"


# --------------------------------------------------------------------- helpers
def probe_report(text) -> dict:
    """Decode the probe pipe's ``PROBE:{json}`` answer (also inside task JSON)."""
    if not isinstance(text, str):
        return {}
    if text.startswith("{"):
        try:
            text = json.loads(text).get("probe", "")
        except ValueError:
            return {}
    if text.startswith("PROBE:"):
        try:
            return json.loads(text[len("PROBE:") :])
        except ValueError:
            return {}
    return {}


_ENCODINGS: dict = {}


def tokens(text: str, name: str = "cl100k_base") -> int:
    """tiktoken count as the tracker computes it (special tokens as text).

    The first call per encoding downloads it into TIKTOKEN_CACHE_DIR (shared
    with the server), so the server's first load of it is a fast cache read.
    """
    if name not in _ENCODINGS:
        import tiktoken  # ships with the Open WebUI image

        _ENCODINGS[name] = tiktoken.get_encoding(name)
    return len(_ENCODINGS[name].encode(text or "", disallowed_special=()))


def tiktoken_cache_file(name: str) -> str:
    url = f"https://{TIKTOKEN_HOST}/encodings/{name}.tiktoken"
    return os.path.join(
        os.environ.get("TIKTOKEN_CACHE_DIR", ""), hashlib.sha1(url.encode()).hexdigest()
    )


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def expected(report: dict, answer: str, name: str = "cl100k_base") -> dict:
    """Record values the tracker should compute (CALCULATE_ALL_MESSAGES on) for
    the messages the probe received plus its answer."""
    messages = report.get("messages") or []
    request = [m for m in messages if m.get("role") in ("user", "system")]
    responses = [message_text(m) for m in messages if m.get("role") == "assistant"]
    responses.append(answer or "")
    return {
        "req": sum(tokens(message_text(m), name) for m in request),
        "resp": sum(tokens(text, name) for text in responses),
        "last": tokens(answer or "", name),
        "req_count": len(request),
        "resp_count": len(responses),
    }


def record_of(entry: Optional[dict]) -> dict:
    body = (entry or {}).get("body")
    if isinstance(body, list) and body and isinstance(body[0], dict):
        return body[0]
    return {}


def parse_timestamp(value) -> Optional[datetime.datetime]:
    try:
        stamp = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:  # naive utcnow() on main
        stamp = stamp.replace(tzinfo=datetime.timezone.utc)
    return stamp


def close(a, b, rel: float = 1e-6) -> bool:
    return (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and abs(a - b) <= rel * max(1.0, abs(b))
    )


def record_problems(entry: dict, exp: dict, model: str, user_id: str) -> list:
    """Differences between one recorded Log Analytics POST and ``exp``.

    ``exp``: req, resp, last, req_count, resp_count; optional ``tps`` (default
    last / responseTime), ``avg`` (False: no average fields), ``chat_id`` (None:
    any UUID), ``message_id`` (compared when given).
    """
    problems = []
    headers = {k.lower(): v for k, v in (entry.get("headers") or {}).items()}
    if entry.get("host") != LA_HOST:
        problems.append(f"host={entry.get('host')}")
    if entry.get("path") != "/api/logs" or entry.get("query") != {
        "api-version": "2016-04-01"
    }:
        problems.append(f"path={entry.get('path')} query={entry.get('query')}")
    if not entry.get("auth_ok"):
        problems.append("SharedKey signature invalid")
    if headers.get("content-type") != "application/json":
        problems.append(f"content-type={headers.get('content-type')}")
    if headers.get("log-type") != LOG_TYPE:
        problems.append(f"log-type={headers.get('log-type')}")
    if headers.get("time-generated-field") != "timestamp":
        problems.append(f"time-generated-field={headers.get('time-generated-field')}")
    try:
        date = parsedate_to_datetime(headers.get("x-ms-date", ""))
        skew = abs(
            (datetime.datetime.now(datetime.timezone.utc) - date).total_seconds()
        )
        if skew > 300 or not headers.get("x-ms-date", "").endswith(" GMT"):
            problems.append(f"x-ms-date={headers.get('x-ms-date')}")
    except (TypeError, ValueError):
        problems.append(f"x-ms-date={headers.get('x-ms-date')}")
    body = entry.get("body")
    if not isinstance(body, list) or len(body) != 1 or not isinstance(body[0], dict):
        return problems + [f"body={short(body, 200)}"]
    rec = body[0]
    want_keys = RECORD_KEYS | (AVG_KEYS if exp.get("avg", True) else set())
    missing, extra = want_keys - set(rec), set(rec) - want_keys - OPTIONAL_KEYS
    if missing or extra:
        problems.append(f"keys missing={sorted(missing)} extra={sorted(extra)}")
    stamp = parse_timestamp(rec.get("timestamp"))
    now = datetime.datetime.now(datetime.timezone.utc)
    if stamp is None or abs((now - stamp).total_seconds()) > 300:
        problems.append(f"timestamp={rec.get('timestamp')}")
    if rec.get("model") != model:
        problems.append(f"model={rec.get('model')}")
    if rec.get("userId") != user_id:
        problems.append(f"userId={rec.get('userId')}")
    if exp.get("chat_id") is None:
        try:
            uuid.UUID(str(rec.get("chatId")))
        except ValueError:
            problems.append(f"chatId not a UUID: {rec.get('chatId')!r}")
    elif rec.get("chatId") != exp["chat_id"]:
        problems.append(f"chatId={rec.get('chatId')} want {exp['chat_id']}")
    if not rec.get("messageId"):
        problems.append("messageId empty")
    elif exp.get("message_id") and rec.get("messageId") != exp["message_id"]:
        problems.append(f"messageId={rec.get('messageId')} want {exp['message_id']}")
    rt = rec.get("responseTime")
    if not isinstance(rt, (int, float)) or not 0 < rt < 60:
        problems.append(f"responseTime={rt}")
    if rec.get("requestTokens") != exp["req"]:
        problems.append(f"requestTokens={rec.get('requestTokens')} want {exp['req']}")
    if rec.get("responseTokens") != exp["resp"]:
        problems.append(
            f"responseTokens={rec.get('responseTokens')} want {exp['resp']}"
        )
    want_tps = exp.get("tps")
    if want_tps is None:
        want_tps = exp["last"] / rt if isinstance(rt, (int, float)) and rt > 0 else 0
    if not close(rec.get("tokensPerSecond"), want_tps):
        problems.append(f"tokensPerSecond={rec.get('tokensPerSecond')} want {want_tps}")
    if exp.get("avg", True):
        want_req = exp["req"] / exp["req_count"] if exp["req_count"] else 0
        want_resp = exp["resp"] / exp["resp_count"] if exp["resp_count"] else 0
        if not close(rec.get("avgRequestTokens"), want_req, 1e-9):
            problems.append(
                f"avgRequestTokens={rec.get('avgRequestTokens')} want {want_req}"
            )
        if not close(rec.get("avgResponseTokens"), want_resp, 1e-9):
            problems.append(
                f"avgResponseTokens={rec.get('avgResponseTokens')} want {want_resp}"
            )
    return problems


def status_text(rec: dict) -> str:
    """Browser status the tracker shows for a record (default valves)."""
    return (
        f"{rec['responseTime']:.2f}s | Req: {rec['requestTokens']} "
        f"(Ø {rec['avgRequestTokens']:.2f}) | Resp: {rec['responseTokens']} "
        f"(Ø {rec['avgResponseTokens']:.2f}) | {rec['tokensPerSecond']:.2f} T/s"
    )


def inlet_lines(t: Suite, mark: int) -> list:
    out = []
    for line in t.log.lines(mark, "Inlet complete:"):
        m = _INLET.search(line)
        if m:
            out.append(
                {
                    "model": m[2],
                    "req": int(m[3]),
                    "counted": int(m[4]),
                    "estimated": m[5],
                }
            )
    return out


def outlet_lines(t: Suite, mark: int) -> list:
    out = []
    for line in t.log.lines(mark, "Outlet complete:"):
        m = _OUTLET.search(line)
        if m:
            out.append(
                {
                    "model": m[2],
                    "response_time": float(m[3]),
                    "req": int(m[4]),
                    "resp": int(m[5]),
                    "estimated": m[7],
                }
            )
    return out


def outlet_errors(t: Suite, mark: int) -> list:
    """time_token_tracker outlet failures logged since ``mark``."""
    return [e for e in t.log.errors(mark) if "outlet filter time_token_tracker" in e]


def send_timeout(t: Suite, mark: int, timeout: float = 8.0) -> float:
    """How long to wait for a Log Analytics record: not at all when the
    outlet already failed (it never got to the send)."""
    return 0.5 if outlet_errors(t, mark) else timeout


async def wait_log(t: Suite, mark: int, needle: str, timeout: float) -> list:
    """Lines containing ``needle`` logged since ``mark`` (polls up to ``timeout``)."""
    deadline = time.time() + timeout
    while True:
        lines = t.log.lines(mark, needle)
        if lines or time.time() >= deadline:
            return lines
        await asyncio.sleep(0.5)


def edit_hosts(line: str, add: bool) -> None:
    """Add / remove a line of /etc/hosts in place (a bind mount: no rename)."""
    with open("/etc/hosts", "r+", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip() != line]
        if add:
            lines.append(line)
        fh.seek(0)
        fh.truncate()
        fh.write("\n".join(lines) + "\n")


async def upsert_derived_model(
    t: Suite, model_id: str, name: str, filter_ids: list
) -> int:
    """Workspace model on top of the probe pipe with its own filters."""
    form = {
        "id": model_id,
        "base_model_id": PROBE_MODEL,
        "name": name,
        "params": {},
        "meta": {"description": "e2e", "filterIds": filter_ids, "capabilities": {}},
    }
    status, existing = await t.owui.api(
        "GET", f"/api/v1/models/model?id={quote(model_id)}"
    )
    if status == 200 and isinstance(existing, dict) and existing.get("id"):
        status, _ = await t.owui.api("POST", "/api/v1/models/model/update", form)
    else:
        status, _ = await t.owui.api("POST", "/api/v1/models/create", form)
    await t.owui.models(refresh=True)
    return status


class LogAnalytics:
    """Control client of mocks/mock_la.py."""

    def __init__(self):
        self.http = httpx.AsyncClient(base_url=LA_CTL, timeout=30)

    async def reset(self) -> None:
        await self.http.post("/__reset")

    async def mode(self, status: int = 200, delay: float = 0.0) -> None:
        await self.http.post("/__mode", json={"status": status, "delay": delay})

    async def records(self) -> list:
        return (await self.http.get("/__requests")).json()

    async def tarpit(self) -> list:
        return (await self.http.get("/__tarpit")).json()

    async def wait(self, n: int = 1, timeout: float = 8.0) -> list:
        """Recorded POSTs once there are ``n`` (or after ``timeout`` seconds)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(await self.records()) >= n:
                await asyncio.sleep(0.3)  # one more would be a duplicate
                break
            await asyncio.sleep(0.25)
        return await self.records()

    async def close(self) -> None:
        await self.http.aclose()


def kill_stale_mock() -> None:
    """mock_la.py left over by an earlier --reuse run."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                if b"mock_la.py" in fh.read():
                    os.kill(int(pid), 9)
        except (OSError, ValueError):
            continue


async def start_la_mock() -> tuple:
    """Test CA + /etc/hosts entry + mock_la.py; returns (process, client)."""
    subprocess.run(["bash", "-c", LA_SETUP, "la-setup", LA_DIR, LA_HOST], check=True)
    edit_hosts(LA_HOSTS_LINE, add=True)
    kill_stale_mock()
    await asyncio.sleep(0.5)
    out = open(LA_OUT, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [
            "python3",
            "-u",
            os.path.join(os.path.dirname(__file__), "..", "mocks", "mock_la.py"),
            WORKSPACE,
            LA_KEY,
            f"{LA_DIR}/srv.crt",
            f"{LA_DIR}/srv.key",
        ],
        stdout=out,
        stderr=subprocess.STDOUT,
    )
    out.close()
    la = LogAnalytics()
    for _ in range(60):
        await asyncio.sleep(0.5)
        try:
            await la.reset()
            return proc, la
        except httpx.HTTPError:
            if proc.poll() is not None:
                break
    await la.close()
    proc.kill()
    raise RuntimeError(f"mocks/mock_la.py did not start (see {LA_OUT})")


# ------------------------------------------------------------------------ run
async def run(t: Suite) -> None:
    if not await t.install(PROBE_FID, "probe", "E2E Probe", "load.probe"):
        return
    for fid, path in FILTERS.items():
        if not await t.install(fid, path, fid, f"load.{fid}"):
            return
        info = await t.owui.function(fid) or {}
        t.check(
            f"load.{fid}.type",
            f"{fid} is registered as a filter",
            info.get("type") == "filter",
            f"type={info.get('type')}",
        )
    try:
        await t.owui.upsert_model(PROBE_MODEL, "E2E Probe", list(FILTERS))
        await t.owui.replace_valves(TRACKER, {})  # defaults (--reuse)
        if any(t.selected(g) for g in TRACKER_GROUPS):
            await warm_up(t)
        if t.selected("model"):
            await per_model(t)
        if t.selected("global"):
            await global_mode(t)
        if t.selected("spec"):
            await spec(t)
        if any(t.selected(g) for g in LA_GROUPS):
            await la_groups(t)
        if t.selected("search"):
            await search(t)
        if t.selected("vertex"):
            await vertex(t)
    finally:
        edit_hosts(TARPIT_LINE, add=False)
        for fid in FILTERS:
            await t.owui.set_global(fid, False)
        for model_id in WORKSPACE_MODELS:
            await t.owui.delete_model(model_id)
        await t.owui.upsert_model(PROBE_MODEL, "E2E Probe", [])
        await t.owui.replace_valves(TRACKER, {})
    if t.selected("correlation"):
        warnings = t.log.warnings(t.log_start, *TRACKER_WARNINGS)
        t.check(
            "server-log.warnings",
            "no 'No inlet data found' / 'Could not emit status event' warning of "
            "time_token_tracker in the server log",
            not warnings,
            f"{len(warnings)} warnings: " + " || ".join(warnings[:3]),
        )
    t.assert_no_secrets(LA_KEY)
    t.scan_log()


async def warm_up(t: Suite) -> None:
    """Load cl100k_base in the driver (fills TIKTOKEN_CACHE_DIR) and repeat a
    request until the tracker counts it exactly: the server's first load may
    take longer than the tracker's 5 s first-request wait."""
    text = "warm up the token counter please"
    want = tokens(text)
    seen = []
    for _ in range(20):
        mark = t.mark()
        await t.owui.chat(PROBE_MODEL, text, features={"web_search": False})
        await t.log.settle(0.5)
        inlet = inlet_lines(t, mark)
        seen.append(inlet[0]["req"] if inlet else None)
        if seen[-1] == want and inlet[0]["estimated"] != "True":
            break
        await asyncio.sleep(2)
    print(f"          warm-up: want={want} seen={seen}", flush=True)


# ------------------------------------------------------------- model / global
async def feature_mapping(t: Suite, tag: str) -> None:
    """features.* in the request -> what the pipe sees in __metadata__."""
    r = await t.owui.chat(PROBE_MODEL, "probe", features={"web_search": True})
    rep = probe_report(r.content)
    features = rep.get("metadata_features") or {}
    t.check(
        f"{tag}.web-search-on",
        "features.web_search=true -> __metadata__.features.google_search_tool",
        r.status == 200
        and features.get("google_search_tool") is True
        and "web_search" not in features,
        f"HTTP {r.status} metadata_features={features}",
    )
    r = await t.owui.chat(PROBE_MODEL, "probe", features={"web_search": False})
    rep = probe_report(r.content)
    t.check(
        f"{tag}.web-search-off",
        "features.web_search=false -> no google_search_tool flag",
        r.status == 200
        and bool(rep)
        and not (rep.get("metadata_features") or {}).get("google_search_tool"),
        f"HTTP {r.status} report={short(rep, 300)}",
    )
    r = await t.owui.chat(
        PROBE_MODEL, "probe", features={"web_search": False, "vertex_ai_search": True}
    )
    rep = probe_report(r.content)
    features = rep.get("metadata_features") or {}
    store = (rep.get("metadata_params") or {}).get("vertex_rag_store")
    t.check(
        f"{tag}.vertex",
        "features.vertex_ai_search -> metadata flag + VERTEX_AI_RAG_STORE env in params",
        r.status == 200
        and features.get("vertex_ai_search") is True
        and store == (VERTEX_RAG_STORE or None),
        f"HTTP {r.status} features={features} vertex_rag_store={store!r}",
    )


async def per_model(t: Suite) -> None:
    filter_ids = list(FILTERS)
    status = await t.owui.upsert_model(PROBE_MODEL, "E2E Probe", filter_ids)
    seen = await t.owui.model_filter_ids(PROBE_MODEL)
    t.check(
        "model.attach",
        "filters attached per model (meta.filterIds, visible after models refresh)",
        status == 200 and seen == filter_ids,
        f"HTTP {status} filterIds={seen}",
    )
    await feature_mapping(t, "model")

    r = await t.owui.chat(PROBE_MODEL, "probe without features")
    await t.log.settle()
    t.check(
        "model.no-features",
        "API request without a 'features' key passes the filters",
        r.status == 200 and bool(probe_report(r.content)),
        r.brief(),
    )

    for stream in (False, True):
        await tracker_api(t, "model", stream)

    async with t.browser() as b:
        c = await b.chat(PROBE_MODEL, "tracker in the browser", stream=True)
        w = await b.chat(PROBE_MODEL, "probe", features={"web_search": True})
    rep = probe_report(c.content)
    t.check(
        "model.browser",
        "browser path: pipe has an event emitter and chat/message/session ids",
        c.done
        and rep.get("has_event_emitter") is True
        and all((rep.get("metadata_ids") or {}).values()),
        f"report={short(rep, 300)}",
    )
    tracker_browser(t, c, rep)
    browser_web_search(t, "model", w)

    status, answer, raw = await t.owui.title_task(
        PROBE_MODEL, [{"role": "user", "content": "Hi"}]
    )
    rep = probe_report(answer)
    t.check(
        "model.task",
        "background title task: __task__ set, no __event_emitter__",
        status == 200
        and rep.get("task") == "title_generation"
        and rep.get("has_event_emitter") is False,
        f"HTTP {status} report={short(rep, 300)} raw={short(raw, 200)}",
    )


def tracker_browser(t: Suite, c, rep: dict) -> None:
    """The tracker's status on the browser path: one status, done, exact."""
    statuses = [s for s in c.status_history if "Req:" in (s.get("description") or "")]
    exp = expected(rep, c.content)
    problems = []
    if len(statuses) != 1:
        problems.append(f"{len(statuses)} tracker statuses")
    else:
        status = statuses[0]
        m = _STATUS_FULL.match(status.get("description") or "")
        if status.get("done") is not True:
            problems.append(f"done={status.get('done')!r}")
        if not m:
            problems.append("format")
        else:
            rt, req, avg_req, resp, avg_resp, tps = m.groups()
            if int(req) != exp["req"] or avg_req != f"{exp['req']:.2f}":
                problems.append(f"Req {req} (Ø {avg_req}) want {exp['req']}")
            if int(resp) != exp["resp"] or avg_resp != f"{exp['resp']:.2f}":
                problems.append(f"Resp {resp} (Ø {avg_resp}) want {exp['resp']}")
            # the shown time is rounded to 0.01 s, T/s used the exact time
            rt = float(rt)
            low = exp["last"] / (rt + 0.005) - 0.01
            high = exp["last"] / max(rt - 0.005, 1e-6) + 0.01
            if not (rt > 0 and low <= float(tps) <= high):
                problems.append(f"{tps} T/s not {exp['last']} tokens / {rt}s")
    t.check(
        "model.tracker-browser",
        "browser path: one time_token_tracker status, done, exact counts "
        "'<t>s | Req: n (Ø a) | Resp: m (Ø b) | x T/s'",
        c.done and bool(rep) and not problems,
        f"problems={problems} status={statuses} want req={exp['req']} "
        f"resp={exp['resp']}",
    )


def browser_web_search(t: Suite, tag: str, w) -> None:
    """features.web_search in a UI chat -> google_search_tool flag at the pipe."""
    features = probe_report(w.content).get("metadata_features") or {}
    t.check(
        f"{tag}.browser-web-search",
        "browser path: features.web_search=true -> the pipe's "
        "__metadata__.features.google_search_tool",
        w.done and features.get("google_search_tool") is True,
        f"done={w.done} metadata_features={features} error={w.error}",
    )


async def tracker_api(t: Suite, tag: str, stream: bool) -> None:
    """time_token_tracker outlet on the API path (no chat, no event emitter):
    no error and the exact counts in its 'Outlet complete' log line."""
    mark = t.mark()
    r = await t.owui.chat(
        PROBE_MODEL, "tracker", stream=stream, features={"web_search": False}
    )
    await t.log.settle(1.5)
    errors = t.log.errors(mark)
    rep = probe_report(r.content)
    exp = expected(rep, r.content)
    lines = outlet_lines(t, mark)
    exact = (
        len(lines) == 1
        and lines[0]["model"] == PROBE_MODEL
        and lines[0]["req"] == exp["req"]
        and lines[0]["resp"] == exp["resp"]
    )
    t.check(
        f"{tag}.tracker-api.{'stream' if stream else 'nonstream'}",
        f"time_token_tracker outlet on the API path (stream={stream}) runs "
        "without errors and logs exact token counts",
        r.status == 200 and bool(rep) and exact and not errors,
        f"{r.brief()} outlet={lines} want req={exp['req']} resp={exp['resp']} "
        f"log_errors={errors[:2]}",
    )


async def global_mode(t: Suite) -> None:
    await t.owui.upsert_model(PROBE_MODEL, "E2E Probe", [])
    try:
        flags = {fid: await t.owui.set_global(fid, True) for fid in FILTERS}
        await t.owui.models(refresh=True)
        t.check(
            "global.switch",
            "filters switched to global (no per-model filterIds)",
            all(flags.values()),
            f"is_global={flags}",
        )
        await feature_mapping(t, "global")
        await tracker_api(t, "global", stream=False)
        async with t.browser() as b:
            w = await b.chat(PROBE_MODEL, "probe", features={"web_search": True})
        browser_web_search(t, "global", w)
    finally:
        for fid in FILTERS:
            await t.owui.set_global(fid, False)
        await t.owui.upsert_model(PROBE_MODEL, "E2E Probe", list(FILTERS))


# ------------------------------------------------------------------------ spec
async def spec(t: Suite) -> None:
    props = (await t.owui.valves_spec(TRACKER)).get("properties") or {}
    default = (props.get("SEND_TO_LOG_ANALYTICS") or {}).get("default")
    env = os.environ.get("SEND_TO_LOG_ANALYTICS")
    t.check(
        "spec.send-env-false",
        "SEND_TO_LOG_ANALYTICS=false in the environment -> valve default False",
        env == "false" and default is False,
        f"default={default!r} env={env!r}",
    )
    await t.owui.update_valves(TRACKER, LOG_ANALYTICS_SHARED_KEY=LA_KEY)
    stored = (await t.owui.get_valves(TRACKER)).get("LOG_ANALYTICS_SHARED_KEY")
    t.check(
        "spec.key-encrypted",
        "LOG_ANALYTICS_SHARED_KEY is stored encrypted",
        isinstance(stored, str)
        and stored.startswith("encrypted:")
        and LA_KEY not in stored,
        f"stored={short(stored, 40)}",
    )
    await t.owui.replace_valves(TRACKER, {})
    names = await t.valve_names(TRACKER)
    t.check(
        "spec.valve-names",
        "time_token_tracker keeps every valve name (public API)",
        set(TRACKER_VALVES) <= set(names),
        f"missing={sorted(set(TRACKER_VALVES) - set(names))} valves={names}",
    )


# ---------------------------------------------------------------- Log Analytics
async def la_groups(t: Suite) -> None:
    proc, la = await start_la_mock()
    try:
        await t.owui.update_valves(
            TRACKER,
            SEND_TO_LOG_ANALYTICS=True,
            LOG_ANALYTICS_WORKSPACE_ID=WORKSPACE,
            LOG_ANALYTICS_SHARED_KEY=LA_KEY,
            LOG_ANALYTICS_LOG_TYPE=LOG_TYPE,
        )
        if t.selected("la"):
            await la_group(t, la)
        if t.selected("valves"):
            await valves_group(t, la)
        if t.selected("correlation"):
            await correlation(t, la)
        if t.selected("encoding"):
            await encoding(t, la)
        if t.selected("offline"):
            await offline(t, la)
        if t.selected("multimodel"):
            await multimodel(t, la)
    finally:
        await la.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        await t.owui.replace_valves(TRACKER, {})


async def la_api(
    t: Suite,
    la: LogAnalytics,
    sid: str,
    title: str,
    text: str,
    stream: bool = False,
    history: tuple = (),
):
    """One API request -> exactly one exact, signed Log Analytics record."""
    await la.reset()
    mark = t.mark()
    messages = [*history, {"role": "user", "content": text}]
    r = await t.owui.chat(
        PROBE_MODEL, messages, stream=stream, features={"web_search": False}
    )
    rep = probe_report(r.content)
    await t.log.settle(0.3)
    recs = await la.wait(1, timeout=send_timeout(t, mark))
    exp = expected(rep, r.content)
    problems = (
        record_problems(recs[0], exp, PROBE_MODEL, t.owui.user.get("id"))
        if len(recs) == 1
        else [f"posts={len(recs)}"]
    )
    t.check(
        sid,
        title,
        r.status == 200 and bool(rep) and not problems,
        f"HTTP {r.status} problems={problems} exp={exp} "
        f"outlet_errors={outlet_errors(t, mark)[:1]} "
        f"rec={short(record_of(recs[0]) if recs else None, 300)}",
    )


async def la_group(t: Suite, la: LogAnalytics) -> None:
    await la_api(
        t,
        la,
        "la.api.nonstream",
        "API non-stream: one signed Log Analytics record with exact token counts",
        "Count my tokens please, exactly.",
    )
    await la_api(
        t,
        la,
        "la.api.stream",
        "API stream: one signed Log Analytics record with exact token counts",
        "Streaming token count check.",
        stream=True,
    )

    # '<|endoftext|>' is counted as text
    text = "say <|endoftext|> now"
    await la.reset()
    r = await t.owui.chat(PROBE_MODEL, text, features={"web_search": False})
    recs = await la.wait(1, timeout=8 if r.status == 200 else 0.5)
    rec = record_of(recs[0]) if recs else {}
    t.check(
        "la.special-token",
        "API: '<|endoftext|>' in a message is counted as text, no error",
        r.status == 200 and len(recs) == 1 and rec.get("requestTokens") == tokens(text),
        f"{r.brief()} posts={len(recs)} requestTokens={rec.get('requestTokens')} "
        f"want {tokens(text)}",
    )

    await la_browser(t, la)
    await la_multiturn(t, la)
    await la_two_user_turns(t)
    await la_off(t, la)
    await la_http_error(t, la)
    await la_slow(t, la)


async def la_browser(t: Suite, la: LogAnalytics) -> None:
    """Browser path: ids of the saved chat, status = record, estimate flag."""
    await la.reset()
    mark = t.mark()
    async with t.browser() as b:
        c = await b.chat(PROBE_MODEL, "Browser token check", stream=True)
    recs = await la.wait(1)
    rep = probe_report(c.content)
    exp = {**expected(rep, c.content), "chat_id": c.chat_id, "message_id": c.message_id}
    problems = (
        record_problems(recs[0], exp, PROBE_MODEL, t.owui.user.get("id"))
        if len(recs) == 1
        else [f"posts={len(recs)}"]
    )
    rec = record_of(recs[0]) if recs else {}
    try:
        want = status_text(rec)
    except (KeyError, TypeError, ValueError):
        want = None
    tracker = [s for s in c.status_history if "Req:" in (s.get("description") or "")]
    t.check(
        "la.browser",
        "browser: record has the chat id and the assistant message id, the status "
        "text shows the record's values, done",
        c.done
        and not problems
        and len(tracker) == 1
        and tracker[0].get("description") == want
        and tracker[0].get("done") is True,
        f"problems={problems} status={tracker} want={want!r}",
    )

    lines = outlet_lines(t, mark)
    flag = rec.get("tokensEstimated", MISSING) if rec else MISSING
    logged = [line["estimated"] for line in lines]
    shown = [s.get("description") for s in tracker]
    if not rec:
        state = "no record"
    elif flag == MISSING and not any(logged):
        state = "tokensEstimated flag missing"
    else:
        state = f"tokensEstimated={flag!r} log tokens_estimated={logged}"
    t.check(
        "la.estimate-marker",
        "exact counts are marked as such: record tokensEstimated=false, "
        "'tokens_estimated=False' in the outlet log line, no '~' in the status",
        flag is False
        and logged == ["False"]
        and len(shown) == 1
        and "~" not in (shown[0] or "~"),
        f"{state} status={shown} keys={sorted(rec)}",
    )


async def la_multiturn(t: Suite, la: LogAnalytics) -> None:
    """Second turn of a saved chat with a system prompt: all user/system and
    assistant messages counted, averages, T/s of the last answer."""
    system = {"role": "system", "content": SYSTEM_PROMPT}
    first = "First question, a bit longer than usual."
    second = "Second question here?"
    async with t.browser() as b:
        c1 = await b.chat(PROBE_MODEL, first, history=[system])
        await la.wait(1)
        await la.reset()
        c2 = await b.chat(
            PROBE_MODEL,
            second,
            history=[
                system,
                {"role": "user", "content": first},
                {"role": "assistant", "content": c1.content},
            ],
            chat_id=c1.chat_id,
            parent_id=c1.message_id,
        )
    recs = await la.wait(1)
    rep = probe_report(c2.content)
    roles = [m.get("role") for m in rep.get("messages") or []]
    # The inlet counts the request's messages (system prompt included); the
    # outlet averages over the saved chat messages, which have no system
    # message: 2 user and 2 assistant messages.
    exp = {
        "req": tokens(SYSTEM_PROMPT) + tokens(first) + tokens(second),
        "resp": tokens(c1.content) + tokens(c2.content),
        "last": tokens(c2.content),
        "req_count": 2,
        "resp_count": 2,
        "chat_id": c1.chat_id,
    }
    problems = (
        record_problems(recs[0], exp, PROBE_MODEL, t.owui.user.get("id"))
        if len(recs) == 1
        else [f"posts={len(recs)}"]
    )
    t.check(
        "la.multiturn",
        "multi-turn chat with a system prompt: requestTokens = system + both user "
        "messages, responseTokens = both answers, averages over the saved user / "
        "assistant messages, T/s of the last answer",
        c1.done
        and c2.done
        and roles == ["system", "user", "assistant", "user"]
        and not problems,
        f"roles={roles} problems={problems} exp={exp} "
        f"rec={short(record_of(recs[0]) if recs else None, 300)}",
    )


async def la_two_user_turns(t: Suite) -> None:
    """CALCULATE_ALL_MESSAGES=false: exactly two user/system messages are
    summed, otherwise only the last one counts (documents the current rule)."""
    u1, u2, u3 = "First user turn here.", "Second user turn.", "And a third turn!"
    two = [
        {"role": "user", "content": u1},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": u2},
    ]
    three = [
        *two,
        {"role": "assistant", "content": "Second answer."},
        {"role": "user", "content": u3},
    ]
    await t.owui.update_valves(TRACKER, CALCULATE_ALL_MESSAGES=False)
    seen = {}
    try:
        for name, messages in (("two", two), ("three", three)):
            mark = t.mark()
            r = await t.owui.chat(PROBE_MODEL, messages, features={"web_search": False})
            await t.log.settle(0.5)
            lines = inlet_lines(t, mark)
            seen[name] = (r.status, [(x["req"], x["counted"]) for x in lines])
    finally:
        await t.owui.update_valves(TRACKER, CALCULATE_ALL_MESSAGES=True)
    want = {
        "two": (200, [(tokens(u1) + tokens(u2), 2)]),
        "three": (200, [(tokens(u3), 1)]),
    }
    t.check(
        "la.two-user-turns",
        "CALCULATE_ALL_MESSAGES=false: two user messages are summed, of three only "
        "the last one counts (inlet log line)",
        seen == want,
        f"seen={seen} want={want}",
    )


async def la_off(t: Suite, la: LogAnalytics) -> None:
    await t.owui.update_valves(TRACKER, SEND_TO_LOG_ANALYTICS=False)
    try:
        await la.reset()
        async with t.browser() as b:
            c = await b.chat(PROBE_MODEL, "no send please")
        await asyncio.sleep(2)
        recs = await la.records()
    finally:
        await t.owui.update_valves(TRACKER, SEND_TO_LOG_ANALYTICS=True)
    t.check(
        "la.off",
        "SEND_TO_LOG_ANALYTICS=False: status shown, nothing posted",
        c.done and any("Req:" in (d or "") for d in c.status_descriptions) and not recs,
        f"done={c.done} status={c.status_descriptions} posts={len(recs)}",
    )


async def la_http_error(t: Suite, la: LogAnalytics) -> None:
    await la.reset()
    await la.mode(status=500)
    mark = t.mark()
    try:
        async with t.browser() as b:
            c = await b.chat(PROBE_MODEL, "la error please")
        lines = await wait_log(t, mark, "Error sending to Log Analytics: 500", 10)
        await t.log.settle(1)
    finally:
        await la.mode(status=200)
    recs = await la.records()
    tracebacks = [b for b in t.log.error_blocks(mark) if "Traceback" in b]
    unclosed = t.log.lines(mark, "Unclosed client session", "Unclosed connector")
    t.expect_errors(mark, ("time_token_tracker", "Error sending to Log Analytics: 500"))
    t.check(
        "la.http-error",
        "Log Analytics HTTP 500: the chat answers, the error is logged without a "
        "traceback, the session is closed",
        c.done
        and not c.error
        and len(recs) == 1
        and bool(lines)
        and not tracebacks
        and not unclosed,
        f"done={c.done} error={c.error} posts={len(recs)} lines={lines[:1]} "
        f"tracebacks={len(tracebacks)} unclosed={unclosed[:1]}",
    )


async def la_slow(t: Suite, la: LogAnalytics) -> None:
    """A slow (6 s) or hanging Log Analytics endpoint must not hold up API
    answers (Open WebUI awaits outlet filters before it answers); the record
    still arrives and a hanging send times out."""
    results = {}
    timeout_lines: list = []
    mark0 = t.mark()
    try:
        for name, delay in (("slow", SLOW_SECONDS), ("hang", HANG_SECONDS)):
            await la.reset()
            await la.mode(status=200, delay=delay)
            mark = t.mark()
            t0 = time.time()
            r = await t.owui.chat(
                PROBE_MODEL,
                f"{name} log analytics please",
                features={"web_search": False},
            )
            elapsed = time.time() - t0
            await t.log.settle(0.5)
            recs = await la.wait(1, timeout=send_timeout(t, mark, 15))
            arrived = [round(e["received"] - t0, 2) for e in recs]
            results[name] = (r.status, round(elapsed, 2), len(recs), arrived)
            if name == "hang" and recs:
                timeout_lines = await wait_log(
                    t, mark, "Exception when sending to Log Analytics", 16
                )
            print(f"          la.slow {name}: {results[name]}", flush=True)
    finally:
        await la.mode(status=200)
    await t.log.settle(0.5)
    t.expect_errors(
        mark0, ("time_token_tracker", "Exception when sending to Log Analytics")
    )
    ok = all(
        status == 200 and elapsed < 2 and posts == 1 and arrived[0] <= 15
        for status, elapsed, posts, arrived in results.values()
    ) and any("TimeoutError" in line for line in timeout_lines)
    detail = " ".join(
        f"{name}: HTTP {s} elapsed={e}s posts={p} arrived={a}"
        for name, (s, e, p, a) in results.items()
    )
    t.check(
        "la.slow-not-blocking",
        f"slow ({SLOW_SECONDS} s) and hanging ({HANG_SECONDS} s) Log Analytics: API "
        "answer < 2 s, record received <= 15 s, the hanging send times out",
        ok,
        f"{detail} timeout_logged={timeout_lines[:1]} "
        f"outlet_errors={outlet_errors(t, mark0)[:1]}",
    )


# ---------------------------------------------------------------------- valves
async def valves_group(t: Suite, la: LogAnalytics) -> None:
    await t.owui.update_valves(
        TRACKER,
        CALCULATE_ALL_MESSAGES=False,
        SHOW_RESPONSE_TIME=False,
        SHOW_TOKENS_PER_SECOND=False,
    )
    try:
        await la.reset()
        async with t.browser() as b:
            c = await b.chat(PROBE_MODEL, "Compact status please", stream=True)
        recs = await la.wait(1)
        rep = probe_report(c.content)
        request = [
            m for m in rep.get("messages") or [] if m.get("role") in ("user", "system")
        ]
        req = tokens(message_text(request[-1])) if request else -1
        resp = tokens(c.content)
        exp = {
            "req": req,
            "resp": resp,
            "last": resp,
            "tps": 0,
            "avg": False,
            "req_count": 0,
            "resp_count": 0,
            "chat_id": c.chat_id,
        }
        problems = (
            record_problems(recs[0], exp, PROBE_MODEL, t.owui.user.get("id"))
            if len(recs) == 1
            else [f"posts={len(recs)}"]
        )
        tracker = [d for d in c.status_descriptions if d and "Req:" in d]
        want = f"Req: {req} | Resp: {resp}"
        t.check(
            "valves.compact",
            "CALCULATE_ALL_MESSAGES / SHOW_RESPONSE_TIME / SHOW_TOKENS_PER_SECOND off: "
            "status 'Req: n | Resp: m', record without averages, tokensPerSecond 0",
            c.done and tracker == [want] and len(request) == 1 and not problems,
            f"status={tracker} want={want!r} problems={problems}",
        )

        await t.owui.replace_valves(
            TRACKER,
            {
                **await t.owui.get_valves(TRACKER),
                "CALCULATE_ALL_MESSAGES": True,
                "SHOW_RESPONSE_TIME": True,
                "SHOW_TOKENS_PER_SECOND": True,
                "SHOW_TOKEN_COUNT": False,
            },
        )
        async with t.browser() as b:
            c = await b.chat(PROBE_MODEL, "No token count please", stream=True)
        shown = [s for s in c.status_history if s.get("description")]
        t.check(
            "valves.no-token-count",
            "SHOW_TOKEN_COUNT off: one status '<t>s | x T/s' without 'Req:', done",
            c.done
            and len(shown) == 1
            and bool(_STATUS_NO_TOKENS.match(shown[0]["description"]))
            and shown[0].get("done") is True,
            f"status={shown}",
        )
    finally:
        await t.owui.update_valves(
            TRACKER,
            CALCULATE_ALL_MESSAGES=True,
            SHOW_RESPONSE_TIME=True,
            SHOW_TOKENS_PER_SECOND=True,
            SHOW_TOKEN_COUNT=True,
        )


# ----------------------------------------------------------------- correlation
async def correlation(t: Suite, la: LogAnalytics) -> None:
    # The legacy code interpreter prompt changes the last user message after
    # the inlet, so its fingerprint differs in the outlet.
    text = "Correlate me with metadata"
    await la.reset()
    mark = t.mark()
    r = await t.owui.chat(
        PROBE_MODEL,
        text,
        features={"code_interpreter": True, "web_search": False},
        params={"function_calling": "legacy"},
    )
    await t.log.settle(1.5)
    rep = probe_report(r.content)
    users = [m for m in rep.get("messages") or [] if m.get("role") == "user"]
    modified = bool(users) and users[-1].get("content") != text
    recs = await la.wait(1, timeout=send_timeout(t, mark))
    rec = record_of(recs[0]) if recs else {}
    lines = outlet_lines(t, mark)
    warn = t.log.lines(mark, "No inlet data found")
    want = tokens(text)
    t.check(
        "correlation.api-modified-message",
        "API, user message changed after the inlet (legacy code interpreter "
        "prompt): the outlet still finds the inlet entry (request tokens, response "
        "time)",
        r.status == 200
        and modified
        and [x["req"] for x in lines] == [want]
        and not warn
        and rec.get("requestTokens") == want
        and (rec.get("responseTime") or 0) > 0,
        f"HTTP {r.status} trigger_modified={modified} outlet={lines} want={want} "
        f"warn={warn[:1]} rec={short(rec, 200)}",
    )

    # Two identical requests in flight at once (same user, model and text).
    text = "Identical concurrent request PROBE_SLEEP=1.5"
    await la.reset()
    mark = t.mark()
    results = await asyncio.gather(
        *(
            t.owui.chat(PROBE_MODEL, text, features={"web_search": False})
            for _ in range(2)
        )
    )
    await t.log.settle(1.5)
    recs = await la.wait(2, timeout=send_timeout(t, mark))
    lines = outlet_lines(t, mark)
    warn = t.log.lines(mark, "No inlet data found")
    want = tokens(text)
    got = sorted(
        (
            record_of(e).get("requestTokens"),
            (record_of(e).get("responseTime") or 0) >= 1.4,
        )
        for e in recs
    )
    t.check(
        "correlation.concurrent-identical",
        "two identical concurrent API requests: each outlet finds its own inlet "
        "entry (request tokens, response time >= the pipe's 1.5 s)",
        all(r.status == 200 for r in results)
        and [x["req"] for x in lines] == [want, want]
        and all(x["response_time"] >= 1.4 for x in lines)
        and not warn
        and got == [(want, True), (want, True)],
        f"HTTP {[r.status for r in results]} outlet={lines} want={want} "
        f"warn={warn[:1]} records={got}",
    )


# -------------------------------------------------------------------- encoding
async def encoding(t: Suite, la: LogAnalytics) -> None:
    text = (
        "Grüße aus Zürich: Übermäßig größere Bäume blühen schön 🌳🌸 日本語のテキスト"
    )
    # Loads o200k_base in the driver first: the server's first load of it is a
    # cache read, so the tracker's first request must already be exact.
    o200, cl100 = tokens(text, "o200k_base"), tokens(text)
    status = await upsert_derived_model(t, GPT4O_MODEL, "gpt-4o e2e", [TRACKER])
    await la.reset()
    async with t.browser() as b:
        c = await b.chat(GPT4O_MODEL, text)
    recs = await la.wait(1)
    rec = record_of(recs[0]) if recs else {}
    want_resp = tokens(c.content, "o200k_base")
    t.check(
        "encoding.per-model",
        "model id gpt-4o-*: o200k_base counts (differ from cl100k_base), exact on "
        "the first request of the encoding",
        status == 200
        and c.done
        and o200 != cl100
        and rec.get("requestTokens") == o200
        and rec.get("responseTokens") == want_resp
        and rec.get("model") == GPT4O_MODEL,
        f"create={status} done={c.done} o200k={o200} cl100k={cl100} "
        f"want_resp={want_resp} rec={short(rec, 250)}",
    )


# --------------------------------------------------------------------- offline
async def offline(t: Suite, la: LogAnalytics) -> None:
    """tiktoken's encoding host points at a TLS peer that hangs for 8 s."""
    model = DAVINCI_MODEL  # tiktoken: p50k_base, used by no other scenario
    cache = tiktoken_cache_file("p50k_base")
    if os.path.exists(cache):
        os.remove(cache)
    precondition = not os.path.exists(cache)
    status = await upsert_derived_model(t, model, "davinci e2e", [TRACKER])
    edit_hosts(TARPIT_LINE, add=True)
    tarpit0 = len(await la.tarpit())

    async def timed(text: str) -> tuple:
        t0 = time.time()
        r = await t.owui.chat(model, text, features={"web_search": False})
        return r, round(time.time() - t0, 2)

    async def health(stop: asyncio.Event, seen: list) -> None:
        while not stop.is_set():
            t0 = time.time()
            try:
                await t.owui.http.get("/health", timeout=10)
            except httpx.HTTPError:
                pass
            seen.append(round(time.time() - t0, 2))
            await asyncio.sleep(0.25)

    flags: list = []
    logged: list = []
    try:
        await la.reset()
        mark = t.mark()
        texts = [f"offline request number {i} " + "x" * (8 * i) for i in range(1, 4)]
        stop, health_seen = asyncio.Event(), []
        probe = asyncio.create_task(health(stop, health_seen))
        results = await asyncio.gather(*(timed(text) for text in texts))
        stop.set()
        await probe
        recs = await la.wait(3)
        records = [record_of(e) for e in recs]
        flags += [r.get("tokensEstimated", MISSING) for r in records]
        got = sorted(r.get("requestTokens", -1) for r in records)
        want = sorted(len(text) // 4 for text in texts)
        latencies = sorted(d for _, d in results)
        errors = [short(r.errors, 160) for r, _ in results if r.errors]
        t.check(
            "offline.concurrent",
            "encoding download hangs: of 3 concurrent requests only the one that "
            "starts the load waits (<= ~5 s), all answer, counts = len(text) // 4, "
            "/health stays responsive",
            precondition
            and status == 200
            and all(r.status == 200 for r, _ in results)
            and latencies[-1] < 7.5
            and sum(1 for d in latencies if d >= 3) <= 1
            and got == want
            and max(health_seen or [99]) < 1,
            f"cache_absent={precondition} create={status} "
            f"HTTP {[r.status for r, _ in results]} latencies={latencies} "
            f"health_max={max(health_seen or [None])} got={got} want={want} "
            f"errors={errors}",
        )

        await asyncio.sleep(TARPIT_SECONDS + 1)  # the tarpit closes: load failed
        await la.reset()
        text = "after the failed load, still estimating"
        r, latency = await timed(text)
        recs = await la.wait(1)
        rec = record_of(recs[0]) if recs else {}
        if rec:
            flags.append(rec.get("tokensEstimated", MISSING))
        await t.log.settle(1)
        warn = t.log.lines(mark, "p50k_base' could not be loaded")
        attempts = len(await la.tarpit()) - tarpit0
        logged = [x["estimated"] for x in outlet_lines(t, mark)]
        t.check(
            "offline.after-failure",
            "after the failed load the next request does not wait and estimates; "
            "one warning, one download attempt (retry window)",
            r.status == 200
            and latency < 2
            and rec.get("requestTokens") == len(text) // 4
            and len(warn) == 1
            and attempts == 1,
            f"HTTP {r.status} latency={latency}s requestTokens="
            f"{rec.get('requestTokens')} want {len(text) // 4} warnings={len(warn)} "
            f"attempts={attempts} errors={short(r.errors, 160)}",
        )
    finally:
        edit_hosts(TARPIT_LINE, add=False)
        await t.owui.delete_model(model)
    if all(f == MISSING for f in flags) and not any(logged):
        state = "tokensEstimated flag missing"
    else:
        state = f"tokensEstimated={flags} log tokens_estimated={logged}"
    t.check(
        "offline.estimate-marker",
        "estimated counts are marked: record tokensEstimated=true and "
        "'tokens_estimated=True' in the outlet log lines",
        len(flags) == 4
        and all(f is True for f in flags)
        and len(logged) == 4
        and all(x == "True" for x in logged),
        state,
    )


# ------------------------------------------------------------------ multimodel
async def multimodel(t: Suite, la: LogAnalytics) -> None:
    """One UI message to three models: two with google_search_tool, one
    without. They share the request's features dict."""
    models = [PROBE_MODEL, MULTI_SEARCH_MODEL, MULTI_PLAIN_MODEL]
    created = [
        await upsert_derived_model(
            t, MULTI_SEARCH_MODEL, "E2E Multi S", [SEARCH, TRACKER]
        ),
        await upsert_derived_model(t, MULTI_PLAIN_MODEL, "E2E Multi P", [TRACKER]),
    ]
    # Answers that finish at the same moment race when Open WebUI saves them
    # into the chat (a lost update can leave one message "not done"): the
    # probe answers 1 s apart.
    text = (
        f"multi model probe PROBE_SLEEP[{MULTI_SEARCH_MODEL}]=1 "
        f"PROBE_SLEEP[{MULTI_PLAIN_MODEL}]=2"
    )
    await la.reset()
    async with t.browser() as b:
        c = await b.chat(
            PROBE_MODEL, text, features={"web_search": True}, models=models
        )
    recs = await la.wait(3)
    answers = {a.model: a for a in c.answers}
    reports = {m: probe_report(a.content) for m, a in answers.items()}
    features = {m: rep.get("metadata_features") or {} for m, rep in reports.items()}
    errors = [
        (a.error.get("content") if isinstance(a.error, dict) else str(a.error))
        if a.error
        else None
        for a in c.answers
    ]
    records = sorted(
        (record_of(e).get("messageId"), record_of(e).get("requestTokens")) for e in recs
    )
    want = sorted((a.message_id, tokens(text)) for a in c.answers)
    search_flags = [features.get(m, {}).get("google_search_tool") for m in models[:2]]
    plain_web_search = features.get(MULTI_PLAIN_MODEL, {}).get("web_search")
    t.check(
        "multimodel.search",
        "multi-model chat with web_search: every answer done without error, both "
        "models with google_search_tool get the flag, the model without it keeps "
        "web_search, one record per assistant message",
        created == [200, 200]
        and all(a.done for a in c.answers)
        and not any(errors)
        and search_flags == [True, True]
        and plain_web_search is True
        and records == want,
        f"create={created} done={[a.done for a in c.answers]} errors={errors} "
        f"google_search_tool={search_flags} plain web_search={plain_web_search} "
        f"records={records} want={want}",
    )


# ---------------------------------------------------------------------- search
async def search(t: Suite) -> None:
    status = await upsert_derived_model(t, SEARCH_MODEL, "E2E Search", [SEARCH])
    for sid, features in (
        ("search.features-empty", {}),
        ("search.features-no-web-search", {"image_generation": False}),
        ("search.features-null", None),
    ):
        r = await t.owui.chat(SEARCH_MODEL, "probe", features=features)
        rep = probe_report(r.content)
        flags = rep.get("metadata_features") or {}
        t.check(
            sid,
            f"features={json.dumps(features)} passes google_search_tool, no "
            "google_search_tool flag",
            status == 200
            and r.status == 200
            and bool(rep)
            and not flags.get("google_search_tool"),
            f"{r.brief()} metadata_features={flags}",
        )
    r = await t.owui.chat(
        SEARCH_MODEL, "probe", features={"web_search": True, "memory": False}
    )
    flags = probe_report(r.content).get("metadata_features")
    t.check(
        "search.keeps-other-features",
        "web_search -> google_search_tool, web_search removed, other feature keys kept",
        r.status == 200 and flags == {"google_search_tool": True, "memory": False},
        f"HTTP {r.status} metadata_features={flags}",
    )
    await permission_limitation(t)


async def permission_limitation(t: Suite) -> None:
    """Documented limitation: google_search_tool does not check the per-user
    web search permission (fails once a permission check is added)."""
    _, before = await t.owui.api("GET", "/api/v1/users/default/permissions")
    user = None
    try:
        if not isinstance(before, dict) or "features" not in before:
            raise RuntimeError(f"default permissions unreadable: {short(before)}")
        denied = {**before, "features": {**before["features"], "web_search": False}}
        perm_status, _ = await t.owui.api(
            "POST", "/api/v1/users/default/permissions", denied
        )
        grant_status, _ = await t.owui.api(
            "POST",
            "/api/v1/models/model/access/update",
            {
                "id": SEARCH_MODEL,
                "access_grants": [
                    {
                        "principal_type": "user",
                        "principal_id": "*",
                        "permission": "read",
                    }
                ],
            },
        )
        base_status, _ = await t.owui.api(
            "POST",
            "/api/v1/models/model/access/update",
            {
                "id": PROBE_MODEL,
                "access_grants": [
                    {
                        "principal_type": "user",
                        "principal_id": "*",
                        "permission": "read",
                    }
                ],
            },
        )
        user = await t.owui.create_user(
            name="E2E Filters User", email="filters-user@example.com"
        )
        r = await user.chat(SEARCH_MODEL, "probe", features={"web_search": True})
        flags = probe_report(r.content).get("metadata_features") or {}
        ok = (
            perm_status == 200
            and grant_status == 200
            and base_status == 200
            and r.status == 200
            and flags.get("google_search_tool") is True
        )
        detail = (
            f"permissions={perm_status} grants={grant_status}/{base_status} "
            f"{r.brief()} metadata_features={flags}"
        )
    except Exception as exc:  # setup through the admin API failed
        ok, detail = False, f"setup failed: {exc!r}"
    finally:
        if isinstance(before, dict) and "features" in before:
            await t.owui.api("POST", "/api/v1/users/default/permissions", before)
        for model_id in (SEARCH_MODEL, PROBE_MODEL):
            await t.owui.api(
                "POST",
                "/api/v1/models/model/access/update",
                {"id": model_id, "access_grants": []},
            )
        if user is not None:
            await user.close()
    t.check(
        "search.permission-limitation",
        "a user without the web search permission still gets google_search_tool "
        "(documented: the filter does not check the permission)",
        ok,
        detail,
    )


# ---------------------------------------------------------------------- vertex
async def vertex(t: Suite) -> None:
    status = await upsert_derived_model(t, VERTEX_MODEL, "E2E Vertex", [VERTEX])
    r = await t.owui.chat(
        VERTEX_MODEL,
        "probe",
        features={"vertex_ai_search": True},
        params={"vertex_rag_store": STORE2},
    )
    rep = probe_report(r.content)
    params, flags = rep.get("metadata_params") or {}, rep.get("metadata_features") or {}
    store = params.get("vertex_rag_store")
    t.check(
        "vertex.per-request-store",
        "params.vertex_rag_store of the request reaches __metadata__.params (and "
        "leaves the body) when the request enables vertex_ai_search",
        status == 200
        and r.status == 200
        and flags.get("vertex_ai_search") is True
        and store == STORE2
        and "vertex_rag_store" not in (rep.get("body_keys") or []),
        f"HTTP {r.status} store={store!r} want {STORE2!r} features={flags} "
        f"body_keys={rep.get('body_keys')}",
    )

    r = await t.owui.chat(
        VERTEX_MODEL,
        "probe",
        features={"vertex_ai_search": False},
        params={"vertex_rag_store": STORE2},
    )
    rep = probe_report(r.content)
    params, flags = rep.get("metadata_params") or {}, rep.get("metadata_features") or {}
    t.check(
        "vertex.store-flag-off",
        "params.vertex_rag_store without vertex_ai_search: no data store and no "
        "flag in __metadata__ (a client cannot redirect the pipe's own search)",
        r.status == 200
        and bool(rep)
        and params.get("vertex_rag_store") is None
        and not flags.get("vertex_ai_search"),
        f"HTTP {r.status} params={params} features={flags}",
    )

    r = await t.owui.chat(VERTEX_MODEL, "probe", features=None)
    rep = probe_report(r.content)
    flags = rep.get("metadata_features") or {}
    t.check(
        "vertex.features-null",
        "features=null passes vertex_ai_search_tool, no vertex flag",
        r.status == 200 and bool(rep) and not flags.get("vertex_ai_search"),
        f"{r.brief()} metadata_features={flags}",
    )
