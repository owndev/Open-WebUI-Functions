"""
Filters suite: filters/{google_search_tool,vertex_ai_search_tool,time_token_tracker}.py
in front of the probe pipe (tests/e2e/probe/probe_pipe.py), which reports what
reached the pipe (messages, __metadata__ features/params, ...), so filter -> pipe
coupling is checked without a provider. The tracker's Azure Log Analytics sends
go to mocks/mock_la.py: this suite maps <workspace>.ods.opinsights.azure.com
(HTTP Data Collector API), the Microsoft Entra ID login hosts and the DCR
ingestion hosts (Logs Ingestion API) to 127.0.0.1 in /etc/hosts, installs a
throw-away test CA into the container's system store and starts the mock (HTTPS
on 127.0.0.1:443, control on :9105, managed identity endpoints on :9107).
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
  ingest       Logs Ingestion API: request shape and 204, token cache (scope and
               tenant in its key) / concurrency / refresh, refresh failure with a
               still-valid token (also inside the back-off), the 30 s token
               back-off (still on after 20 s) and its end, token and HTTP errors
               with one log line each (a secret echoed at the cut points of the
               error text, a second revocation, a late 401 for a replaced
               token), undecryptable secret, https-only (endpoint and authority)
               and invalid settings, slow / unreachable endpoints, mode selection
               and fallback, both APIs (also with one side incomplete),
               deprecation warning, sovereign cloud, App Service (also its error
               shape) / IMDS / workload identity (environment set through the
               probe pipe's PROBE_ENV hook)

While the Log Analytics groups run, the probe pipe's PROBE_LOG hook writes
everything the tracker logs, DEBUG included, to /e2e/out/tracker_debug.log;
log.no-secrets-debug scans it (the server log runs at INFO).
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
from urllib.parse import quote, quote_plus

import httpx

from harness import Suite, short
from harness.config import VERTEX_RAG_STORE
from harness.logs import ServerLog

GROUPS = (
    "model",
    "global",
    "spec",
    "la",
    "ingest",
    "valves",
    "correlation",
    "encoding",
    "offline",
    "multimodel",
    "search",
    "vertex",
)
# Groups that need the Log Analytics mock (and a warm tracker).
LA_GROUPS = (
    "la",
    "ingest",
    "valves",
    "correlation",
    "encoding",
    "offline",
    "multimodel",
)
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
ENV_MODEL = "e2e-probe-env"  # no filters: carries PROBE_ENV (ingest group)
WORKSPACE_MODELS = (
    GPT4O_MODEL,
    DAVINCI_MODEL,
    SEARCH_MODEL,
    VERTEX_MODEL,
    MULTI_SEARCH_MODEL,
    MULTI_PLAIN_MODEL,
    ENV_MODEL,
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
    "LOG_ANALYTICS_INGESTION_API",
    "LOG_ANALYTICS_DCR_ENDPOINT",
    "LOG_ANALYTICS_DCR_IMMUTABLE_ID",
    "LOG_ANALYTICS_DCR_STREAM_NAME",
    "LOG_ANALYTICS_AUTH_MODE",
    "LOG_ANALYTICS_TENANT_ID",
    "LOG_ANALYTICS_CLIENT_ID",
    "LOG_ANALYTICS_CLIENT_SECRET",
    "LOG_ANALYTICS_AUTHORITY_HOST",
    "LOG_ANALYTICS_INGESTION_SCOPE",
)

# Log Analytics mock (mocks/mock_la.py)
WORKSPACE = "e2ews"
LA_HOST = f"{WORKSPACE}.ods.opinsights.azure.com"
LA_KEY = base64.b64encode(b"e2e-log-analytics-shared-key-0123456789").decode()
LOG_TYPE = "E2E_Metrics"
LA_CTL = "http://127.0.0.1:9105"
LA_DIR = "/tmp/e2e-la"
LA_OUT = "/e2e/out/mock_la.txt"
# la-setup <dir> <san1,san2,...>: the CA is kept across --reuse runs (OpenSSL
# caches CA certificates by subject hash in the long-running server process, so
# a new CA with the same subject could fail verification); the server
# certificate is reissued when it lacks a name or expires within an hour.
LA_SETUP = r"""
set -e
mkdir -p "$1" && cd "$1"
IFS=, read -ra sans <<< "$2"
if [ ! -f ca.crt ] || [ ! -f ca.key ] \
    || ! openssl x509 -in ca.crt -noout -checkend 3600 >/dev/null 2>&1; then
  openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.crt -days 30 \
    -subj "/CN=E2E Log Analytics CA" 2>/dev/null
  rm -f srv.crt
fi
issue=0
if [ ! -f srv.crt ] \
    || ! openssl x509 -in srv.crt -noout -checkend 3600 >/dev/null 2>&1; then
  issue=1
else
  text=$(openssl x509 -in srv.crt -noout -text)
  for san in "${sans[@]}"; do
    case "$text" in *"DNS:$san"*) ;; *) issue=1 ;; esac
  done
fi
if [ "$issue" = 1 ]; then
  ext="DNS:${sans[0]}"
  for san in "${sans[@]:1}"; do ext="$ext,DNS:$san"; done
  openssl req -newkey rsa:2048 -nodes -keyout srv.key -out srv.csr \
    -subj "/CN=${sans[0]}" 2>/dev/null
  printf "subjectAltName=%s\n" "$ext" > ext.cnf
  openssl x509 -req -in srv.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out srv.crt -days 30 -extfile ext.cnf 2>/dev/null
fi
installed=/usr/local/share/ca-certificates/e2e-la-ca.crt
if ! cmp -s ca.crt "$installed"; then
  cp ca.crt "$installed"
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

# Logs Ingestion API (mocks/mock_la.py, ingest group)
TENANT = "11111111-2222-3333-4444-555555555555"
CLIENT_SECRET = "e2e~Secret+/=&value.42"  # needs URL encoding; >= 6 chars
DCR_ID = "dcr-0123456789abcdef0123456789abcdef"
DCR_HOST = "e2e-dcr-ab12-westeurope.logs.z1.ingest.monitor.azure.com"
GOV_DCR_HOST = "e2e-dce-cd34.usgovvirginia-1.ingest.monitor.azure.us"
REFUSED_HOST = "e2e-dcr-down-westeurope.logs.z1.ingest.monitor.azure.com"
LOGIN_HOSTS = ("login.microsoftonline.com", "login.microsoftonline.us")
STREAM = f"Custom-{LOG_TYPE}_CL"  # default stream name derived from LOG_TYPE
DEFAULT_SCOPE = "https://monitor.azure.com/.default"
TOKEN_PATH = f"/{TENANT}/oauth2/v2.0/token"
MI_PORT = 9107  # MSI_PORT of mocks/mock_la.py
MI_URL = f"http://127.0.0.1:{MI_PORT}"
MI_IMDS_URL = f"http://127.0.0.2:{MI_PORT}"  # not covered by NO_PROXY (mi.imds)
PROBE_ENV_FILE = f"{LA_DIR}/probe-env.json"
WI_TOKEN_FILE = f"{LA_DIR}/wi-token"
DEPRECATION = "HTTP Data Collector API, which Microsoft deprecated"
INGEST_HOSTS_LINES = [
    f"127.0.0.1 {host}" for host in (*LOGIN_HOSTS, DCR_HOST, GOV_DCR_HOST)
] + [f"127.0.0.9 {REFUSED_HOST}"]
LA_SANS = (LA_HOST, *LOGIN_HOSTS, DCR_HOST, GOV_DCR_HOST)
# Environment variables the probe pipe may set (tests/e2e/probe/probe_pipe.py).
PROBE_ENV_NAMES = (
    "IDENTITY_ENDPOINT",
    "IDENTITY_HEADER",
    "IDENTITY_SERVER_THUMBPRINT",
    "IMDS_ENDPOINT",
    "MSI_ENDPOINT",
    "MSI_SECRET",
    "AZURE_POD_IDENTITY_AUTHORITY_HOST",
    "AZURE_FEDERATED_TOKEN_FILE",
    "AZURE_KUBERNETES_TOKEN_PROXY",
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "AZURE_AUTHORITY_HOST",
    "HTTP_PROXY",
    "http_proxy",
    "NO_PROXY",
    "no_proxy",
)
# The container's own values of those variables: run.sh sets IDENTITY_ENDPOINT
# and IDENTITY_HEADER at docker run (App Service managed identity for the azure
# suite, mocks/mock_search.py). The driver runs in the same container and sees
# them too; the ingest group removes them while it runs and puts them back at
# the end, for the suites after it and for --reuse runs.
CONTAINER_ENV = {
    name: os.environ[name] for name in PROBE_ENV_NAMES if name in os.environ
}
# Issued tokens, identity headers, assertions etc. of the ingest group for the
# final log.no-secrets check (collected in the group's finally).
INGEST_SECRETS: list = []
# The server log runs at INFO: while the Log Analytics groups run, the probe
# pipe (PROBE_LOG) also writes everything the tracker's logger logs, DEBUG
# included, to this file; log.no-secrets-debug scans it.
DEBUG_LOG = "/e2e/out/tracker_debug.log"
DEBUG_CAPTURE: dict = {"started": False, "problems": []}
TRACKER_LOADED = "Loaded module: function_time_token_tracker"

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
    """Differences between one recorded HTTP Data Collector POST and ``exp``:
    the signed 2.6.2 request (``dc_transport_problems``) and the record
    (``record_body_problems``)."""
    return dc_transport_problems(entry) + record_body_problems(
        entry, exp, model, user_id
    )


def dc_transport_problems(entry: dict) -> list:
    """Host, path, SharedKey signature and headers of a Data Collector POST."""
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
    return problems


def li_transport_problems(
    entry: dict,
    host: str = DCR_HOST,
    dcr: str = DCR_ID,
    stream: str = STREAM,
    status: Optional[int] = 204,
) -> list:
    """URI, Bearer token and headers of a Logs Ingestion API POST."""
    problems = []
    headers = {k.lower(): v for k, v in (entry.get("headers") or {}).items()}
    if entry.get("host") != host:
        problems.append(f"host={entry.get('host')}")
    if entry.get("path") != f"/dataCollectionRules/{dcr}/streams/{stream}":
        problems.append(f"path={entry.get('path')}")
    if entry.get("query") != {"api-version": "2023-01-01"}:
        problems.append(f"query={entry.get('query')}")
    if not entry.get("token_ok") or entry.get("token_expired"):
        problems.append(
            f"token_ok={entry.get('token_ok')} expired={entry.get('token_expired')}"
        )
    if not str(headers.get("authorization", "")).startswith("Bearer <token#"):
        problems.append(f"authorization={headers.get('authorization')}")
    if headers.get("content-type") != "application/json":
        problems.append(f"content-type={headers.get('content-type')}")
    if "content-encoding" in headers:
        problems.append(f"content-encoding={headers['content-encoding']}")
    try:
        uuid.UUID(str(headers.get("x-ms-client-request-id")))
    except ValueError:
        problems.append(
            f"x-ms-client-request-id={headers.get('x-ms-client-request-id')}"
        )
    dc_headers = {"log-type", "x-ms-date", "time-generated-field"} & set(headers)
    if dc_headers or "SharedKey" in str(headers.get("authorization", "")):
        problems.append(f"Data Collector headers {sorted(dc_headers)}")
    if status is not None and entry.get("status") != status:
        problems.append(f"status={entry.get('status')}")
    return problems


def record_body_problems(entry: dict, exp: dict, model: str, user_id: str) -> list:
    """Differences between the record of one recorded POST and ``exp``.

    ``exp``: req, resp, last, req_count, resp_count; optional ``tps`` (default
    last / responseTime), ``avg`` (False: no average fields), ``chat_id`` (None:
    any UUID), ``message_id`` (compared when given).
    """
    problems = []
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


async def wait_log(
    t: Suite, mark: int, needle: str, timeout: float, n: int = 1
) -> list:
    """Lines containing ``needle`` logged since ``mark``, once there are ``n``
    (polls up to ``timeout``)."""
    deadline = time.time() + timeout
    while True:
        lines = t.log.lines(mark, needle)
        if len(lines) >= n or time.time() >= deadline:
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
        """Clear the records and reset every mode (config and tokens stay)."""
        await self.http.post("/__reset")

    async def mode(
        self, status: int = 200, delay: float = 0.0, target: str = "dc", **extra
    ) -> None:
        """Replace the whole mode of ``target`` (dc, token, ingest or msi);
        ``extra``: expires_in, retry_after, error_code, echo_secret."""
        await self.http.post(
            "/__mode",
            json={"target": target, "status": status, "delay": delay, **extra},
        )

    async def records(self, kind: str = "dc") -> list:
        """Recorded requests of ``kind`` (dc, token, ingest, msi or all)."""
        return (await self.http.get("/__requests", params={"kind": kind})).json()

    async def config(self, **values) -> None:
        """tenant, client_secret, dcr_id, stream, identity_header, token_file."""
        await self.http.post("/__config", json=values)

    async def tokens(self) -> list:
        """Every access token the mock issued."""
        return (await self.http.get("/__tokens")).json()

    async def revoke(self) -> None:
        """Invalidate every issued token (the next ingestion POST gets 401)."""
        await self.http.post("/__revoke")

    async def tarpit(self) -> list:
        return (await self.http.get("/__tarpit")).json()

    async def wait(self, n: int = 1, timeout: float = 8.0, kind: str = "dc") -> list:
        """Recorded requests of ``kind`` once there are ``n`` (or after
        ``timeout`` seconds)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(await self.records(kind)) >= n:
                await asyncio.sleep(0.3)  # one more would be a duplicate
                break
            await asyncio.sleep(0.25)
        return await self.records(kind)

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
    """Test CA + /etc/hosts entries + mock_la.py; returns (process, client)."""
    subprocess.run(
        ["bash", "-c", LA_SETUP, "la-setup", LA_DIR, ",".join(LA_SANS)], check=True
    )
    for line in (LA_HOSTS_LINE, *INGEST_HOSTS_LINES):
        edit_hosts(line, add=True)
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
            # Over loopback to the mock: the secret never reaches the server log.
            await la.config(
                tenant=TENANT,
                client_secret=CLIENT_SECRET,
                dcr_id=DCR_ID,
                stream=STREAM,
                token_file=WI_TOKEN_FILE,
            )
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
        for line in INGEST_HOSTS_LINES:  # unlike LA_HOSTS_LINE, never left mapped
            edit_hosts(line, add=False)
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
    t.assert_no_secrets(LA_KEY, *INGEST_SECRETS)
    if DEBUG_CAPTURE["started"]:
        debug_no_secrets(t)
    t.scan_log()


async def tracker_debug_capture(t: Suite, target: str) -> list:
    """Start (``target`` an absolute path) or stop (``"off"``) the probe
    pipe's PROBE_LOG capture of the tracker's logger at DEBUG. Returns
    problems."""
    await upsert_derived_model(t, ENV_MODEL, "E2E Probe Env", [])
    r = await t.owui.chat(ENV_MODEL, f"PROBE_LOG={target}")
    rep = probe_report(r.content)
    if r.status == 200 and rep.get("log_capture") == target:
        return []
    return [
        f"PROBE_LOG={target}: HTTP {r.status} capture={rep.get('log_capture')} "
        f"error={rep.get('log_error')}"
    ]


def debug_no_secrets(t: Suite) -> None:
    """log.no-secrets for the tracker's DEBUG output (DEBUG_LOG): the same
    secrets as the server log check."""
    secrets = dict(t.owui.secrets)
    for value in (LA_KEY, *INGEST_SECRETS):
        if value:
            secrets.setdefault(value, "secret")
    capture = ServerLog(DEBUG_LOG)
    debug = sum(
        " DEBUG time_token_tracker " in line for line in capture.since(0).splitlines()
    )
    problems = DEBUG_CAPTURE["problems"] + capture.secrets(0, secrets)
    t.check(
        "log.no-secrets-debug",
        "no plaintext secrets in what time_token_tracker logs, DEBUG included "
        "(captured through the probe pipe's PROBE_LOG; the server log runs at INFO)",
        debug > 0 and not problems,
        f"{debug} DEBUG lines captured; "
        + (" || ".join(problems[:5]) if problems else "no secret logged"),
    )


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
    await t.owui.update_valves(TRACKER, LOG_ANALYTICS_CLIENT_SECRET=CLIENT_SECRET)
    stored = (await t.owui.get_valves(TRACKER)).get("LOG_ANALYTICS_CLIENT_SECRET")
    secret_spec = props.get("LOG_ANALYTICS_CLIENT_SECRET") or {}
    input_type = (secret_spec.get("input") or {}).get("type")
    t.check(
        "spec.client-secret-encrypted",
        "LOG_ANALYTICS_CLIENT_SECRET is a password valve and stored encrypted",
        isinstance(stored, str)
        and stored.startswith("encrypted:")
        and CLIENT_SECRET not in stored
        and input_type == "password",
        f"stored={short(stored, 40)} input.type={input_type!r}",
    )
    want = {
        "LOG_ANALYTICS_INGESTION_API": "auto",
        "LOG_ANALYTICS_AUTH_MODE": "client_secret",
        "LOG_ANALYTICS_AUTHORITY_HOST": "https://login.microsoftonline.com",
        "LOG_ANALYTICS_INGESTION_SCOPE": DEFAULT_SCOPE,
        "LOG_ANALYTICS_LOG_TYPE": "OpenWebuiMetrics",
        "LOG_ANALYTICS_DCR_ENDPOINT": "",
        "LOG_ANALYTICS_DCR_IMMUTABLE_ID": "",
        "LOG_ANALYTICS_DCR_STREAM_NAME": "",
        "LOG_ANALYTICS_TENANT_ID": "",
        "LOG_ANALYTICS_CLIENT_ID": "",
        "LOG_ANALYTICS_CLIENT_SECRET": "",
    }
    got = {name: (props.get(name) or {}).get("default", MISSING) for name in want}
    diff = {name: got[name] for name in want if got[name] != want[name]}
    t.check(
        "spec.ingest-defaults",
        "Logs Ingestion valve defaults (auto, client_secret, public cloud "
        "authority and scope, OpenWebuiMetrics, the others empty)",
        not diff,
        f"differs: {diff}" if diff else f"defaults={got}",
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
        # After the install: a module load resets the tracker's logger level.
        DEBUG_CAPTURE["problems"] = await tracker_debug_capture(t, DEBUG_LOG)
        DEBUG_CAPTURE["started"] = True
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
        if t.selected("ingest"):
            await ingest_group(t, la)
    finally:
        if DEBUG_CAPTURE["started"]:
            DEBUG_CAPTURE["problems"] += await tracker_debug_capture(t, "off")
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


# ---------------------------------------------------------- Logs Ingestion API
TOKEN_ERROR = "Could not get a Microsoft Entra ID token"
INGEST_ERROR = "Error sending to Logs Ingestion API"
INGEST_EXCEPTION = "Exception when sending to Logs Ingestion API"
INGEST_SENT = "sent via the Logs Ingestion API"
DROPPED = "Log Analytics record dropped"
# premise of ingest.mi.imds: a trust_env=True request through the same proxy
# environment does not reach the IMDS mock
PROXY_PREMISE = """
import asyncio, sys, aiohttp
async def main():
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as s:
            async with s.get(sys.argv[1], headers={"Metadata": "true"}) as r:
                print("STATUS", r.status)
    except Exception as e:
        print("ERROR", type(e).__name__)
asyncio.run(main())
"""
PROXY_ENV = {
    "HTTP_PROXY": "http://127.0.0.1:9",  # nothing listens
    "http_proxy": "http://127.0.0.1:9",
    "NO_PROXY": "localhost,127.0.0.1",  # the server's other local traffic
    "no_proxy": "localhost,127.0.0.1",
}


def fresh_client_id(prefix: str = "e2e-app") -> str:
    """A client id no earlier request used, i.e. a cold token cache (the cache
    lives in the server process and survives --reuse)."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def li_valves(**over) -> dict:
    """Every Logs Ingestion valve with an explicit value and a fresh client id,
    then ``over``: update_valves merges into the stored valves, so a valve one
    scenario overrides must not leak into the next one."""
    return {
        "SEND_TO_LOG_ANALYTICS": True,
        "LOG_ANALYTICS_INGESTION_API": "auto",
        "LOG_ANALYTICS_DCR_ENDPOINT": f"https://{DCR_HOST}",
        "LOG_ANALYTICS_DCR_IMMUTABLE_ID": DCR_ID,
        "LOG_ANALYTICS_DCR_STREAM_NAME": "",
        "LOG_ANALYTICS_AUTH_MODE": "client_secret",
        "LOG_ANALYTICS_TENANT_ID": TENANT,
        "LOG_ANALYTICS_CLIENT_ID": fresh_client_id(),
        "LOG_ANALYTICS_CLIENT_SECRET": CLIENT_SECRET,
        "LOG_ANALYTICS_AUTHORITY_HOST": "https://login.microsoftonline.com",
        "LOG_ANALYTICS_INGESTION_SCOPE": DEFAULT_SCOPE,
        **over,
    }


async def set_li_valves(t: Suite, **over) -> dict:
    valves = li_valves(**over)
    await t.owui.update_valves(TRACKER, **valves)
    return valves


async def set_env(t: Suite, mapping: Optional[dict] = None) -> list:
    """Set the managed identity / proxy environment of the server process
    through the probe pipe (PROBE_ENV): every allow-listed name not in
    ``mapping`` is removed, so a scenario never inherits one. The values go
    through a file, never through the chat text. Returns problems."""
    mapping = mapping or {}
    unknown = sorted(set(mapping) - set(PROBE_ENV_NAMES))
    with open(PROBE_ENV_FILE, "w", encoding="utf-8") as fh:
        json.dump({name: mapping.get(name) for name in PROBE_ENV_NAMES}, fh)
    r = await t.owui.chat(ENV_MODEL, f"PROBE_ENV={PROBE_ENV_FILE}")
    rep = probe_report(r.content)
    if (
        r.status == 200
        and rep.get("env_applied") == sorted(PROBE_ENV_NAMES)
        and not rep.get("env_rejected")
        and not rep.get("env_error")
        and not unknown
    ):
        return []
    return [
        f"set_env: HTTP {r.status} applied={rep.get('env_applied')} "
        f"rejected={rep.get('env_rejected')} error={rep.get('env_error')} "
        f"unknown={unknown}"
    ]


async def li_chat(t: Suite, text: str, stream: bool = False) -> tuple:
    """API request through the tracker: (result, seconds until the answer)."""
    t0 = time.time()
    r = await t.owui.chat(
        PROBE_MODEL, text, stream=stream, features={"web_search": False}
    )
    return r, round(time.time() - t0, 2)


def token_problems(
    toks: list,
    client_ids: list,
    host: str = LOGIN_HOSTS[0],
    scope: str = DEFAULT_SCOPE,
    auth: str = "secret",
) -> list:
    """Recorded token requests vs. one expected client id per request."""
    if len(toks) != len(client_ids):
        return [f"token requests={len(toks)} want {len(client_ids)}"]
    problems = []
    for tok, client_id in zip(toks, client_ids):
        content_type = str(tok.get("content_type") or "").split(";")[0].strip()
        want = {
            "host": host,
            "path": TOKEN_PATH,
            "grant_type": "client_credentials",
            "client_id": client_id,
            "scope": scope,
            "auth": auth,
            "status": 200,
        }
        problems += [
            f"{name}={tok.get(name)!r}"
            for name, value in want.items()
            if tok.get(name) != value
        ]
        if content_type != "application/x-www-form-urlencoded":
            problems.append(f"content_type={tok.get('content_type')!r}")
        if not tok.get(f"{auth}_ok"):
            problems.append(f"{auth}_ok=False")
    return problems


def unclean(t: Suite, mark: int) -> list:
    """Tracebacks and unclosed aiohttp sessions logged since ``mark``."""
    tracebacks = [b for b in t.log.error_blocks(mark) if "Traceback" in b]
    unclosed = t.log.lines(mark, "Unclosed client session", "Unclosed connector")
    return ([f"{len(tracebacks)} tracebacks"] if tracebacks else []) + unclosed[:1]


def header(entry: dict, name: str) -> Optional[str]:
    for key, value in (entry.get("headers") or {}).items():
        if key.lower() == name.lower():
            return value
    return None


def once_per_process(t: Suite, needle: str) -> tuple:
    """(ok, detail) for a warning the tracker logs once per process, i.e. per
    module load: exactly once in this run when the module was loaded during
    it (the install loads it, also with --reuse), otherwise at most once in
    this run and at least once since the server started."""
    run_lines = t.log.lines(t.log_start, needle)
    if t.log.lines(t.log_start, TRACKER_LOADED):
        return len(run_lines) == 1, f"{len(run_lines)} in this run (module loaded)"
    all_lines = t.log.lines(0, needle)
    return (
        len(run_lines) <= 1 and len(all_lines) >= 1,
        f"{len(run_lines)} in this run, {len(all_lines)} since the server started",
    )


async def ingest_group(t: Suite, la: LogAnalytics) -> None:
    state: dict = {"secrets": []}
    try:
        await upsert_derived_model(t, ENV_MODEL, "E2E Probe Env", [])
        # A crashed --reuse run may have left a proxy or identity endpoint set.
        state["env_problems"] = await set_env(t)
        await ingest_api(t, la)
        await ingest_token_cache(t, la)
        retry = await ingest_token_errors(t, la, state)
        await ingest_http_401(t, la)
        await ingest_token_backoff(t, la, retry)  # 20 s after token-error
        await ingest_http_errors(t, la)
        await ingest_refresh(t, la)
        await ingest_slow(t, la)
        await ingest_modes(t, la)
        await ingest_clouds(t, la)
        await ingest_managed_identity(t, la, state)
        await ingest_retry_after_window(t, la, retry)
    finally:
        try:
            INGEST_SECRETS.extend(await la.tokens())
            await la.reset()
            await la.config(stream=STREAM, identity_header="")
        except httpx.HTTPError:
            pass
        INGEST_SECRETS.extend(state["secrets"])
        await set_env(t, CONTAINER_ENV)
        await t.owui.replace_valves(
            TRACKER,
            {
                "SEND_TO_LOG_ANALYTICS": True,
                "LOG_ANALYTICS_WORKSPACE_ID": WORKSPACE,
                "LOG_ANALYTICS_SHARED_KEY": LA_KEY,
                "LOG_ANALYTICS_LOG_TYPE": LOG_TYPE,
            },
        )


async def ingest_api(t: Suite, la: LogAnalytics) -> None:
    """API and browser path: one exact record per response, 204, one token."""
    valves = await set_li_valves(t)
    user_id = t.owui.user.get("id")
    for stream in (False, True):
        await la.reset()
        mark = t.mark()
        r, _ = await li_chat(
            t, f"Logs Ingestion API token count, stream={stream}.", stream
        )
        rep = probe_report(r.content)
        posts = await la.wait(1, timeout=send_timeout(t, mark, 10), kind="ingest")
        sent = await wait_log(t, mark, INGEST_SENT, 5)
        await t.log.settle(0.3)
        dc = await la.records()
        toks = await la.records("token")
        exp = expected(rep, r.content)
        problems = (
            li_transport_problems(posts[0])
            + record_body_problems(posts[0], exp, PROBE_MODEL, user_id)
            if len(posts) == 1
            else [f"ingest posts={len(posts)}"]
        )
        rec = record_of(posts[0]) if posts else {}
        ids = f"(chat={rec.get('chatId')}, message={rec.get('messageId')})"
        if len(sent) != 1 or ids not in sent[0]:
            problems.append(f"success lines={sent[:2]}")
        if dc:
            problems.append(f"dc posts={len(dc)}")
        if stream:
            if toks:
                problems.append(f"token requests={len(toks)} (cached token expected)")
        else:
            problems += token_problems(toks, [valves["LOG_ANALYTICS_CLIENT_ID"]])
        tag = "stream" if stream else "nonstream"
        t.check(
            f"ingest.api.{tag}",
            f"API {tag}: one exact record to "
            "{endpoint}/dataCollectionRules/{dcr}/streams/Custom-<log type>_CL with "
            "a Bearer token, 204 logged as success; "
            + (
                "the cached token is reused"
                if stream
                else "one form-encoded client credentials token request"
            ),
            r.status == 200 and bool(rep) and not problems,
            f"HTTP {r.status} problems={problems} exp={exp} "
            f"outlet_errors={outlet_errors(t, mark)[:1]} rec={short(rec, 250)}",
        )

    await la.reset()
    async with t.browser() as b:
        c = await b.chat(PROBE_MODEL, "Browser via the Logs Ingestion API", stream=True)
    posts = await la.wait(1, timeout=10, kind="ingest")
    rep = probe_report(c.content)
    exp = {**expected(rep, c.content), "chat_id": c.chat_id, "message_id": c.message_id}
    problems = (
        li_transport_problems(posts[0])
        + record_body_problems(posts[0], exp, PROBE_MODEL, user_id)
        if len(posts) == 1
        else [f"ingest posts={len(posts)}"]
    )
    rec = record_of(posts[0]) if posts else {}
    try:
        want = status_text(rec)
    except (KeyError, TypeError, ValueError):
        want = None
    tracker = [s for s in c.status_history if "Req:" in (s.get("description") or "")]
    t.check(
        "ingest.browser",
        "browser: the Logs Ingestion record has the chat id and the assistant "
        "message id, the status text shows the record's values, done",
        c.done
        and not problems
        and len(tracker) == 1
        and tracker[0].get("description") == want
        and tracker[0].get("done") is True,
        f"problems={problems} status={tracker} want={want!r}",
    )


async def ingest_token_cache(t: Suite, la: LogAnalytics) -> None:
    """One token for many records: sequential and concurrent."""
    await la.reset()
    await set_li_valves(t)
    statuses = []
    for i in range(3):
        r, _ = await li_chat(t, f"cached token request {i}")
        statuses.append(r.status)
        await la.wait(i + 1, timeout=10, kind="ingest")
    posts = await la.records("ingest")
    toks = await la.records("token")
    indexes = [p.get("token_index") for p in posts]
    t.check(
        "ingest.token-cached",
        "3 sequential records: 3 posts with 204, 1 token request, the same token",
        statuses == [200] * 3
        and len(posts) == 3
        and all(p.get("status") == 204 for p in posts)
        and len(toks) == 1
        and indexes == [toks[0].get("token_index")] * 3,
        f"HTTP {statuses} posts={[p.get('status') for p in posts]} "
        f"token_indexes={indexes} token_requests={len(toks)}",
    )

    await la.reset()
    await set_li_valves(t)
    await la.mode(target="token", delay=4)
    mark = t.mark()
    t0 = time.time()
    results = await asyncio.gather(
        *(li_chat(t, f"concurrent token request {i}") for i in range(5))
    )
    posts = await la.wait(5, timeout=15, kind="ingest")
    sent = await wait_log(t, mark, INGEST_SENT, max(0.5, t0 + 15 - time.time()), n=5)
    toks = await la.records("token")
    elapsed = sorted(e for _, e in results)
    arrived = sorted(round(p["received"] - t0, 2) for p in posts)
    t.check(
        "ingest.token-concurrent",
        "5 concurrent records while the token request takes 4 s: every API answer "
        "< 3 s (outlet does not wait), exactly 1 token request, 5 posts with 204 "
        "and 5 success lines within 15 s",
        all(r.status == 200 for r, _ in results)
        and elapsed[-1] < 3
        and len(toks) == 1
        and len(posts) == 5
        and all(p.get("status") == 204 for p in posts)
        and bool(arrived)
        and arrived[-1] <= 15
        and len(sent) == 5,
        f"HTTP {[r.status for r, _ in results]} answers={elapsed}s "
        f"token_requests={len(toks)} posts={[p.get('status') for p in posts]} "
        f"arrived={arrived} success_lines={len(sent)}",
    )

    # Scope and tenant are part of the cache key: the other scenarios change
    # the client id as well, this one keeps it.
    await la.reset()
    client_id = (await set_li_valves(t))["LOG_ANALYTICS_CLIENT_ID"]
    await li_chat(t, "cache key: default scope")
    await la.wait(1, timeout=10, kind="ingest")
    alt_scope = "https://monitor.azure.com//.default"
    await set_li_valves(
        t, LOG_ANALYTICS_CLIENT_ID=client_id, LOG_ANALYTICS_INGESTION_SCOPE=alt_scope
    )
    await li_chat(t, "cache key: other scope")
    await la.wait(2, timeout=10, kind="ingest")
    other_tenant = "22222222-3333-4444-5555-666666666666"  # unknown to the mock
    await set_li_valves(
        t, LOG_ANALYTICS_CLIENT_ID=client_id, LOG_ANALYTICS_TENANT_ID=other_tenant
    )
    mark = t.mark()
    await li_chat(t, "cache key: other tenant")
    await wait_log(t, mark, TOKEN_ERROR, 10)
    await t.log.settle(1)
    errors = t.log.lines(mark, TOKEN_ERROR)
    posts = await la.records("ingest")
    toks = await la.records("token")
    t.expect_errors(mark, ("time_token_tracker", TOKEN_ERROR))
    got = [(tok.get("tenant"), tok.get("scope"), tok.get("client_id")) for tok in toks]
    want = [
        (TENANT, DEFAULT_SCOPE, client_id),
        (TENANT, alt_scope, client_id),
        (other_tenant, DEFAULT_SCOPE, client_id),
    ]
    statuses = [(p.get("status"), p.get("token_index")) for p in posts]
    t.check(
        "ingest.token-cache-key",
        "the same client id with another scope, then with another tenant: one "
        "new token request each (scope and tenant are part of the cache key); "
        "the unknown tenant logs one ERROR with the AADSTS90002 hint, nothing "
        "is sent with the old tenant's token",
        got == want
        and statuses
        == [(204, toks[0].get("token_index")), (204, toks[1].get("token_index"))]
        and len(errors) == 1
        and "AADSTS90002" in errors[0]
        and "LOG_ANALYTICS_TENANT_ID" in errors[0],
        f"token_requests={got} posts={statuses} errors={errors[:2]}",
    )


async def ingest_token_errors(t: Suite, la: LogAnalytics, state: dict) -> dict:
    """Wrong secret (one ERROR with the AADSTS hint, then the 30 s back-off),
    redaction of an echoed secret, an undecryptable stored secret."""
    await la.reset()
    client_id = fresh_client_id()
    await set_li_valves(
        t,
        LOG_ANALYTICS_CLIENT_ID=client_id,
        LOG_ANALYTICS_CLIENT_SECRET="e2e-wrong-secret-1",
    )
    mark = t.mark()
    r1, _ = await li_chat(t, "token error request 1")
    errors = await wait_log(t, mark, TOKEN_ERROR, 12)
    toks = await la.records("token")
    failed_at = toks[0]["received"] if toks else time.time()
    mark2 = t.mark()
    r2, _ = await li_chat(t, "token error request 2")
    dropped = await wait_log(t, mark2, DROPPED, 8)
    await t.log.settle(1)
    toks = await la.records("token")
    posts = await la.records("ingest")
    errors = t.log.lines(mark, TOKEN_ERROR)
    dirty = unclean(t, mark)
    t.expect_errors(mark, ("time_token_tracker", TOKEN_ERROR))
    needles = ("AADSTS7000215", "invalid_client", "not its ID")
    t.check(
        "ingest.token-error",
        "wrong client secret: one ERROR with AADSTS7000215, invalid_client and the "
        "'value, not its ID' hint, nothing posted; the next record within 30 s "
        "sends no token request and logs one 'record dropped' WARNING",
        r1.status == 200
        and r2.status == 200
        and len(errors) == 1
        and all(n in errors[0] for n in needles)
        and not posts
        and len(toks) == 1
        and len(dropped) == 1
        and not dirty,
        f"errors={errors[:2]} token_requests={len(toks)} posts={len(posts)} "
        f"dropped={dropped[:1]} unclean={dirty}",
    )
    retry = {"client_id": client_id, "failed_at": failed_at}

    wrong = "e2e-wrong+secret/2="  # changes when URL-encoded
    state["secrets"].append(quote_plus(wrong))
    await la.reset()
    await set_li_valves(t, LOG_ANALYTICS_CLIENT_SECRET=wrong)
    await la.mode(target="token", echo_secret=True)
    mark = t.mark()
    await li_chat(t, "token error echo")
    await wait_log(t, mark, TOKEN_ERROR, 12)
    await t.log.settle(0.5)
    blocks = [b for b in t.log.error_blocks(mark) if TOKEN_ERROR in b]
    forms = (wrong, quote_plus(wrong), quote(wrong, safe=""))
    leaked = [i for i, form in enumerate(forms) if any(form in b for b in blocks)]
    t.expect_errors(mark, ("time_token_tracker", TOKEN_ERROR))
    t.check(
        "ingest.token-error-echo",
        "an error text that echoes the client secret (plain and form-encoded) is "
        "logged redacted (***)",
        len(blocks) == 1 and "echo ***" in blocks[0] and not leaked,
        f"blocks={len(blocks)} leaked_forms={leaked} "
        f"redacted={'***' in (blocks[0] if blocks else '')}",
    )

    cut = "e2e-cut+secret/3=x"
    state["secrets"].append(quote_plus(cut))
    await la.reset()
    await set_li_valves(t, LOG_ANALYTICS_CLIENT_SECRET=cut)
    await la.mode(target="token", echo_secret=True, echo_at_cut=True)
    mark = t.mark()
    await li_chat(t, "token error echo at the cut points")
    await wait_log(t, mark, TOKEN_ERROR, 12)
    await t.log.settle(0.5)
    blocks = [b for b in t.log.error_blocks(mark) if TOKEN_ERROR in b]
    # The first 8 characters of the plain and the form-encoded secret: the
    # mock puts 10 of them before each cut point.
    leaked = [
        i
        for i, prefix in enumerate((cut[:8], quote_plus(cut)[:8]))
        if any(prefix in b for b in blocks)
    ]
    redacted = blocks[0].count("***") if blocks else 0
    t.expect_errors(mark, ("time_token_tracker", TOKEN_ERROR))
    t.check(
        "ingest.token-error-echo-cut",
        "the client secret echoed across the cut points of the error text "
        "(error, first line of error_description, trace_id, correlation_id) is "
        "redacted before the text is cut: not even a prefix of it is logged",
        len(blocks) == 1 and redacted == 4 and not leaked,
        f"blocks={len(blocks)} leaked_prefixes={leaked} redacted={redacted}",
    )

    value = "encrypted:e2e-not-a-fernet-token"
    state["secrets"].append(value)
    await la.reset()
    await set_li_valves(t, LOG_ANALYTICS_CLIENT_SECRET=value)
    stored = (await t.owui.get_valves(TRACKER)).get("LOG_ANALYTICS_CLIENT_SECRET")
    mark = t.mark()
    r, _ = await li_chat(t, "undecryptable secret")
    lines = await wait_log(t, mark, "could not be decrypted", 10)
    await t.log.settle(1)
    lines = t.log.lines(mark, "could not be decrypted")
    toks = await la.records("token")
    posts = await la.records("ingest")
    t.expect_errors(mark, ("time_token_tracker", TOKEN_ERROR))
    t.check(
        "ingest.secret-undecryptable",
        "a stored client secret that cannot be decrypted (WEBUI_SECRET_KEY "
        "changed): one ERROR 'could not be decrypted', nothing sent to Entra ID",
        r.status == 200
        and stored == value
        and len(lines) == 1
        and not toks
        and not posts,
        f"HTTP {r.status} stored_unchanged={stored == value} lines={lines[:1]} "
        f"token_requests={len(toks)} posts={len(posts)}",
    )
    return retry


async def ingest_http_401(t: Suite, la: LogAnalytics) -> None:
    """A revoked token (twice, with a 204 in between), every token rejected."""
    await la.reset()
    await set_li_valves(t)
    results = [(await li_chat(t, "401 warm-up (caches token A)"))[0]]
    await la.wait(1, timeout=10, kind="ingest")
    await la.revoke()
    mark = t.mark()
    results.append((await li_chat(t, "401 with the revoked token A"))[0])
    await wait_log(t, mark, f"{INGEST_ERROR}: 401", 10)
    results.append((await li_chat(t, "401 then a new token B"))[0])
    await la.wait(3, timeout=10, kind="ingest")
    # A later, single revocation: the 204 in between ended the "last token got
    # a 401" state, so this is no second 401 in a row (no token back-off).
    await la.revoke()
    results.append((await li_chat(t, "401 with the revoked token B"))[0])
    await wait_log(t, mark, f"{INGEST_ERROR}: 401", 10, n=2)
    results.append((await li_chat(t, "401 then a new token C"))[0])
    posts = await la.wait(5, timeout=10, kind="ingest")
    await t.log.settle(1)
    lines = t.log.lines(mark, INGEST_ERROR)
    dropped = t.log.lines(mark, DROPPED)
    toks = await la.records("token")
    t.expect_errors(mark, ("time_token_tracker", f"{INGEST_ERROR}: 401"))
    statuses = [p.get("status") for p in posts]
    indexes = [p.get("token_index") for p in posts]
    tok = [x.get("token_index") for x in toks]
    t.check(
        "ingest.http-401",
        "a revoked token (401): one line with the scope hint, the next record "
        "requests exactly one new token and gets 204; a second revocation after "
        "that 204 is handled the same way (no token back-off, nothing dropped)",
        [r.status for r in results] == [200] * 5
        and len(lines) == 2
        and all("LOG_ANALYTICS_INGESTION_SCOPE" in line for line in lines)
        and statuses == [204, 401, 204, 401, 204]
        and len(tok) == 3
        and indexes == [tok[0], tok[0], tok[1], tok[1], tok[2]]
        and not dropped,
        f"lines={lines[:2]} statuses={statuses} token_indexes={indexes} "
        f"token_requests={len(toks)} dropped={dropped[:1]}",
    )

    await la.reset()
    await set_li_valves(t)
    await la.mode(target="ingest", status=401)
    mark = t.mark()
    for i in (1, 2):
        await li_chat(t, f"persistent 401 request {i}")
        await wait_log(t, mark, f"{INGEST_ERROR}: 401", 10, n=i)
    mark3 = t.mark()
    await li_chat(t, "persistent 401 request 3")
    dropped = await wait_log(t, mark3, DROPPED, 8)
    await t.log.settle(1)
    lines = t.log.lines(mark, f"{INGEST_ERROR}: 401")
    toks = await la.records("token")
    posts = await la.records("ingest")
    t.expect_errors(mark, ("time_token_tracker", f"{INGEST_ERROR}: 401"))
    t.check(
        "ingest.http-401-persistent",
        "every token gets 401: two records with a new token each (one 401 line "
        "each), then the token back-off: the third record sends nothing and logs "
        "'record dropped'",
        len(lines) == 2
        and len(toks) == 2
        and len(posts) == 2
        and [p.get("token_index") for p in posts]
        == [tok.get("token_index") for tok in toks]
        and len(dropped) == 1,
        f"lines={len(lines)} token_requests={len(toks)} posts={len(posts)} "
        f"dropped={dropped[:1]}",
    )


async def ingest_token_backoff(t: Suite, la: LogAnalytics, retry: dict) -> None:
    """The token back-off lasts 30 s, not just a few: 20 s after the failed
    token request of ingest.token-error, a record of that identity is still
    dropped although the secret is fixed. ingest.token-retry-after-window
    checks the end of the same back-off."""
    client_id, failed_at, note = retry["client_id"], retry["failed_at"], ""
    if time.time() > failed_at + 24:
        # A slow run: too late for that back-off, start another one.
        await la.reset()
        client_id = fresh_client_id()
        await set_li_valves(
            t,
            LOG_ANALYTICS_CLIENT_ID=client_id,
            LOG_ANALYTICS_CLIENT_SECRET="e2e-wrong-secret-3",
        )
        mark = t.mark()
        await li_chat(t, "token back-off: failed token request")
        await wait_log(t, mark, TOKEN_ERROR, 12)
        toks = await la.records("token")
        failed_at = toks[0]["received"] if toks else time.time()
        t.expect_errors(mark, ("time_token_tracker", TOKEN_ERROR))
        note = " (own failed token request, the run was slow)"
    await asyncio.sleep(max(0.0, failed_at + 20 - time.time()))
    await la.reset()
    await set_li_valves(t, LOG_ANALYTICS_CLIENT_ID=client_id)
    mark = t.mark()
    sent_at = round(time.time() - failed_at, 1)
    await li_chat(t, "token back-off: 20 s later, secret fixed")
    dropped = await wait_log(t, mark, DROPPED, 8)
    await t.log.settle(1)
    toks = await la.records("token")
    posts = await la.records("ingest")
    t.check(
        "ingest.token-backoff-duration",
        "20 s after a failed token request the identity is still in its 30 s "
        "back-off: the record is dropped (one 'record dropped' WARNING), no token "
        "request although the secret is fixed now",
        20 <= sent_at < 28 and not toks and not posts and len(dropped) == 1,
        f"sent {sent_at}s after the failure{note} token_requests={len(toks)} "
        f"posts={len(posts)} dropped={dropped[:1]}",
    )


async def ingest_http_errors(t: Suite, la: LogAnalytics) -> None:
    """A 401 that arrives after the token was replaced; one actionable log
    line per failed record, no retry."""
    await la.reset()
    await set_li_valves(t)
    await li_chat(t, "late 401 warm-up (caches token A)")
    await la.wait(1, timeout=10, kind="ingest")
    await la.revoke()
    await la.mode(target="ingest", delay=4)
    mark = t.mark()
    await li_chat(t, "late 401: slow post with token A")  # its 401 comes 4 s later
    await la.wait(2, timeout=5, kind="ingest")
    await la.mode(target="ingest")
    await li_chat(t, "late 401: fast post with token A")  # 401 at once
    await wait_log(t, mark, f"{INGEST_ERROR}: 401", 4)
    await li_chat(t, "late 401: new token B")
    sent = await wait_log(t, mark, INGEST_SENT, 4)
    late = await wait_log(t, mark, f"{INGEST_ERROR}: 401", 8, n=2)
    await li_chat(t, "late 401: after the late 401")  # still token B
    posts = await la.wait(5, timeout=10, kind="ingest")
    await t.log.settle(1)
    toks = await la.records("token")
    dropped = t.log.lines(mark, DROPPED)
    t.expect_errors(mark, ("time_token_tracker", f"{INGEST_ERROR}: 401"))
    shape = [(p.get("status"), p.get("token_index")) for p in posts]
    tok = [x.get("token_index") for x in toks]
    # Token B must have been cached before the slow post's 401 came back.
    ordered = len(toks) >= 2 and len(posts) >= 2
    ordered = ordered and toks[1]["received"] < posts[1]["received"] + 3.5
    t.check(
        "ingest.http-401-late",
        "two records in flight with a revoked token: the 401 of the slow one "
        "arrives after the new token was cached and does not drop it (the next "
        "record reuses it: 2 token requests in all)",
        len(tok) == 2
        and shape
        == [(204, tok[0]), (401, tok[0]), (401, tok[0]), (204, tok[1]), (204, tok[1])]
        and len(sent) == 1
        and len(late) == 2
        and ordered
        and not dropped,
        f"posts={shape} token_requests={len(toks)} success_lines={len(sent)} "
        f"401_lines={len(late)} ordered={ordered} dropped={dropped[:1]}",
    )

    cases = (
        (
            "ingest.http-403",
            "403: one ERROR line with the Monitoring Metrics Publisher role and the "
            "30 minute propagation hint, no retry",
            {"status": 403},
            {},
            (": 403", "Monitoring Metrics Publisher", "30 minutes"),
        ),
        (
            "ingest.http-404",
            "404 (unknown DCR immutable ID): one line naming the DCR ID and stream "
            "name valves",
            None,
            {"LOG_ANALYTICS_DCR_IMMUTABLE_ID": "dcr-" + "f" * 32},
            (
                ": 404",
                "LOG_ANALYTICS_DCR_IMMUTABLE_ID",
                "LOG_ANALYTICS_DCR_STREAM_NAME",
            ),
        ),
        (
            "ingest.http-413",
            "413: one line with the 1 MB limit, no retry",
            {"status": 413},
            {},
            (": 413", "1 MB"),
        ),
        (
            "ingest.http-429",
            "429: one line with Retry-After as sent, no retry (one post 3 s later)",
            {"status": 429, "retry_after": 17},
            {},
            (": 429", "Retry-After: 17"),
        ),
        (
            "ingest.http-500",
            "500: one line with the response body and the x-ms-client-request-id "
            "of the request, no traceback, session closed",
            {"status": 500},
            {},
            (f"{INGEST_ERROR}: 500", "e2e mock"),
        ),
    )
    for sid, title, ingest_mode, over, needles in cases:
        await la.reset()
        await set_li_valves(t, **over)
        if ingest_mode:
            await la.mode(target="ingest", **ingest_mode)
        mark = t.mark()
        r, _ = await li_chat(t, f"{sid} please")
        await wait_log(t, mark, INGEST_ERROR, 10)
        await asyncio.sleep(3 if sid.endswith("429") else 1.5)  # a retry would show
        await t.log.settle(0.5)
        lines = t.log.lines(mark, INGEST_ERROR)
        posts = await la.records("ingest")
        dirty = unclean(t, mark)
        t.expect_errors(mark, ("time_token_tracker", INGEST_ERROR))
        problems = []
        if len(lines) != 1 or not all(n in lines[0] for n in needles):
            problems.append(f"lines={lines[:2]}")
        if len(posts) != 1:
            problems.append(f"posts={len(posts)}")
        if sid.endswith("500"):
            logged = re.search(r"request=([0-9a-f-]{36})", lines[0] if lines else "")
            sent_id = header(posts[0], "x-ms-client-request-id") if posts else None
            if not logged or logged.group(1) != sent_id:
                problems.append(
                    f"request id logged={logged.group(1) if logged else None} "
                    f"sent={sent_id}"
                )
            problems += dirty
        t.check(sid, title, r.status == 200 and not problems, f"problems={problems}")


async def ingest_refresh(t: Suite, la: LogAnalytics) -> None:
    """Token refresh before expiry, and a failed refresh with a valid token."""
    await la.reset()
    await set_li_valves(t)
    await la.mode(target="token", expires_in=8)  # refresh after 4 s
    await li_chat(t, "refresh request 1")
    posts = await la.wait(1, timeout=10, kind="ingest")
    # The filter posts right after it received the token: its refresh time is
    # at most 4 s after this post (the mock's token receipt is earlier).
    anchor = posts[0]["received"] if posts else time.time()
    await li_chat(t, "refresh request 1b, well before half the lifetime")
    posts = await la.wait(2, timeout=10, kind="ingest")
    toks = await la.records("token")
    early = (
        round(posts[1]["received"] - toks[0]["received"], 2)
        if len(posts) > 1 and toks
        else None
    )
    await asyncio.sleep(max(0.0, anchor + 5 - time.time()))
    await li_chat(t, "refresh request 2")
    posts = await la.wait(3, timeout=10, kind="ingest")
    toks = await la.records("token")
    gap = round(toks[1]["received"] - toks[0]["received"], 2) if len(toks) > 1 else None
    t.check(
        "ingest.token-refresh",
        "a token with 8 s lifetime is reused for a record in the first half of "
        "its lifetime and refreshed after half of it: 2 token requests, the "
        "second before the first token expired, the last post uses the new "
        "token, no expired token is sent",
        len(toks) == 2
        and gap is not None
        and gap < 8
        and early is not None
        and early < 3.5
        and len(posts) == 3
        and posts[1].get("token_index") == toks[0].get("token_index")
        and posts[2].get("token_index") == toks[1].get("token_index")
        and all(p.get("status") == 204 and not p.get("token_expired") for p in posts),
        f"token_requests={len(toks)} early={early}s gap={gap}s posts="
        f"{[(p.get('status'), p.get('token_index')) for p in posts]}",
    )

    await la.reset()
    await set_li_valves(t)
    await la.mode(target="token", expires_in=12)  # refresh 6 s, usable until 9 s
    await li_chat(t, "refresh fallback request 1")
    posts = await la.wait(1, timeout=10, kind="ingest")
    toks = await la.records("token")
    t0 = toks[0]["received"] if toks else time.time()
    # refresh due <= anchor + 6, usable until >= t0 + 9 and <= anchor + 9
    anchor = posts[0]["received"] if posts else time.time()
    await la.mode(target="token", status=500)
    await asyncio.sleep(max(0.0, anchor + 6.3 - time.time()))
    mark2 = t.mark()
    sent2 = round(time.time() - t0, 2)
    await li_chat(t, "refresh fallback request 2")
    await la.wait(2, timeout=10, kind="ingest")
    # Inside the 30 s token back-off that failure started, the token is still
    # usable (until >= t0 + 9): sent with it, no token request.
    sent2b = round(time.time() - anchor, 2)
    await li_chat(t, "refresh fallback request 2b, in the back-off")
    await la.wait(3, timeout=5, kind="ingest")
    keep = await wait_log(t, mark2, "keeping the cached token", 5)
    await asyncio.sleep(max(0.0, anchor + 9.8 - time.time()))
    mark3 = t.mark()
    sent3 = round(time.time() - t0, 2)
    await li_chat(t, "refresh fallback request 3")
    await wait_log(t, mark3, DROPPED, 6)
    await t.log.settle(1)
    toks = await la.records("token")
    posts = await la.records("ingest")
    keep = t.log.lines(mark2, "keeping the cached token")
    dropped = t.log.lines(mark2, DROPPED)
    t.expect_errors(mark2, ("time_token_tracker", "keeping the cached token"))
    t.check(
        "ingest.token-refresh-fallback",
        "the refresh fails while the token is still valid: one ERROR 'keeping the "
        "cached token', the record is sent with the old token (204), and so is "
        "the next one inside the 30 s back-off; after the token's usable time "
        "the next record is dropped without a token request",
        len(keep) == 1
        and len(toks) == 2
        and toks[1].get("status") == 500
        and len(posts) == 3
        and all(p.get("token_index") == toks[0].get("token_index") for p in posts)
        and all(p.get("status") == 204 for p in posts)
        and sent2b < 8.5
        and len(dropped) == 1
        and dropped[0] in t.log.lines(mark3, DROPPED),
        f"request2_at={sent2}s request2b_at=post1+{sent2b}s request3_at={sent3}s "
        f"keep={keep[:1]} posts="
        f"{[(p.get('status'), p.get('token_index')) for p in posts]} token_requests="
        f"{[(x.get('status'), x.get('token_index')) for x in toks]} dropped={dropped[:2]}",
    )


async def ingest_slow(t: Suite, la: LogAnalytics) -> None:
    """Hanging token endpoint, slow and hanging ingestion, refused connection:
    the API answer never waits."""
    mark0 = t.mark()
    results = {}
    await la.reset()
    await set_li_valves(t)
    await la.mode(target="token", delay=HANG_SECONDS)
    mark = t.mark()
    r, elapsed = await li_chat(t, "token endpoint hangs")
    lines = await wait_log(t, mark, TOKEN_ERROR, 15)
    posts = await la.records("ingest")
    results["token-hang"] = (r.status, elapsed, lines[:1], len(posts))
    ok_a = (
        r.status == 200
        and elapsed < 2
        and len(lines) == 1
        and "TimeoutError" in lines[0]
        and not posts
    )

    await la.reset()
    await set_li_valves(t)  # another client id: (a)'s key is in its back-off
    await la.mode(target="ingest", delay=SLOW_SECONDS)
    mark = t.mark()
    t0 = time.time()
    r, elapsed = await li_chat(t, "slow ingestion")
    posts = await la.wait(1, timeout=15, kind="ingest")
    sent = await wait_log(t, mark, INGEST_SENT, max(0.5, t0 + 15 - time.time()))
    sent_at = round(time.time() - t0, 2)
    results["slow"] = (r.status, elapsed, len(posts), sent_at, bool(sent))
    ok_b = r.status == 200 and elapsed < 2 and len(posts) == 1 and bool(sent)
    ok_b = ok_b and sent_at <= 15

    await la.reset()
    await la.mode(target="ingest", delay=HANG_SECONDS)
    mark = t.mark()
    r, elapsed = await li_chat(t, "hanging ingestion")
    lines = await wait_log(t, mark, INGEST_EXCEPTION, 16)
    results["hang"] = (r.status, elapsed, lines[:1])
    ok_c = (
        r.status == 200 and elapsed < 2 and bool(lines) and "TimeoutError" in lines[0]
    )
    await la.mode(target="ingest")
    t.expect_errors(
        mark0,
        ("time_token_tracker", TOKEN_ERROR),
        ("time_token_tracker", INGEST_EXCEPTION),
    )
    print(f"          ingest.slow: {results}", flush=True)
    t.check(
        "ingest.slow-not-blocking",
        f"hanging token endpoint, slow ({SLOW_SECONDS} s) and hanging "
        f"({HANG_SECONDS} s) ingestion: API answer < 2 s each; the token request "
        "times out (ERROR, nothing posted), the slow record arrives <= 15 s, the "
        "hanging send times out",
        ok_a and ok_b and ok_c,
        f"{results}",
    )

    await la.reset()
    await set_li_valves(t, LOG_ANALYTICS_DCR_ENDPOINT=f"https://{REFUSED_HOST}")
    mark = t.mark()
    r, elapsed = await li_chat(t, "unreachable ingestion endpoint")
    lines = await wait_log(t, mark, INGEST_EXCEPTION, 5)
    await t.log.settle(0.5)
    lines = t.log.lines(mark, INGEST_EXCEPTION)
    t.expect_errors(mark, ("time_token_tracker", INGEST_EXCEPTION))
    t.check(
        "ingest.unreachable",
        "ingestion endpoint refuses the connection: API answer < 2 s, one line "
        "'Exception when sending to Logs Ingestion API: ClientConnectorError'",
        r.status == 200
        and elapsed < 2
        and len(lines) == 1
        and "ClientConnectorError" in lines[0],
        f"HTTP {r.status} elapsed={elapsed}s lines={lines[:2]}",
    )


async def ingest_modes(t: Suite, la: LogAnalytics) -> None:
    """LOG_ANALYTICS_INGESTION_API: auto, data_collector, logs_ingestion, both,
    unknown values, invalid settings, deprecation warning."""
    user_id = t.owui.user.get("id")
    await la.reset()
    await set_li_valves(t)
    r, _ = await li_chat(t, "auto with both APIs configured")
    posts = await la.wait(1, timeout=10, kind="ingest")
    await asyncio.sleep(1)
    dc = await la.records()
    t.check(
        "ingest.mode.auto-li",
        "auto with both APIs configured: Logs Ingestion API only",
        r.status == 200
        and len(posts) == 1
        and posts[0].get("status") == 204
        and not dc,
        f"HTTP {r.status} ingest={[p.get('status') for p in posts]} dc={len(dc)}",
    )

    await la.reset()
    await set_li_valves(t, LOG_ANALYTICS_CLIENT_SECRET="")
    problems = []
    for i in range(2):
        r, _ = await li_chat(t, f"auto with incomplete Logs Ingestion settings {i}")
        dc = await la.wait(i + 1, timeout=10)
        rep = probe_report(r.content)
        if len(dc) != i + 1:
            problems.append(f"request {i}: dc posts={len(dc)}")
        else:
            problems += record_problems(
                dc[i], expected(rep, r.content), PROBE_MODEL, user_id
            )
    await asyncio.sleep(1)
    posts = await la.records("ingest")
    once, warnings = once_per_process(
        t,
        "not fully configured (missing: LOG_ANALYTICS_CLIENT_SECRET); using the "
        "HTTP Data Collector API",
    )
    t.check(
        "ingest.mode.auto-incomplete",
        "auto with incomplete Logs Ingestion settings: the 2.6.2 Data Collector "
        "request (signature, headers, record), one WARNING naming the missing "
        "valve (once per process)",
        not problems and not posts and once,
        f"problems={problems} ingest={len(posts)} warnings: {warnings}",
    )

    await la.reset()
    await set_li_valves(
        t,
        LOG_ANALYTICS_AUTH_MODE="managed_identity",
        LOG_ANALYTICS_DCR_ENDPOINT="",
        LOG_ANALYTICS_DCR_IMMUTABLE_ID="",
        LOG_ANALYTICS_TENANT_ID="",
        LOG_ANALYTICS_CLIENT_ID="",
        LOG_ANALYTICS_CLIENT_SECRET="",
    )
    r, _ = await li_chat(t, "auto, only LOG_ANALYTICS_AUTH_MODE is set")
    dc = await la.wait(1, timeout=10)
    await asyncio.sleep(1)
    other = [e.get("kind") for e in await la.records("all") if e.get("kind") != "dc"]
    problems = (
        record_problems(
            dc[0], expected(probe_report(r.content), r.content), PROBE_MODEL, user_id
        )
        if len(dc) == 1
        else [f"dc posts={len(dc)}"]
    )
    once, warnings = once_per_process(
        t,
        "not fully configured (missing: LOG_ANALYTICS_DCR_ENDPOINT, "
        "LOG_ANALYTICS_DCR_IMMUTABLE_ID); using the HTTP Data Collector API",
    )
    t.check(
        "ingest.mode.auto-partial-auth",
        "auto with LOG_ANALYTICS_AUTH_MODE=managed_identity as the only Logs "
        "Ingestion setting: the 2.6.2 Data Collector request, one WARNING naming "
        "the missing endpoint and DCR ID (once per process)",
        r.status == 200 and not problems and not other and once,
        f"HTTP {r.status} problems={problems} other_requests={other} "
        f"warnings: {warnings}",
    )

    await la.reset()
    await set_li_valves(t, LOG_ANALYTICS_INGESTION_API="data_collector")
    r, _ = await li_chat(t, "data_collector forced")
    dc = await la.wait(1, timeout=10)
    await asyncio.sleep(1)
    posts = await la.records("ingest")
    toks = await la.records("token")
    problems = (
        record_problems(
            dc[0], expected(probe_report(r.content), r.content), PROBE_MODEL, user_id
        )
        if len(dc) == 1
        else [f"dc posts={len(dc)}"]
    )
    t.check(
        "ingest.mode.data-collector",
        "data_collector with complete Logs Ingestion settings: only the 2.6.2 "
        "Data Collector request, no token request",
        r.status == 200 and not problems and not posts and not toks,
        f"HTTP {r.status} problems={problems} ingest={len(posts)} "
        f"token_requests={len(toks)}",
    )

    await la.reset()
    await set_li_valves(
        t, LOG_ANALYTICS_INGESTION_API="logs_ingestion", LOG_ANALYTICS_DCR_ENDPOINT=""
    )
    statuses = []
    for i in range(2):
        r, _ = await li_chat(t, f"logs_ingestion incomplete {i}")
        statuses.append(r.status)
    await asyncio.sleep(2)
    everything = await la.records("all")
    once, warnings = once_per_process(
        t,
        "LOG_ANALYTICS_INGESTION_API=logs_ingestion, but the Logs Ingestion API is "
        "not fully configured (missing: LOG_ANALYTICS_DCR_ENDPOINT)",
    )
    t.check(
        "ingest.mode.logs-ingestion-incomplete",
        "logs_ingestion without LOG_ANALYTICS_DCR_ENDPOINT: nothing sent, one "
        "WARNING naming the valve (once per process)",
        statuses == [200, 200] and not everything and once,
        f"HTTP {statuses} requests={[e.get('kind') for e in everything]} "
        f"warnings: {warnings}",
    )

    await la.reset()
    await set_li_valves(t, LOG_ANALYTICS_INGESTION_API="both")
    mark = t.mark()
    r, _ = await li_chat(t, "both APIs please")
    rep = probe_report(r.content)
    dc = await la.wait(1, timeout=10)
    posts = await la.wait(1, timeout=10, kind="ingest")
    sent_dc = await wait_log(t, mark, "Log Analytics data sent successfully", 5)
    sent_li = await wait_log(t, mark, INGEST_SENT, 5)
    problems = []
    if len(dc) != 1:
        problems.append(f"dc posts={len(dc)}")
    else:
        problems += record_problems(
            dc[0], expected(rep, r.content), PROBE_MODEL, user_id
        )
    if len(posts) != 1:
        problems.append(f"ingest posts={len(posts)}")
    else:
        problems += li_transport_problems(posts[0])
    if dc and posts and dc[0].get("body") != posts[0].get("body"):
        problems.append("dc body != ingest body")
    if len(sent_dc) != 1 or len(sent_li) != 1:
        problems.append(f"success lines dc={len(sent_dc)} li={len(sent_li)}")
    t.check(
        "ingest.mode.both",
        "both: the same record to the Data Collector API (2.6.2 request) and the "
        "Logs Ingestion API, one success line each",
        r.status == 200 and not problems,
        f"HTTP {r.status} problems={problems}",
    )

    # both, one side incomplete: the record still goes through the other one.
    await la.reset()
    await set_li_valves(
        t, LOG_ANALYTICS_INGESTION_API="both", LOG_ANALYTICS_CLIENT_SECRET=""
    )
    r, _ = await li_chat(t, "both, Logs Ingestion settings incomplete")
    dc = await la.wait(1, timeout=10)
    await asyncio.sleep(1)
    other = [e.get("kind") for e in await la.records("all") if e.get("kind") != "dc"]
    problems = (
        record_problems(
            dc[0], expected(probe_report(r.content), r.content), PROBE_MODEL, user_id
        )
        if len(dc) == 1
        else [f"dc posts={len(dc)}"]
    )
    problems += [f"Logs Ingestion requests={other}"] if other else []
    once_li, warnings_li = once_per_process(
        t,
        "LOG_ANALYTICS_INGESTION_API=both, but the Logs Ingestion API is not fully "
        "configured (missing: LOG_ANALYTICS_CLIENT_SECRET); sending only through "
        "the other one",
    )
    await la.reset()
    await set_li_valves(t, LOG_ANALYTICS_INGESTION_API="both")
    try:
        await t.owui.update_valves(TRACKER, LOG_ANALYTICS_WORKSPACE_ID="")
        r2, _ = await li_chat(t, "both, Data Collector settings incomplete")
        posts = await la.wait(1, timeout=10, kind="ingest")
        await asyncio.sleep(1)
        dc = await la.records()
    finally:
        await t.owui.update_valves(TRACKER, LOG_ANALYTICS_WORKSPACE_ID=WORKSPACE)
    problems += (
        li_transport_problems(posts[0])
        + record_body_problems(
            posts[0],
            expected(probe_report(r2.content), r2.content),
            PROBE_MODEL,
            user_id,
        )
        if len(posts) == 1
        else [f"ingest posts={len(posts)}"]
    )
    problems += [f"dc posts={len(dc)}"] if dc else []
    once_dc, warnings_dc = once_per_process(
        t,
        "LOG_ANALYTICS_INGESTION_API=both, but the HTTP Data Collector API is not "
        "fully configured (missing: LOG_ANALYTICS_WORKSPACE_ID); sending only "
        "through the other one",
    )
    t.check(
        "ingest.mode.both-one-side",
        "both with one API incomplete: the record goes through the other one "
        "(Logs Ingestion incomplete: the 2.6.2 Data Collector request; Data "
        "Collector incomplete: Logs Ingestion 204), one WARNING each (once per "
        "process)",
        r.status == 200 and r2.status == 200 and not problems and once_li and once_dc,
        f"HTTP {r.status}/{r2.status} problems={problems} warnings: "
        f"Logs Ingestion side {warnings_li}; Data Collector side {warnings_dc}",
    )

    hexa = uuid.uuid4().hex[:4]
    endpoint_a = f"http://{DCR_HOST}"
    dcr_c = f"my-dcr-{hexa}"
    mark_a = t.mark()
    seen = {}
    for name, over in (
        ("a", {"LOG_ANALYTICS_DCR_ENDPOINT": endpoint_a}),
        ("b", {"LOG_ANALYTICS_AUTH_MODE": f"bogus-{hexa}"}),
        ("c", {"LOG_ANALYTICS_DCR_IMMUTABLE_ID": dcr_c}),
        # The client secret must never go out over plain HTTP.
        ("d", {"LOG_ANALYTICS_AUTHORITY_HOST": "http://login.microsoftonline.com"}),
    ):
        await la.reset()
        await set_li_valves(t, LOG_ANALYTICS_INGESTION_API="logs_ingestion", **over)
        mark = t.mark()
        # c: two records with the same value, the hint is logged once
        for i in range(2 if name == "c" else 1):
            r, _ = await li_chat(t, f"invalid Logs Ingestion setting {name} {i}")
            if name == "c":
                await wait_log(t, mark, f"{INGEST_ERROR}: 404", 10, n=i + 1)
        await asyncio.sleep(1.5)
        seen[name] = (r.status, [e.get("kind") for e in await la.records("all")])
    t.expect_errors(mark_a, ("time_token_tracker", f"{INGEST_ERROR}: 404"))
    once_https, https = once_per_process(
        t, "LOG_ANALYTICS_DCR_ENDPOINT (must use https)"
    )
    once_authority, authority = once_per_process(
        t, "LOG_ANALYTICS_AUTHORITY_HOST (must use https)"
    )
    auth = t.log.lines(mark_a, "LOG_ANALYTICS_AUTH_MODE (unknown value")
    dcr_hint = t.log.lines(mark_a, "does not look like an immutable ID")
    token_errors = t.log.lines(mark_a, TOKEN_ERROR)
    # The DCR ID is no secret: the 404 hint names it on purpose.
    shown = [
        line
        for line in t.log.lines(mark_a, endpoint_a, dcr_c)
        if f"{INGEST_ERROR}: 404" not in line
    ]
    t.check(
        "ingest.config-invalid",
        "invalid settings: an http:// endpoint, an http:// authority and an "
        "unknown auth mode count as missing (WARNING, nothing sent, no token "
        "request); a DCR ID that is not dcr-<32 hex> logs a hint once per value "
        "and is still used; the warnings do not show the values",
        seen.get("a") == (200, [])
        and seen.get("b") == (200, [])
        and seen.get("c") == (200, ["token", "ingest", "ingest"])
        and seen.get("d") == (200, [])
        and once_https
        and once_authority
        and len(auth) == 1
        and len(dcr_hint) == 1
        and not token_errors
        and not shown,
        f"requests={seen} https warnings: endpoint {https}, authority {authority}; "
        f"auth={auth[:1]} dcr_hint={len(dcr_hint)} token_errors={token_errors[:1]} "
        f"values_shown={shown[:1]}",
    )

    await la.reset()
    await set_li_valves(t, LOG_ANALYTICS_INGESTION_API=f" BoGuS-{hexa} ")
    mark = t.mark()
    for i in range(2):
        await li_chat(t, f"unknown mode {i}")
    posts = await la.wait(2, timeout=10, kind="ingest")
    await asyncio.sleep(1)
    dc = await la.records()
    lines = t.log.lines(mark, "unknown LOG_ANALYTICS_INGESTION_API value")
    t.check(
        "ingest.mode.unknown",
        "an unknown LOG_ANALYTICS_INGESTION_API value: one WARNING, behaves as auto",
        len(lines) == 1
        and len(posts) == 2
        and all(p.get("status") == 204 for p in posts)
        and not dc,
        f"warnings={lines[:2]} ingest={[p.get('status') for p in posts]} dc={len(dc)}",
    )

    once, warnings = once_per_process(t, DEPRECATION)
    t.check(
        "ingest.deprecation-warning",
        "sending through the HTTP Data Collector API logs one deprecation WARNING "
        "per process",
        once,
        f"{warnings}: {t.log.lines(0, DEPRECATION)[-1:]}",
    )


async def ingest_clouds(t: Suite, la: LogAnalytics) -> None:
    """Sovereign cloud valves and endpoint normalization."""
    await la.reset()
    scope = "https://monitor.azure.us/.default"
    valves = await set_li_valves(
        t,
        LOG_ANALYTICS_AUTHORITY_HOST="login.microsoftonline.us/",
        LOG_ANALYTICS_INGESTION_SCOPE=scope,
        LOG_ANALYTICS_DCR_ENDPOINT=f"https://{GOV_DCR_HOST}",
    )
    r, _ = await li_chat(t, "US Government cloud")
    posts = await la.wait(1, timeout=10, kind="ingest")
    toks = await la.records("token")
    problems = token_problems(
        toks, [valves["LOG_ANALYTICS_CLIENT_ID"]], host=LOGIN_HOSTS[1], scope=scope
    ) + (
        li_transport_problems(posts[0], host=GOV_DCR_HOST)
        if len(posts) == 1
        else [f"ingest posts={len(posts)}"]
    )
    t.check(
        "ingest.sovereign",
        "US Government valves (authority without scheme and with a trailing "
        "slash): token from login.microsoftonline.us with the .us scope, record "
        "to the .azure.us endpoint, 204",
        r.status == 200 and not problems,
        f"HTTP {r.status} problems={problems}",
    )

    stream = "Custom-E2E_Explicit"
    await la.reset()
    await la.config(stream=stream)
    try:
        await set_li_valves(
            t,
            LOG_ANALYTICS_DCR_ENDPOINT=f"{DCR_HOST}/",
            LOG_ANALYTICS_DCR_STREAM_NAME=stream,
        )
        r, _ = await li_chat(t, "endpoint without scheme, explicit stream")
        posts = await la.wait(1, timeout=10, kind="ingest")
    finally:
        await la.config(stream=STREAM)
    problems = (
        li_transport_problems(posts[0], stream=stream)
        if len(posts) == 1
        else [f"ingest posts={len(posts)}"]
    )
    t.check(
        "ingest.endpoint-normalized",
        "endpoint without https:// and with a trailing slash, explicit "
        "LOG_ANALYTICS_DCR_STREAM_NAME: exact URI, 204",
        r.status == 200 and not problems,
        f"HTTP {r.status} problems={problems}",
    )


async def ingest_managed_identity(t: Suite, la: LogAnalytics, state: dict) -> None:
    """App Service, IMDS (without proxy), AKS workload identity, unsupported."""
    env_problems = list(state.get("env_problems") or [])
    mi_valves = {
        "LOG_ANALYTICS_AUTH_MODE": "managed_identity",
        "LOG_ANALYTICS_CLIENT_SECRET": "",
    }

    header_a = f"e2e-idh-{uuid.uuid4().hex}"
    header_b = f"e2e-idh-{uuid.uuid4().hex}"
    state["secrets"] += [header_a, header_b]
    client_id = fresh_client_id("e2e-uami")
    env = {"IDENTITY_ENDPOINT": f"{MI_URL}/msi/token", "IDENTITY_HEADER": header_a}
    await la.reset()
    await la.config(identity_header=header_a)
    await set_li_valves(t, LOG_ANALYTICS_CLIENT_ID=client_id, **mi_valves)
    await la.mode(target="msi", expires_in=8)
    problems = env_problems + await set_env(t, env)
    try:
        await li_chat(t, "App Service managed identity 1")
        posts = await la.wait(1, timeout=10, kind="ingest")
        anchor = posts[0]["received"] if posts else time.time()
        problems += await set_env(t, {**env, "IDENTITY_HEADER": header_b})
        await la.config(identity_header=header_b)
        await asyncio.sleep(max(0.0, anchor + 5 - time.time()))
        await li_chat(t, "App Service managed identity 2")
        posts = await la.wait(2, timeout=10, kind="ingest")
    finally:
        problems += await set_env(t)
        await la.config(identity_header="")
    msi = await la.records("msi")
    toks = await la.records("token")
    for rec in msi:
        query = rec.get("query") or {}
        want = {
            "api-version": "2019-08-01",
            "resource": "https://monitor.azure.com",
            "client_id": client_id,
        }
        problems += [
            f"{k}={query.get(k)!r}" for k, v in want.items() if query.get(k) != v
        ]
        if rec.get("flavour") != "app_service" or not rec.get("header_ok"):
            problems.append(
                f"flavour={rec.get('flavour')} header_ok={rec.get('header_ok')}"
            )
    if len(msi) != 2:
        problems.append(f"msi requests={len(msi)}")
    problems += [p for e in posts for p in li_transport_problems(e)]
    t.check(
        "ingest.mi.app-service",
        "managed identity on App Service: IDENTITY_ENDPOINT with X-IDENTITY-HEADER "
        "(read per request: the rotated header is used), api-version 2019-08-01, "
        "resource without /.default, user-assigned client_id; 2 records with 204",
        len(posts) == 2 and not toks and not problems,
        f"problems={problems} posts={len(posts)} token_route={len(toks)}",
    )

    # App Service / Container Apps answer errors as {statusCode, message,
    # correlationId}, not {error, error_description}.
    header_c = f"e2e-idh-{uuid.uuid4().hex}"
    state["secrets"].append(header_c)
    env = {"IDENTITY_ENDPOINT": f"{MI_URL}/msi/token", "IDENTITY_HEADER": header_c}
    await la.reset()
    await la.config(identity_header=header_c)
    await set_li_valves(
        t, LOG_ANALYTICS_CLIENT_ID=fresh_client_id("e2e-unknown-uami"), **mi_valves
    )
    problems = env_problems + await set_env(t, env)
    mark = t.mark()
    try:
        await li_chat(t, "App Service managed identity, client ID not assigned")
        await wait_log(t, mark, TOKEN_ERROR, 10)
        await t.log.settle(1)
    finally:
        problems += await set_env(t)
        await la.config(identity_header="")
    blocks = [b for b in t.log.error_blocks(mark) if TOKEN_ERROR in b]
    msi = await la.records("msi")
    posts = await la.records("ingest")
    t.expect_errors(mark, ("time_token_tracker", TOKEN_ERROR))
    correlation_id = msi[0].get("correlation_id") if len(msi) == 1 else None
    needles = (
        "(managed identity (App Service)): 400 - check LOG_ANALYTICS_CLIENT_ID",
        "Detail: Unable to load the proper Managed Identity.",
        f"(correlationId={correlation_id})",
    )
    t.check(
        "ingest.mi.app-service-error",
        "the App Service / Container Apps token service answers 400 {statusCode, "
        "message, correlationId} (client ID not assigned to the app): one ERROR "
        "with the LOG_ANALYTICS_CLIENT_ID hint, the service's message and its "
        "correlationId; nothing sent",
        bool(correlation_id)
        and len(blocks) == 1
        and all(n in blocks[0] for n in needles)
        and not posts
        and not problems,
        f"errors={[b[:400] for b in blocks[:2]]} msi={len(msi)} posts={len(posts)} "
        f"problems={problems}",
    )

    await la.reset()
    await set_li_valves(t, LOG_ANALYTICS_CLIENT_ID="", **mi_valves)
    await la.mode(target="msi", expires_in=8)  # no stale cache in a --reuse rerun
    problems = list(env_problems)
    problems += await set_env(
        t, {"AZURE_POD_IDENTITY_AUTHORITY_HOST": MI_IMDS_URL, **PROXY_ENV}
    )
    try:
        r, _ = await li_chat(t, "IMDS managed identity")
        posts = await la.wait(1, timeout=10, kind="ingest")
    finally:
        problems += await set_env(t)
    premise = await asyncio.to_thread(
        subprocess.run,
        [
            "python3",
            "-c",
            PROXY_PREMISE,
            f"{MI_IMDS_URL}/metadata/identity/oauth2/token?api-version=2018-02-01"
            "&resource=https://monitor.azure.com",
        ],
        env={**os.environ, **PROXY_ENV},
        capture_output=True,
        text=True,
        timeout=30,
    )
    premise_out = (premise.stdout or premise.stderr).strip()
    msi = await la.records("msi")
    for rec in msi:
        query = rec.get("query") or {}
        if (
            rec.get("flavour") != "imds"
            or rec.get("local_ip") != "127.0.0.2"
            or not rec.get("header_ok")
            or query.get("api-version") != "2018-02-01"
            or query.get("resource") != "https://monitor.azure.com"
            or "client_id" in query
        ):
            problems.append(f"msi={short(rec, 200)}")
    if len(msi) != 1:
        problems.append(f"msi requests={len(msi)}")
    problems += (
        li_transport_problems(posts[0]) if len(posts) == 1 else [f"posts={len(posts)}"]
    )
    t.check(
        "ingest.mi.imds",
        "managed identity on a VM (IMDS): Metadata: true, api-version 2018-02-01, "
        "system-assigned (no client_id), reached directly although HTTP_PROXY is "
        "set (a trust_env request through the same environment fails at the "
        "proxy); record with 204",
        r.status == 200 and not problems and premise_out.startswith("ERROR Client"),
        f"problems={problems} premise={premise_out[:120]!r}",
    )

    hex_a, hex_b = uuid.uuid4().hex, uuid.uuid4().hex
    assertion_a, assertion_b = f"e2e-fed-{hex_a}", f"e2e-fed-{hex_b}"
    state["secrets"] += [assertion_a, assertion_b]
    client_1, client_2 = fresh_client_id("e2e-wi"), fresh_client_id("e2e-wi")
    env = {
        "AZURE_FEDERATED_TOKEN_FILE": WI_TOKEN_FILE,
        "AZURE_TENANT_ID": TENANT,
        "AZURE_CLIENT_ID": client_1,
    }
    with open(WI_TOKEN_FILE, "w", encoding="utf-8") as fh:
        fh.write(assertion_a + "\n")
    await la.reset()
    await set_li_valves(
        t, LOG_ANALYTICS_TENANT_ID="", LOG_ANALYTICS_CLIENT_ID="", **mi_valves
    )
    await la.mode(target="token", expires_in=8)
    problems = env_problems + await set_env(t, env)
    try:
        await li_chat(t, "workload identity 1")
        posts = await la.wait(1, timeout=10, kind="ingest")
        anchor = posts[0]["received"] if posts else time.time()
        with open(WI_TOKEN_FILE, "w", encoding="utf-8") as fh:
            fh.write(assertion_b + "\n")
        await asyncio.sleep(max(0.0, anchor + 5 - time.time()))
        await li_chat(t, "workload identity 2")
        await la.wait(2, timeout=10, kind="ingest")
        problems += await set_env(t, {**env, "AZURE_CLIENT_ID": client_2})
        await li_chat(t, "workload identity 3")
        posts = await la.wait(3, timeout=10, kind="ingest")
    finally:
        problems += await set_env(t)
    toks = await la.records("token")
    problems += token_problems(toks, [client_1, client_1, client_2], auth="assertion")
    problems += [
        f"form={tok.get('form')}"
        for tok in toks
        if "client_secret" in (tok.get("form") or {})
        or (tok.get("form") or {}).get("client_assertion_type")
        != "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
    ]
    problems += (
        [p for e in posts for p in li_transport_problems(e)]
        if len(posts) == 3
        else [f"posts={len(posts)}"]
    )
    t.check(
        "ingest.mi.workload-identity",
        "AKS workload identity: client assertion from AZURE_FEDERATED_TOKEN_FILE "
        "(read per token request: the rotated file is used), tenant and client id "
        "from AZURE_TENANT_ID / AZURE_CLIENT_ID (a changed AZURE_CLIENT_ID gets its "
        "own token), no client secret; 3 records with 204",
        not problems,
        f"problems={problems}",
    )

    await la.reset()
    await set_li_valves(t, **mi_valves)
    problems = env_problems + await set_env(
        t, {"IDENTITY_ENDPOINT": f"{MI_URL}/arc", "IMDS_ENDPOINT": MI_URL}
    )
    mark = t.mark()
    try:
        await li_chat(t, "Azure Arc managed identity")
        lines = await wait_log(t, mark, "Azure Arc is not supported", 8)
        await t.log.settle(1)
    finally:
        problems += await set_env(t)
    lines = t.log.lines(mark, "Azure Arc is not supported")
    msi = await la.records("msi")
    posts = await la.records("ingest")
    t.expect_errors(mark, ("time_token_tracker", TOKEN_ERROR))
    t.check(
        "ingest.mi.unsupported",
        "managed identity on Azure Arc: one ERROR 'not supported', no request",
        len(lines) == 1 and not msi and not posts and not problems,
        f"lines={lines[:1]} msi={len(msi)} posts={len(posts)} problems={problems}",
    )


async def ingest_retry_after_window(t: Suite, la: LogAnalytics, retry: dict) -> None:
    """The 30 s token back-off of ingest.token-error ends; the fixed secret
    works without a restart (the secret is not part of the cache key)."""
    await la.reset()
    await set_li_valves(t, LOG_ANALYTICS_CLIENT_ID=retry["client_id"])
    waited = max(0.0, retry["failed_at"] + 31 - time.time())
    await asyncio.sleep(waited)
    r, _ = await li_chat(t, "after the token back-off")
    posts = await la.wait(1, timeout=10, kind="ingest")
    toks = await la.records("token")
    problems = token_problems(toks, [retry["client_id"]]) + (
        [p.get("status") for p in posts if p.get("status") != 204]
        if len(posts) == 1
        else [f"posts={len(posts)}"]
    )
    t.check(
        "ingest.token-retry-after-window",
        "31 s after the failed token request, the same identity with the fixed "
        "secret gets exactly one new token and its record is sent (204)",
        r.status == 200 and not problems,
        f"waited={waited:.1f}s problems={problems}",
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
