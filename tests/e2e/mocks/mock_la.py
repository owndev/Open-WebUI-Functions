"""
Azure Log Analytics mock for the filters suite: the HTTP Data Collector API, the
Logs Ingestion API (data collection rules), the Microsoft Entra ID token
endpoint and the managed identity endpoints.

``filters/time_token_tracker.py`` posts to
``https://<workspace>.ods.opinsights.azure.com/api/logs?api-version=2016-04-01``
(Data Collector) or to
``https://<dcr endpoint>/dataCollectionRules/<dcr>/streams/<stream>?api-version=2023-01-01``
(Logs Ingestion) with a token from ``https://login.microsoftonline.<tld>`` or a
managed identity endpoint. The filters suite (tests/e2e/suites/filters.py) maps
those hosts to 127.0.0.1 in /etc/hosts, installs a throw-away test CA into the
container's system store and starts this mock; it is not part of serve_all.py
because it needs the certificate and port 443.

- HTTPS on 127.0.0.1:443 (one certificate for every mapped host):
  - ``POST /api/logs`` (record kind ``dc``), validated like Azure: SharedKey
    signature over "POST\\n<len>\\napplication/json\\nx-ms-date:<date>\\n/api/logs"
    (403 InvalidAuthorization), Content-Type exactly application/json (400
    UnsupportedContentType), Log-Type letters/digits/_ up to 100 characters (400
    InvalidLogType), api-version=2016-04-01 (400 InvalidApiVersion /
    MissingApiVersion).
  - ``POST /<tenant>/oauth2/v2.0/token`` (kind ``token``): form-encoded client
    credentials with a client secret or a federated assertion (the current
    content of the configured token file); Entra-shaped errors (``error``,
    ``error_description``, ``error_codes``, ``trace_id``, ...). Issues
    ``e2e-tok-<hex>`` tokens. The secret and the assertion are never recorded.
  - ``POST /dataCollectionRules/<dcr>/streams/<stream>`` (kind ``ingest``):
    api-version=2023-01-01, a known, unrevoked, unexpired Bearer token of the
    host's cloud (401), Content-Type application/json, configured DCR / stream
    (404), JSON array of objects (400), at most 1 MiB (413); 204 on success.
- Managed identity on http://127.0.0.1:9106 and http://127.0.0.2:9106 (kind
  ``msi``): ``GET /msi/token`` (App Service, ``X-IDENTITY-HEADER``; a
  ``client_id`` starting with ``e2e-unknown`` gets App Service's 400
  ``{statusCode, message, correlationId}``) and
  ``GET /metadata/identity/oauth2/token`` (IMDS, ``Metadata: true``).
- Tarpit on 127.0.0.3:443: accepts TLS connections, waits 8 s and closes them
  (a hanging download for tiktoken's encoding host, offline group).
- Control on http://127.0.0.1:9105: ``GET /__requests[?kind=dc|token|ingest|msi|all]``
  (recorded requests, default ``dc``), ``POST /__reset`` (records, modes),
  ``POST /__mode`` ``{"target": "dc"|"token"|"ingest"|"msi", "status": 500,
  "delay": 6, "expires_in", "retry_after", "error_code", "echo_secret",
  "echo_at_cut"}`` (each
  call replaces the whole mode of its target; target defaults to dc),
  ``POST /__config`` ``{tenant, client_secret, dcr_id, stream,
  identity_header, token_file}``, ``GET /__tokens`` (issued access tokens),
  ``POST /__revoke`` (invalidate every issued token), ``GET /__tarpit``
  (connection times seen by the tarpit).

Every request is recorded before it is answered; stdout gets one line per
request without tokens, secrets, identity headers or assertions.

usage: python3 mock_la.py <workspace_id> <shared_key_b64> <certfile> <keyfile>
"""

import asyncio
import base64
import datetime
import hashlib
import hmac
import json
import re
import secrets
import ssl
import sys
import time
import uuid
from urllib.parse import parse_qs, quote_plus

from aiohttp import web

CONTROL_PORT = 9105
MSI_PORT = 9106
MSI_HOSTS = ("127.0.0.1", "127.0.0.2")
TARPIT_HOST = "127.0.0.3"
TARPIT_SECONDS = 8
API_VERSION = "2016-04-01"
INGEST_API_VERSION = "2023-01-01"
MAX_BODY = 1024 * 1024
JWT_BEARER = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
MSI_RESOURCES = ("https://monitor.azure.com", "https://monitor.azure.com/")
TARGETS = ("dc", "token", "ingest", "msi")
LOGIN_CLOUDS = {"login.microsoftonline.com": "com", "login.microsoftonline.us": "us"}


def default_mode() -> dict:
    return {
        "status": 200,
        "delay": 0.0,
        "expires_in": 3599,
        "retry_after": None,
        "error_code": None,
        "echo_secret": False,
        "echo_at_cut": False,
    }


STATE = {
    "records": [],
    "modes": {target: default_mode() for target in TARGETS},
    "tarpit": [],
    "config": {
        "tenant": "",
        "client_secret": "",
        "dcr_id": "",
        "stream": "",
        "identity_header": "",
        "token_file": "",
    },
    # access token -> {"index", "expires" (monotonic), "cloud", "revoked"}
    "tokens": {},
}


def expected_auth(workspace: str, key: str, raw: bytes, date: str) -> str:
    """Authorization header Azure expects for this body and x-ms-date."""
    to_sign = f"POST\n{len(raw)}\napplication/json\nx-ms-date:{date}\n/api/logs"
    digest = hmac.new(base64.b64decode(key), to_sign.encode("utf-8"), hashlib.sha256)
    return f"SharedKey {workspace}:{base64.b64encode(digest.digest()).decode()}"


def mode(target: str) -> dict:
    return STATE["modes"][target]


def issue_token(cloud: str) -> tuple:
    """New access token of ``cloud`` with the target mode's lifetime."""
    token = f"e2e-tok-{secrets.token_hex(16)}"
    index = len(STATE["tokens"])
    STATE["tokens"][token] = {
        "index": index,
        "expires": time.monotonic(),  # set by the caller
        "cloud": cloud,
        "revoked": False,
    }
    return token, index


def bare_host(request: web.Request) -> str:
    return (request.host or "").split(":")[0].lower()


def entra_error(
    host: str,
    status: int,
    error: str,
    code: int,
    text: str,
    echo: str = "",
    at_cut: bool = False,
) -> web.Response:
    """An error response shaped like Microsoft Entra ID's token endpoint.

    ``echo``: the secret / assertion is echoed in the first line of
    error_description (plain and form-encoded). With ``at_cut`` it straddles
    the filter's cut points instead: ``error`` (100 characters), the first
    line of ``error_description`` (200), ``trace_id`` and ``correlation_id``
    (64 each), each with the first 10 characters of the secret (plain or
    form-encoded) before the cut. A filter that truncates before it redacts
    logs those characters.
    """
    first = f"AADSTS{code}: {text} e2e mock"
    trace_id, correlation_id = str(uuid.uuid4()), str(uuid.uuid4())
    if echo and at_cut:
        head = f"AADSTS{code}: "
        error = "e" * 90 + echo
        first = head + "p" * (190 - len(head)) + quote_plus(echo)
        trace_id = "t" * 54 + echo
        correlation_id = "c" * 54 + quote_plus(echo)
    elif echo:
        first = f"AADSTS{code}: echo {echo} {quote_plus(echo)} e2e mock"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    body = {
        "error": error,
        "error_description": (
            f"{first}\r\nTrace ID: {trace_id}\r\nCorrelation ID: {correlation_id}"
            f"\r\nTimestamp: {stamp}"
        ),
        "error_codes": [code],
        "timestamp": stamp,
        "trace_id": trace_id,
        "correlation_id": correlation_id,
        "error_uri": f"https://{host}/error?code={code}",
    }
    return web.json_response(body, status=status)


def make_api(workspace: str, key: str) -> web.Application:
    async def logs(request: web.Request) -> web.Response:
        raw = await request.read()
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        date = request.headers.get("x-ms-date", "")
        auth_ok = request.headers.get("Authorization") == expected_auth(
            workspace, key, raw, date
        )
        log_type = request.headers.get("Log-Type", "")
        STATE["records"].append(
            {
                "kind": "dc",
                "method": request.method,
                "host": request.host,
                "path": request.path,
                "query": dict(request.query),
                "headers": dict(request.headers),
                "raw_len": len(raw),
                "body": body,
                "auth_ok": auth_ok,
                "received": time.time(),
            }
        )
        print(
            f"LA POST {request.path} auth_ok={auth_ok} body={raw[:300]!r}", flush=True
        )
        dc = mode("dc")
        if dc["delay"]:
            await asyncio.sleep(dc["delay"])
        error = None
        api_version = request.query.get("api-version")
        if api_version is None:
            error = (400, "MissingApiVersion")
        elif api_version != API_VERSION:
            error = (400, "InvalidApiVersion")
        elif request.headers.get("Content-Type") != "application/json":
            error = (400, "UnsupportedContentType")
        elif not log_type:
            error = (400, "MissingLogType")
        elif not re.fullmatch(r"[A-Za-z0-9_]{1,100}", log_type):
            error = (400, "InvalidLogType")
        elif not auth_ok:
            error = (403, "InvalidAuthorization")
        elif body is None:
            error = (400, "InvalidDataFormat")
        elif dc["status"] != 200:
            error = (dc["status"], "E2EForced")
        if error:
            return web.json_response(
                {"Error": error[1], "Message": "e2e mock"}, status=error[0]
            )
        return web.Response(status=200)

    async def token(request: web.Request) -> web.Response:
        """Microsoft Entra ID client credentials (secret or federated assertion)."""
        raw = await request.read()
        host = bare_host(request)
        content_type = request.headers.get("Content-Type", "")
        form = {}
        if content_type.split(";")[0].strip() == "application/x-www-form-urlencoded":
            form = {
                k: v[0]
                for k, v in parse_qs(
                    raw.decode("utf-8", "replace"), keep_blank_values=True
                ).items()
            }
        cfg = STATE["config"]
        secret = form.get("client_secret")
        assertion = form.get("client_assertion")
        file_content = ""
        if cfg["token_file"]:
            try:
                with open(cfg["token_file"], encoding="utf-8") as fh:
                    file_content = fh.read().strip()
            except OSError:
                file_content = ""
        rec = {
            "kind": "token",
            "method": request.method,
            "host": host,
            "path": request.path,
            "tenant": request.match_info["tenant"],
            "content_type": content_type,
            "form": {
                k: ("<redacted>" if k in ("client_secret", "client_assertion") else v)
                for k, v in form.items()
            },
            "grant_type": form.get("grant_type"),
            "client_id": form.get("client_id"),
            "scope": form.get("scope"),
            "auth": "secret"
            if secret is not None
            else ("assertion" if assertion is not None else None),
            "secret_ok": secret is not None
            and bool(cfg["client_secret"])
            and secret == cfg["client_secret"],
            "assertion_ok": assertion is not None
            and form.get("client_assertion_type") == JWT_BEARER
            and bool(file_content)
            and assertion == file_content,
            "token_index": None,
            "status": None,
            "received": time.time(),
        }
        STATE["records"].append(rec)
        m = mode("token")
        if m["delay"]:
            await asyncio.sleep(m["delay"])
        echo = (secret or assertion or "") if m["echo_secret"] else ""
        cloud = LOGIN_CLOUDS.get(host, "")
        scopes = (
            f"https://monitor.azure.{cloud}/.default",
            f"https://monitor.azure.{cloud}//.default",
        )

        def error(status: int, name: str, code: int, text: str) -> web.Response:
            return entra_error(host, status, name, code, text, echo, m["echo_at_cut"])

        if rec["content_type"].split(";")[0].strip() != (
            "application/x-www-form-urlencoded"
        ):
            resp = error(400, "invalid_request", 900144, "no form")
        elif form.get("grant_type") != "client_credentials":
            resp = error(400, "unsupported_grant_type", 70003, "grant type")
        elif rec["tenant"] != cfg["tenant"]:
            resp = error(400, "invalid_request", 90002, "Tenant not found.")
        elif not cloud or form.get("scope") not in scopes:
            resp = error(400, "invalid_scope", 70011, "The provided scope is invalid.")
        elif str(form.get("client_id") or "").startswith("e2e-unknown"):
            resp = error(
                400,
                "unauthorized_client",
                700016,
                "Application not found in the directory.",
            )
        elif rec["auth"] == "secret" and not rec["secret_ok"]:
            resp = error(
                401, "invalid_client", 7000215, "Invalid client secret provided."
            )
        elif rec["auth"] == "assertion" and not rec["assertion_ok"]:
            resp = error(
                401,
                "invalid_client",
                700212,
                "No matching federated identity record found.",
            )
        elif rec["auth"] is None:
            resp = error(401, "invalid_client", 7000216, "Client credential missing.")
        elif m["status"] != 200:
            resp = error(
                m["status"],
                "temporarily_unavailable" if m["status"] >= 500 else "invalid_request",
                int(m["error_code"] or 90000),
                "Forced by the e2e mock.",
            )
        else:
            access, index = issue_token(cloud)
            lifetime = int(m["expires_in"])
            STATE["tokens"][access]["expires"] = time.monotonic() + lifetime
            rec["token_index"] = index
            resp = web.json_response(
                {
                    "token_type": "Bearer",
                    "expires_in": lifetime,
                    "ext_expires_in": lifetime,
                    "access_token": access,
                }
            )
        rec["status"] = resp.status
        print(
            f"LA TOKEN {host} tenant={rec['tenant']} client_id={rec['client_id']} "
            f"auth={rec['auth']} -> {resp.status} "
            f"token#{rec['token_index']}",
            flush=True,
        )
        return resp

    async def ingest(request: web.Request) -> web.Response:
        """Logs Ingestion API: POST a JSON array to a DCR stream."""
        raw = await request.read()
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        host = bare_host(request)
        auth = request.headers.get("Authorization", "")
        bearer = auth[len("Bearer ") :] if auth.startswith("Bearer ") else ""
        known = STATE["tokens"].get(bearer)
        headers = dict(request.headers)
        for name in list(headers):
            if name.lower() == "authorization":
                headers[name] = (
                    f"Bearer <token#{known['index']}>"
                    if known
                    else ("Bearer <unknown>" if bearer else "<not bearer>")
                )
        expired = bool(known) and time.monotonic() >= known["expires"]
        rec = {
            "kind": "ingest",
            "method": request.method,
            "host": host,
            "path": request.path,
            "query": dict(request.query),
            "headers": headers,
            "raw_len": len(raw),
            "body": body,
            "token_ok": bool(known) and not known["revoked"],
            "token_index": known["index"] if known else None,
            "token_expired": expired,
            "status": None,
            "received": time.time(),
        }
        STATE["records"].append(rec)
        m = mode("ingest")
        if m["delay"]:
            await asyncio.sleep(m["delay"])
        cfg = STATE["config"]
        cloud = "us" if host.endswith(".azure.us") else "com"
        extra = {}
        if request.query.get("api-version") != INGEST_API_VERSION:
            error = (400, "InvalidApiVersion")
        elif not known or known["revoked"]:
            error = (401, "InvalidToken")
        elif expired:
            error = (401, "TokenExpired")
        elif known["cloud"] != cloud:
            error = (401, "InvalidAudience")
        elif request.headers.get("Content-Type") != "application/json":
            error = (400, "UnsupportedContentType")
        elif request.match_info["dcr"] != cfg["dcr_id"]:
            error = (404, "DcrNotFound")
        elif request.match_info["stream"] != cfg["stream"]:
            error = (404, "StreamNotFound")
        elif not (
            isinstance(body, list) and all(isinstance(item, dict) for item in body)
        ):
            error = (400, "InvalidJson")
        elif len(raw) > MAX_BODY:
            error = (413, "ContentLengthLimitExceeded")
        elif m["status"] != 200:
            error = (m["status"], m["error_code"] or "E2EForced")
            if m["retry_after"] is not None:
                extra["Retry-After"] = str(m["retry_after"])
        else:
            error = None
        if error:
            resp = web.json_response(
                {"error": {"code": error[1], "message": "e2e mock"}},
                status=error[0],
                headers={"x-ms-error-code": error[1], **extra},
            )
        else:
            resp = web.Response(status=204)
        rec["status"] = resp.status
        print(
            f"LA INGEST {host}{request.path} token#{rec['token_index']} "
            f"token_ok={rec['token_ok']} expired={expired} -> {resp.status} "
            f"body={raw[:200]!r}",
            flush=True,
        )
        return resp

    app = web.Application(client_max_size=4 * MAX_BODY)
    app.router.add_post("/api/logs", logs)
    app.router.add_post("/{tenant}/oauth2/v2.0/token", token)
    app.router.add_post("/dataCollectionRules/{dcr}/streams/{stream}", ingest)
    return app


def make_msi() -> web.Application:
    """Managed identity endpoints (App Service and IMDS), plain HTTP."""

    async def handle(request: web.Request, flavour: str) -> web.Response:
        sockname = (
            request.transport.get_extra_info("sockname") if request.transport else None
        )
        query = dict(request.query)
        cfg = STATE["config"]
        if flavour == "app_service":
            header_ok = (
                bool(cfg["identity_header"])
                and request.headers.get("X-IDENTITY-HEADER") == cfg["identity_header"]
            )
            version = "2019-08-01"
        elif flavour == "imds":
            header_ok = request.headers.get("Metadata") == "true"
            version = "2018-02-01"
        else:
            header_ok, version = False, ""
        rec = {
            "kind": "msi",
            "flavour": flavour,
            "method": request.method,
            "local_ip": sockname[0] if sockname else None,
            "path": request.path,
            "query": query,
            "client_id": query.get("client_id"),
            "header_ok": header_ok,
            "token_index": None,
            "status": None,
            "received": time.time(),
        }
        STATE["records"].append(rec)
        m = mode("msi")
        if m["delay"]:
            await asyncio.sleep(m["delay"])
        if flavour == "unknown":
            resp = web.json_response({"error": "not_found"}, status=404)
        elif (
            flavour == "app_service"
            and header_ok
            and str(query.get("client_id") or "").startswith("e2e-unknown")
        ):
            # What App Service / Container Apps answer for a client ID that is
            # not assigned to the app (microsoft/azure-container-apps#442).
            rec["correlation_id"] = str(uuid.uuid4())
            resp = web.json_response(
                {
                    "statusCode": 400,
                    "message": "Unable to load the proper Managed Identity.",
                    "correlationId": rec["correlation_id"],
                },
                status=400,
            )
        elif not header_ok:
            if flavour == "imds":
                resp = web.json_response(
                    {
                        "error": "invalid_request",
                        "error_description": "Required metadata header not specified",
                    },
                    status=400,
                )
            else:
                resp = web.json_response({"error": "invalid_request"}, status=401)
        elif query.get("api-version") != version:
            resp = web.json_response(
                {"error": "invalid_request", "error_description": "api-version"},
                status=400,
            )
        elif query.get("resource") not in MSI_RESOURCES:
            resp = web.json_response(
                {"error": "invalid_resource", "error_description": "resource"},
                status=400,
            )
        elif m["status"] != 200:
            resp = web.json_response(
                {"error": "e2e_forced", "error_description": "Forced by the e2e mock."},
                status=m["status"],
            )
        else:
            access, index = issue_token("com")
            lifetime = int(m["expires_in"])
            STATE["tokens"][access]["expires"] = time.monotonic() + lifetime
            rec["token_index"] = index
            expires_on = str(int(time.time()) + lifetime)
            if flavour == "app_service":
                payload = {
                    "access_token": access,
                    "expires_on": expires_on,
                    "resource": query.get("resource"),
                    "token_type": "Bearer",
                    "client_id": query.get("client_id") or "e2e-system-assigned",
                }
            else:
                payload = {
                    "access_token": access,
                    "refresh_token": "",
                    "expires_in": str(lifetime),
                    "expires_on": expires_on,
                    "not_before": str(int(time.time())),
                    "resource": query.get("resource"),
                    "token_type": "Bearer",
                }
            resp = web.json_response(payload)
        rec["status"] = resp.status
        print(
            f"LA MSI {flavour} {rec['local_ip']} {request.path} "
            f"client_id={rec['client_id']} header_ok={header_ok} -> {resp.status} "
            f"token#{rec['token_index']}",
            flush=True,
        )
        return resp

    async def app_service(request: web.Request) -> web.Response:
        return await handle(request, "app_service")

    async def imds(request: web.Request) -> web.Response:
        return await handle(request, "imds")

    async def unknown(request: web.Request) -> web.Response:
        return await handle(request, "unknown")

    app = web.Application()
    app.router.add_get("/msi/token", app_service)
    app.router.add_get("/metadata/identity/oauth2/token", imds)
    app.router.add_route("*", "/{tail:.*}", unknown)
    return app


def make_control() -> web.Application:
    async def requests(request: web.Request) -> web.Response:
        kind = request.query.get("kind", "dc")
        records = STATE["records"]
        if kind != "all":
            records = [r for r in records if r.get("kind") == kind]
        return web.json_response(records)

    async def reset(_: web.Request) -> web.Response:
        STATE["records"] = []
        STATE["modes"] = {target: default_mode() for target in TARGETS}
        return web.json_response({"ok": True})

    async def set_mode(request: web.Request) -> web.Response:
        data = await request.json()
        target = data.get("target", "dc")
        if target not in TARGETS:
            return web.json_response({"error": f"unknown target {target}"}, status=400)
        new = default_mode()
        new["status"] = int(data.get("status", 200))
        new["delay"] = float(data.get("delay", 0))
        new["expires_in"] = int(data.get("expires_in", 3599))
        new["retry_after"] = data.get("retry_after")
        new["error_code"] = data.get("error_code")
        new["echo_secret"] = bool(data.get("echo_secret", False))
        new["echo_at_cut"] = bool(data.get("echo_at_cut", False))
        STATE["modes"][target] = new
        return web.json_response({"ok": True})

    async def config(request: web.Request) -> web.Response:
        data = await request.json()
        for name in STATE["config"]:
            if name in data:
                STATE["config"][name] = str(data[name] or "")
        return web.json_response({"ok": True})

    async def tokens(_: web.Request) -> web.Response:
        return web.json_response(sorted(STATE["tokens"]))

    async def revoke(_: web.Request) -> web.Response:
        for info in STATE["tokens"].values():
            info["revoked"] = True
        return web.json_response({"ok": True})

    async def tarpit_log(_: web.Request) -> web.Response:
        return web.json_response(STATE["tarpit"])

    app = web.Application()
    app.router.add_get("/__requests", requests)
    app.router.add_post("/__reset", reset)
    app.router.add_post("/__mode", set_mode)
    app.router.add_post("/__config", config)
    app.router.add_get("/__tokens", tokens)
    app.router.add_post("/__revoke", revoke)
    app.router.add_get("/__tarpit", tarpit_log)
    return app


async def tarpit(_reader, writer) -> None:
    """Slow TLS peer: accept, say nothing for TARPIT_SECONDS, close."""
    STATE["tarpit"].append(time.time())
    await asyncio.sleep(TARPIT_SECONDS)
    writer.close()


async def main(workspace: str, key: str, certfile: str, keyfile: str) -> None:
    tls = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    tls.load_cert_chain(certfile, keyfile)
    msi = make_msi()
    sites = [
        (make_api(workspace, key), ("127.0.0.1",), 443, tls),
        (make_control(), ("127.0.0.1",), CONTROL_PORT, None),
        (msi, MSI_HOSTS, MSI_PORT, None),
    ]
    for app, hosts, port, context in sites:
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        for host in hosts:
            await web.TCPSite(runner, host, port, ssl_context=context).start()
    await asyncio.start_server(tarpit, TARPIT_HOST, 443)
    print("mock LA ready", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:5]))
