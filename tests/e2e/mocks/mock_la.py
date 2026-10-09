"""
Azure Monitor HTTP Data Collector API mock (Log Analytics) for the filters suite.

``filters/time_token_tracker.py`` posts to
``https://<workspace>.ods.opinsights.azure.com/api/logs?api-version=2016-04-01``.
The filters suite (tests/e2e/suites/filters.py) maps that host to 127.0.0.1 in
/etc/hosts, installs a throw-away test CA into the container's system store and
starts this mock; it is not part of serve_all.py because it needs the
certificate and port 443.

- HTTPS on 127.0.0.1:443: ``POST /api/logs``, validated like Azure: SharedKey
  signature over "POST\\n<len>\\napplication/json\\nx-ms-date:<date>\\n/api/logs"
  (403 InvalidAuthorization), Content-Type exactly application/json (400
  UnsupportedContentType), Log-Type letters/digits/_ up to 100 characters (400
  InvalidLogType), api-version=2016-04-01 (400 InvalidApiVersion /
  MissingApiVersion). Every POST is recorded before it is answered.
- Tarpit on 127.0.0.3:443: accepts TLS connections, waits 8 s and closes them
  (a hanging download for tiktoken's encoding host, offline group).
- Control on http://127.0.0.1:9105: ``GET /__requests`` (recorded POSTs),
  ``POST /__reset`` (records, mode), ``POST /__mode`` ``{"status": 500,
  "delay": 6}`` (answer status / seconds before answering), ``GET /__tarpit``
  (connection times seen by the tarpit).

usage: python3 mock_la.py <workspace_id> <shared_key_b64> <certfile> <keyfile>
"""

import asyncio
import base64
import hashlib
import hmac
import json
import re
import ssl
import sys
import time

from aiohttp import web

CONTROL_PORT = 9105
TARPIT_HOST = "127.0.0.3"
TARPIT_SECONDS = 8
API_VERSION = "2016-04-01"

STATE = {"records": [], "status": 200, "delay": 0.0, "tarpit": []}


def expected_auth(workspace: str, key: str, raw: bytes, date: str) -> str:
    """Authorization header Azure expects for this body and x-ms-date."""
    to_sign = f"POST\n{len(raw)}\napplication/json\nx-ms-date:{date}\n/api/logs"
    digest = hmac.new(base64.b64decode(key), to_sign.encode("utf-8"), hashlib.sha256)
    return f"SharedKey {workspace}:{base64.b64encode(digest.digest()).decode()}"


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
        if STATE["delay"]:
            await asyncio.sleep(STATE["delay"])
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
        elif STATE["status"] != 200:
            error = (STATE["status"], "E2EForced")
        if error:
            return web.json_response(
                {"Error": error[1], "Message": "e2e mock"}, status=error[0]
            )
        return web.Response(status=200)

    app = web.Application()
    app.router.add_post("/api/logs", logs)
    return app


def make_control() -> web.Application:
    async def requests(_: web.Request) -> web.Response:
        return web.json_response(STATE["records"])

    async def reset(_: web.Request) -> web.Response:
        STATE.update(records=[], status=200, delay=0.0)
        return web.json_response({"ok": True})

    async def mode(request: web.Request) -> web.Response:
        data = await request.json()
        STATE["status"] = int(data.get("status", 200))
        STATE["delay"] = float(data.get("delay", 0))
        return web.json_response({"ok": True})

    async def tarpit_log(_: web.Request) -> web.Response:
        return web.json_response(STATE["tarpit"])

    app = web.Application()
    app.router.add_get("/__requests", requests)
    app.router.add_post("/__reset", reset)
    app.router.add_post("/__mode", mode)
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
    sites = (
        (make_api(workspace, key), "127.0.0.1", 443, tls),
        (make_control(), "127.0.0.1", CONTROL_PORT, None),
    )
    for app, host, port, context in sites:
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, host, port, ssl_context=context).start()
    await asyncio.start_server(tarpit, TARPIT_HOST, 443)
    print("mock LA ready", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:5]))
