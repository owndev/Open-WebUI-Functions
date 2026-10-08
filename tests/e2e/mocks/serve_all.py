"""
Run every provider mock in one process (used by tests/e2e/run.sh).

  gemini      127.0.0.1:9101   mock_gemini.py
  azure       127.0.0.1:9102   mock_azure.py
  n8n         127.0.0.1:9103   mock_n8n.py
  infomaniak  127.0.0.1:9104   mock_infomaniak.py

usage:
  python serve_all.py              # serve until stopped
  python serve_all.py --shutdown   # stop a running instance (ignores errors)
"""

import argparse
import asyncio
import urllib.request

from aiohttp import web

import mock_azure
import mock_gemini
import mock_infomaniak
import mock_n8n
from common import fault_status

HOST = "127.0.0.1"
MOCKS = {
    "gemini": (9101, mock_gemini.make_app),
    "azure": (9102, mock_azure.make_app),
    "n8n": (9103, mock_n8n.make_app),
    "infomaniak": (9104, mock_infomaniak.make_app),
}


async def serve() -> None:
    runners = []
    for name, (port, make_app) in MOCKS.items():
        runner = web.AppRunner(make_app(), access_log=None)
        await runner.setup()
        await web.TCPSite(runner, HOST, port).start()
        runners.append(runner)
        print(f"mock {name} listening on http://{HOST}:{port}", flush=True)
    if fault_status():
        print(f"E2E_MOCK_FAULT: provider routes answer {fault_status()}", flush=True)
    await asyncio.Event().wait()


def shutdown_running() -> None:
    """Ask a running serve_all process to exit (any mock port will do)."""
    for port, _ in MOCKS.values():
        request = urllib.request.Request(
            f"http://{HOST}:{port}/__shutdown", data=b"{}", method="POST"
        )
        try:
            urllib.request.urlopen(request, timeout=2)
            return
        except OSError:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all provider mocks")
    parser.add_argument("--shutdown", action="store_true")
    if parser.parse_args().shutdown:
        shutdown_running()
    else:
        asyncio.run(serve())
