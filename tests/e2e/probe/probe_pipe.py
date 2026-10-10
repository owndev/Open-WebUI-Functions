"""
title: E2E Metadata Probe
author: owndev
version: 0.4.0
license: Apache License 2.0
description: Test-only pipe for tests/e2e. Answers with a JSON report of what Open WebUI hands a pipe (body keys, model and messages, __metadata__ features/params/model_id, __task__, whether an __event_emitter__ exists), so filter -> pipe coupling and background-task behaviour can be asserted without a provider. "PROBE_SLEEP=<seconds>" in the last user message delays the answer (at most 10 s), so concurrent requests overlap; "PROBE_SLEEP[<model id>]=<seconds>" delays only the answer of that model (multi-model chats). "PROBE_ENV=<absolute path>" sets (or, for null, removes) the environment variables listed in a JSON object in that file in the server process, for an allow-list of managed identity, workload identity and proxy variables only (the report lists env_applied / env_rejected); the values go through the file, never through the chat text. "PROBE_LOG=<absolute path>" writes everything the time_token_tracker logger logs, DEBUG included, to that file (the server log runs at INFO); "PROBE_LOG=off" stops it (the report has log_capture).
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

PREFIX = "PROBE:"
SLEEP = re.compile(r"PROBE_SLEEP(?:\[([^\]]+)\])?=(\d+(?:\.\d+)?)")
MAX_SLEEP = 10.0
ENV = re.compile(r"PROBE_ENV=(/\S+)")
LOG = re.compile(r"PROBE_LOG=(/\S+|off)")
# Loggers whose DEBUG output PROBE_LOG may capture (never the server's own).
LOG_ALLOWED = ("time_token_tracker",)
# Environment variables the filters suite may change in the server process
# (managed identity / workload identity detection, proxy). run.sh sets only
# IDENTITY_ENDPOINT / IDENTITY_HEADER at docker run (the azure suite's managed
# identity mock); the filters suite puts the container's values back at the end.
ENV_ALLOWED = (
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


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


async def _apply_env(path: str) -> dict:
    """Set / pop the allow-listed variables of the JSON object in ``path``."""
    applied, rejected, error = [], [], None
    try:
        values = await asyncio.to_thread(_read_json, path)
        if not isinstance(values, dict):
            raise ValueError("not a JSON object")
        for name, value in values.items():
            if name not in ENV_ALLOWED:
                rejected.append(name)
            elif value is None:
                os.environ.pop(name, None)
                applied.append(name)
            else:
                os.environ[name] = str(value)
                applied.append(name)
    except (OSError, ValueError) as exc:
        error = type(exc).__name__
    return {
        "env_applied": sorted(applied),
        "env_rejected": sorted(rejected),
        "env_error": error,
    }


def _capture_log(target: str) -> dict:
    """Attach a DEBUG FileHandler to the allow-listed loggers (replacing an
    earlier one), or remove it again for ``off``. The logger level is raised
    to DEBUG while the capture runs and restored afterwards; the records still
    propagate to the server log, whose handler keeps its own (INFO) level.
    A filter module that is loaded again resets its logger level in
    __init__, so start the capture after the module is loaded."""
    try:
        for name in LOG_ALLOWED:
            logger = logging.getLogger(name)
            for handler in list(logger.handlers):
                if getattr(handler, "_probe_capture", False):
                    logger.removeHandler(handler)
                    handler.close()
                    logger.setLevel(handler._probe_level)
            if target == "off":
                continue
            handler = logging.FileHandler(target, mode="w", encoding="utf-8")
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            )
            handler._probe_capture = True
            handler._probe_level = logger.level
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        return {"log_capture": target}
    except (OSError, ValueError) as exc:
        return {"log_capture": None, "log_error": type(exc).__name__}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


class Pipe:
    def __init__(self):
        self.type = "manifold"

    def pipes(self):
        return [{"id": "echo", "name": "E2E Probe"}]

    async def pipe(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __event_emitter__=None,
        __task__: Optional[str] = None,
        __user__: Optional[dict] = None,
    ) -> Any:
        metadata = __metadata__ or {}
        messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
        report = {
            "body_keys": sorted(body.keys()),
            "body_model": body.get("model"),
            "body_features": body.get("features"),
            "metadata_features": metadata.get("features"),
            "metadata_params": metadata.get("params"),
            "metadata_model_id": metadata.get("model_id"),
            "metadata_ids": {
                key: bool(metadata.get(key))
                for key in ("chat_id", "message_id", "session_id")
            },
            "messages": [
                {"role": m.get("role"), "content": m.get("content")} for m in messages
            ],
            "task": __task__,
            "has_event_emitter": __event_emitter__ is not None,
            "has_user": bool(__user__),
            "stream": body.get("stream"),
        }
        users = [m for m in messages if m.get("role") == "user"]
        last = _text(users[-1].get("content")) if users else ""
        env = ENV.search(last) if not __task__ else None
        if env:
            report.update(await _apply_env(env.group(1)))
        log = LOG.search(last) if not __task__ else None
        if log:
            report.update(_capture_log(log.group(1)))
        text = PREFIX + json.dumps(report, sort_keys=True, default=str)
        if __task__:
            # Background tasks parse JSON out of the answer: keep it valid.
            return json.dumps({"title": "Probe Title", "tags": [], "probe": text})
        model_id = metadata.get("model_id") or body.get("model")
        for match in SLEEP.finditer(last):
            if match.group(1) in (None, model_id):
                await asyncio.sleep(min(float(match.group(2)), MAX_SLEEP))
                break
        return text
