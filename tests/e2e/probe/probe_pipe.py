"""
title: E2E Metadata Probe
author: owndev
version: 0.2.0
license: Apache License 2.0
description: Test-only pipe for tests/e2e. Answers with a JSON report of what Open WebUI hands a pipe (body keys, model and messages, __metadata__ features/params/model_id, __task__, whether an __event_emitter__ exists), so filter -> pipe coupling and background-task behaviour can be asserted without a provider. "PROBE_SLEEP=<seconds>" in the last user message delays the answer (at most 10 s), so concurrent requests overlap; "PROBE_SLEEP[<model id>]=<seconds>" delays only the answer of that model (multi-model chats).
"""

import asyncio
import json
import re
from typing import Any, Optional

PREFIX = "PROBE:"
SLEEP = re.compile(r"PROBE_SLEEP(?:\[([^\]]+)\])?=(\d+(?:\.\d+)?)")
MAX_SLEEP = 10.0


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
        text = PREFIX + json.dumps(report, sort_keys=True, default=str)
        if __task__:
            # Background tasks parse JSON out of the answer: keep it valid.
            return json.dumps({"title": "Probe Title", "tags": [], "probe": text})
        users = [m for m in messages if m.get("role") == "user"]
        model_id = metadata.get("model_id") or body.get("model")
        for match in SLEEP.finditer(_text(users[-1].get("content")) if users else ""):
            if match.group(1) in (None, model_id):
                await asyncio.sleep(min(float(match.group(2)), MAX_SLEEP))
                break
        return text
