"""
title: E2E Metadata Probe
author: owndev
version: 0.1.0
license: Apache License 2.0
description: Test-only pipe for tests/e2e. Answers with a JSON report of what Open WebUI hands a pipe (body keys, __metadata__ features/params, __task__, whether an __event_emitter__ exists), so filter -> pipe coupling and background-task behaviour can be asserted without a provider.
"""

import json
from typing import Any, Optional

PREFIX = "PROBE:"


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
        report = {
            "body_keys": sorted(body.keys()),
            "body_features": body.get("features"),
            "metadata_features": metadata.get("features"),
            "metadata_params": metadata.get("params"),
            "metadata_ids": {
                key: bool(metadata.get(key))
                for key in ("chat_id", "message_id", "session_id")
            },
            "task": __task__,
            "has_event_emitter": __event_emitter__ is not None,
            "has_user": bool(__user__),
            "stream": body.get("stream"),
        }
        text = PREFIX + json.dumps(report, sort_keys=True, default=str)
        if __task__:
            # Background tasks parse JSON out of the answer: keep it valid.
            return json.dumps({"title": "Probe Title", "tags": [], "probe": text})
        return text
