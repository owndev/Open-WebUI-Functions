"""
title: Google Search Tool Filter for https://github.com/owndev/Open-WebUI-Functions/blob/main/pipelines/google/google_gemini.py
author: owndev, olivier-lacroix
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 1.0.1
required_open_webui_version: 0.9.0
license: Apache License 2.0
requirements:
  - https://github.com/owndev/Open-WebUI-Functions/blob/main/pipelines/google/google_gemini.py
description: Replacing web_search tool with google search grounding
"""

import logging
from open_webui.env import SRC_LOG_LEVELS


class Filter:
    def __init__(self):
        self.log = logging.getLogger("google_ai.pipe")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

    def inlet(self, body: dict) -> dict:
        features = body.get("features")
        # API clients, channel replies and automations may send no `features`.
        if not isinstance(features, dict) or not features.get("web_search"):
            return body

        self.log.debug("Replacing web_search tool with google search grounding")

        # Open WebUI decides on its own web search from body["features"] after
        # the inlet filters (and >= 0.11 copies it to metadata["features"]).
        # Replace the dict instead of popping from it: in multi-model chats all
        # models share it, and the other models keep Open WebUI's web search.
        body["features"] = {
            **{k: v for k, v in features.items() if k != "web_search"},
            "google_search_tool": True,
        }

        # The pipeline reads __metadata__["features"]. Set the flag in place:
        # for chats sent from the UI, Open WebUI >= 0.11 passes the
        # request-level metadata["features"] dict to the pipe, not a copy.
        metadata = body.get("metadata")
        if not isinstance(metadata, dict):
            metadata = body["metadata"] = {}
        metadata_features = metadata.get("features")
        if not isinstance(metadata_features, dict):
            metadata_features = metadata["features"] = {}
        metadata_features["google_search_tool"] = True
        return body
