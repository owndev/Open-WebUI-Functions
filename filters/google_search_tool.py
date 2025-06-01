"""
title: Google Search Tool Filter for https://github.com/owndev/Open-WebUI-Functions/blob/master/pipelines/google/google_gemini.py
author: owndev, olivier-lacroix
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 1.0.0
license: Apache License 2.0
requirements:
  - https://github.com/owndev/Open-WebUI-Functions/blob/master/pipelines/google/google_gemini.py
description: Replacing web_search tool with google search grounding
"""

import logging
from open_webui.env import SRC_LOG_LEVELS


class Filter:
    def __init__(self):
        self.log = logging.getLogger("google_ai.pipe")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

    def inlet(self, body: dict) -> dict:
        features = body.get("features", {})

        # Ensure metadata structure exists and add new feature
        metadata = body.setdefault("metadata", {})
        metadata_features = metadata.setdefault("features", {})

        if features.pop("web_search"):
            self.log.debug("Replacing web_search tool with google search grounding")
            metadata_features["google_search_tool"] = True
        return body
