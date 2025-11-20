"""
title: Vertex AI Search Tool Filter for https://github.com/owndev/Open-WebUI-Functions/blob/main/pipelines/google/google_gemini.py
author: owndev, eun2ce
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 1.0.0
license: Apache License 2.0
requirements:
  - https://github.com/owndev/Open-WebUI-Functions/blob/main/pipelines/google/google_gemini.py
description: Enable Vertex AI Search grounding for RAG
"""

import logging
import os
from open_webui.env import SRC_LOG_LEVELS


class Filter:
    def __init__(self):
        self.log = logging.getLogger("google_ai.pipe")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

    def inlet(self, body: dict) -> dict:
        features = body.get("features", {})

        metadata = body.setdefault("metadata", {})
        metadata_features = metadata.setdefault("features", {})
        metadata_params = metadata.setdefault("params", {})

        if features.pop("vertex_ai_search", False):
            self.log.debug("Enabling Vertex AI Search grounding")
            metadata_features["vertex_ai_search"] = True

            if "vertex_rag_store" not in metadata_params:
                vertex_rag_store = os.getenv("VERTEX_AI_RAG_STORE")
                if vertex_rag_store:
                    metadata_params["vertex_rag_store"] = vertex_rag_store
                else:
                    self.log.warning(
                        "vertex_ai_search enabled but vertex_rag_store not provided in params or VERTEX_AI_RAG_STORE env var"
                    )
        return body

