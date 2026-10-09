"""
title: Vertex AI Search Tool Filter for https://github.com/owndev/Open-WebUI-Functions/blob/main/pipelines/google/google_gemini.py
author: owndev, eun2ce
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 1.0.1
required_open_webui_version: 0.9.0
license: Apache License 2.0
requirements:
  - https://github.com/owndev/Open-WebUI-Functions/blob/main/pipelines/google/google_gemini.py
description: Enable Vertex AI Search grounding for RAG
features:
  - Turns on Vertex AI Search grounding when features.vertex_ai_search is on (sets vertex_ai_search in the request metadata features).
  - Data store from the request's params.vertex_rag_store, else from the VERTEX_AI_RAG_STORE environment variable. The request's data store is used only when the request also turns on vertex_ai_search; any client that can use the model can choose it.
changelog:
  - 1.0.1 - Open WebUI >= 0.10 compatibility. Open WebUI moves unknown request params to the top level of the body before the inlet filters run, so the per-request params.vertex_rag_store never reached the pipe; it is now read from there and passed on only when the request turns on vertex_ai_search. vertex_ai_search is no longer popped from body["features"], and a request with "features": null no longer fails with a NoneType error. The flag and the data store are set in place in the request-level metadata, which UI chats pass to the pipe. Requires Open WebUI 0.9.0.
"""

import logging
import os
from open_webui.env import SRC_LOG_LEVELS


class Filter:
    def __init__(self):
        self.log = logging.getLogger("google_ai.pipe")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

    def inlet(self, body: dict) -> dict:
        # The pipeline reads __metadata__["features"] and __metadata__["params"].
        # Edit both dicts in place: for chats sent from the UI, Open WebUI
        # >= 0.11 passes the request-level metadata dicts to the pipe, not
        # copies of them.
        metadata = body.get("metadata")
        if not isinstance(metadata, dict):
            metadata = body["metadata"] = {}
        metadata_features = metadata.get("features")
        if not isinstance(metadata_features, dict):
            metadata_features = metadata["features"] = {}
        metadata_params = metadata.get("params")
        if not isinstance(metadata_params, dict):
            metadata_params = metadata["params"] = {}

        # Open WebUI moves request params it does not know, such as a
        # per-request `params.vertex_rag_store`, to the top level of the body
        # before the inlet filters run. Take it out of the body in any case.
        vertex_rag_store = body.pop("vertex_rag_store", None)
        body_params = body.get("params")
        if not vertex_rag_store and isinstance(body_params, dict):
            vertex_rag_store = body_params.get("vertex_rag_store")

        # Leave the flag in body["features"] (do not pop it): Open WebUI >= 0.11
        # copies body["features"] to metadata["features"] after the inlet
        # filters.
        features = body.get("features")
        if isinstance(features, dict) and features.get("vertex_ai_search"):
            self.log.debug("Enabling Vertex AI Search grounding")
            metadata_features["vertex_ai_search"] = True

            # The request's data store is passed on only together with the
            # feature. The pipe also searches its own data store without the
            # feature (USE_VERTEX_AI plus a store) and prefers
            # params.vertex_rag_store, which a client must not redirect.
            if vertex_rag_store and not metadata_params.get("vertex_rag_store"):
                metadata_params["vertex_rag_store"] = vertex_rag_store

            if not metadata_params.get("vertex_rag_store"):
                vertex_rag_store = os.getenv("VERTEX_AI_RAG_STORE")
                if vertex_rag_store:
                    metadata_params["vertex_rag_store"] = vertex_rag_store
                else:
                    self.log.warning(
                        "vertex_ai_search enabled but vertex_rag_store not provided in params or VERTEX_AI_RAG_STORE env var"
                    )
        elif vertex_rag_store:
            self.log.debug(
                "vertex_rag_store ignored: the request does not enable vertex_ai_search"
            )
        return body
