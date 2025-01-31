"""
title: Time Token Tracker
author: owndev
author_url: https://github.com/owndev
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/owndev/Open-WebUI-Functions
version: 2.0.0
license: MIT
description: A Python-based filter for tracking the response time and token usage of a request.
features:
  - Tracks the response time of a request.
  - Tracks Token Usage.
"""

import time
from typing import Optional
import tiktoken
from pydantic import BaseModel, Field

# Global variables to track start time and token counts
global start_time, request_token_count, response_token_count

class Filter:
    class Valves(BaseModel):
        CALCULATE_ALL_MESSAGES: bool = Field(
            default=True,
            description="If true, calculate tokens for all messages. If false, only use the last user and assistant messages."
        )
        SHOW_AVERAGE_TOKENS: bool = Field(
            default=False,
            description="Show average tokens per message (only used if CALCULATE_ALL_MESSAGES is true)."
        )
        SHOW_RESPONSE_TIME: bool = Field(
            default=True,
            description="Show the response time."
        )
        SHOW_TOKEN_COUNT: bool = Field(
            default=True,
            description="Show the token count."
        )

    def __init__(self):
        self.name = "Time Token Tracker"
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None
    ) -> dict:
        global start_time, request_token_count
        start_time = time.time()

        model = body.get("model", "default-model")
        all_messages = body.get("messages", [])

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        # If CALCULATE_ALL_MESSAGES is true, use all "user" and "system" messages
        if self.valves.CALCULATE_ALL_MESSAGES:
            request_messages = [
                m for m in all_messages if m.get("role") in ("user", "system")
            ]
        else:
            # If CALCULATE_ALL_MESSAGES is false and there are exactly two messages
            # (one user and one system), sum them both.
            request_user_system = [m for m in all_messages if m.get("role") in ("user", "system")]
            if len(request_user_system) == 2:
                request_messages = request_user_system
            else:
                # Otherwise, take only the last "user" or "system" message if any
                reversed_messages = list(reversed(all_messages))
                last_user_system = next(
                    (m for m in reversed_messages if m.get("role") in ("user", "system")), None
                )
                request_messages = [last_user_system] if last_user_system else []

        request_token_count = sum(
            len(encoding.encode(m["content"]))
            for m in request_messages
        )

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None
    ) -> dict:
        global start_time, request_token_count, response_token_count
        end_time = time.time()
        response_time = end_time - start_time

        model = body.get("model", "default-model")
        all_messages = body.get("messages", [])

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        # If CALCULATE_ALL_MESSAGES is true, use all "assistant" messages
        if self.valves.CALCULATE_ALL_MESSAGES:
            assistant_messages = [
                m for m in all_messages if m.get("role") == "assistant"
            ]
        else:
            # Take only the last "assistant" message if any
            reversed_messages = list(reversed(all_messages))
            last_assistant = next(
                (m for m in reversed_messages if m.get("role") == "assistant"), None
            )
            assistant_messages = [last_assistant] if last_assistant else []

        response_token_count = sum(
            len(encoding.encode(m["content"]))
            for m in assistant_messages
        )

        # Calculate averages only if CALCULATE_ALL_MESSAGES is true
        avg_request_tokens = avg_response_tokens = 0
        if self.valves.SHOW_AVERAGE_TOKENS and self.valves.CALCULATE_ALL_MESSAGES:
            # Count how many user/system messages were actually used
            req_count = len([m for m in all_messages if m.get("role") in ("user", "system")])
            
            # Count how many assistant messages were actually used
            resp_count = len([m for m in all_messages if m.get("role") == "assistant"])
            avg_request_tokens = request_token_count / req_count if req_count else 0
            avg_response_tokens = response_token_count / resp_count if resp_count else 0

        # Build the output description
        description = ""
        if self.valves.SHOW_RESPONSE_TIME:
            description += f"Response time: {response_time:.2f}s"

        if self.valves.SHOW_TOKEN_COUNT:
            if description:
                description += ", "
            description += f"Request tokens: {request_token_count}"
            # Only show average if enabled and CALCULATE_ALL_MESSAGES is true
            if self.valves.SHOW_AVERAGE_TOKENS and self.valves.CALCULATE_ALL_MESSAGES:
                description += f" (∅ {avg_request_tokens:.2f})"
            description += f", Response tokens: {response_token_count}"
            if self.valves.SHOW_AVERAGE_TOKENS and self.valves.CALCULATE_ALL_MESSAGES:
                description += f" (∅ {avg_response_tokens:.2f})"

        await __event_emitter__({
            "type": "status",
            "data": {"description": description, "done": True},
        })
        return body