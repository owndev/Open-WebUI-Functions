"""
title: Time Token Tracker
author: owndev
author_url: https://github.com/owndev
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/owndev/Open-WebUI-Functions
version: 2.4.0
license: Apache License 2.0
description: A filter for tracking the response time and token usage of a request.
features:
  - Tracks the response time of a request.
  - Tracks Token Usage.
  - Calculates the average tokens per message.
  - Calculates the tokens per second.
"""

import time
from typing import Optional
import tiktoken
from pydantic import BaseModel, Field

# Global variables to track start time and token counts
global start_time, request_token_count, response_token_count

class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        CALCULATE_ALL_MESSAGES: bool = Field(
            default=True,
            description="If true, calculate tokens for all messages. If false, only use the last user and assistant messages."
        )
        SHOW_AVERAGE_TOKENS: bool = Field(
            default=True,
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
        SHOW_TOKENS_PER_SECOND: bool = Field(
            default=True,
            description="Show tokens per second for the response."
        )

    def __init__(self):
        self.name = "Time Token Tracker"
        self.valves = self.Valves()

    async def inlet(self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None) -> dict:
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
            request_messages = [m for m in all_messages if m.get("role") in ("user", "system")]
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
                    (m for m in reversed_messages if m.get("role") in ("user", "system")),
                    None
                )
                request_messages = [last_user_system] if last_user_system else []

        request_token_count = sum(len(encoding.encode(m["content"])) for m in request_messages)

        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None) -> dict:
        global start_time, request_token_count, response_token_count
        end_time = time.time()
        response_time = end_time - start_time

        model = body.get("model", "default-model")
        all_messages = body.get("messages", [])

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        reversed_messages = list(reversed(all_messages))

        # If CALCULATE_ALL_MESSAGES is true, use all "assistant" messages
        if self.valves.CALCULATE_ALL_MESSAGES:
            assistant_messages = [m for m in all_messages if m.get("role") == "assistant"]
        else:
            # Take only the last "assistant" message if any
            last_assistant = next(
                (m for m in reversed_messages if m.get("role") == "assistant"),
                None
            )
            assistant_messages = [last_assistant] if last_assistant else []

        response_token_count = sum(len(encoding.encode(m["content"])) for m in assistant_messages)

        # Calculate tokens per second (only for the last assistant response)
        if self.valves.SHOW_TOKENS_PER_SECOND:
            last_assistant_msg = next((m for m in reversed_messages if m.get("role") == "assistant"), None)
            last_assistant_tokens = len(encoding.encode(last_assistant_msg["content"])) if last_assistant_msg else 0
            resp_tokens_per_sec = 0 if response_time == 0 else last_assistant_tokens / response_time

        # Calculate averages only if CALCULATE_ALL_MESSAGES is true
        avg_request_tokens = avg_response_tokens = 0
        if self.valves.SHOW_AVERAGE_TOKENS and self.valves.CALCULATE_ALL_MESSAGES:
            req_count = len([m for m in all_messages if m.get("role") in ("user", "system")])
            resp_count = len([m for m in all_messages if m.get("role") == "assistant"])
            avg_request_tokens = request_token_count / req_count if req_count else 0
            avg_response_tokens = response_token_count / resp_count if resp_count else 0

        # Shorter style, e.g.: "10.90s | Req: 175 (Ø 87.50) | Resp: 439 (Ø 219.50) | 40.18 T/s"
        description_parts = []
        if self.valves.SHOW_RESPONSE_TIME:
            description_parts.append(f"{response_time:.2f}s")
        if self.valves.SHOW_TOKEN_COUNT:
            if self.valves.SHOW_AVERAGE_TOKENS and self.valves.CALCULATE_ALL_MESSAGES:
                # Add averages (Ø) into short output
                short_str = (
                    f"Req: {request_token_count} (Ø {avg_request_tokens:.2f}) | "
                    f"Resp: {response_token_count} (Ø {avg_response_tokens:.2f})"
                )
            else:
                short_str = f"Req: {request_token_count} | Resp: {response_token_count}"
            description_parts.append(short_str)
        if self.valves.SHOW_TOKENS_PER_SECOND:
            description_parts.append(f"{resp_tokens_per_sec:.2f} T/s")
        description = " | ".join(description_parts)

        await __event_emitter__({
            "type": "status",
            "data": {"description": description, "done": True},
        })
        return body