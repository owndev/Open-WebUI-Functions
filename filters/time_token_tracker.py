"""
title: Time Token Tracker
author: owndev
author_url: https://github.com/owndev
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/owndev/Open-WebUI-Functions
version: 1.0.0
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
            default=True, description="Calculate tokens for all messages or only the last one."
        )
        N_LAST_MESSAGES: int = Field(
            default=1, description="Number of last messages to calculate tokens for."
        )
        SHOW_AVERAGE_TOKENS: bool = Field(
            default=False, description="Show average tokens per message."
        )

    def __init__(self):
        self.name = "Time Token Tracker"
        self.valves = self.Valves()

    async def inlet(
        self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None
    ) -> dict:
        global start_time, request_token_count

        # Start the timer
        start_time = time.time()

        # Get the model and messages from the body
        model = body.get("model", "default-model")
        messages = body.get("messages", [])
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")  # Fallback encoding

        # Calculate the number of tokens for all or the last N messages
        if self.valves.CALCULATE_ALL_MESSAGES:
            request_token_count = sum(len(encoding.encode(message["content"])) for message in messages)
        else:
            messages = messages[-self.valves.N_LAST_MESSAGES:]
            request_token_count = sum(len(encoding.encode(message["content"])) for message in messages)

        return body

    async def outlet(
        self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None
    ) -> dict:
        global start_time, request_token_count, response_token_count

        # Stop the timer and calculate the response time
        end_time = time.time()
        response_time = end_time - start_time

        # Get the model and messages from the body
        model = body.get("model", "default-model")
        messages = body.get("messages", [])
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")  # Fallback encoding

        # Calculate the number of tokens for all or the last N messages
        if self.valves.CALCULATE_ALL_MESSAGES:
            response_token_count = sum(len(encoding.encode(message["content"])) for message in messages)
        else:
            messages = messages[-self.valves.N_LAST_MESSAGES:]
            response_token_count = sum(len(encoding.encode(message["content"])) for message in messages)

        # Calculate the average number of tokens per message if enabled
        avg_request_tokens = avg_response_tokens = 0
        if self.valves.SHOW_AVERAGE_TOKENS:
            num_messages = len(messages)
            avg_request_tokens = request_token_count / num_messages if num_messages > 0 else 0
            avg_response_tokens = response_token_count / num_messages if num_messages > 0 else 0

        # Create the description for the output
        description = (
            f"Response time: {response_time:.2f}s, "
            f"Request tokens: {request_token_count}"
        )
        if self.valves.SHOW_AVERAGE_TOKENS:
            description += f" (∅ {avg_request_tokens:.2f})"
        description += f", Response tokens: {response_token_count}"
        if self.valves.SHOW_AVERAGE_TOKENS:
            description += f" (∅ {avg_response_tokens:.2f})"

        # Send the status data
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "done": True,
                },
            }
        )
        return body