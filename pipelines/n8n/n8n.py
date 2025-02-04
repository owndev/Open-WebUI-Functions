"""
title: n8n Pipeline
author: owndev
author_url: https://github.com/owndev
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/owndev/Open-WebUI-Functions
n8n_template: https://github.com/owndev/Open-WebUI-Functions/tree/master/pipelines/n8n
version: 1.1.0
license: MIT
description: A pipeline for interacting with N8N workflows, enabling seamless communication with various N8N workflows via configurable headers and robust error handling. This includes support for dynamic message handling and real-time interaction with N8N workflows.
features:
    - Integrates with N8N for seamless communication.
    - Supports dynamic message handling.
    - Enables real-time interaction with N8N workflows.
    - Provides configurable status emissions.
    - Cloudflare Access support for secure communication.
"""

from typing import Optional, Callable, Awaitable
from pydantic import BaseModel, Field
import time
import requests


class Pipe:
    class Valves(BaseModel):
        n8n_url: str = Field(
            default="https://<your-endpoint>/webhook/<your-webhook>",
            description="URL for the N8N webhook"
        )
        n8n_bearer_token: str = Field(
            default="",
            description="Bearer token for authenticating with the N8N webhook"
        )
        input_field: str = Field(
            default="chatInput",
            description="Field name for the input message in the N8N payload"
        )
        response_field: str = Field(
            default="output",
            description="Field name for the response message in the N8N payload"
        )
        emit_interval: float = Field(
            default=2.0,
            description="Interval in seconds between status emissions"
        )
        enable_status_indicator: bool = Field(
            default=True,
            description="Enable or disable status indicator emissions"
        )
        CF_Access_Client_Id: str = Field(
            default="",
            description="Only if behind Cloudflare: https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/"
        )
        CF_Access_Client_Secret: str = Field(
            default="",
            description="Only if behind Cloudflare: https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/"
        )

    def __init__(self):
        self.name = "N8N Agent"
        self.valves = self.Valves()
        self.last_emit_time = 0
        pass

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        level: str,
        message: str,
        done: bool,
    ):
        current_time = time.time()
        if (
            __event_emitter__
            and self.valves.enable_status_indicator
            and (
                current_time - self.last_emit_time >= self.valves.emit_interval or done
            )
        ):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )
            self.last_emit_time = current_time

    def extract_event_info(self, event_emitter):
        if not event_emitter or not event_emitter.__closure__:
            return None, None
        for cell in event_emitter.__closure__:
            if isinstance(request_info := cell.cell_contents, dict):
                chat_id = request_info.get("chat_id")
                message_id = request_info.get("message_id")
                return chat_id, message_id
        return None, None

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Optional[dict]:
        await self.emit_status(
            __event_emitter__, "info", f"Calling {self.name} ...", False
        )

        messages = body.get("messages", [])

        # Verify a message is available
        if messages:
            question = messages[-1]["content"]
            if "Prompt: " in question:
                question = question.split("Prompt: ")[-1]
            try:
                # Extract chat_id and message_id
                chat_id, message_id = self.extract_event_info(__event_emitter__)

                # Invoke N8N workflow
                headers = {
                    "Authorization": f"Bearer {self.valves.n8n_bearer_token}",
                    "Content-Type": "application/json",
                    "CF-Access-Client-Id": self.valves.CF_Access_Client_Id,
                    "CF-Access-Client-Secret": self.valves.CF_Access_Client_Secret
                }
                payload = {
                    "systemPrompt": f"{messages[0]['content'].split('Prompt: ')[-1]}",
                    "user_id": __user__.get("id"),
                    "user_email": __user__.get("email"),
                    "user_name": __user__.get("name"),
                    "user_role": __user__.get("role"),
                    "chat_id": chat_id,
                    "message_id": message_id,
                }
                payload[self.valves.input_field] = question
                response = requests.post(
                    self.valves.n8n_url, json=payload, headers=headers
                )
                if response.status_code == 200:
                    n8n_response = response.json()[self.valves.response_field]
                else:
                    raise Exception(f"Error: {response.status_code} - {response.text}")

                # Set assistant message with chain reply
                body["messages"].append({"role": "assistant", "content": n8n_response})
            except Exception as e:
                await self.emit_status(
                    __event_emitter__,
                    "error",
                    f"Error during sequence execution: {str(e)}",
                    True,
                )
                return {"error": str(e)}
            
        # If no message is available alert user
        else:
            await self.emit_status(
                __event_emitter__,
                "error",
                "No messages found in the request body",
                True,
            )
            body["messages"].append(
                {
                    "role": "assistant",
                    "content": "No messages found in the request body",
                }
            )

        await self.emit_status(__event_emitter__, "info", "Complete", True)
        return n8n_response
