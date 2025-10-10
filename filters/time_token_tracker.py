"""
title: Time Token Tracker
author: owndev
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 2.5.1
license: Apache License 2.0
description: A filter for tracking the response time and token usage of a request with Azure Log Analytics integration.
features:
  - Tracks the response time of a request.
  - Tracks Token Usage.
  - Calculates the average tokens per message.
  - Calculates the tokens per second.
  - Sends metrics to Azure Log Analytics.
"""

import time
import json
import uuid
import hmac
import base64
import hashlib
import datetime
import os
import logging
import aiohttp
from typing import Optional, Any
from open_webui.env import AIOHTTP_CLIENT_TIMEOUT, SRC_LOG_LEVELS
from cryptography.fernet import Fernet, InvalidToken
import tiktoken
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

# Global variables to track start time and token counts
global start_time, request_token_count, response_token_count
start_time = 0
request_token_count = 0
response_token_count = 0


# Simplified encryption implementation with automatic handling
class EncryptedStr(str):
    """A string type that automatically handles encryption/decryption"""

    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        """
        Generate encryption key from WEBUI_SECRET_KEY if available
        Returns None if no key is configured
        """
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None

        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        """
        Encrypt a string value if a key is available
        Returns the original value if no key is available
        """
        if not value or value.startswith("encrypted:"):
            return value

        key = cls._get_encryption_key()
        if not key:  # No encryption if no key
            return value

        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return f"encrypted:{encrypted.decode()}"

    @classmethod
    def decrypt(cls, value: str) -> str:
        """
        Decrypt an encrypted string value if a key is available
        Returns the original value if no key is available or decryption fails
        """
        if not value or not value.startswith("encrypted:"):
            return value

        key = cls._get_encryption_key()
        if not key:  # No decryption if no key
            return value[len("encrypted:") :]  # Return without prefix

        try:
            encrypted_part = value[len("encrypted:") :]
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except (InvalidToken, Exception):
            return value

    # Pydantic integration
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(
                            lambda value: cls(cls.encrypt(value) if value else value)
                        ),
                    ]
                ),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance)
            ),
        )


# Helper functions
async def cleanup_response(
    response: Optional[aiohttp.ClientResponse],
    session: Optional[aiohttp.ClientSession],
) -> None:
    """
    Clean up the response and session objects.

    Args:
        response: The ClientResponse object to close
        session: The ClientSession object to close
    """
    if response:
        response.close()
    if session:
        await session.close()


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        CALCULATE_ALL_MESSAGES: bool = Field(
            default=True,
            description="If true, calculate tokens for all messages. If false, only use the last user and assistant messages.",
        )
        SHOW_AVERAGE_TOKENS: bool = Field(
            default=True,
            description="Show average tokens per message (only used if CALCULATE_ALL_MESSAGES is true).",
        )
        SHOW_RESPONSE_TIME: bool = Field(
            default=True, description="Show the response time."
        )
        SHOW_TOKEN_COUNT: bool = Field(
            default=True, description="Show the token count."
        )
        SHOW_TOKENS_PER_SECOND: bool = Field(
            default=True, description="Show tokens per second for the response."
        )
        SEND_TO_LOG_ANALYTICS: bool = Field(
            default=bool(os.getenv("SEND_TO_LOG_ANALYTICS", False)),
            description="Send logs to Azure Log Analytics workspace",
        )
        LOG_ANALYTICS_WORKSPACE_ID: str = Field(
            default=os.getenv("LOG_ANALYTICS_WORKSPACE_ID", ""),
            description="Azure Log Analytics Workspace ID",
        )
        LOG_ANALYTICS_SHARED_KEY: EncryptedStr = Field(
            default=os.getenv("LOG_ANALYTICS_SHARED_KEY", ""),
            description="Azure Log Analytics Workspace Shared Key",
        )
        LOG_ANALYTICS_LOG_TYPE: str = Field(
            default="OpenWebuiMetrics", description="Log Analytics log type name."
        )

    def __init__(self):
        self.name = "Time Token Tracker"
        self.valves = self.Valves()

    def _build_signature(self, date, content_length, method, content_type, resource):
        """Build the signature for Log Analytics authentication."""
        x_headers = "x-ms-date:" + date
        string_to_hash = (
            method
            + "\n"
            + str(content_length)
            + "\n"
            + content_type
            + "\n"
            + x_headers
            + "\n"
            + resource
        )
        bytes_to_hash = string_to_hash.encode("utf-8")
        decoded_key = base64.b64decode(
            EncryptedStr.decrypt(self.valves.LOG_ANALYTICS_SHARED_KEY)
        )
        encoded_hash = base64.b64encode(
            hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest()
        ).decode("utf-8")
        authorization = (
            f"SharedKey {self.valves.LOG_ANALYTICS_WORKSPACE_ID}:{encoded_hash}"
        )
        return authorization

    async def _send_to_log_analytics_async(self, data):
        """Send data to Azure Log Analytics asynchronously using aiohttp."""
        if (
            not self.valves.SEND_TO_LOG_ANALYTICS
            or not self.valves.LOG_ANALYTICS_WORKSPACE_ID
            or not self.valves.LOG_ANALYTICS_SHARED_KEY
        ):
            return False

        log = logging.getLogger("time_token_tracker._send_to_log_analytics_async")
        log.setLevel(SRC_LOG_LEVELS["OPENAI"])

        method = "POST"
        content_type = "application/json"
        resource = "/api/logs"
        rfc1123date = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        )
        content_length = len(json.dumps(data))

        signature = self._build_signature(
            rfc1123date, content_length, method, content_type, resource
        )

        uri = f"https://{self.valves.LOG_ANALYTICS_WORKSPACE_ID}.ods.opinsights.azure.com{resource}?api-version=2016-04-01"

        headers = {
            "Content-Type": content_type,
            "Authorization": signature,
            "Log-Type": self.valves.LOG_ANALYTICS_LOG_TYPE,
            "x-ms-date": rfc1123date,
            "time-generated-field": "timestamp",
        }

        session = None
        response = None

        try:
            session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
            )

            response = await session.request(
                method="POST",
                url=uri,
                json=data,
                headers=headers,
            )

            if response.status == 200:
                return True
            else:
                response_text = await response.text()
                log.error(
                    f"Error sending to Log Analytics: {response.status} - {response_text}"
                )
                return False

        except Exception as e:
            log.error(
                f"Exception when sending to Log Analytics asynchronously: {str(e)}"
            )
            return False
        finally:
            await cleanup_response(response, session)

    async def inlet(
        self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None
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
            request_user_system = [
                m for m in all_messages if m.get("role") in ("user", "system")
            ]
            if len(request_user_system) == 2:
                request_messages = request_user_system
            else:
                # Otherwise, take only the last "user" or "system" message if any
                reversed_messages = list(reversed(all_messages))
                last_user_system = next(
                    (
                        m
                        for m in reversed_messages
                        if m.get("role") in ("user", "system")
                    ),
                    None,
                )
                request_messages = [last_user_system] if last_user_system else []

        request_token_count = sum(
            len(encoding.encode(self._get_message_content(m)))
            for m in request_messages
            if m
        )

        return body

    def _get_message_content(self, message):
        """Extract content from a message, handling different formats."""
        content = message.get("content", "")

        # Handle None content
        if content is None:
            content = ""

        # Handle string content
        if isinstance(content, str):
            return content

        # Handle list content (e.g., for messages with multiple content parts)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                else:
                    # Try to convert other types to string
                    try:
                        text_parts.append(str(part))
                    except:  # noqa: E722
                        pass
            return " ".join(text_parts)

        # Handle function_call in message
        if message.get("function_call"):
            try:
                func_call = message["function_call"]
                func_str = f"function: {func_call.get('name', '')}, arguments: {func_call.get('arguments', '')}"
                return func_str
            except:  # noqa: E722
                return ""

        # If nothing else works, try converting to string or return empty
        try:
            return str(content)
        except:  # noqa: E722
            return ""

    async def outlet(
        self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None
    ) -> dict:
        log = logging.getLogger("time_token_tracker.outlet")
        log.setLevel(SRC_LOG_LEVELS["OPENAI"])

        global start_time, request_token_count, response_token_count
        end_time = time.time()
        response_time = end_time - start_time

        model = body.get("model", "default-model")
        all_messages = body.get("messages", [])

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        reversed_messages = list(
            reversed(all_messages)
        )  # If CALCULATE_ALL_MESSAGES is true, use all "assistant" messages
        if self.valves.CALCULATE_ALL_MESSAGES:
            assistant_messages = [
                m for m in all_messages if m.get("role") == "assistant"
            ]
        else:
            # Take only the last "assistant" message if any
            last_assistant = next(
                (m for m in reversed_messages if m.get("role") == "assistant"), None
            )
            assistant_messages = [last_assistant] if last_assistant else []

        response_token_count = sum(
            len(encoding.encode(self._get_message_content(m)))
            for m in assistant_messages
            if m
        )  # Calculate tokens per second (only for the last assistant response)
        resp_tokens_per_sec = 0
        if self.valves.SHOW_TOKENS_PER_SECOND:
            last_assistant_msg = next(
                (m for m in reversed_messages if m.get("role") == "assistant"), None
            )
            last_assistant_tokens = (
                len(encoding.encode(self._get_message_content(last_assistant_msg)))
                if last_assistant_msg
                else 0
            )
            resp_tokens_per_sec = (
                0 if response_time == 0 else last_assistant_tokens / response_time
            )

        # Calculate averages only if CALCULATE_ALL_MESSAGES is true
        avg_request_tokens = avg_response_tokens = 0
        if self.valves.SHOW_AVERAGE_TOKENS and self.valves.CALCULATE_ALL_MESSAGES:
            req_count = len(
                [m for m in all_messages if m.get("role") in ("user", "system")]
            )
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

        # Send event with description
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": description, "done": True},
            }
        )

        # If Log Analytics integration is enabled, send the data
        if self.valves.SEND_TO_LOG_ANALYTICS:
            # Create chat and message IDs for tracking
            chat_id = body.get("chat_id", str(uuid.uuid4()))
            message_id = str(uuid.uuid4())
            # User ID if available
            user_id = __user__.get("id", "unknown") if __user__ else "unknown"

            # Create log data for Log Analytics
            log_data = [
                {
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "chatId": chat_id,
                    "messageId": message_id,
                    "model": model,
                    "userId": user_id,
                    "responseTime": response_time,
                    "requestTokens": request_token_count,
                    "responseTokens": response_token_count,
                    "tokensPerSecond": resp_tokens_per_sec,
                }
            ]

            # Add averages if calculated
            if self.valves.SHOW_AVERAGE_TOKENS and self.valves.CALCULATE_ALL_MESSAGES:
                log_data[0]["avgRequestTokens"] = avg_request_tokens
                log_data[0]["avgResponseTokens"] = avg_response_tokens

            # Send to Log Analytics asynchronously (non-blocking)
            try:
                result = await self._send_to_log_analytics_async(log_data)
                if result:
                    log.info("Log Analytics data sent successfully")
                else:
                    log.warning("Failed to send data to Log Analytics")
            except Exception as e:
                # Handle exceptions during sending to Log Analytics
                log.error(f"Error sending to Log Analytics: {e}")

        return body
