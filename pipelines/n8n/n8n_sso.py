"""
title: n8n Pipeline with SSO Authentication
author: owndev
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 1.0.0
license: Apache License 2.0
description: An n8n pipeline with SSO authentication support. Users authenticate via ID/PW, and the authentication token is passed to n8n workflows. Supports both streaming and non-streaming modes.
features:
  - SSO authentication before n8n communication
  - User credential collection via chat interface
  - Authentication token caching per user
  - Custom authentication API integration
  - Secure encrypted storage of credentials
  - Full n8n streaming support
  - Compatible with Open WebUI streaming architecture
"""

from typing import (
    Optional,
    Callable,
    Awaitable,
    Any,
    Dict,
    AsyncIterator,
    Union,
    Generator,
    Iterator,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from starlette.background import BackgroundTask
from cryptography.fernet import Fernet, InvalidToken
import aiohttp
import os
import base64
import hashlib
import logging
import json
import asyncio
from open_webui.env import AIOHTTP_CLIENT_TIMEOUT, SRC_LOG_LEVELS
from pydantic_core import core_schema
import time
import re


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


# Helper functions for resource cleanup
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


async def stream_processor(
    content: aiohttp.StreamReader,
    __event_emitter__=None,
    response: Optional[aiohttp.ClientResponse] = None,
    session: Optional[aiohttp.ClientSession] = None,
    logger: Optional[logging.Logger] = None,
) -> AsyncIterator[str]:
    """
    Process streaming content from n8n and yield chunks for StreamingResponse.

    Args:
        content: The streaming content from the response
        __event_emitter__: Optional event emitter for status updates
        response: The response object for cleanup
        session: The session object for cleanup
        logger: Logger for debugging

    Yields:
        String content from the streaming response
    """
    try:
        if logger:
            logger.info("Starting stream processing...")

        buffer = ""
        # Attempt to read preserve flag later via closure if needed
        async for chunk_bytes in content:
            chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
            if not chunk_str:
                continue
            buffer += chunk_str

            # Process complete lines (retain trailing newline info)
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                had_newline = True
                original_line = line  # without \n
                if line.endswith("\r"):
                    line = line[:-1]

                if logger:
                    logger.debug(f"Raw line received: {repr(line)}")

                # Preserve blank lines
                if line == "":
                    yield "\n"
                    continue

                content_text = ""

                if line.startswith("data: "):
                    data_part = line[6:]
                    if logger:
                        logger.debug(f"SSE data part: {repr(data_part)}")
                    if data_part == "[DONE]":
                        if logger:
                            logger.debug("Received [DONE] signal")
                        buffer = ""
                        break
                    try:
                        event_data = json.loads(data_part)
                        if logger:
                            logger.debug(f"Parsed SSE JSON: {event_data}")
                        for key in ("content", "text", "output", "data"):
                            val = event_data.get(key)
                            if isinstance(val, str) and val:
                                content_text = val
                                break
                    except json.JSONDecodeError:
                        content_text = data_part
                        if logger:
                            logger.debug(
                                f"Using raw data as content: {repr(content_text)}"
                            )
                elif not line.startswith(":"):
                    # Plain text (non-SSE)
                    content_text = original_line
                    if logger:
                        logger.debug(f"Plain text content: {repr(content_text)}")

                if content_text:
                    if not content_text.endswith("\n"):
                        content_text += "\n"
                    if logger:
                        logger.debug(f"Yielding content: {repr(content_text)}")
                    yield content_text

        # Send completion status update when streaming is done
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete",
                        "description": "N8N streaming completed successfully",
                        "done": True,
                    },
                }
            )

        if logger:
            logger.info("Stream processing completed successfully")

    except Exception as e:
        if logger:
            logger.error(f"Error processing stream: {e}")

        # Send error status update
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "error",
                        "description": f"N8N streaming error: {str(e)}",
                        "done": True,
                    },
                }
            )
        raise
    finally:
        # Always attempt to close response and session to avoid resource leaks
        await cleanup_response(response, session)


class Pipe:
    class Valves(BaseModel):
        N8N_URL: str = Field(
            default="https://<your-endpoint>/webhook/<your-webhook>",
            description="URL for the N8N webhook",
        )
        N8N_BEARER_TOKEN: EncryptedStr = Field(
            default="",
            description="Bearer token for authenticating with the N8N webhook",
        )
        INPUT_FIELD: str = Field(
            default="chatInput",
            description="Field name for the input message in the N8N payload",
        )
        RESPONSE_FIELD: str = Field(
            default="output",
            description="Field name for the response message in the N8N payload",
        )
        CF_ACCESS_CLIENT_ID: EncryptedStr = Field(
            default="",
            description="Only if behind Cloudflare: https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/",
        )
        CF_ACCESS_CLIENT_SECRET: EncryptedStr = Field(
            default="",
            description="Only if behind Cloudflare: https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/",
        )
        # SSO Authentication Settings
        AUTH_API_URL: str = Field(
            default="https://<your-auth-server>/api/auth/login",
            description="URL for the SSO authentication API endpoint",
        )
        AUTH_ENABLED: bool = Field(
            default=True,
            description="Enable SSO authentication requirement",
        )
        AUTH_TOKEN_EXPIRY: int = Field(
            default=3600,
            description="Authentication token expiry time in seconds (default: 1 hour)",
        )

    def __init__(self):
        self.name = "N8N SSO Agent"
        self.valves = self.Valves()
        self.log = logging.getLogger("n8n_sso_pipeline")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        # Store authentication tokens per user {user_id: {"token": str, "expires_at": float}}
        self.auth_tokens: Dict[str, Dict[str, Any]] = {}

    async def emit_simple_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        status: str,
        message: str,
        done: bool = False,
    ):
        """Simplified status emission without intervals"""
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": message,
                        "done": done,
                    },
                }
            )

    def extract_event_info(self, event_emitter):
        if not event_emitter or not event_emitter.__closure__:
            return None, None
        for cell in event_emitter.__closure__:
            if isinstance(request_info := cell.cell_contents, dict):
                chat_id = request_info.get("chat_id")
                message_id = request_info.get("message_id")
                return chat_id, message_id
        return None, None

    def get_headers(self) -> Dict[str, str]:
        """
        Constructs the headers for the API request.

        Returns:
            Dictionary containing the required headers for the API request.
        """
        headers = {"Content-Type": "application/json"}

        # Add bearer token if available
        bearer_token = EncryptedStr.decrypt(self.valves.N8N_BEARER_TOKEN)
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"

        # Add Cloudflare Access headers if available
        cf_client_id = EncryptedStr.decrypt(self.valves.CF_ACCESS_CLIENT_ID)
        if cf_client_id:
            headers["CF-Access-Client-Id"] = cf_client_id

        cf_client_secret = EncryptedStr.decrypt(self.valves.CF_ACCESS_CLIENT_SECRET)
        if cf_client_secret:
            headers["CF-Access-Client-Secret"] = cf_client_secret

        return headers

    async def authenticate_user(
        self, username: str, password: str
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with the SSO authentication API.

        Args:
            username: User's username/ID
            password: User's password

        Returns:
            Authentication response data if successful, None otherwise
        """
        try:
            self.log.info(f"Attempting authentication for user: {username}")

            auth_payload = {
                "username": username,
                "password": password,
            }

            session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=30),
            )

            try:
                async with session.post(
                    self.valves.AUTH_API_URL,
                    json=auth_payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        auth_data = await response.json()
                        self.log.info(f"Authentication successful for user: {username}")
                        return auth_data
                    else:
                        error_text = await response.text()
                        self.log.error(
                            f"Authentication failed: {response.status} - {error_text}"
                        )
                        return None
            finally:
                await session.close()

        except Exception as e:
            self.log.error(f"Authentication error: {str(e)}")
            return None

    def is_authenticated(self, user_id: str) -> bool:
        """
        Check if user has a valid authentication token.

        Args:
            user_id: User's ID

        Returns:
            True if user is authenticated and token is not expired
        """
        if user_id not in self.auth_tokens:
            return False

        token_data = self.auth_tokens[user_id]
        expires_at = token_data.get("expires_at", 0)

        # Check if token is expired
        if time.time() > expires_at:
            self.log.info(f"Token expired for user: {user_id}")
            del self.auth_tokens[user_id]
            return False

        return True

    def get_auth_token(self, user_id: str) -> Optional[str]:
        """
        Get authentication token for user.

        Args:
            user_id: User's ID

        Returns:
            Authentication token if available and valid
        """
        if self.is_authenticated(user_id):
            return self.auth_tokens[user_id].get("token")
        return None

    def store_auth_token(self, user_id: str, auth_data: Dict[str, Any]):
        """
        Store authentication token for user.

        Args:
            user_id: User's ID
            auth_data: Authentication response data
        """
        # Extract token from auth_data (adjust based on your API response format)
        token = auth_data.get("token") or auth_data.get("access_token") or auth_data.get("auth_token")

        if token:
            self.auth_tokens[user_id] = {
                "token": token,
                "expires_at": time.time() + self.valves.AUTH_TOKEN_EXPIRY,
                "auth_data": auth_data,
            }
            self.log.info(f"Stored auth token for user: {user_id}")

    def parse_login_command(self, message: str) -> Optional[Dict[str, str]]:
        """
        Parse login command from user message.
        Expected format: /login username password
        or: /login username:password

        Args:
            message: User message

        Returns:
            Dictionary with username and password if command found
        """
        # Pattern 1: /login username password
        pattern1 = r'^/login\s+(\S+)\s+(\S+)$'
        match = re.match(pattern1, message.strip())
        if match:
            return {
                "username": match.group(1),
                "password": match.group(2),
            }

        # Pattern 2: /login username:password
        pattern2 = r'^/login\s+(\S+):(\S+)$'
        match = re.match(pattern2, message.strip())
        if match:
            return {
                "username": match.group(1),
                "password": match.group(2),
            }

        return None

    def parse_n8n_streaming_chunk(self, chunk_text: str) -> Optional[str]:
        """Parse N8N streaming chunk and extract content, filtering out metadata"""
        if not chunk_text.strip():
            return None

        try:
            data = json.loads(chunk_text.strip())

            if isinstance(data, dict):
                # Skip N8N metadata chunks but be more selective
                chunk_type = data.get("type", "")
                if chunk_type in ["begin", "end", "error", "metadata"]:
                    self.log.debug(f"Skipping N8N metadata chunk: {chunk_type}")
                    return None

                # Skip metadata-only chunks
                if "metadata" in data and len(data) <= 2:
                    return None

                # Extract content from various possible field names
                content = (
                    data.get("text")
                    or data.get("content")
                    or data.get("output")
                    or data.get("message")
                    or data.get("delta")
                    or data.get("data")
                    or data.get("response")
                    or data.get("result")
                )

                # Handle OpenAI-style streaming format
                if not content and "choices" in data:
                    choices = data.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")

                if content:
                    self.log.debug(
                        f"Extracted content from JSON: {repr(content[:100])}"
                    )
                    return str(content)

                # Return non-metadata objects as strings (be more permissive)
                if not any(
                    key in data
                    for key in [
                        "type",
                        "metadata",
                        "nodeId",
                        "nodeName",
                        "timestamp",
                        "id",
                    ]
                ):
                    # For smaller models, return the entire object if it's simple
                    self.log.debug(
                        f"Returning entire object as content: {repr(str(data)[:100])}"
                    )
                    return str(data)

        except json.JSONDecodeError:
            # Handle plain text content - be more permissive
            stripped = chunk_text.strip()
            if stripped and not stripped.startswith("{"):
                self.log.debug(f"Returning plain text content: {repr(stripped[:100])}")
                return stripped

        return None

    def extract_content_from_mixed_stream(self, raw_text: str) -> str:
        """Extract content from mixed stream containing both metadata and content"""
        content_parts = []

        # First try to handle concatenated JSON objects
        if "{" in raw_text and "}" in raw_text:
            parts = raw_text.split("}{")

            for i, part in enumerate(parts):
                # Reconstruct valid JSON
                if i > 0:
                    part = "{" + part
                if i < len(parts) - 1:
                    part = part + "}"

                extracted = self.parse_n8n_streaming_chunk(part)
                if extracted:
                    content_parts.append(extracted)

        # If no JSON content found, treat as plain text
        if not content_parts:
            # Remove common streaming artifacts but preserve actual content
            cleaned = raw_text.strip()
            if (
                cleaned
                and not cleaned.startswith("data:")
                and not cleaned.startswith(":")
            ):
                self.log.debug(f"Using raw text as content: {repr(cleaned[:100])}")
                return cleaned

        return "".join(content_parts)

    def dedupe_system_prompt(self, text: str) -> str:
        """Remove duplicated content from the system prompt.

        Strategies:
        1. Detect full duplication where the prompt text is repeated twice consecutively.
        2. Remove duplicate lines (keeping first occurrence, preserving order & spacing where possible).
        3. Preserve blank lines but collapse consecutive duplicate non-blank lines.
        """
        if not text:
            return text

        original = text
        stripped = text.strip()

        # 1. Full duplication detection (exact repeat of first half == second half)
        half = len(stripped) // 2
        if len(stripped) % 2 == 0:
            first_half = stripped[:half].strip()
            second_half = stripped[half:].strip()
            if first_half and first_half == second_half:
                text = first_half

        # 2. Line-level dedupe
        lines = text.splitlines()
        seen = set()
        deduped = []
        for line in lines:
            key = line.strip()
            # Allow empty lines to pass through (formatting), but avoid repeating identical non-empty lines
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped.append(line)

        deduped_text = "\n".join(deduped).strip()

        if deduped_text != original.strip():
            self.log.debug("System prompt deduplicated")
        return deduped_text

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Union[str, Generator, Iterator, Dict[str, Any], StreamingResponse]:
        """
        Main method for sending requests to the N8N endpoint with SSO authentication.

        Args:
            body: The request body containing messages and other parameters
            __user__: User information
            __event_emitter__: Optional event emitter function for status updates
            __event_call__: Optional event call function

        Returns:
            Response from N8N API, which could be a string, dictionary or streaming response
        """
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

        messages = body.get("messages", [])

        # Verify a message is available
        if not messages:
            error_msg = "No messages found in the request body"
            self.log.warning(error_msg)
            await self.emit_simple_status(
                __event_emitter__,
                "error",
                error_msg,
                True,
            )
            return error_msg

        # Get user ID
        user_id = __user__.get("id") if __user__ else "anonymous"

        # Extract last message
        last_message = messages[-1]["content"]

        # Check if it's a login command
        login_credentials = self.parse_login_command(last_message)

        if login_credentials:
            # Process login
            await self.emit_simple_status(
                __event_emitter__, "in_progress", "Authenticating...", False
            )

            auth_data = await self.authenticate_user(
                login_credentials["username"],
                login_credentials["password"]
            )

            if auth_data:
                self.store_auth_token(user_id, auth_data)

                success_msg = f"✅ Authentication successful! You are now logged in as {login_credentials['username']}.\n\nYou can now proceed with your queries."

                await self.emit_simple_status(
                    __event_emitter__, "complete", "Authentication successful", True
                )

                return success_msg
            else:
                error_msg = "❌ Authentication failed. Please check your credentials and try again.\n\nUsage: `/login username password` or `/login username:password`"

                await self.emit_simple_status(
                    __event_emitter__, "error", "Authentication failed", True
                )

                return error_msg

        # Check authentication if enabled
        if self.valves.AUTH_ENABLED:
            if not self.is_authenticated(user_id):
                auth_required_msg = """🔐 **Authentication Required**

Please authenticate to use this service.

**Usage:**
- `/login username password`
- `/login username:password`

**Example:**
```
/login john.doe mypassword123
```

Please provide your credentials to continue."""

                await self.emit_simple_status(
                    __event_emitter__, "error", "Authentication required", True
                )

                return auth_required_msg

        # User is authenticated, proceed with N8N request
        await self.emit_simple_status(
            __event_emitter__, "in_progress", f"Calling {self.name} ...", False
        )

        session = None
        n8n_response = ""

        question = last_message
        if "Prompt: " in question:
            question = question.split("Prompt: ")[-1]

        try:
            # Extract chat_id and message_id
            chat_id, message_id = self.extract_event_info(__event_emitter__)

            self.log.info(f"Starting N8N workflow request for chat ID: {chat_id}")

            # Extract system prompt correctly
            system_prompt = ""
            if messages and messages[0].get("role") == "system":
                system_prompt = self.dedupe_system_prompt(messages[0]["content"])

            # Include full conversation history
            conversation_history = []
            for msg in messages:
                if msg.get("role") in ["user", "assistant"]:
                    conversation_history.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )

            # Get authentication token
            auth_token = self.get_auth_token(user_id)
            auth_data = None
            if auth_token:
                auth_data = self.auth_tokens[user_id].get("auth_data", {})

            # Prepare payload for N8N workflow (with auth data)
            payload = {
                "systemPrompt": system_prompt,
                "messages": conversation_history,
                "currentMessage": question,
                "user_id": user_id,
                "user_email": __user__.get("email") if __user__ else None,
                "user_name": __user__.get("name") if __user__ else None,
                "user_role": __user__.get("role") if __user__ else None,
                "chat_id": chat_id,
                "message_id": message_id,
                # SSO Authentication data
                "auth_token": auth_token,
                "auth_data": auth_data,
            }
            # Keep backward compatibility
            payload[self.valves.INPUT_FIELD] = question

            # Get headers for the request
            headers = self.get_headers()

            # Create session
            session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
            )

            self.log.debug(f"Sending request to N8N: {self.valves.N8N_URL}")

            # Send status update via event emitter if available
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "in_progress",
                            "description": "Sending request to N8N...",
                            "done": False,
                        },
                    }
                )

            # Make the request
            request = session.post(
                self.valves.N8N_URL, json=payload, headers=headers
            )

            response = await request.__aenter__()
            self.log.debug(f"Response status: {response.status}")
            self.log.debug(f"Response headers: {dict(response.headers)}")

            if response.status == 200:
                # Enhanced streaming detection
                content_type = response.headers.get("Content-Type", "").lower()
                is_streaming = (
                    "stream" in content_type
                    or "text/plain" in content_type
                    or "text/event-stream" in content_type
                    or "application/x-ndjson" in content_type
                    or response.headers.get("Transfer-Encoding") == "chunked"
                )

                if is_streaming:
                    # Enhanced streaming
                    self.log.info("Processing streaming response from N8N")
                    n8n_response = ""
                    buffer = ""
                    completed_thoughts: list[str] = []

                    try:
                        async for chunk in response.content.iter_any():
                            if not chunk:
                                continue

                            text = chunk.decode(errors="ignore")
                            buffer += text

                            # Handle different streaming formats
                            if "{" in buffer and "}" in buffer:
                                # Process complete JSON objects
                                while True:
                                    start_idx = buffer.find("{")
                                    if start_idx == -1:
                                        break

                                    # Find matching closing brace
                                    brace_count = 0
                                    end_idx = -1

                                    for i in range(start_idx, len(buffer)):
                                        if buffer[i] == "{":
                                            brace_count += 1
                                        elif buffer[i] == "}":
                                            brace_count -= 1
                                            if brace_count == 0:
                                                end_idx = i
                                                break

                                    if end_idx == -1:
                                        # Incomplete JSON, wait for more data
                                        break

                                    # Extract and process the JSON chunk
                                    json_chunk = buffer[start_idx : end_idx + 1]
                                    buffer = buffer[end_idx + 1 :]

                                    # Parse N8N streaming chunk
                                    content = self.parse_n8n_streaming_chunk(
                                        json_chunk
                                    )
                                    if content:
                                        # Normalize escaped newlines to actual newlines
                                        content = content.replace("\\n", "\n")

                                        # Just accumulate content
                                        n8n_response += content

                                        # Emit delta
                                        if __event_emitter__:
                                            await __event_emitter__(
                                                {
                                                    "type": "chat:message:delta",
                                                    "data": {
                                                        "role": "assistant",
                                                        "content": content,
                                                    },
                                                }
                                            )
                            else:
                                # Handle plain text streaming
                                while "\n" in buffer:
                                    line, buffer = buffer.split("\n", 1)
                                    if line.strip():
                                        self.log.debug(
                                            f"Processing plain text line: {repr(line[:100])}"
                                        )

                                        # Normalize content
                                        content = line.replace("\\n", "\n")
                                        n8n_response += content + "\n"

                                        # Emit delta for plain text
                                        if __event_emitter__:
                                            await __event_emitter__(
                                                {
                                                    "type": "chat:message:delta",
                                                    "data": {
                                                        "role": "assistant",
                                                        "content": content + "\n",
                                                    },
                                                }
                                            )

                        # Process any remaining content in buffer
                        if buffer.strip():
                            self.log.debug(
                                f"Processing remaining buffer content: {repr(buffer[:100])}"
                            )

                            # Try to extract from mixed content first
                            remaining_content = (
                                self.extract_content_from_mixed_stream(buffer)
                            )

                            # If that doesn't work, use buffer as-is
                            if not remaining_content:
                                remaining_content = buffer.strip()

                            if remaining_content:
                                # Normalize escaped newlines
                                remaining_content = remaining_content.replace(
                                    "\\n", "\n"
                                )

                                # Accumulate final buffer content
                                n8n_response += remaining_content

                                # Emit final buffer delta
                                if __event_emitter__:
                                    await __event_emitter__(
                                        {
                                            "type": "chat:message:delta",
                                            "data": {
                                                "role": "assistant",
                                                "content": remaining_content,
                                            },
                                        }
                                    )

                        # Process think blocks
                        if n8n_response and "<think>" in n8n_response.lower():
                            think_pattern = re.compile(
                                r"<think>\s*(.*?)\s*</think>",
                                re.IGNORECASE | re.DOTALL,
                            )

                            think_counter = 0

                            def replace_think_block(match):
                                nonlocal think_counter
                                think_counter += 1
                                thought_content = match.group(1).strip()
                                if thought_content:
                                    completed_thoughts.append(thought_content)

                                    # Format blockquote
                                    quoted_lines = []
                                    for line in thought_content.split("\n"):
                                        quoted_lines.append(f"> {line}")
                                    quoted_content = "\n".join(quoted_lines)

                                    return f"""<details>
<summary>Thought {think_counter}</summary>

{quoted_content}

</details>"""
                                return ""

                            # Replace all think blocks
                            n8n_response = think_pattern.sub(
                                replace_think_block, n8n_response
                            )

                        # Emit final message
                        if __event_emitter__:
                            if not n8n_response.strip():
                                n8n_response = "(Empty response received from N8N)"
                                self.log.warning(
                                    "Empty response received from N8N"
                                )

                            await __event_emitter__(
                                {
                                    "type": "chat:message",
                                    "data": {
                                        "role": "assistant",
                                        "content": n8n_response,
                                    },
                                }
                            )
                            if completed_thoughts:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "action": "thinking",
                                            "done": True,
                                            "hidden": True,
                                        },
                                    }
                                )

                        self.log.info(
                            f"Streaming completed. Length: {len(n8n_response)}"
                        )

                    except Exception as e:
                        self.log.error(f"Streaming error: {e}")

                        # Emit whatever we have
                        if n8n_response:
                            if __event_emitter__:
                                await __event_emitter__(
                                    {
                                        "type": "chat:message",
                                        "data": {
                                            "role": "assistant",
                                            "content": n8n_response,
                                        },
                                    }
                                )
                        else:
                            error_msg = f"Streaming error occurred: {str(e)}"
                            n8n_response = error_msg
                            if __event_emitter__:
                                await __event_emitter__(
                                    {
                                        "type": "chat:message",
                                        "data": {
                                            "role": "assistant",
                                            "content": error_msg,
                                        },
                                    }
                                )
                    finally:
                        await cleanup_response(response, session)

                    # Update conversation
                    body["messages"].append(
                        {"role": "assistant", "content": n8n_response}
                    )
                    await self.emit_simple_status(
                        __event_emitter__, "complete", "Streaming complete", True
                    )
                    return n8n_response
                else:
                    # Non-streaming response
                    self.log.info("Processing regular response from N8N")

                    async def read_body_safely():
                        text_body = None
                        json_body = None
                        lowered = content_type.lower()
                        try:
                            if "application/json" in lowered or "json" in lowered:
                                try:
                                    json_body = await response.json(
                                        content_type=None
                                    )
                                except Exception as je:
                                    self.log.warning(
                                        f"Direct JSON parse failed: {je}"
                                    )
                            if json_body is None:
                                text_body = await response.text()
                                try:
                                    json_body = json.loads(text_body)
                                except Exception:
                                    pass
                        except Exception as e_inner:
                            self.log.error(
                                f"Error reading response body: {e_inner}"
                            )
                        return json_body, text_body

                    response_json, response_text = await read_body_safely()

                    def extract_message(data) -> str:
                        if data is None:
                            return ""
                        if isinstance(data, dict):
                            if self.valves.RESPONSE_FIELD in data and isinstance(
                                data[self.valves.RESPONSE_FIELD], (str, list)
                            ):
                                val = data[self.valves.RESPONSE_FIELD]
                                if isinstance(val, list):
                                    return "\n".join(str(v) for v in val if v)
                                return str(val)
                            for key in (
                                "content",
                                "text",
                                "output",
                                "answer",
                                "message",
                            ):
                                if key in data and isinstance(
                                    data[key], (str, list)
                                ):
                                    val = data[key]
                                    return (
                                        "\n".join(val)
                                        if isinstance(val, list)
                                        else str(val)
                                    )
                            try:
                                flat = []
                                for k, v in data.items():
                                    if isinstance(v, (str, int, float)):
                                        flat.append(f"{k}: {v}")
                                return "\n".join(flat)
                            except Exception:
                                return ""
                        if isinstance(data, list):
                            for item in data:
                                m = extract_message(item)
                                if m:
                                    return m
                            return ""
                        if isinstance(data, (str, int, float)):
                            return str(data)
                        return ""

                    n8n_response = extract_message(response_json)
                    if not n8n_response and response_text:
                        n8n_response = response_text.rstrip()

                    if not n8n_response:
                        n8n_response = "(Received empty response from N8N)"

                    # Process think blocks
                    try:
                        if n8n_response and "<think>" in n8n_response.lower():
                            normalized_response = n8n_response.replace("\\n", "\n")

                            think_pattern = re.compile(
                                r"<think>\s*(.*?)\s*</think>",
                                re.IGNORECASE | re.DOTALL,
                            )

                            think_counter = 0

                            def replace_think_block(match):
                                nonlocal think_counter
                                think_counter += 1
                                thought_content = match.group(1).strip()

                                quoted_lines = []
                                for line in thought_content.split("\n"):
                                    quoted_lines.append(f"> {line}")
                                quoted_content = "\n".join(quoted_lines)

                                return f"""<details>
<summary>Thought {think_counter}</summary>

{quoted_content}

</details>"""

                            n8n_response = think_pattern.sub(
                                replace_think_block, normalized_response
                            )
                    except Exception as post_e:
                        self.log.debug(
                            f"Non-streaming thinking parse failed: {post_e}"
                        )

                    # Cleanup
                    await cleanup_response(response, session)
                    session = None

                    # Append assistant message
                    body["messages"].append(
                        {"role": "assistant", "content": n8n_response}
                    )

                    await self.emit_simple_status(
                        __event_emitter__, "complete", "Complete", True
                    )
                    return n8n_response

            else:
                error_text = await response.text()
                self.log.error(
                    f"N8N error: Status {response.status} - {error_text}"
                )
                await cleanup_response(response, session)

                # Parse error message
                user_error_msg = f"N8N Error {response.status}"
                try:
                    error_json = json.loads(error_text)
                    if "message" in error_json:
                        user_error_msg = f"N8N Error: {error_json['message']}"
                    if "hint" in error_json:
                        user_error_msg += f"\n\nHint: {error_json['hint']}"
                except:
                    if error_text:
                        truncated = (
                            error_text[:200] + "..."
                            if len(error_text) > 200
                            else error_text
                        )
                        user_error_msg = f"N8N Error {response.status}: {truncated}"

                await self.emit_simple_status(
                    __event_emitter__, "error", user_error_msg, True
                )
                return user_error_msg

        except Exception as e:
            error_msg = f"Connection or processing error: {str(e)}"
            self.log.exception(error_msg)

            # Clean up session
            if session:
                await session.close()

            await self.emit_simple_status(
                __event_emitter__,
                "error",
                error_msg,
                True,
            )
            return error_msg
