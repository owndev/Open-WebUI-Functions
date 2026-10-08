"""
title: n8n Pipeline with StreamingResponse Support
author: owndev
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
n8n_template: https://github.com/owndev/Open-WebUI-Functions/blob/main/pipelines/n8n/Open_WebUI_Test_Agent_Streaming.json
version: 2.3.1
required_open_webui_version: 0.8.0
license: Apache License 2.0
description: An optimized streaming-enabled pipeline for interacting with N8N workflows, consistent response handling for both streaming and non-streaming modes, robust error handling, and simplified status management. Supports Server-Sent Events (SSE) streaming and various N8N workflow formats. Now includes configurable AI Agent tool usage display with three verbosity levels (minimal, compact, detailed) and customizable length limits for tool inputs/outputs (non-streaming mode only).
features:
  - Integrates with N8N for seamless streaming communication.
  - Uses FastAPI StreamingResponse for real-time streaming.
  - Enables real-time interaction with N8N workflows.
  - Provides configurable status emissions and chunk streaming.
  - Cloudflare Access support for secure communication.
  - Encrypted storage of sensitive API keys.
  - Fallback support for non-streaming responses.
  - Compatible with Open WebUI streaming architecture.
  - Displays N8N AI Agent tool usage with configurable verbosity (non-streaming mode only).
  - Three display modes: minimal (tool names only), compact (names + preview), detailed (full collapsible sections).
  - Customizable length limits for tool inputs and outputs.
  - Shows tool calls, inputs, and results from intermediateSteps in non-streaming mode (N8N limitation - streaming responses do not include intermediateSteps).
  - Parses n8n native streaming (NDJSON), Server-Sent Events (data lines, [DONE], comments) and plain-text streams line by line without leaking SSE framing.
  - Forwards token usage from a non-streamed JSON reply of the workflow to Open WebUI, for streaming and non-streaming chats.
  - Sends Open WebUI tasks (title, tags, follow-ups, ...) without chat_id/message_id, so they stay out of the workflow's chat memory.
  - Final status on completion, on error or when the response is stopped.
  - Shows n8n stream error chunks as 'N8N Error: ...' and ends a failed or interrupted stream with an error status.
  - Parses large JSON objects streamed in small chunks in linear time, without blocking Open WebUI.
"""

from typing import (
    Optional,
    Callable,
    Awaitable,
    Any,
    Dict,
    Union,
    Generator,
    Iterator,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
import aiohttp
import os
import base64
import codecs
import hashlib
import logging
import json
import asyncio
from open_webui.env import AIOHTTP_CLIENT_TIMEOUT, SRC_LOG_LEVELS
from pydantic_core import core_schema
import time
import re
import uuid


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


# Characters the JSON object scanner has to look at, outside and inside strings
_JSON_STRUCTURE = re.compile(r'[{}"]')
_JSON_STRING_SPECIAL = re.compile(r'["\\]')
_NON_SPACE = re.compile(r"\S")


def _scan_json_object(text: str, pos: int, depth: int, in_string: bool) -> tuple:
    """
    Scan text from pos for the brace that closes a JSON object, ignoring braces
    inside JSON strings. The scan can be resumed with the returned state once more
    text has been appended, so an object that arrives in many small chunks is
    scanned only once (linear instead of quadratic time).

    Returns:
        (index of the closing brace or -1, pos, depth, in_string) where the last
        three values are the state to resume from
    """
    end = len(text)
    while pos < end:
        if in_string:
            match = _JSON_STRING_SPECIAL.search(text, pos)
            if match is None:
                return -1, end, depth, True
            i = match.start()
            if text[i] == "\\":
                pos = i + 2  # Skip the escaped character (it may arrive later)
                continue
            in_string = False
            pos = i + 1
        else:
            match = _JSON_STRUCTURE.search(text, pos)
            if match is None:
                return -1, end, depth, False
            i = match.start()
            pos = i + 1
            if text[i] == '"':
                in_string = True
            elif text[i] == "{":
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    return i, pos, depth, False
    return -1, pos, depth, in_string


def _find_json_object_end(text: str, start: int = 0) -> int:
    """
    Return the index of the brace closing the JSON object that starts at
    text[start], or -1 if the object is not complete yet. Braces inside JSON
    strings are ignored.
    """
    return _scan_json_object(text, start, 0, False)[0]


def _split_json_objects(text: str) -> tuple:
    """
    Split back-to-back JSON objects ('{..}{..}' or NDJSON) from the start of text.

    Returns:
        (list of parsed objects, unparsed remainder)
    """
    objects = []
    pos = 0
    while True:
        match = _NON_SPACE.search(text, pos)
        if match is None or text[match.start()] != "{":
            break
        start = match.start()
        end = _find_json_object_end(text, start)
        if end == -1:
            break
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            break
        objects.append(obj)
        pos = end + 1
    return objects, text[pos:]


class N8NStreamParser:
    """
    Incremental parser for streamed n8n replies.

    Text is processed line by line first:
      - SSE framing is removed: 'data:' prefixes are stripped, '[DONE]' sentinels,
        ':' comments and 'event:'/'id:'/'retry:' fields are skipped, consecutive
        'data:' lines of one event are joined with a newline (SSE spec)
      - JSON payloads (n8n native streaming / NDJSON, OpenAI-style chunks) are
        parsed and their content extracted, 'intermediateSteps' are collected
      - n8n error chunks ({"type": "error", ...}) are kept as content and
        collected in 'errors'
      - anything else is plain text and kept, including its line break
    A string-aware brace matcher is used only for JSON objects, so objects written
    back to back ('{..}{..}') or split across network chunks are still parsed
    without dropping plain text that arrived in the same chunk. The matcher keeps
    its state across feed() calls, so every character is scanned only once.
    """

    _SSE_FIELD = re.compile(r"^(event|id|retry)\s*:")

    def __init__(
        self,
        extract_content: Callable[[dict], Optional[str]],
        sse: bool = False,
    ):
        self._extract_content = extract_content
        self.sse = sse
        self.intermediate_steps: list = []
        self.errors: list = []  # Contents of n8n error chunks
        self._has_output = False
        self._buffer = ""
        self._start = 0  # Start of the unprocessed text in _buffer
        self._sse_data: list = []
        self._reset_scan()

    def _reset_scan(self) -> None:
        """
        Forget what is known about the unprocessed text. Positions are relative
        to self._start and stay valid while text is only appended.
        """
        self._space = 0  # Leading characters known to be whitespace
        self._obj = -1  # Start of a JSON object at the beginning, -1 if none
        self._obj_scan = (0, 0, False)  # Brace scan state: pos, depth, in_string
        self._obj_end = -1  # Closing brace of that object, -1 while incomplete
        self._obj_is_text = False  # Complete, but not JSON: handle it as text
        self._newline_from = 0  # There is no line break before this position

    def _consume(self, end: int) -> None:
        """Mark the text before the absolute position end as processed."""
        self._start = end
        self._reset_scan()

    def feed(self, text: str) -> list:
        """Add decoded text and return the content pieces that are complete."""
        self._buffer += text
        return self._drain(final=False)

    def close(self) -> list:
        """Flush everything that is still buffered at the end of the stream."""
        out = self._drain(final=True)
        line = self._buffer
        self._buffer = ""
        self._reset_scan()
        if line:
            # Last line without a trailing newline
            out += self._handle_line(line, newline=False)
        out += self._flush_sse_event()
        return out

    def _drain(self, final: bool) -> list:
        out = []
        while self._start < len(self._buffer):
            # NDJSON / concatenated JSON: parse an object as soon as it is complete
            if not self._sse_data:
                end = self._json_object_end()
                if end >= 0 and not self._obj_is_text:
                    try:
                        obj = json.loads(
                            self._buffer[self._start + self._obj : end + 1]
                        )
                    except json.JSONDecodeError:
                        self._obj_is_text = True  # Not JSON: handle it as text
                    else:
                        self._consume(end + 1)
                        out += self._handle_json(obj)
                        continue
                elif end == -1 and not final:
                    break  # Incomplete JSON object, wait for more data
            newline = self._buffer.find("\n", self._start + self._newline_from)
            if newline == -1:
                self._newline_from = len(self._buffer) - self._start
                break
            line = self._buffer[self._start : newline]
            self._consume(newline + 1)
            out += self._handle_line(line, newline=True)
        if self._start:
            # Drop the processed text once per call (positions are relative)
            self._buffer = self._buffer[self._start :]
            self._start = 0
        return out

    def _json_object_end(self) -> int:
        """
        Return the absolute index of the brace closing the JSON object at the
        start of the unprocessed text, -1 while that object is incomplete, or -2
        if the text does not start with '{'. Resumes the previous scan.
        """
        if self._obj_end >= 0:
            return self._start + self._obj_end
        if self._obj < 0:
            match = _NON_SPACE.search(self._buffer, self._start + self._space)
            if match is None:
                self._space = len(self._buffer) - self._start
                return -2
            self._space = match.start() - self._start
            if self._buffer[match.start()] != "{":
                return -2
            self._obj = self._space
            self._obj_scan = (self._obj, 0, False)
        pos, depth, in_string = self._obj_scan
        end, pos, depth, in_string = _scan_json_object(
            self._buffer, self._start + pos, depth, in_string
        )
        self._obj_scan = (pos - self._start, depth, in_string)
        if end == -1:
            return -1
        self._obj_end = end - self._start
        return end

    def _handle_line(self, line: str, newline: bool) -> list:
        line = line.rstrip("\r")
        if line.startswith("data:"):
            self.sse = True
            payload = line[5:]
            if payload.startswith(" "):
                payload = payload[1:]
            self._sse_data.append(payload)
            return []

        # Any other line ends the pending SSE event
        out = self._flush_sse_event()
        if not line.strip():
            return out  # Blank line (SSE event separator)
        if self.sse and (line.startswith(":") or self._SSE_FIELD.match(line)):
            return out  # SSE comment / keep-alive or non-data field
        return out + self._handle_payload(line, "\n" if newline else "")

    def _flush_sse_event(self) -> list:
        if not self._sse_data:
            return []
        payload = "\n".join(self._sse_data)
        self._sse_data = []
        if payload.strip() == "[DONE]":
            return []
        return self._handle_payload(payload, "")

    def _handle_payload(self, payload: str, plain_suffix: str) -> list:
        stripped = payload.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                objects, rest = _split_json_objects(stripped)
                if objects:
                    out = []
                    for obj in objects:
                        out += self._handle_json(obj)
                    if rest.strip():
                        out += self._output(rest + plain_suffix)
                    return out
            else:
                if isinstance(parsed, dict):
                    parsed = [parsed]
                if (
                    isinstance(parsed, list)
                    and parsed
                    and all(isinstance(item, dict) for item in parsed)
                ):
                    out = []
                    for obj in parsed:
                        out += self._handle_json(obj)
                    return out
        return self._output(payload + plain_suffix)

    def _handle_json(self, obj: Any) -> list:
        if not isinstance(obj, dict):
            return []
        steps = obj.get("intermediateSteps")
        if isinstance(steps, list) and steps:
            self.intermediate_steps.extend(steps)
        content = self._extract_content(obj)
        if content and obj.get("type") == "error":
            self.errors.append(content)
            if self._has_output:
                content = "\n\n" + content  # Separate it from the answer so far
        return self._output(content)

    def _output(self, text: Optional[str]) -> list:
        if not text:
            return []
        self._has_output = True
        return [text]


class Pipe:
    class Valves(BaseModel):
        N8N_URL: str = Field(
            default="https://<your-endpoint>/webhook/<your-webhook>",
            description="URL for the N8N webhook",
        )
        N8N_BEARER_TOKEN: EncryptedStr = Field(
            default="",
            description="Bearer token for authenticating with the N8N webhook",
            json_schema_extra={"input": {"type": "password"}},
        )
        INPUT_FIELD: str = Field(
            default="chatInput",
            description="Field name for the input message in the N8N payload",
        )
        RESPONSE_FIELD: str = Field(
            default="output",
            description="Field name for the response message in the N8N payload",
        )
        SEND_CONVERSATION_HISTORY: bool = Field(
            default=False,
            description="Whether to include conversation history when sending requests to N8N",
        )
        TOOL_DISPLAY_VERBOSITY: str = Field(
            default="detailed",
            description="Verbosity level for tool usage display: 'minimal' (only tool names), 'compact' (names + short preview), 'detailed' (full info with collapsible sections)",
        )
        TOOL_INPUT_MAX_LENGTH: int = Field(
            default=500,
            description="Maximum length for tool input display (0 = unlimited). Longer inputs will be truncated.",
        )
        TOOL_OUTPUT_MAX_LENGTH: int = Field(
            default=500,
            description="Maximum length for tool output/observation display (0 = unlimited). Longer outputs will be truncated.",
        )
        CF_ACCESS_CLIENT_ID: EncryptedStr = Field(
            default="",
            description="Only if behind Cloudflare: https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/",
            json_schema_extra={"input": {"type": "password"}},
        )
        CF_ACCESS_CLIENT_SECRET: EncryptedStr = Field(
            default="",
            description="Only if behind Cloudflare: https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/",
            json_schema_extra={"input": {"type": "password"}},
        )

    def __init__(self):
        self.name = "N8N Agent"
        self.valves = self.Valves()
        self.log = logging.getLogger("n8n_streaming_pipeline")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

    def _format_tool_calls_section(
        self, intermediate_steps: list, for_streaming: bool = False
    ) -> str:
        """
        Creates a formatted tool calls section using collapsible details elements.

        Args:
            intermediate_steps: List of intermediate step objects from N8N response
            for_streaming: If True, format for streaming (with escaping), else for regular response

        Returns:
            Formatted tool calls section with HTML details elements
        """
        if not intermediate_steps:
            return ""

        verbosity = self.valves.TOOL_DISPLAY_VERBOSITY.lower()
        input_max_len = self.valves.TOOL_INPUT_MAX_LENGTH
        output_max_len = self.valves.TOOL_OUTPUT_MAX_LENGTH

        # Helper function to truncate text
        def truncate_text(text: str, max_length: int) -> str:
            if max_length <= 0 or len(text) <= max_length:
                return text
            return text[:max_length] + "..."

        # Minimal mode: just list tool names
        if verbosity == "minimal":
            tool_names = []
            for i, step in enumerate(intermediate_steps, 1):
                if isinstance(step, dict):
                    tool_name = step.get("action", {}).get("tool", "Unknown Tool")
                    tool_names.append(f"{i}. {tool_name}")

            tool_list = "\\n" if for_streaming else "\n"
            tool_list = tool_list.join(tool_names)

            if for_streaming:
                return f"\\n\\n<details>\\n<summary>🛠️ Tool Calls ({len(intermediate_steps)} steps)</summary>\\n\\n{tool_list}\\n\\n</details>\\n"
            else:
                return f"\n\n<details>\n<summary>🛠️ Tool Calls ({len(intermediate_steps)} steps)</summary>\n\n{tool_list}\n\n</details>\n"

        # Compact mode: tool names with short preview
        if verbosity == "compact":
            tool_summaries = []
            for i, step in enumerate(intermediate_steps, 1):
                if not isinstance(step, dict):
                    continue

                action = step.get("action", {})
                observation = step.get("observation", "")
                tool_name = action.get("tool", "Unknown Tool")

                # Get short preview of output
                preview = ""
                if observation:
                    obs_str = str(observation)
                    # If output_max_len is 0 (unlimited), use a reasonable default preview length for compact mode
                    # Otherwise, use the configured limit
                    if output_max_len > 0:
                        preview_len = min(100, output_max_len)
                    else:
                        preview_len = 100  # Default preview length for compact mode when unlimited
                    preview = truncate_text(obs_str, preview_len)

                summary = f"**{i}. {tool_name}**"
                if preview:
                    summary += f" → {preview}"
                tool_summaries.append(summary)

            summary_text = "\\n" if for_streaming else "\n"
            summary_text = summary_text.join(tool_summaries)

            if for_streaming:
                return f"\\n\\n<details>\\n<summary>🛠️ Tool Calls ({len(intermediate_steps)} steps)</summary>\\n\\n{summary_text}\\n\\n</details>\\n"
            else:
                return f"\n\n<details>\n<summary>🛠️ Tool Calls ({len(intermediate_steps)} steps)</summary>\n\n{summary_text}\n\n</details>\n"

        # Detailed mode: full collapsible sections (default)
        tool_entries = []

        for i, step in enumerate(intermediate_steps, 1):
            if not isinstance(step, dict):
                continue

            action = step.get("action", {})
            observation = step.get("observation", "")

            tool_name = action.get("tool", "Unknown Tool")
            tool_input = action.get("toolInput", {})
            tool_call_id = action.get("toolCallId", "")
            log_message = action.get("log", "")

            # Build individual tool call details
            tool_info = []
            tool_info.append(f"🔧 **Tool:** {tool_name}")

            if tool_call_id:
                tool_info.append(f"🆔 **Call ID:** `{tool_call_id}`")

            # Format tool input
            if tool_input:
                try:
                    if isinstance(tool_input, dict):
                        input_json = json.dumps(tool_input, indent=2)

                        # Apply max length limit
                        if input_max_len > 0:
                            input_json = truncate_text(input_json, input_max_len)

                        if for_streaming:
                            # Escape for streaming
                            input_json = (
                                input_json.replace("\\", "\\\\")
                                .replace('"', '\\"')
                                .replace("\n", "\\n")
                            )
                            tool_info.append(
                                f"📥 **Input:**\\n```json\\n{input_json}\\n```"
                            )
                        else:
                            tool_info.append(
                                f"📥 **Input:**\n```json\n{input_json}\n```"
                            )
                    else:
                        input_str = str(tool_input)
                        if input_max_len > 0:
                            input_str = truncate_text(input_str, input_max_len)
                        tool_info.append(f"📥 **Input:** `{input_str}`")
                except Exception:
                    input_str = str(tool_input)
                    if input_max_len > 0:
                        input_str = truncate_text(input_str, input_max_len)
                    tool_info.append(f"📥 **Input:** `{input_str}`")

            # Format observation/result
            if observation:
                try:
                    # Try to parse as JSON for better formatting
                    if isinstance(observation, str) and (
                        observation.startswith("[") or observation.startswith("{")
                    ):
                        obs_json = json.loads(observation)
                        obs_formatted = json.dumps(obs_json, indent=2)

                        # Apply max length limit
                        if output_max_len > 0:
                            obs_formatted = truncate_text(obs_formatted, output_max_len)

                        if for_streaming:
                            obs_formatted = (
                                obs_formatted.replace("\\", "\\\\")
                                .replace('"', '\\"')
                                .replace("\n", "\\n")
                            )
                            tool_info.append(
                                f"📤 **Result:**\\n```json\\n{obs_formatted}\\n```"
                            )
                        else:
                            tool_info.append(
                                f"📤 **Result:**\n```json\n{obs_formatted}\n```"
                            )
                    else:
                        # Plain text observation
                        obs_str = str(observation)
                        # Apply configured limit (0 = unlimited, don't truncate)
                        obs_preview = (
                            truncate_text(obs_str, output_max_len)
                            if output_max_len > 0
                            else obs_str
                        )

                        if for_streaming:
                            obs_preview = (
                                obs_preview.replace("\\", "\\\\")
                                .replace('"', '\\"')
                                .replace("\n", "\\n")
                            )
                        tool_info.append(f"📤 **Result:** {obs_preview}")
                except Exception:
                    obs_str = str(observation)
                    # Apply configured limit (0 = unlimited, don't truncate)
                    obs_preview = (
                        truncate_text(obs_str, output_max_len)
                        if output_max_len > 0
                        else obs_str
                    )
                    tool_info.append(f"📤 **Result:** {obs_preview}")

            # Add log if available
            if log_message:
                log_preview = truncate_text(log_message, 200)
                tool_info.append(f"📝 **Log:** {log_preview}")

            # Create collapsible details for individual tool call
            tool_info_text = "\\n" if for_streaming else "\n"
            tool_info_text = tool_info_text.join(tool_info)

            if for_streaming:
                tool_entry = f"<details>\\n<summary>Step {i}: {tool_name}</summary>\\n\\n{tool_info_text}\\n\\n</details>"
            else:
                tool_entry = f"<details>\n<summary>Step {i}: {tool_name}</summary>\n\n{tool_info_text}\n\n</details>"

            tool_entries.append(tool_entry)

        # Combine all tool calls into main collapsible section
        if for_streaming:
            all_tools = "\\n\\n".join(tool_entries)
            result = f"\\n\\n<details>\\n<summary>🛠️ Tool Calls ({len(tool_entries)} steps)</summary>\\n\\n{all_tools}\\n\\n</details>\\n"
        else:
            all_tools = "\n\n".join(tool_entries)
            result = f"\n\n<details>\n<summary>🛠️ Tool Calls ({len(tool_entries)} steps)</summary>\n\n{all_tools}\n\n</details>\n"

        return result

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

    def _stream_with_usage(
        self, model: str, content: str, usage: Any
    ) -> Generator[dict, None, None]:
        """
        Yield a complete answer and its token usage as OpenAI chat.completion.chunk
        objects, for streaming requests answered by a non-streaming n8n reply.

        This is a plain (sync) generator on purpose: Open WebUI appends the
        finish_reason "stop" chunk and "data: [DONE]" to a Generator result in
        every supported version, but in older versions (e.g. 0.8.0) not to an
        async generator.
        """
        base = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
        }
        yield {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        }
        yield {**base, "choices": [], "usage": usage}

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

    def extract_stream_chunk_content(self, data: dict) -> Optional[str]:
        """Extract the content of one parsed N8N streaming JSON object, skipping metadata"""
        # Check if this chunk contains intermediateSteps (will be handled separately)
        # Note: Don't skip chunks just because they have a type field
        chunk_type = data.get("type", "")

        # n8n reports a failed node as {"type": "error", "content": "<message>"}:
        # show it instead of dropping it (the stream then ends with an error status)
        if chunk_type == "error":
            message = data.get("content") or data.get("message")
            return f"N8N Error: {message or 'The workflow reported an error.'}"

        # Skip only true metadata chunks that have no content or intermediateSteps
        if (
            chunk_type in ["begin", "end", "metadata"]
            and "intermediateSteps" not in data
        ):
            self.log.debug(f"Skipping N8N metadata chunk: {chunk_type}")
            return None

        # Skip metadata-only chunks (but allow intermediateSteps)
        if "metadata" in data and len(data) <= 2 and "intermediateSteps" not in data:
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
                delta = choices[0].get("delta") or {}
                content = delta.get("content", "") if isinstance(delta, dict) else ""

        if content:
            self.log.debug(f"Extracted content from JSON: {repr(str(content)[:100])}")
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
                "choices",
                "usage",
            ]
        ):
            # For smaller models, return the entire object if it's simple
            self.log.debug(
                f"Returning entire object as content: {repr(str(data)[:100])}"
            )
            return str(data)

        return None

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
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
        __task__: Optional[str] = None,
    ) -> Union[str, Generator, Iterator, Dict[str, Any], StreamingResponse]:
        """
        Main method for sending requests to the N8N endpoint.

        Args:
            body: The request body containing messages and other parameters
            __event_emitter__: Optional event emitter function for status updates
            __chat_id__: Chat ID passed by Open WebUI (empty for API calls without a chat)
            __message_id__: Assistant message ID passed by Open WebUI
            __task__: Open WebUI task type (title, tags, follow-ups, ...), None for chat turns

        Returns:
            Response from N8N API: a string, an OpenAI-style dict (non-streaming
            request with usage) or a generator of OpenAI chunks (streaming request
            with usage)
        """
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

        await self.emit_simple_status(
            __event_emitter__, "in_progress", f"Calling {self.name} ...", False
        )

        session = None
        n8n_response = ""
        messages = body.get("messages", [])

        # Verify a message is available
        if messages:
            question = messages[-1]["content"]
            if "Prompt: " in question:
                question = question.split("Prompt: ")[-1]
            try:
                if __task__:
                    # Open WebUI tasks (title, tag, follow-up, search query
                    # generation, ...) are not chat turns. Send them without
                    # chat_id / message_id (title, tag and follow-up tasks were
                    # already sent that way), so workflows that key their memory on
                    # chat_id (like the templates) do not store task prompts in the
                    # chat's memory.
                    chat_id, message_id = None, None
                    self.log.debug(f"Open WebUI task '{__task__}': no chat context")
                else:
                    # chat_id / message_id are passed by Open WebUI; fall back to
                    # the event emitter closure if they are missing
                    chat_id, message_id = __chat_id__, __message_id__
                    if not chat_id or not message_id:
                        fallback_chat_id, fallback_message_id = self.extract_event_info(
                            __event_emitter__
                        )
                        chat_id = chat_id or fallback_chat_id
                        message_id = message_id or fallback_message_id

                self.log.info(f"Starting N8N workflow request for chat ID: {chat_id}")

                # Extract system prompt correctly
                system_prompt = ""
                if messages and messages[0].get("role") == "system":
                    system_prompt = self.dedupe_system_prompt(messages[0]["content"])

                # Optionally include full conversation history (controlled by valve)
                conversation_history = []
                if self.valves.SEND_CONVERSATION_HISTORY:
                    for msg in messages:
                        if msg.get("role") in ["user", "assistant"]:
                            conversation_history.append(
                                {"role": msg["role"], "content": msg["content"]}
                            )

                # Prepare payload for N8N workflow (improved version)
                payload = {
                    "systemPrompt": system_prompt,
                    # Include messages only when enabled in valves for privacy/control
                    "messages": (
                        conversation_history
                        if self.valves.SEND_CONVERSATION_HISTORY
                        else []
                    ),
                    "currentMessage": question,  # Current user message
                    "user_id": __user__.get("id") if __user__ else None,
                    "user_email": __user__.get("email") if __user__ else None,
                    "user_name": __user__.get("name") if __user__ else None,
                    "user_role": __user__.get("role") if __user__ else None,
                    "chat_id": chat_id,
                    "message_id": message_id,
                }
                # Keep backward compatibility
                payload[self.valves.INPUT_FIELD] = question

                # Get headers for the request
                headers = self.get_headers()

                # Create session with no timeout like in stream-example.py
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
                    # Enhanced streaming detection (n8n controls streaming)
                    content_type = response.headers.get("Content-Type", "").lower()

                    # Check for explicit streaming indicators
                    # Note: Don't rely solely on Transfer-Encoding: chunked as regular JSON can also be chunked
                    is_streaming = (
                        "text/event-stream" in content_type
                        or "application/x-ndjson" in content_type
                        or (
                            "application/json" in content_type
                            and response.headers.get("Transfer-Encoding") == "chunked"
                            and "Cache-Control" in response.headers
                            and "no-cache"
                            in response.headers.get("Cache-Control", "").lower()
                        )
                    )

                    # Additional check: if content-type is text/html or application/json without streaming headers, it's likely not streaming
                    if "text/html" in content_type:
                        is_streaming = False
                    elif (
                        "application/json" in content_type
                        and "Cache-Control" not in response.headers
                    ):
                        is_streaming = False

                    if is_streaming:
                        # Enhanced streaming like in stream-example.py
                        self.log.info("Processing streaming response from N8N")
                        n8n_response = ""
                        completed_thoughts: list[str] = []
                        # Line-oriented parser: strips SSE framing, parses NDJSON /
                        # concatenated JSON and keeps plain text in the same chunk
                        parser = N8NStreamParser(
                            self.extract_stream_chunk_content,
                            sse="text/event-stream" in content_type,
                        )
                        # Tool calls found in streamed JSON objects (future-proof:
                        # works automatically if N8N adds intermediateSteps to streams)
                        intermediate_steps = parser.intermediate_steps
                        # Incremental decoder so multi-byte characters split across
                        # network chunks are not dropped
                        decoder = codecs.getincrementaldecoder("utf-8")(
                            errors="replace"
                        )

                        async def emit_stream_content(pieces: list) -> None:
                            nonlocal n8n_response
                            for content in pieces:
                                if not content:
                                    continue
                                # Normalize escaped newlines to actual newlines (like non-streaming)
                                content = content.replace("\\n", "\n")

                                # Just accumulate content without processing think blocks yet
                                n8n_response += content

                                # Emit delta without think block processing
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

                        # Exception that ended the stream early, if any
                        stream_error: Optional[Exception] = None
                        try:
                            async for chunk in response.content.iter_any():
                                if not chunk:
                                    continue
                                await emit_stream_content(
                                    parser.feed(decoder.decode(chunk))
                                )

                            # Flush the decoder and whatever is left in the buffer
                            await emit_stream_content(
                                parser.feed(decoder.decode(b"", final=True))
                                + parser.close()
                            )
                            if intermediate_steps:
                                self.log.info(
                                    f"✓ Found {len(intermediate_steps)} intermediate steps in streaming response"
                                )

                            # NOW process all think blocks in the complete response
                            if n8n_response and "<think>" in n8n_response.lower():
                                # Use regex to find and replace all think blocks at once
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

                                        # Format each line with > for blockquote while preserving formatting
                                        quoted_lines = []
                                        for line in thought_content.split("\n"):
                                            quoted_lines.append(f"> {line}")
                                        quoted_content = "\n".join(quoted_lines)

                                        # Return details block with custom thought formatting
                                        return f"""<details>
<summary>Thought {think_counter}</summary>

{quoted_content}

</details>"""
                                    return ""

                                # Replace all think blocks with details blocks in the complete response
                                n8n_response = think_pattern.sub(
                                    replace_think_block, n8n_response
                                )

                            # ALWAYS emit final complete message (critical for UI update)
                            if __event_emitter__:
                                # Ensure we have some response to show
                                if not n8n_response.strip():
                                    n8n_response = "(Empty response received from N8N)"
                                    self.log.warning(
                                        "Empty response received from N8N, using fallback message"
                                    )

                                # Add tool calls section if present
                                if intermediate_steps:
                                    tool_calls_section = (
                                        self._format_tool_calls_section(
                                            intermediate_steps, for_streaming=False
                                        )
                                    )
                                    if tool_calls_section:
                                        n8n_response += tool_calls_section
                                        self.log.info(
                                            f"Added {len(intermediate_steps)} tool calls to response"
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
                                    # Clear any thinking status indicator
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
                                f"Streaming completed successfully. Total response length: {len(n8n_response)}"
                            )

                        except Exception as e:
                            stream_error = e
                            self.log.error(f"Streaming error: {e}")

                            # In case of streaming errors, try to emit whatever we have
                            if n8n_response:
                                self.log.info(
                                    f"Emitting partial response due to error: {len(n8n_response)} chars"
                                )
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
                                # If no response was accumulated, provide error message
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

                        # Update conversation with response
                        body["messages"].append(
                            {"role": "assistant", "content": n8n_response}
                        )
                        # Final status: an interrupted stream or an n8n error chunk
                        # must not end with a success status
                        if stream_error is not None:
                            await self.emit_simple_status(
                                __event_emitter__,
                                "error",
                                f"N8N streaming error: {stream_error}",
                                True,
                            )
                        elif parser.errors:
                            self.log.warning(
                                f"Error chunk in the N8N stream: {parser.errors[-1]}"
                            )
                            await self.emit_simple_status(
                                __event_emitter__, "error", parser.errors[-1], True
                            )
                        else:
                            await self.emit_simple_status(
                                __event_emitter__,
                                "complete",
                                "Streaming complete",
                                True,
                            )
                        return n8n_response
                    else:
                        # Fallback to non-streaming response (robust parsing)
                        self.log.info(
                            "Processing regular response from N8N (non-streaming)"
                        )

                        async def read_body_safely():
                            text_body = None
                            json_body = None
                            try:
                                # Read as text first (works for all content types)
                                text_body = await response.text()

                                # Try to parse as JSON regardless of content-type
                                # (N8N might return JSON with text/html content-type)
                                try:
                                    json_body = json.loads(text_body)
                                    self.log.debug(
                                        f"Successfully parsed response body as JSON (content-type was: {content_type})"
                                    )
                                except json.JSONDecodeError:
                                    # If it starts with [{ or { it might be JSON wrapped in something
                                    if text_body.strip().startswith(
                                        "[{"
                                    ) or text_body.strip().startswith("{"):
                                        self.log.warning(
                                            f"Response looks like JSON but failed to parse (content-type: {content_type})"
                                        )
                                    else:
                                        self.log.debug(
                                            f"Response is not JSON, will use as plain text (content-type: {content_type})"
                                        )
                            except Exception as e_inner:
                                self.log.error(
                                    f"Error reading response body: {e_inner}"
                                )
                            return json_body, text_body

                        response_json, response_text = await read_body_safely()
                        self.log.debug(f"Parsed JSON body: {response_json}")
                        if response_json is None and response_text:
                            snippet = (
                                (response_text[:300] + "...")
                                if len(response_text) > 300
                                else response_text
                            )
                            self.log.debug(f"Raw text body snippet: {snippet}")

                        # Extract intermediateSteps from non-streaming response
                        intermediate_steps = []
                        if isinstance(response_json, list):
                            # Handle array response format
                            self.log.debug(
                                f"Response is an array with {len(response_json)} items"
                            )
                            for item in response_json:
                                if (
                                    isinstance(item, dict)
                                    and "intermediateSteps" in item
                                ):
                                    steps = item.get("intermediateSteps", [])
                                    intermediate_steps.extend(steps)
                                    self.log.debug(
                                        f"Found {len(steps)} intermediate steps in array item"
                                    )
                        elif isinstance(response_json, dict):
                            # Handle single object response format
                            self.log.debug(
                                f"Response is a dict with keys: {list(response_json.keys())}"
                            )
                            intermediate_steps = response_json.get(
                                "intermediateSteps", []
                            )
                            if intermediate_steps:
                                self.log.debug(
                                    f"Found intermediateSteps field with {len(intermediate_steps)} items"
                                )
                        else:
                            self.log.debug(
                                f"Response is not JSON (type: {type(response_json)}), cannot extract intermediateSteps"
                            )

                        if intermediate_steps:
                            self.log.info(
                                f"✓ Found {len(intermediate_steps)} intermediate steps in non-streaming response"
                            )
                        else:
                            self.log.debug(
                                "No intermediate steps found in non-streaming response"
                            )

                        def extract_message(data) -> str:
                            if data is None:
                                return ""
                            if isinstance(data, dict):
                                # Prefer configured field
                                if self.valves.RESPONSE_FIELD in data and isinstance(
                                    data[self.valves.RESPONSE_FIELD], (str, list)
                                ):
                                    val = data[self.valves.RESPONSE_FIELD]
                                    if isinstance(val, list):
                                        return "\n".join(str(v) for v in val if v)
                                    return str(val)
                                # Common generic keys fallback
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
                                # Flatten simple dict of scalars
                                try:
                                    flat = []
                                    for k, v in data.items():
                                        if isinstance(v, (str, int, float)):
                                            flat.append(f"{k}: {v}")
                                    return "\n".join(flat)
                                except Exception:
                                    return ""
                            if isinstance(data, list):
                                # Take first meaningful element
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
                            # Use raw text fallback (strip trailing whitespace only)
                            n8n_response = response_text.rstrip()

                        if not n8n_response:
                            n8n_response = (
                                "(Received empty response or unknown format from N8N)"
                            )

                        # Post-process for <think> blocks (non-streaming mode)
                        try:
                            if n8n_response and "<think>" in n8n_response.lower():
                                # First, normalize escaped newlines to actual newlines
                                normalized_response = n8n_response.replace("\\n", "\n")

                                # Use case-insensitive patterns to find and replace each think block
                                think_pattern = re.compile(
                                    r"<think>\s*(.*?)\s*</think>",
                                    re.IGNORECASE | re.DOTALL,
                                )

                                think_counter = 0

                                def replace_think_block(match):
                                    nonlocal think_counter
                                    think_counter += 1
                                    thought_content = match.group(1).strip()

                                    # Format each line with > for blockquote while preserving formatting
                                    quoted_lines = []
                                    for line in thought_content.split("\n"):
                                        quoted_lines.append(f"> {line}")
                                    quoted_content = "\n".join(quoted_lines)

                                    return f"""<details>
<summary>Thought {think_counter}</summary>

{quoted_content}

</details>"""

                                # Replace each <think>...</think> with its own details block
                                n8n_response = think_pattern.sub(
                                    replace_think_block, normalized_response
                                )
                        except Exception as post_e:
                            self.log.debug(
                                f"Non-streaming thinking parse failed: {post_e}"
                            )

                        # Add tool calls section if present (non-streaming mode)
                        if intermediate_steps:
                            tool_calls_section = self._format_tool_calls_section(
                                intermediate_steps, for_streaming=False
                            )
                            if tool_calls_section:
                                n8n_response += tool_calls_section
                                self.log.info(
                                    f"Added {len(intermediate_steps)} tool calls to non-streaming response"
                                )

                        # Extract token usage from N8N response (best-effort)
                        usage = None
                        if isinstance(response_json, dict):
                            usage = response_json.get("usage")
                        elif isinstance(response_json, list):
                            for item in response_json:
                                if isinstance(item, dict) and "usage" in item:
                                    usage = item["usage"]
                                    break

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

                        # Return OpenAI-format data with usage so the middleware saves it to DB
                        if usage:
                            if body.get("stream"):
                                # Open WebUI forwards a dict returned to a streaming
                                # request as a single SSE event without choices[].delta,
                                # so the answer would be saved empty. Stream the answer
                                # and the usage as chat.completion.chunk events instead
                                # (sync generator: Open WebUI appends the finish chunk
                                # and [DONE], see _stream_with_usage).
                                return self._stream_with_usage(
                                    body.get("model", ""), n8n_response, usage
                                )
                            return {
                                "choices": [
                                    {
                                        "message": {
                                            "role": "assistant",
                                            "content": n8n_response,
                                        }
                                    }
                                ],
                                "usage": usage,
                            }
                        return n8n_response

                else:
                    error_text = await response.text()
                    self.log.error(
                        f"N8N error: Status {response.status} - {error_text}"
                    )
                    await cleanup_response(response, session)

                    # Parse error message for better user experience
                    user_error_msg = f"N8N Error {response.status}"
                    try:
                        error_json = json.loads(error_text)
                        if "message" in error_json:
                            user_error_msg = f"N8N Error: {error_json['message']}"
                        if "hint" in error_json:
                            user_error_msg += f"\n\nHint: {error_json['hint']}"
                    except (ValueError, TypeError):
                        # If not JSON, use raw text but truncate if too long
                        if error_text:
                            truncated = (
                                error_text[:200] + "..."
                                if len(error_text) > 200
                                else error_text
                            )
                            user_error_msg = f"N8N Error {response.status}: {truncated}"

                    # Return error as chat message string
                    await self.emit_simple_status(
                        __event_emitter__, "error", user_error_msg, True
                    )
                    return user_error_msg

            except Exception as e:
                error_msg = f"Connection or processing error: {str(e)}"
                self.log.exception(error_msg)

                # Clean up session if it exists
                if session:
                    await session.close()

                # Return error as chat message string
                await self.emit_simple_status(
                    __event_emitter__,
                    "error",
                    error_msg,
                    True,
                )
                return error_msg
            except asyncio.CancelledError:
                # Stopped by the user (or the client went away): close the request
                # and end the status, so the message keeps no in-progress indicator
                self.log.info("N8N request stopped before completion")
                if session:
                    await session.close()
                try:
                    await self.emit_simple_status(
                        __event_emitter__, "cancelled", "Stopped", True
                    )
                except Exception as emit_error:
                    self.log.debug(f"Could not emit stopped status: {emit_error}")
                raise

        # If no message is available alert user
        else:
            error_msg = "No messages found in the request body"
            self.log.warning(error_msg)
            await self.emit_simple_status(
                __event_emitter__,
                "error",
                error_msg,
                True,
            )
            return error_msg
