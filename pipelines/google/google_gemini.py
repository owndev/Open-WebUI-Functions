"""
title: Google Gemini Pipeline
author: owndev, olivier-lacroix
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 1.19.1
required_open_webui_version: 0.9.0
requirements: google-genai>=1.68.0, google-genai<3
license: Apache License 2.0
description: Highly optimized Google Gemini pipeline with advanced image and video generation capabilities, intelligent compression, and streamlined processing workflows.
features:
  - Optimized asynchronous API calls for maximum performance
  - Intelligent model caching with configurable TTL, refreshed at once when model valves change
  - Streamlined dynamic model specification with automatic prefix handling
  - Smart streaming response handling with safety checks
  - Retries of temporary API errors for streaming and non-streaming requests
  - Advanced multimodal input support (text and images)
  - Unified image generation and editing with Gemini 2.5 Flash Image Preview
  - Image editing across turns: earlier generated and uploaded images of a saved chat are sent with the edit request (the newest kept within the image limit, each image once; the pipeline reads only files of the requesting user)
  - Nano Banana image models (gemini-3.1-flash-image, gemini-3.1-flash-lite-image, gemini-nano-banana-2.1)
  - Extra image generation model IDs configurable without a code change
  - Interim thought images skipped, so each generated image is uploaded once (the last one is kept if no final image arrives)
  - Native tools and URL context left out for image models, which do not support them
  - Search grounding left out for image models without Search support (gemini-2.5-flash-image, gemini-3.1-flash-lite-image)
  - Intelligent image optimization with size-aware compression algorithms
  - Automated image upload to Open WebUI with robust fallback support
  - Generated images and videos linked in the answer for API clients
  - Optimized text-to-image and image-to-image workflows
  - Non-streaming mode for image generation to prevent chunk overflow
  - Progressive status updates, each closed when a request ends, fails or is stopped
  - Consolidated error handling and comprehensive logging
  - Seamless Google Generative AI and Vertex AI integration
  - Advanced generation parameters (temperature, max tokens, etc.)
  - Configurable safety settings with environment variable support
  - Military-grade encrypted storage of sensitive API keys
  - Intelligent grounding with Google search integration, shown with Open WebUI's own localized search statuses (queries, searched sites)
  - Vertex AI Search grounding for RAG
  - Native tool calling through Open WebUI's tool loop (built-in, workspace, MCP, OpenAPI, terminal and direct tools, tool approval)
  - Thought signatures carried across tool rounds and turns
  - Tool calls returned to API clients as OpenAI tool_calls (streaming and non-streaming)
  - Google Search grounding together with function calling on Gemini 3
  - URL context grounding for specified web pages
  - Unified image processing with consolidated helper methods
  - Optimized payload creation for image generation models
  - Configurable image processing parameters (size, quality, compression)
  - Flexible upload fallback options and optimization controls
  - Configurable thinking levels for Gemini 3 models with model-specific validation
  - Thinking shown as Open WebUI's native reasoning block, live while Gemini thinks, with a title in the user's language
  - Thinking summaries stripped from replayed history to save tokens (configurable)
  - Configurable thinking budgets (0-32768 tokens) for Gemini 2.5 models
  - Configurable image generation aspect ratio (1:1, 16:9, etc.) and resolution (1K, 2K, 4K)
  - Model whitelist for filtering available models
  - Additional model support for SDK-unsupported models
  - Video generation with Google Veo models (Veo 3.1, 3, 2)
  - Configurable video generation parameters (aspect ratio, resolution, duration)
  - Asynchronous video generation with progressive polling status updates
  - Automatic video upload to Open WebUI with chat file attachments
  - Image-to-video generation support for Veo models
  - Negative prompt and person generation controls for video
  - Token usage reporting for streaming and non-streaming responses
  - Usable as task model for title, tag and follow-up generation
"""

import os
import re
import sys
import time
import asyncio
import base64
import copy
import hashlib
import importlib
import importlib.metadata
import json
import logging
import io
import uuid
import aiofiles
from PIL import Image
from typing import (
    List,
    Union,
    Optional,
    Dict,
    Any,
    Tuple,
    AsyncIterator,
    Awaitable,
    Callable,
)
from pydantic_core import core_schema
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
from open_webui.env import SRC_LOG_LEVELS
from open_webui.internal.db import get_async_db_context
from fastapi import Request, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
from open_webui.routers.files import upload_file
from open_webui.models.chats import Chats
from open_webui.models.files import Files
from open_webui.models.users import UserModel, Users
from starlette.datastructures import Headers

try:  # Open WebUI 0.9+: the message chain of a saved chat (image history)
    from open_webui.utils.misc import get_message_list
except ImportError:  # without it, image history comes from the request only
    get_message_list = None


def _unload_stale_modules() -> None:
    """
    Drop already imported modules whose installed version changed on disk.

    Open WebUI >= 0.11.4 no longer bundles google-genai, so it is installed at
    runtime from the `requirements` header. google-genai requires websockets<17
    (https://github.com/googleapis/python-genai/issues/2835), so pip downgrades the
    websockets 17.x that uvicorn has already imported. Importing google.genai would
    then mix in-memory 17.x modules with 16.x files and fail with
    "cannot import name 'OP_BINARY' from 'websockets.frames'" until Open WebUI is
    restarted. Unloading the stale modules makes the next import load them from disk.
    """
    unloaded = False
    for distribution, package, version_module in (
        ("websockets", "websockets", "websockets"),
        ("google-genai", "google.genai", "google.genai.version"),
    ):
        loaded = sys.modules.get(version_module)
        if loaded is None:
            continue
        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        loaded_version = getattr(loaded, "__version__", None)
        if loaded_version == installed_version:
            continue
        logging.getLogger("google_ai.pipe").info(
            f"Reloading stale {package} {loaded_version} as {installed_version}"
        )
        for name in [
            n for n in list(sys.modules) if n == package or n.startswith(f"{package}.")
        ]:
            sys.modules.pop(name, None)
        parent_name, _, child = package.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and hasattr(parent, child):
            delattr(parent, child)
        unloaded = True
    if unloaded:
        importlib.invalidate_caches()


_unload_stale_modules()

try:
    importlib.metadata.version("google-genai")
except importlib.metadata.PackageNotFoundError:
    raise ImportError(
        "google-genai is not installed. Open WebUI 0.11.4+ no longer bundles it: "
        "keep ENABLE_PIP_INSTALL_FRONTMATTER_REQUIREMENTS enabled or install "
        "'google-genai>=1.68.0,<3' into the Open WebUI environment."
    ) from None

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from google.genai.errors import ClientError, ServerError, APIError  # noqa: E402

# Function names Gemini accepts: the intersection of the Gemini API rules
# (FunctionCall.name: letters, digits, "_", "-") and the Vertex AI rules (start
# with a letter or "_", at most 64 characters).
_GEMINI_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")

# Thought signature Gemini accepts in place of a missing one (e.g. for a tool
# call an API client sent back without its reasoning_details).
SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"

# Prefix of tool call ids the pipeline makes up when Gemini sends none. They are
# never sent back to Gemini.
SYNTHETIC_TOOL_CALL_ID_PREFIX = "owui_"

# reasoning_details formats: the thought signature of one function call, and the
# complete model content of a round with server-side tool parts.
REASONING_FORMAT_SIGNATURE = "google-gemini-v1"
REASONING_FORMAT_CONTENT = "google-gemini-v1-content"

# Text of the user message Open WebUI adds after tool results that held images.
TOOL_IMAGES_TEXT = (
    "Here are the images from the tool results above. Please analyze them."
)

# Chat ids Open WebUI does not save: temporary chats (also the legacy "local:"
# prefix) and channel messages. Their history exists only in the request.
UNSAVED_CHAT_ID_PREFIXES = ("temporary:", "local:", "channel:")

# Seconds between two updates of the live thinking block in the chat (Open WebUI
# saves the message content on every update).
LIVE_THINKING_INTERVAL = 0.4


def _gemini_function_name(name: str) -> str:
    """Map an Open WebUI tool name to a function name Gemini accepts.

    Valid names stay as they are; any other name gets its invalid characters
    replaced and a hash suffix, so the mapping is deterministic (no state is
    needed across rounds and turns) and distinct names stay distinct.
    """
    # fullmatch: "$" alone would also accept a name with a trailing newline
    if _GEMINI_FUNCTION_NAME_RE.fullmatch(name):
        return name
    base = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    if not re.match(r"[A-Za-z_]", base[:1] or "0"):
        base = "_" + base
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base[:55]}_{digest}"  # <= 64 chars


ASPECT_RATIO_OPTIONS: List[str] = [
    "default",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]

RESOLUTION_OPTIONS: List[str] = [
    "default",
    "1K",
    "2K",
    "4K",
]

VIDEO_ASPECT_RATIO_OPTIONS: List[str] = [
    "default",
    "16:9",
    "9:16",
]

VIDEO_RESOLUTION_OPTIONS: List[str] = [
    "default",
    "720p",
    "1080p",
    "4k",
]

VIDEO_DURATION_OPTIONS: List[str] = [
    "default",
    "4",
    "5",
    "6",
    "8",
]

VIDEO_PERSON_GENERATION_OPTIONS: List[str] = [
    "default",
    "allow_all",
    "allow_adult",
    "dont_allow",
]


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


class Pipe:
    """
    Pipeline for interacting with Google Gemini models.
    """

    # User-overridable configuration valves
    class UserValves(BaseModel):
        IMAGE_GENERATION_ASPECT_RATIO: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_ASPECT_RATIO", "default"),
            description="Default aspect ratio for image generation.",
            json_schema_extra={"enum": ASPECT_RATIO_OPTIONS},
        )
        IMAGE_GENERATION_RESOLUTION: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_RESOLUTION", "default"),
            description="Default resolution for image generation.",
            json_schema_extra={"enum": RESOLUTION_OPTIONS},
        )
        VIDEO_GENERATION_ASPECT_RATIO: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_ASPECT_RATIO", "default"),
            description="Default aspect ratio for video generation (16:9 landscape or 9:16 portrait).",
            json_schema_extra={"enum": VIDEO_ASPECT_RATIO_OPTIONS},
        )
        VIDEO_GENERATION_RESOLUTION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_RESOLUTION", "default"),
            description="Default resolution for video generation (720p, 1080p, or 4k).",
            json_schema_extra={"enum": VIDEO_RESOLUTION_OPTIONS},
        )
        VIDEO_GENERATION_DURATION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_DURATION", "default"),
            description="Default duration in seconds for video generation (4, 5, 6, or 8 - availability varies by model).",
            json_schema_extra={"enum": VIDEO_DURATION_OPTIONS},
        )

    # Configuration valves for the pipeline
    class Valves(BaseModel):
        BASE_URL: str = Field(
            default=os.getenv(
                "GOOGLE_GENAI_BASE_URL", "https://generativelanguage.googleapis.com/"
            ),
            description="Base URL for the Google Generative AI API.",
        )
        GOOGLE_API_KEY: EncryptedStr = Field(
            default=os.getenv("GOOGLE_API_KEY", ""),
            description="API key for Google Generative AI (used if USE_VERTEX_AI is false).",
            json_schema_extra={"input": {"type": "password"}},
        )
        API_VERSION: str = Field(
            default=os.getenv("GOOGLE_API_VERSION", "v1alpha"),
            description="API version to use for Google Generative AI (e.g., v1alpha, v1beta, v1).",
        )
        STREAMING_ENABLED: bool = Field(
            default=os.getenv("GOOGLE_STREAMING_ENABLED", "true").lower() == "true",
            description="Enable streaming responses (set false to force non-streaming mode).",
        )
        INCLUDE_THOUGHTS: bool = Field(
            default=os.getenv("GOOGLE_INCLUDE_THOUGHTS", "true").lower() == "true",
            description="Enable Gemini thoughts outputs (set false to disable).",
        )
        STRIP_THINKING_FROM_HISTORY: bool = Field(
            default=os.getenv("GOOGLE_STRIP_THINKING_FROM_HISTORY", "true").lower()
            == "true",
            description="Remove previously rendered thinking summaries from assistant "
            "messages before they are replayed to the API (saves tokens and avoids "
            "distraction). Set false to send them as before.",
        )
        THINKING_BUDGET: int = Field(
            default=int(os.getenv("GOOGLE_THINKING_BUDGET", "-1")),
            description="Thinking budget for Gemini 2.5 models (0=disabled, -1=dynamic, 1-32768=fixed token limit). "
            "Not used for Gemini 3 models which use THINKING_LEVEL instead.",
        )
        THINKING_LEVEL: str = Field(
            default=os.getenv("GOOGLE_THINKING_LEVEL", ""),
            description="Thinking level for Gemini 3 models. Most Gemini 3 models support 'low'/'high', "
            "while gemini-3.1-flash-image and gemini-3.1-flash-lite-image support 'minimal'/'high' "
            "and gemini-nano-banana-2.1 supports 'minimal'/'medium'/'high'. "
            "Ignored for other models. Empty string means use model default.",
        )
        USE_VERTEX_AI: bool = Field(
            default=os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true",
            description="Whether to use Google Cloud Vertex AI instead of the Google Generative AI API.",
        )
        VERTEX_PROJECT: str | None = Field(
            default=os.getenv("GOOGLE_CLOUD_PROJECT"),
            description="The Google Cloud project ID to use with Vertex AI.",
        )
        VERTEX_LOCATION: str = Field(
            default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            description="The Google Cloud region to use with Vertex AI.",
        )
        VERTEX_AI_RAG_STORE: str | None = Field(
            default=os.getenv("GOOGLE_VERTEX_AI_RAG_STORE"),
            description="Vertex AI RAG Store path for grounding (e.g., projects/PROJECT/locations/LOCATION/ragCorpora/DATA_STORE_ID). Only used when USE_VERTEX_AI is true.",
        )
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=os.getenv("GOOGLE_USE_PERMISSIVE_SAFETY", "false").lower()
            == "true",
            description="Use permissive safety settings for content generation.",
        )
        MODEL_CACHE_TTL: int = Field(
            default=int(os.getenv("GOOGLE_MODEL_CACHE_TTL", "600")),
            description="Time in seconds to cache the model list before refreshing",
        )
        RETRY_COUNT: int = Field(
            default=int(os.getenv("GOOGLE_RETRY_COUNT", "2")),
            description="Number of times to retry API calls on temporary failures",
        )
        DEFAULT_SYSTEM_PROMPT: str = Field(
            default=os.getenv("GOOGLE_DEFAULT_SYSTEM_PROMPT", ""),
            description="Default system prompt applied to all chats. If a user-defined system prompt exists, "
            "this is prepended to it. Leave empty to disable.",
        )
        ENABLE_FORWARD_USER_INFO_HEADERS: bool = Field(
            default=os.getenv(
                "GOOGLE_ENABLE_FORWARD_USER_INFO_HEADERS", "false"
            ).lower()
            == "true",
            description="Whether to forward user information headers.",
        )
        MODEL_ADDITIONAL: str = Field(
            default=os.getenv("GOOGLE_MODEL_ADDITIONAL", ""),
            description="A comma-separated list of model IDs to manually add to the list of available models. "
            "These are models not returned by the SDK but that you want to make available. "
            "Non-Gemini model IDs must be explicitly included in MODEL_WHITELIST to be available.",
        )
        MODEL_WHITELIST: str = Field(
            default=os.getenv("GOOGLE_MODEL_WHITELIST", ""),
            description="A comma-separated list of model IDs to show in the models list. "
            "If set, only these models will be available (after MODEL_ADDITIONAL is applied). "
            "Leave empty to show all models.",
        )
        USE_ENTERPRISE_WEB_SEARCH: bool = Field(
            default=os.getenv("GOOGLE_USE_ENTERPRISE_WEB_SEARCH", "false").lower()
            == "true",
            description="Whether to use Enterprise Web Search instead of standard Google search when grounding is enabled. "
            "Only available on Vertex AI.",
        )

        # Image Processing Configuration
        IMAGE_GENERATION_ASPECT_RATIO: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_ASPECT_RATIO", "default"),
            description="Default aspect ratio for image generation.",
            json_schema_extra={"enum": ASPECT_RATIO_OPTIONS},
        )
        IMAGE_GENERATION_RESOLUTION: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_RESOLUTION", "default"),
            description="Default resolution for image generation.",
            json_schema_extra={"enum": RESOLUTION_OPTIONS},
        )
        IMAGE_GENERATION_MODELS: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_MODELS", ""),
            description="A comma-separated list of extra model IDs to treat as Gemini 3 "
            "image models: called without streaming, images uploaded to the chat, and "
            "given the aspect ratio/resolution (ImageConfig) and thinking level settings. "
            "List image models released after this pipeline version whose IDs use "
            "neither Gemini 3 nor Nano Banana naming (e.g. a future gemini-4-flash-image); "
            "without an entry they get no ImageConfig and no thinking_level. Known "
            "Gemini 3 and Nano Banana image models need not be listed. Imagen IDs "
            "(imagen-*) are ignored.",
        )
        IMAGE_MAX_SIZE_MB: float = Field(
            default=float(os.getenv("GOOGLE_IMAGE_MAX_SIZE_MB", "15.0")),
            description="Maximum image size in MB before compression is applied",
        )
        IMAGE_MAX_DIMENSION: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_MAX_DIMENSION", "2048")),
            description="Maximum width or height in pixels before resizing",
        )
        IMAGE_COMPRESSION_QUALITY: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_COMPRESSION_QUALITY", "85")),
            description="JPEG compression quality (1-100, higher = better quality but larger size)",
        )
        IMAGE_ENABLE_OPTIMIZATION: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_ENABLE_OPTIMIZATION", "true").lower()
            == "true",
            description="Enable intelligent image optimization for API compatibility",
        )
        IMAGE_PNG_COMPRESSION_THRESHOLD_MB: float = Field(
            default=float(os.getenv("GOOGLE_IMAGE_PNG_THRESHOLD_MB", "0.5")),
            description="PNG files above this size (MB) will be converted to JPEG for better compression",
        )
        IMAGE_HISTORY_MAX_REFERENCES: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_HISTORY_MAX_REFERENCES", "5")),
            description="Maximum total number of images (history + current message) to include in a generation call; the current message's images are kept, then the newest history images",
        )
        IMAGE_ADD_LABELS: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_ADD_LABELS", "true").lower() == "true",
            description="If true, add small text labels like [Image 1] before each image part so the model can reference them.",
        )
        IMAGE_DEDUP_HISTORY: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_DEDUP_HISTORY", "true").lower() == "true",
            description="If true, deduplicate identical images (by hash) when constructing history context",
        )
        IMAGE_HISTORY_FIRST: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_HISTORY_FIRST", "true").lower() == "true",
            description="If true (default), history images precede current message images; if false, current images first.",
        )

        # Video Generation Configuration (Veo models)
        VIDEO_GENERATION_ASPECT_RATIO: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_ASPECT_RATIO", "default"),
            description="Default aspect ratio for video generation (16:9 landscape or 9:16 portrait).",
            json_schema_extra={"enum": VIDEO_ASPECT_RATIO_OPTIONS},
        )
        VIDEO_GENERATION_RESOLUTION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_RESOLUTION", "default"),
            description="Default resolution for video generation (720p, 1080p, or 4k).",
            json_schema_extra={"enum": VIDEO_RESOLUTION_OPTIONS},
        )
        VIDEO_GENERATION_DURATION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_DURATION", "default"),
            description="Default duration in seconds for video generation (4, 5, 6, or 8 - availability varies by model).",
            json_schema_extra={"enum": VIDEO_DURATION_OPTIONS},
        )
        VIDEO_GENERATION_NEGATIVE_PROMPT: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_NEGATIVE_PROMPT", ""),
            description="Default negative prompt for video generation (describes what not to include).",
        )
        VIDEO_GENERATION_PERSON_GENERATION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_PERSON_GENERATION", "default"),
            description="Controls generation of people in videos (allow_all, allow_adult, dont_allow).",
            json_schema_extra={"enum": VIDEO_PERSON_GENERATION_OPTIONS},
        )
        VIDEO_GENERATION_ENHANCE_PROMPT: bool = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_ENHANCE_PROMPT", "true").lower()
            == "true",
            description="Enable prompt enhancement for video generation.",
        )
        VIDEO_POLL_INTERVAL: int = Field(
            default=int(os.getenv("GOOGLE_VIDEO_POLL_INTERVAL", "10")),
            description="Polling interval in seconds when waiting for video generation to complete.",
        )
        VIDEO_POLL_TIMEOUT: int = Field(
            default=int(os.getenv("GOOGLE_VIDEO_POLL_TIMEOUT", "600")),
            description="Maximum time in seconds to wait for video generation before timing out (0=no limit).",
        )

    # ---------------- Internal Helpers ---------------- #
    async def _collect_history_images(
        self,
        sources: List[List[str]],
        room: int,
        current: List[Dict[str, Any]],
        optimization_stats: List[Dict[str, Any]],
        __user__: Optional[dict],
    ) -> List[Dict[str, Any]]:
        """Images of earlier messages for the ``room`` the current message's
        images leave under IMAGE_HISTORY_MAX_REFERENCES, oldest first.

        ``sources`` holds the image URLs of each earlier message, oldest message
        first. They are read newest first and only until the room is filled, so
        a long chat does not read and re-encode all of its images. With
        IMAGE_DEDUP_HISTORY an image counts once, at its newest place, and an
        image of the current message is not sent again as history.
        """
        if room <= 0:
            return []
        dedup = self.valves.IMAGE_DEDUP_HISTORY
        seen = {self._image_part_hash(p) for p in current} if dedup else set()
        picked: List[Dict[str, Any]] = []
        for urls in reversed(sources):
            for url in reversed(urls):
                part = await self._load_image_part(url, optimization_stats, __user__)
                if part is None:
                    continue
                if dedup:
                    digest = self._image_part_hash(part)
                    if digest in seen:
                        continue
                    seen.add(digest)
                picked.append(part)
                if len(picked) >= room:
                    return picked[::-1]
        return picked[::-1]

    @staticmethod
    def _image_part_hash(part: Dict[str, Any]) -> str:
        return hashlib.sha256(
            str((part.get("inline_data") or {}).get("data") or "").encode()
        ).hexdigest()

    @classmethod
    def _trailing_user_messages(
        cls, messages: List[Dict[str, Any]], saved: bool = False
    ) -> int:
        """How many user messages end ``messages`` (after the last answer).

        In the request, Open WebUI's message with the images of tool results is
        not counted. An answer without content, output and tool calls is
        skipped in the request and in a saved chain alike: Open WebUI leaves
        such an answer out of the request (up to 0.11 only a failed one, since
        0.12 every one, e.g. an answer stopped before its first token), so both
        sides count the same on every Open WebUI version.
        """
        count = 0
        for msg in reversed(messages):
            role = msg.get("role")
            if role == "user":
                if not saved and cls._is_tool_images_message(msg):
                    continue
                count += 1
            elif not (
                role == "assistant"
                and not msg.get("content")
                and not msg.get("output")
                and not msg.get("tool_calls")
            ):
                break
        return count

    async def _load_saved_chat_chain(
        self,
        __metadata__: Optional[Dict[str, Any]],
        __user__: Optional[dict],
    ) -> Optional[List[Dict[str, Any]]]:
        """The messages of a saved chat from the first one to the current user
        message (the last entry), as Open WebUI stores them, files included.

        Returns None when there is no saved chat (API clients, temporary and
        channel chats), for background tasks (titles, follow-ups and the like
        get Open WebUI's task prompt, not the chat's images), when the chat
        belongs to another user (admins may use any chat, like in Open WebUI),
        on Open WebUI versions without get_message_list and on errors: the
        image history then comes from the request's messages.
        """
        metadata = __metadata__ or {}
        user = __user__ or {}
        chat_id = str(metadata.get("chat_id") or "")
        user_message_id = metadata.get("user_message_id")
        if (
            get_message_list is None
            or metadata.get("task")
            or not chat_id
            or not user_message_id
            or not user.get("id")
            or chat_id.startswith(UNSAVED_CHAT_ID_PREFIXES)
        ):
            return None
        try:
            if user.get("role") != "admin" and not await Chats.is_chat_owner(
                chat_id, user["id"]
            ):
                return None
            messages_map = await Chats.get_messages_map_by_chat_id(chat_id)
            chain = get_message_list(messages_map or {}, user_message_id)
        except Exception as e:
            self.log.warning(f"Could not load chat {chat_id} for image history: {e}")
            return None
        return [m for m in chain if isinstance(m, dict)] or None

    @staticmethod
    def _saved_image_file_url(file: Any) -> Optional[str]:
        """Where to read an image file of a saved message from.

        data: URLs stay as they are, Open WebUI file URLs
        (/api/v1/files/<id>/content, the generated images) too, and a bare file
        id (what the web UI stores for an upload) becomes such a URL. Other
        files and URLs give None.
        """
        if not isinstance(file, dict):
            return None
        content_type = str(file.get("content_type") or "")
        if file.get("type") != "image" and not content_type.startswith("image/"):
            return None
        url = str(file.get("url") or "")
        if url.startswith("data:image") or "/files/" in url:
            return url
        file_id = url or str(file.get("id") or "")
        if re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
            return f"/api/v1/files/{file_id}/content"
        return None

    def _saved_message_image_urls(self, msg: Dict[str, Any]) -> List[str]:
        """Image URLs of a message of a saved chat, in order.

        Open WebUI turns the image files of user messages into image_url
        parts, but drops the files of assistant messages, and generated images
        are attached there only (not repeated as markdown in the content). So
        the image history of a saved chat is read from the chat itself: the
        markdown image links in the content (answers of pipeline versions
        before 1.15.2, data: URLs of a failed upload), then the image files of
        user and assistant messages.
        """
        if msg.get("role") not in {"user", "assistant"}:
            return []
        text = msg.get("content")
        _texts, urls = self._content_image_sources(
            text if isinstance(text, str) else ""
        )
        files = [self._saved_image_file_url(f) for f in msg.get("files") or []]
        return urls + [url for url in files if url]

    def _deduplicate_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.valves.IMAGE_DEDUP_HISTORY:
            return images
        seen: set[str] = set()
        result: List[Dict[str, Any]] = []
        for part in images:
            try:
                data = part["inline_data"]["data"]
                # Hash full base64 payload for stronger dedup reliability
                h = hashlib.sha256(data.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
            except Exception as e:
                # Skip images with malformed or missing data, but log for debugging.
                self.log.debug(f"Skipping image in deduplication due to error: {e}")
            result.append(part)
        return result

    def _combine_system_prompts(
        self, user_system_prompt: Optional[str]
    ) -> Optional[str]:
        """Combine default system prompt with user-defined system prompt.

        If DEFAULT_SYSTEM_PROMPT is set and user_system_prompt exists,
        the default is prepended to the user's prompt.
        If only DEFAULT_SYSTEM_PROMPT is set, it is used as the system prompt.
        If only user_system_prompt is set, it is used as-is.

        Args:
            user_system_prompt: The user-defined system prompt from messages (may be None)

        Returns:
            Combined system prompt or None if neither is set
        """
        default_prompt = self.valves.DEFAULT_SYSTEM_PROMPT.strip()
        user_prompt = user_system_prompt.strip() if user_system_prompt else ""

        if default_prompt and user_prompt:
            combined = f"{default_prompt}\n\n{user_prompt}"
            self.log.debug(
                f"Combined system prompts: default ({len(default_prompt)} chars) + "
                f"user ({len(user_prompt)} chars) = {len(combined)} chars"
            )
            return combined
        elif default_prompt:
            self.log.debug(f"Using default system prompt ({len(default_prompt)} chars)")
            return default_prompt
        elif user_prompt:
            return user_prompt
        return None

    def _apply_order_and_limit(
        self,
        history: List[Dict[str, Any]],
        current: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[bool]]:
        """Combine history & current image parts honoring order & global limit.

        Over IMAGE_HISTORY_MAX_REFERENCES the oldest history images are dropped
        first: the current message's images are always kept (only more of them
        than the limit are cut), the rest of the limit goes to the newest
        history images, so an edit request keeps the image it refers to. Both
        lists keep their order (oldest first).

        Returns:
            (combined_parts, reused_flags) where reused_flags[i] == True indicates
            the image originated from history, False if from current message.
        """
        limit = max(1, self.valves.IMAGE_HISTORY_MAX_REFERENCES)
        current = current[:limit]
        room = limit - len(current)
        history = history[-room:] if room > 0 else []
        if self.valves.IMAGE_HISTORY_FIRST:
            combined = history + current
            reused_flags = [True] * len(history) + [False] * len(current)
        else:
            combined = current + history
            reused_flags = [False] * len(current) + [True] * len(history)
        return combined, reused_flags

    async def _emit_image_stats(
        self,
        ordered_stats: List[Dict[str, Any]],
        reused_flags: List[bool],
        total_limit: int,
        __event_emitter__: Optional[Callable],
    ) -> None:
        """Emit per-image optimization stats aligned with final combined order.

        ordered_stats: stats list in the exact order images will be sent (same length as combined image list)
        reused_flags: parallel list indicating whether image originated from history
        """
        if not ordered_stats or not __event_emitter__:
            return
        for idx, stat in enumerate(ordered_stats, start=1):
            reused = reused_flags[idx - 1] if idx - 1 < len(reused_flags) else False
            stat_copy = dict(stat) if stat else {}
            stat_copy.update({"index": idx, "reused": reused})
            if stat and stat.get("original_size_mb") is not None:
                desc = f"Image {idx}: {stat['original_size_mb']:.2f}MB -> {stat['final_size_mb']:.2f}MB"
                if stat.get("quality") is not None:
                    desc += f" (Q{stat['quality']})"
            else:
                desc = f"Image {idx}: (no metrics)"
            reasons = stat.get("reasons") if stat else None
            if reasons:
                desc += " | " + ", ".join(reasons[:3])
            await self._safe_emit(
                __event_emitter__,
                {
                    "type": "status",
                    "data": {
                        "action": "image_optimization",
                        "description": desc,
                        "index": idx,
                        "done": False,
                        "details": stat_copy,
                    },
                },
            )
        await self._safe_emit(
            __event_emitter__,
            {
                "type": "status",
                "data": {
                    "action": "image_optimization",
                    "description": f"{len(ordered_stats)} image(s) processed (limit {total_limit}).",
                    "done": True,
                },
            },
        )

    async def _build_image_generation_contents(
        self,
        messages: List[Dict[str, Any]],
        __event_emitter__: Optional[Callable],
        __metadata__: Optional[Dict[str, Any]] = None,
        __user__: Optional[dict] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Construct the contents payload for image-capable models.

        The prompt and the current images come from the last user message of
        the request. Images of earlier turns come from the saved chat when
        there is one (see _saved_message_image_urls), else from the request.

        Returns tuple (contents, system_instruction) where system_instruction is extracted from system messages.
        """
        # Extract user-defined system instruction first
        user_system_instruction = next(
            (msg["content"] for msg in messages if msg.get("role") == "system"),
            None,
        )

        # Combine with default system prompt if configured
        system_instruction = self._combine_system_prompts(user_system_instruction)

        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if not last_user_msg:
            raise ValueError("No user message found")

        optimization_stats: List[Dict[str, Any]] = []
        prompt, current_images = await self._extract_images_from_message(
            last_user_msg, stats_list=optimization_stats, __user__=__user__
        )
        current_images = self._deduplicate_images(current_images)

        chain = await self._load_saved_chat_chain(__metadata__, __user__)
        if chain:
            # The chain ends with the saved current user message. Open WebUI's
            # guided regeneration appends the guidance as one more user
            # message after it: then the saved message is history as well.
            end = next(
                i
                for i in range(len(messages) - 1, -1, -1)
                if messages[i] is last_user_msg
            )
            guided = (
                self._trailing_user_messages(messages[: end + 1])
                == self._trailing_user_messages(chain, saved=True) + 1
            )
            sources = [
                self._saved_message_image_urls(m)
                for m in (chain if guided else chain[:-1])
            ]
        else:
            sources = [
                self._content_image_sources(m.get("content", ""))[1]
                for m in messages
                if m is not last_user_msg and m.get("role") in {"user", "assistant"}
            ]
        limit = max(1, self.valves.IMAGE_HISTORY_MAX_REFERENCES)
        history_images = await self._collect_history_images(
            sources,
            limit - min(len(current_images), limit),
            current_images,
            optimization_stats,
            __user__,
        )

        combined, reused_flags = self._apply_order_and_limit(
            history_images, current_images
        )

        if not prompt and not combined:
            raise ValueError("No prompt or images provided")
        if not prompt and combined:
            prompt = "Analyze and describe the provided images."

        # Build ordered stats aligned with combined list
        ordered_stats: List[Dict[str, Any]] = []
        if optimization_stats:
            # Build map from final_hash -> stat (first wins)
            hash_map: Dict[str, Dict[str, Any]] = {}
            for s in optimization_stats:
                fh = s.get("final_hash")
                if fh and fh not in hash_map:
                    hash_map[fh] = s
            for part in combined:
                try:
                    fh = hashlib.sha256(
                        part["inline_data"]["data"].encode()
                    ).hexdigest()
                    ordered_stats.append(hash_map.get(fh) or {})
                except Exception:
                    ordered_stats.append({})
        # Emit stats AFTER final ordering so labels match
        await self._emit_image_stats(
            ordered_stats,
            reused_flags,
            self.valves.IMAGE_HISTORY_MAX_REFERENCES,
            __event_emitter__,
        )

        # Emit mapping
        if combined:
            mapping = [
                {
                    "index": i + 1,
                    "label": (
                        f"Image {i + 1}" if self.valves.IMAGE_ADD_LABELS else str(i + 1)
                    ),
                    "reused": reused_flags[i],
                    "origin": "history" if reused_flags[i] else "current",
                }
                for i in range(len(combined))
            ]
            await self._safe_emit(
                __event_emitter__,
                {
                    "type": "status",
                    "data": {
                        "action": "image_reference_map",
                        "description": f"{len(combined)} image(s) included (limit {self.valves.IMAGE_HISTORY_MAX_REFERENCES}).",
                        "images": mapping,
                        "done": True,
                    },
                },
            )

        # Build parts
        parts: List[Dict[str, Any]] = []

        # For image generation models, prepend system instruction to the prompt
        # since system_instruction parameter may not be supported
        final_prompt = prompt
        if system_instruction and prompt:
            final_prompt = f"{system_instruction}\n\n{prompt}"
            self.log.debug(
                f"Prepended system instruction to prompt for image generation. "
                f"System instruction length: {len(system_instruction)}, "
                f"Original prompt length: {len(prompt)}, "
                f"Final prompt length: {len(final_prompt)}"
            )
        elif system_instruction and not prompt:
            final_prompt = system_instruction
            self.log.debug(
                f"Using system instruction as prompt for image generation "
                f"(length: {len(system_instruction)})"
            )

        if final_prompt:
            parts.append({"text": final_prompt})
        if self.valves.IMAGE_ADD_LABELS:
            for idx, part in enumerate(combined, start=1):
                parts.append({"text": f"[Image {idx}]"})
                parts.append(part)
        else:
            parts.extend(combined)

        self.log.debug(
            f"Image-capable payload: history={len(history_images)} current={len(current_images)} used={len(combined)} limit={self.valves.IMAGE_HISTORY_MAX_REFERENCES} history_first={self.valves.IMAGE_HISTORY_FIRST} prompt_len={len(final_prompt)}"
        )
        # Return None for system_instruction since we've incorporated it into the prompt
        return [{"role": "user", "parts": parts}], None

    def __init__(self):
        """Initializes the Pipe instance and configures the genai library."""
        self.valves = self.Valves()
        self.name: str = "Google Gemini: "

        # Setup logging
        self.log = logging.getLogger("google_ai.pipe")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

        # Model cache
        self._model_cache: Optional[List[Dict[str, str]]] = None
        self._model_cache_time: float = 0
        self._model_cache_key: Optional[Tuple[Any, ...]] = None

    def _get_client(self, user: Optional[UserModel] = None) -> genai.Client:
        """
        Validates API credentials and returns a genai.Client instance.

        Args:
            user: The requesting user, whose X-OpenWebUI-User-* headers are sent
                when ENABLE_FORWARD_USER_INFO_HEADERS is on. Passed per call
                (never kept on the shared Pipe instance), so concurrent
                requests of different users cannot forward each other's headers.
        """
        self._validate_api_key()

        if self.valves.USE_VERTEX_AI:
            self.log.debug(
                f"Initializing Vertex AI client (Project: {self.valves.VERTEX_PROJECT}, Location: {self.valves.VERTEX_LOCATION})"
            )
            return genai.Client(
                vertexai=True,
                project=self.valves.VERTEX_PROJECT,
                location=self.valves.VERTEX_LOCATION,
            )
        else:
            self.log.debug("Initializing Google Generative AI client with API Key")
            headers = {}
            if self.valves.ENABLE_FORWARD_USER_INFO_HEADERS and user:

                def sanitize_header_value(value: Any, max_length: int = 255) -> str:
                    if value is None:
                        return ""
                    # Convert to string and remove all control characters
                    sanitized = re.sub(r"[\x00-\x1F\x7F]", "", str(value))
                    sanitized = sanitized.strip()
                    return (
                        sanitized[:max_length]
                        if len(sanitized) > max_length
                        else sanitized
                    )

                user_attrs = {
                    "X-OpenWebUI-User-Name": sanitize_header_value(
                        getattr(user, "name", None)
                    ),
                    "X-OpenWebUI-User-Id": sanitize_header_value(
                        getattr(user, "id", None)
                    ),
                    "X-OpenWebUI-User-Email": sanitize_header_value(
                        getattr(user, "email", None)
                    ),
                    "X-OpenWebUI-User-Role": sanitize_header_value(
                        getattr(user, "role", None)
                    ),
                }
                headers = {k: v for k, v in user_attrs.items() if v not in (None, "")}
            options = types.HttpOptions(
                api_version=self.valves.API_VERSION,
                base_url=self.valves.BASE_URL,
                headers=headers,
            )
            return genai.Client(
                api_key=EncryptedStr.decrypt(self.valves.GOOGLE_API_KEY),
                http_options=options,
            )

    def _validate_api_key(self) -> None:
        """
        Validates that the necessary Google API credentials are set.

        Raises:
            ValueError: If the required credentials are not set.
        """
        if self.valves.USE_VERTEX_AI:
            if not self.valves.VERTEX_PROJECT:
                self.log.error("USE_VERTEX_AI is true, but VERTEX_PROJECT is not set.")
                raise ValueError(
                    "VERTEX_PROJECT is not set. Please provide the Google Cloud project ID."
                )
            # For Vertex AI, location has a default, so project is the main thing to check.
            # Actual authentication will be handled by ADC or environment.
            self.log.debug(
                "Using Vertex AI. Ensure ADC or service account is configured."
            )
        else:
            if not self.valves.GOOGLE_API_KEY:
                self.log.error("GOOGLE_API_KEY is not set (and not using Vertex AI).")
                raise ValueError(
                    "GOOGLE_API_KEY is not set. Please provide the API key in the environment variables or valves."
                )
            self.log.debug("Using Google Generative AI API with API Key.")

    def strip_prefix(self, model_name: str) -> str:
        """
        Extract the model identifier using regex, handling various naming conventions.
        e.g., "google_gemini_pipeline.gemini-2.5-flash-preview-04-17" -> "gemini-2.5-flash-preview-04-17"
        e.g., "models/gemini-1.5-flash-001" -> "gemini-1.5-flash-001"
        e.g., "publishers/google/models/gemini-1.5-pro" -> "gemini-1.5-pro"
        """
        # Use regex to remove everything up to and including the last '/' or the first '.'
        stripped = re.sub(r"^(?:.*/|[^.]*\.)", "", model_name)
        return stripped

    def _get_model_cache_key(self) -> Tuple[Any, ...]:
        """The valves that decide which models are listed and how they are named."""
        return (
            self.valves.BASE_URL,
            self.valves.API_VERSION,
            self.valves.USE_VERTEX_AI,
            self.valves.VERTEX_PROJECT,
            self.valves.VERTEX_LOCATION,
            self.valves.MODEL_WHITELIST,
            self.valves.MODEL_ADDITIONAL,
            self.valves.IMAGE_GENERATION_MODELS,
        )

    def get_google_models(self, force_refresh: bool = False) -> List[Dict[str, str]]:
        """
        Retrieve available Google models suitable for content generation.
        Uses caching to reduce API calls. The cache is only used while the
        valves that shape the list (endpoint, whitelist, additional and image
        generation models) are unchanged, so a valve change shows on the next
        model list refresh instead of after MODEL_CACHE_TTL.

        Args:
            force_refresh: Whether to force refreshing the model cache

        Returns:
            List of dictionaries containing model id and name.
        """
        # Check cache first
        current_time = time.time()
        cache_key = self._get_model_cache_key()
        if (
            not force_refresh
            and self._model_cache is not None
            and self._model_cache_key == cache_key
            and (current_time - self._model_cache_time) < self.valves.MODEL_CACHE_TTL
        ):
            self.log.debug("Using cached model list")
            return self._model_cache

        try:
            client = self._get_client()
            self.log.debug("Fetching models from Google API")
            models = list(client.models.list())

            # Process additional models (models not returned by SDK but that we want to add)
            additional = self.valves.MODEL_ADDITIONAL
            if additional:
                self.log.debug(f"Processing additional models: {additional}")
                existing_model_names = {self.strip_prefix(m.name) for m in models}
                additional_ids = set(re.findall(r"[^,\s]+", additional))

                for model_id in additional_ids.difference(existing_model_names):
                    self.log.debug(f"Adding additional model '{model_id}'.")
                    models.append(types.Model(name=f"models/{model_id}"))

            available_models = []
            for model in models:
                actions = model.supported_actions
                model_id_stripped = self.strip_prefix(model.name)
                is_content_model = actions is None or "generateContent" in actions
                is_video_model = (
                    actions is not None and "generateVideos" in actions
                ) or model_id_stripped.startswith("veo-")
                if is_content_model or is_video_model:
                    model_id = model_id_stripped
                    model_name = model.display_name or model_id

                    # Check if model supports image generation
                    supports_image_generation = self._check_image_generation_support(
                        model_id
                    )
                    if supports_image_generation:
                        model_name += " 🎨"  # Add image generation indicator

                    # Check if model supports video generation
                    supports_video_generation = self._check_video_generation_support(
                        model_id
                    )
                    if supports_video_generation:
                        model_name += " 🎬"  # Add video generation indicator

                    available_models.append(
                        {
                            "id": model_id,
                            "name": model_name,
                            "image_generation": supports_image_generation,
                            "video_generation": supports_video_generation,
                        }
                    )

            model_map = {model["id"]: model for model in available_models}

            # Apply MODEL_WHITELIST filter if configured (takes priority)
            whitelist = self.valves.MODEL_WHITELIST
            if whitelist:
                self.log.debug(f"Applying model whitelist: {whitelist}")
                whitelisted_ids = set(re.findall(r"[^,\s]+", whitelist))
                # Filter to only include whitelisted models
                filtered_models = {
                    k: v for k, v in model_map.items() if k in whitelisted_ids
                }
                self.log.debug(f"After whitelist filter: {len(filtered_models)} models")
            else:
                # If no whitelist, filter to only include models starting with 'gemini-' or 'veo-'
                filtered_models = {
                    k: v
                    for k, v in model_map.items()
                    if k.startswith("gemini-") or k.startswith("veo-")
                }
                self.log.debug(f"After prefix filter: {len(filtered_models)} models")

            # Update cache
            self._model_cache = list(filtered_models.values())
            self._model_cache_time = current_time
            self._model_cache_key = cache_key
            self.log.debug(f"Found {len(self._model_cache)} Gemini models")
            return self._model_cache

        except Exception as e:
            self.log.exception(f"Could not fetch models from Google: {str(e)}")
            # Return a specific error entry for the UI
            return [{"id": "error", "name": f"Could not fetch models: {str(e)}"}]

    # A Gemini model ID with "image" as its own dash-separated segment.
    _GEMINI_IMAGE_MODEL_RE = re.compile(r"(?:^|/)gemini-[^/]*-image(?:-|$)")

    # gemini-nano-banana-2.1 and its variants ("-preview", "@001"), but not
    # e.g. gemini-nano-banana-2.10.
    _NANO_BANANA_2_1_RE = re.compile(r"gemini-nano-banana-2\.1(?:[-@]|$)")

    # Image models whose model pages list Search grounding as "Not supported"
    # (https://ai.google.dev/gemini-api/docs/models), incl. their preview IDs.
    _NO_SEARCH_GROUNDING_MODELS = (
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-lite-image",
    )

    def _supports_search_grounding(self, model_id: str) -> bool:
        """Return False for models that do not support Google Search grounding."""
        model_lower = model_id.rsplit("/", 1)[-1].lower()
        return not model_lower.startswith(self._NO_SEARCH_GROUNDING_MODELS)

    def _is_configured_image_model(self, model_id: str) -> bool:
        """Return True if the IMAGE_GENERATION_MODELS valve lists the model."""
        configured = {
            listed.rsplit("/", 1)[-1].lower()
            for listed in re.findall(
                r"[^,\s]+", self.valves.IMAGE_GENERATION_MODELS or ""
            )
        }
        return model_id.rsplit("/", 1)[-1].lower() in configured

    @staticmethod
    def _is_nano_banana_model(model_id: str) -> bool:
        """Return True for Nano Banana IDs such as "gemini-nano-banana-2.1".

        Unlike the older image models, these IDs carry neither "image" nor a
        Gemini version in their name.
        """
        return "nano-banana" in model_id.lower()

    def _check_image_generation_support(self, model_id: str) -> bool:
        """
        Check if a model supports image generation.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model supports image generation, False otherwise
        """
        model_lower = model_id.lower()

        # Imagen models ("imagen-...") use the predict API, not generateContent,
        # so they are never Gemini image models, even if IMAGE_GENERATION_MODELS
        # lists them.
        if model_lower.startswith("imagen-") or "/imagen-" in model_lower:
            return False

        # Models the admin declared as image models (IMAGE_GENERATION_MODELS)
        if self._is_configured_image_model(model_id):
            return True

        # Known image generation models (both Gemini 2.5 and Gemini 3)
        image_generation_models = [
            "gemini-2.5-flash-image",
            "gemini-2.5-flash-image-preview",
            "gemini-3-flash-image",
            "gemini-3-flash-image-preview",
            "gemini-3.1-flash-image",
            "gemini-3.1-flash-image-preview",
            "gemini-3.1-flash-lite-image",
            "gemini-3-pro-image",
            "gemini-3-pro-image-preview",
        ]

        # Check for exact matches or pattern matches
        for pattern in image_generation_models:
            if model_lower == pattern or pattern in model_lower:
                return True

        # Nano Banana models, e.g. "gemini-nano-banana-2.1"
        if self._is_nano_banana_model(model_id):
            return True

        # Gemini image models carry "image" as a separate name segment, both as
        # preview and as released (GA) ID, e.g. "gemini-3.1-flash-image",
        # "gemini-2.0-flash-preview-image-generation".
        if self._GEMINI_IMAGE_MODEL_RE.search(model_lower):
            return True

        # Additional pattern checking for future models
        if "image" in model_lower and (
            "generation" in model_lower or "preview" in model_lower
        ):
            return True

        return False

    def _is_gemini_3_family_model(self, model_id: str) -> bool:
        """Return True for Gemini 3.x model IDs, including Gemini 3.1."""
        model_lower = model_id.lower()
        return model_lower.startswith("gemini-3-") or model_lower.startswith(
            "gemini-3."
        )

    def _is_gemini_3_image_model(self, model_id: str) -> bool:
        """Return True for image models with the features of Gemini 3 image models.

        These take ImageConfig (aspect ratio, resolution) and thinking_level:
        Gemini 3.x image IDs, Nano Banana IDs (gemini-nano-banana-*), whose names
        carry no Gemini version, and the IDs listed in IMAGE_GENERATION_MODELS.
        """
        if not self._check_image_generation_support(model_id):
            return False
        return (
            self._is_gemini_3_family_model(model_id)
            or self._is_nano_banana_model(model_id)
            or self._is_configured_image_model(model_id)
        )

    def _check_image_config_support(self, model_id: str) -> bool:
        """
        Check if a model supports ImageConfig (aspect_ratio and image_size parameters).

        ImageConfig is only supported by Gemini 3 image generation models,
        including the Nano Banana IDs and the IDs in IMAGE_GENERATION_MODELS.
        Gemini 2.5 image models support image generation but not ImageConfig.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model supports ImageConfig, False otherwise
        """
        return self._is_gemini_3_image_model(model_id)

    def _check_thinking_support(self, model_id: str) -> bool:
        """
        Check if a model supports the thinking feature.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model supports thinking, False otherwise
        """
        # Models that do NOT support thinking
        non_thinking_models = [
            "gemini-2.5-flash-image-preview",
            "gemini-2.5-flash-image",
        ]

        # Check for exact matches
        for pattern in non_thinking_models:
            if model_id == pattern or pattern in model_id:
                return False

        # Gemini 3 image models support thinking and thinking-level controls.
        if self._is_gemini_3_image_model(model_id):
            return True

        # Older image generation preview models typically don't support thinking.
        if "image" in model_id.lower() and (
            "generation" in model_id.lower() or "preview" in model_id.lower()
        ):
            return False

        # By default, assume models support thinking
        return True

    def _check_thinking_level_support(self, model_id: str) -> bool:
        """
        Check if a model supports the thinking_level parameter.

        Gemini 3 models (and image models with Gemini 3 image features, such as
        the Nano Banana IDs) support thinking_level and should NOT use
        thinking_budget. Other models (like Gemini 2.5) use thinking_budget instead.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model supports thinking_level, False otherwise
        """
        return self._is_gemini_3_family_model(
            model_id
        ) or self._is_gemini_3_image_model(model_id)

    def _get_supported_thinking_levels(self, model_id: str) -> List[str]:
        """Return the supported thinking levels for a specific Gemini 3 model.

        An empty list means the levels are not known; the configured level is
        then sent unchanged (e.g. other Nano Banana IDs, IMAGE_GENERATION_MODELS).
        """
        model_lower = model_id.lower()

        # https://ai.google.dev/gemini-api/docs/generate-content/image-generation
        if self._NANO_BANANA_2_1_RE.match(model_lower):
            return ["minimal", "medium", "high"]

        if model_lower.startswith(
            ("gemini-3.1-flash-image", "gemini-3.1-flash-lite-image")
        ):
            return ["minimal", "high"]

        if self._is_gemini_3_family_model(model_id):
            return ["low", "high"]

        return []

    @staticmethod
    def _get_supported_image_sizes(model_id: str) -> Optional[List[str]]:
        """Return the image_size values a model accepts, or None if not restricted."""
        if model_id.lower().startswith("gemini-3.1-flash-lite-image"):
            # Nano Banana 2 Lite only generates 1K images.
            return ["1K"]
        return None

    def _coerce_thinking_level(
        self, requested_level: str, supported_levels: List[str]
    ) -> Optional[str]:
        """Map unsupported thinking levels to the closest supported level."""
        if not supported_levels:
            return None

        level_rank = {"minimal": 0, "low": 1, "medium": 2, "high": 3}
        requested_rank = level_rank.get(requested_level)
        if requested_rank is None:
            return None

        supported_ranks = sorted(
            (level_rank[level], level)
            for level in supported_levels
            if level in level_rank
        )
        if not supported_ranks:
            return None

        best_rank, best_level = min(
            supported_ranks,
            key=lambda item: (abs(item[0] - requested_rank), -item[0]),
        )
        _ = best_rank
        return best_level

    def _validate_thinking_level(self, level: str, model_id: str = "") -> Optional[str]:
        """
        Validate and normalize the thinking level value for the current model.

        Args:
            level: The thinking level string to validate
            model_id: The model ID used to determine supported levels

        Returns:
            Supported thinking level string or None if invalid/empty
        """
        if not level:
            return None

        normalized = level.strip().lower()
        valid_levels = ["minimal", "low", "medium", "high"]

        if normalized not in valid_levels:
            self.log.warning(
                f"Invalid thinking level '{level}'. Valid values are: {', '.join(valid_levels)}. "
                "Falling back to model default."
            )
            return None

        supported_levels = self._get_supported_thinking_levels(model_id)
        if not supported_levels or normalized in supported_levels:
            return normalized

        coerced_level = self._coerce_thinking_level(normalized, supported_levels)
        if coerced_level:
            self.log.warning(
                f"Thinking level '{level}' is not supported for model '{model_id}'. "
                f"Using '{coerced_level}' instead. Supported values: {', '.join(supported_levels)}."
            )
            return coerced_level

        self.log.warning(
            f"Thinking level '{level}' is not supported for model '{model_id}'. Supported values: {', '.join(supported_levels)}. "
            "Falling back to model default."
        )
        return None

    def _validate_thinking_budget(self, budget: int) -> int:
        """
        Validate and normalize the thinking budget value.

        Args:
            budget: The thinking budget integer to validate

        Returns:
            Validated budget: -1 for dynamic, 0 to disable, or 1-32768 for fixed limit
        """
        # -1 means dynamic thinking (let the model decide)
        if budget == -1:
            return -1

        # 0 means disable thinking
        if budget == 0:
            return 0

        # Validate positive range (1-32768)
        if budget > 0:
            if budget > 32768:
                self.log.warning(
                    f"Thinking budget {budget} exceeds maximum of 32768. Clamping to 32768."
                )
                return 32768
            return budget

        # Negative values (except -1) are invalid, treat as -1 (dynamic)
        self.log.warning(
            f"Invalid thinking budget {budget}. Only -1 (dynamic), 0 (disabled), or 1-32768 are valid. "
            "Falling back to dynamic thinking."
        )
        return -1

    def _validate_aspect_ratio(self, aspect_ratio: str) -> Optional[str]:
        """
        Validate and normalize the aspect ratio value.

        Args:
            aspect_ratio: The aspect ratio string to validate

        Returns:
            Validated aspect ratio string, None for "default", or "1:1" as fallback for invalid values
        """
        if not aspect_ratio or aspect_ratio == "default":
            self.log.debug("Using default aspect ratio (None)")
            return None

        normalized = aspect_ratio.strip()
        valid_ratios = [r for r in ASPECT_RATIO_OPTIONS if r != "default"]

        if normalized in valid_ratios:
            return normalized

        self.log.warning(
            f"Invalid aspect ratio '{aspect_ratio}'. Valid values are: {', '.join(valid_ratios)}. "
            "Using default '1:1'."
        )
        return "1:1"

    def _validate_resolution(self, resolution: str) -> Optional[str]:
        """
        Validate and normalize the resolution value.

        Args:
            resolution: The resolution string to validate

        Returns:
            Validated resolution string, None for "default", or "2K" as fallback for invalid values
        """
        if not resolution or resolution.lower() == "default":
            self.log.debug("Using default resolution (None)")
            return None

        normalized = resolution.strip().upper()
        valid_resolutions = [r for r in RESOLUTION_OPTIONS if r.lower() != "default"]

        if normalized in valid_resolutions:
            return normalized

        self.log.warning(
            f"Invalid resolution '{resolution}'. Valid values are: {', '.join(valid_resolutions)}. "
            "Using default '2K'."
        )
        return "2K"

    def _check_video_generation_support(self, model_id: str) -> bool:
        model_lower = model_id.lower()
        return model_lower.startswith("veo-") or (
            "veo" in model_lower and "generate" in model_lower
        )

    @staticmethod
    def _image_data_hash(image_data: Any) -> str:
        """Build a stable hash for generated image data across bytes/str inputs."""
        if isinstance(image_data, bytes):
            return hashlib.sha256(image_data).hexdigest()
        return hashlib.sha256(str(image_data).encode("utf-8")).hexdigest()

    async def _safe_emit(
        self, __event_emitter__: Optional[Callable], event: Dict[str, Any]
    ) -> None:
        """Send an event if an emitter is available; never raise.

        Open WebUI passes no event emitter (None) to background tasks such as
        title, tag and follow-up generation, so every emit must be optional.
        """
        if not __event_emitter__:
            return
        try:
            await __event_emitter__(event)
        except Exception as emit_error:
            self.log.warning(
                f"Failed to emit {event.get('type', 'unknown')} event: {emit_error}"
            )

    async def _close_client(self, client: Optional[genai.Client]) -> None:
        """Close both transports (async and sync) of a per-request genai client."""
        if client is None:
            return
        try:
            await client.aio.aclose()
        except Exception as close_error:
            self.log.debug(f"Failed to close Gemini client: {close_error}")
        try:
            client.close()
        except Exception as close_error:
            self.log.debug(f"Failed to close Gemini sync client: {close_error}")

    @staticmethod
    def _track_statuses(
        __event_emitter__: Optional[Callable], running: Dict[str, Dict[str, Any]]
    ) -> Optional[Callable]:
        """Wrap an event emitter to record the status actions still running.

        `running` maps every status action whose last event had done=False
        (image_processing, video_generation, ...) to that event's data; an
        event of the same action with done=True removes it.
        """
        if not __event_emitter__:
            return __event_emitter__

        async def emit(event: Dict[str, Any]) -> Any:
            data = event.get("data") if event.get("type") == "status" else None
            if isinstance(data, dict) and data.get("action"):
                if data.get("done") is False:
                    running[data["action"]] = data
                elif data.get("done"):
                    running.pop(data["action"], None)
            return await __event_emitter__(event)

        return emit

    async def _finish_running_statuses(
        self,
        __event_emitter__: Optional[Callable],
        running: Dict[str, Dict[str, Any]],
        cancelled: bool,
    ) -> None:
        """Send a final done=True status for every action still running.

        Called from `finally`, so a stopped (cancelled) or failed request leaves
        no "Processing image request..." or "Generating video..." status behind.
        """
        for action in list(running):
            label = action.replace("_", " ").capitalize()
            data: Dict[str, Any] = {
                "action": action,
                "done": True,
                "description": f"{label} {'stopped' if cancelled else 'failed'}",
            }
            await self._safe_emit(__event_emitter__, {"type": "status", "data": data})
        running.clear()

    @staticmethod
    def _is_chat_message(__metadata__: Optional[Dict[str, Any]]) -> bool:
        """Whether the request belongs to a chat message (browser path).

        API requests (POST /api/chat/completions without chat_id) get an event
        emitter in Open WebUI 0.9+ too, but with an empty chat_id/message_id:
        events such as "files" are neither saved nor shown to the client then.
        """
        metadata = __metadata__ or {}
        return bool(metadata.get("chat_id")) and bool(metadata.get("message_id"))

    @staticmethod
    def _chunk(
        delta: Dict[str, Any],
        model: str,
        finish_reason: Optional[str] = None,
        response_id: Optional[str] = None,
        created: Optional[int] = None,
    ) -> Dict[str, Any]:
        """An OpenAI chat.completion.chunk carrying `delta`.

        Yielded instead of a plain str: Open WebUI forwards a str chunk that
        starts with "data:" as a raw SSE line, which loses such an answer. The
        chunks of one pipe call share `response_id` and `created`.
        """
        return {
            "id": response_id or f"{model}-{uuid.uuid4()}",
            "created": int(time.time()) if created is None else created,
            "model": model,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                    "delta": delta,
                }
            ],
        }

    @staticmethod
    def _sse(
        chunks: Union[List[Dict[str, Any]], AsyncIterator[Dict[str, Any]]],
    ) -> StreamingResponse:
        """Return chunk dicts as an SSE StreamingResponse that ends with [DONE].

        Open WebUI passes a StreamingResponse on unchanged. For a generator it
        appends a finish_reason "stop" chunk of its own, which would make API
        clients report "stop" for a round that ended with tool calls; and on
        the browser path with stream=false only a StreamingResponse starts
        Open WebUI's tool loop.
        """

        async def body() -> AsyncIterator[str]:
            try:
                if isinstance(chunks, list):
                    for chunk in chunks:
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                else:
                    async for chunk in chunks:
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # Stops the inner generator (and closes its client) also when
                # the client goes away in the middle of the stream.
                aclose = getattr(chunks, "aclose", None)
                if aclose is not None:
                    await aclose()

        return StreamingResponse(body(), media_type="text/event-stream")

    async def _emit_generated_image_files(
        self,
        image_files: List[Dict[str, Any]],
        __event_emitter__: Optional[Callable],
    ) -> bool:
        """Persist generated images on the assistant message via Open WebUI files."""
        if not image_files or not __event_emitter__:
            return False

        try:
            await __event_emitter__(
                {
                    "type": "files",
                    "data": {"files": image_files},
                }
            )
            return True
        except Exception as emit_error:
            self.log.warning(f"Failed to emit generated image files: {emit_error}")
            return False

    async def _emit_generated_video_files(
        self,
        video_files: List[Dict[str, Any]],
        __event_emitter__: Optional[Callable],
    ) -> bool:
        """Persist generated videos on the assistant message via Open WebUI files."""
        if not video_files or not __event_emitter__:
            return False

        try:
            await __event_emitter__(
                {
                    "type": "files",
                    "data": {"files": video_files},
                }
            )
            return True
        except Exception as emit_error:
            self.log.warning(f"Failed to emit generated video files: {emit_error}")
            return False

    @staticmethod
    def _build_generated_image_file(
        content_url: str,
        mime_type: str,
        name: str = "Generated Image",
    ) -> Dict[str, Any]:
        """Build a chat image entry matching Open WebUI's image attachment shape."""
        return {
            "type": "image",
            "url": content_url,
            "content_type": mime_type,
            "name": name,
            "meta": {"content_type": mime_type},
        }

    async def _collect_generated_image(
        self,
        inline_data: Any,
        seen_hashes: set[str],
        generated_images: List[str],
        generated_image_files: List[Dict[str, Any]],
        __request__: Optional[Request],
        __user__: Optional[dict],
        __event_emitter__: Optional[Callable],
    ) -> None:
        """Upload one generated image (or inline it) and record how to show it.

        Uploaded images go to generated_image_files (attached via a "files"
        event); data URLs go to generated_images as markdown.
        """
        mime_type = inline_data.mime_type or "image/png"
        image_data = inline_data.data

        image_hash = self._image_data_hash(image_data)
        if image_hash in seen_hashes:
            self.log.debug(
                "Skipping duplicate generated image part from Gemini response"
            )
            return
        seen_hashes.add(image_hash)

        if __request__ and __user__:
            # Handle generated images with unified upload method
            self.log.debug(
                f"Processing generated image: mime_type={mime_type}, data_type={type(image_data)}, data_length={len(image_data)}"
            )
            image_url = await self._upload_image_with_status(
                image_data,
                mime_type,
                __request__,
                __user__,
                __event_emitter__,
            )
            if image_url.startswith("data:"):
                generated_images.append(f"![Generated Image]({image_url})")
            else:
                generated_image_files.append(
                    self._build_generated_image_file(
                        content_url=image_url,
                        mime_type=mime_type,
                    )
                )
            return

        # Fallback: return as base64 data URL if no request/user context
        if isinstance(image_data, bytes):
            image_data_b64 = base64.b64encode(image_data).decode("utf-8")
        else:
            image_data_b64 = str(image_data)
        data_url = f"data:{mime_type};base64,{image_data_b64}"
        generated_images.append(f"![Generated Image]({data_url})")

    async def _append_generated_images(
        self,
        content: str,
        answer_text: str,
        generated_images: List[str],
        generated_image_files: List[Dict[str, Any]],
        __event_emitter__: Optional[Callable],
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Attach collected images to the message and return the final content.

        Uploaded images are attached to the chat message with a "files" event.
        Without a chat message (API clients) that event reaches nobody, so the
        answer links the uploaded images as markdown instead.
        """
        files_emitted = False
        if self._is_chat_message(__metadata__):
            files_emitted = await self._emit_generated_image_files(
                generated_image_files, __event_emitter__
            )

        if generated_image_files and not files_emitted:
            generated_images.extend(
                f"![Generated Image]({image_file['url']})"
                for image_file in generated_image_files
            )

        if generated_image_files and files_emitted and not answer_text.strip():
            if content:
                content += "\n\n"
            content += "Generated image."

        # Add generated images
        if generated_images:
            if content:
                content += "\n\n"
            content += "\n\n".join(generated_images)

        return content

    @staticmethod
    def _build_generated_video_file(
        file_id: str,
        content_url: str,
        filename: str,
        mime_type: str,
        size: int,
    ) -> Dict[str, Any]:
        """Build a chat file entry that matches Open WebUI's file attachment shape."""
        return {
            "id": file_id,
            "type": "file",
            "url": content_url,
            "name": filename,
            "filename": filename,
            "size": size,
            "content_type": mime_type,
            "meta": {
                "content_type": mime_type,
                "size": size,
            },
        }

    def _check_veo_3_1_support(self, model_id: str) -> bool:
        """Check if a Veo model is version 3.1 (supports reference images, interpolation, 4k, extension)."""
        return "veo-3.1" in model_id.lower()

    def _get_veo_model_capabilities(self, model_id: str) -> Dict[str, Any]:
        """Return per-model feature support matrix based on official Google Veo documentation."""
        model_lower = model_id.lower()
        is_fast = "fast" in model_lower

        # `enhance_prompt` is no longer accepted by any current Veo model in the
        # Gemini API (it was a legacy Vertex-only parameter and the public Veo
        # API parameter table no longer lists it). Sending it now produces
        # `400 INVALID_ARGUMENT: enhancePrompt isn't supported by this model`.
        if "veo-3.1" in model_lower:
            return {
                "version": "3.1",
                "is_fast": is_fast,
                "supports_enhance_prompt": False,
                "supports_resolution": True,
                "valid_resolutions": ["720p", "1080p", "4k"],
                "valid_durations": [4, 6, 8],
                "max_videos": 1,
                "supports_reference_images": True,
                "supports_last_frame": True,
                "supports_extension": True,
            }
        if "veo-3" in model_lower:
            return {
                "version": "3",
                "is_fast": is_fast,
                "supports_enhance_prompt": False,
                "supports_resolution": True,
                "valid_resolutions": ["720p", "1080p"],
                "valid_durations": [8],
                "max_videos": 1,
                "supports_reference_images": False,
                "supports_last_frame": True,
                "supports_extension": False,
            }
        if "veo-2" in model_lower:
            return {
                "version": "2",
                "is_fast": False,
                "supports_enhance_prompt": False,
                "supports_resolution": False,
                "valid_resolutions": [],
                "valid_durations": [5, 6, 8],
                "max_videos": 2,
                "supports_reference_images": False,
                "supports_last_frame": True,
                "supports_extension": False,
            }
        return {
            "version": "unknown",
            "is_fast": is_fast,
            "supports_enhance_prompt": False,
            "supports_resolution": False,
            "valid_resolutions": [],
            "valid_durations": [8],
            "max_videos": 1,
            "supports_reference_images": False,
            "supports_last_frame": False,
            "supports_extension": False,
        }

    def _validate_video_aspect_ratio(self, aspect_ratio: str) -> Optional[str]:
        if not aspect_ratio or aspect_ratio == "default":
            return None
        normalized = aspect_ratio.strip()
        valid = [r for r in VIDEO_ASPECT_RATIO_OPTIONS if r != "default"]
        if normalized in valid:
            return normalized
        self.log.warning(
            f"Invalid video aspect ratio '{aspect_ratio}'. Valid: {', '.join(valid)}. Using default."
        )
        return None

    def _validate_video_resolution(self, resolution: str) -> Optional[str]:
        if not resolution or resolution.lower() == "default":
            return None
        normalized = resolution.strip().lower()
        valid = [r for r in VIDEO_RESOLUTION_OPTIONS if r.lower() != "default"]
        if normalized in valid:
            return normalized
        self.log.warning(
            f"Invalid video resolution '{resolution}'. Valid: {', '.join(valid)}. Using default."
        )
        return None

    def _validate_video_duration(self, duration: str) -> Optional[int]:
        if not duration or duration.lower() == "default":
            return None
        valid = {int(d) for d in VIDEO_DURATION_OPTIONS if d != "default"}
        try:
            val = int(duration)
            if val in valid:
                return val
        except (ValueError, TypeError):
            pass
        self.log.warning(
            f"Invalid video duration '{duration}'. Valid: {', '.join(str(v) for v in sorted(valid))}. Using default."
        )
        return None

    def _build_video_generation_config(
        self,
        body: Dict[str, Any],
        __user__: Optional[dict] = None,
        model_id: str = "",
    ) -> types.GenerateVideosConfig:
        """Build GenerateVideosConfig from valves, user overrides, and model capabilities."""
        caps = self._get_veo_model_capabilities(model_id)

        user_ar = self._get_user_valve_value(__user__, "VIDEO_GENERATION_ASPECT_RATIO")
        aspect_ratio = self._validate_video_aspect_ratio(
            body.get(
                "aspect_ratio", user_ar or self.valves.VIDEO_GENERATION_ASPECT_RATIO
            )
        )

        user_res = self._get_user_valve_value(__user__, "VIDEO_GENERATION_RESOLUTION")
        resolution = self._validate_video_resolution(
            body.get("resolution", user_res or self.valves.VIDEO_GENERATION_RESOLUTION)
        )

        user_dur = self._get_user_valve_value(__user__, "VIDEO_GENERATION_DURATION")
        duration_seconds = self._validate_video_duration(
            body.get("duration", user_dur or self.valves.VIDEO_GENERATION_DURATION)
        )

        negative_prompt = (
            body.get("negative_prompt", self.valves.VIDEO_GENERATION_NEGATIVE_PROMPT)
            or None
        )

        person_generation_raw = body.get(
            "person_generation", self.valves.VIDEO_GENERATION_PERSON_GENERATION
        )
        person_generation = None
        if person_generation_raw and person_generation_raw != "default":
            valid_person_values = [
                v for v in VIDEO_PERSON_GENERATION_OPTIONS if v != "default"
            ]
            if person_generation_raw in valid_person_values:
                person_generation = person_generation_raw
            else:
                self.log.warning(
                    f"Invalid person_generation '{person_generation_raw}'. "
                    f"Valid: {', '.join(valid_person_values)}. Ignoring."
                )

        enhance_prompt = body.get(
            "enhance_prompt", self.valves.VIDEO_GENERATION_ENHANCE_PROMPT
        )

        number_of_videos_raw = body.get("number_of_videos", 1)
        try:
            number_of_videos = int(number_of_videos_raw)
        except (ValueError, TypeError):
            self.log.warning(
                f"Invalid number_of_videos '{number_of_videos_raw}', defaulting to 1"
            )
            number_of_videos = 1

        config_params: Dict[str, Any] = {
            "number_of_videos": min(max(number_of_videos, 1), caps["max_videos"]),
        }

        # enhance_prompt: not supported by Fast models or Veo 2
        if caps["supports_enhance_prompt"] and enhance_prompt:
            config_params["enhance_prompt"] = enhance_prompt

        if aspect_ratio:
            config_params["aspect_ratio"] = aspect_ratio

        # Resolution: not supported by Veo 2; model-specific valid values
        if resolution and caps["supports_resolution"]:
            if resolution in caps["valid_resolutions"]:
                config_params["resolution"] = resolution
            else:
                self.log.warning(
                    f"Resolution '{resolution}' not supported by {model_id}. "
                    f"Valid: {', '.join(caps['valid_resolutions'])}. Using default."
                )

        # Duration: model-specific valid values
        if duration_seconds:
            if duration_seconds in caps["valid_durations"]:
                config_params["duration_seconds"] = duration_seconds
            else:
                self.log.warning(
                    f"Duration {duration_seconds}s not supported by {model_id}. "
                    f"Valid: {', '.join(str(d) for d in caps['valid_durations'])}. Using default."
                )

        if negative_prompt:
            config_params["negative_prompt"] = negative_prompt
        if person_generation:
            config_params["person_generation"] = person_generation

        self.log.debug(f"Video generation config for {model_id}: {config_params}")
        return types.GenerateVideosConfig(**config_params)

    def pipes(self) -> List[Dict[str, str]]:
        """
        Returns a list of available Google Gemini models for the UI.

        Returns:
            List of dictionaries containing model id and name.
        """
        try:
            self.name = "Google Gemini: "
            return self.get_google_models()
        except ValueError as e:
            # Handle the case where API key is missing during pipe listing
            self.log.error(f"Error during pipes listing (validation): {e}")
            return [{"id": "error", "name": str(e)}]
        except Exception as e:
            # Handle other potential errors during model fetching
            self.log.exception(
                f"An unexpected error occurred during pipes listing: {str(e)}"
            )
            return [{"id": "error", "name": f"An unexpected error occurred: {str(e)}"}]

    def _prepare_model_id(self, model_id: str) -> str:
        """
        Prepare and validate the model ID for use with the API.

        Args:
            model_id: The original model ID from the user

        Returns:
            Properly formatted model ID

        Raises:
            ValueError: If the model ID is invalid or unsupported
        """
        original_model_id = model_id
        model_id = self.strip_prefix(model_id)

        valid_prefixes = ("gemini-", "veo-")

        # If the model ID doesn't match a known prefix, try to find it by name
        if not model_id.startswith(valid_prefixes):
            models_list = self.get_google_models()
            found_model = next(
                (m["id"] for m in models_list if m["name"] == original_model_id), None
            )
            if found_model and found_model.startswith(valid_prefixes):
                model_id = found_model
                self.log.debug(
                    f"Mapped model name '{original_model_id}' to model ID '{model_id}'"
                )
            else:
                if not model_id.startswith(valid_prefixes):
                    self.log.error(
                        f"Invalid or unsupported model ID: '{original_model_id}'"
                    )
                    raise ValueError(
                        f"Invalid or unsupported Google model ID or name: '{original_model_id}'"
                    )

        return model_id

    # Matches the thinking summary this pipeline prepends to its own answers, i.e. a
    # <details> block whose <summary> is exactly "Thought (12s)": the plain
    # <details> of chats saved before 1.19.0 and the <details type="reasoning" ...>
    # block (done="false" when the answer was stopped while Gemini was thinking).
    # The summary shape is kept strict so an answer that merely talks about
    # <details> blocks survives.
    # Non-greedy on purpose: a "</details>" inside the quoted thoughts ends the match
    # early and leaves a harmless remainder rather than eating the real answer.
    _THINKING_DETAILS_RE = re.compile(
        r"<details[^>]*>\s*<summary>\s*Thought \(\d+s\)\s*</summary>"
        r".*?</details>\s*",
        re.DOTALL | re.IGNORECASE,
    )

    def _strip_thinking_from_content(self, content: Any) -> Any:
        """Remove rendered thinking summaries from an assistant message.

        Open WebUI stores what the user sees, so the "<details ...><summary>Thought
        (Ns)</summary>" block emitted for the UI comes back verbatim in the message
        history and would otherwise be replayed to the API (issue #176). Gemini has
        no use for it: reasoning context is carried by the API itself, not by the
        rendered markdown.

        Args:
            content: Message content, either a plain string or a multimodal list

        Returns:
            The content with thinking summaries removed (same shape as the input)
        """
        if isinstance(content, str):
            return self._THINKING_DETAILS_RE.sub("", content)
        if isinstance(content, list):
            return [
                (
                    {**item, "text": self._THINKING_DETAILS_RE.sub("", item["text"])}
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                    else item
                )
                for item in content
            ]
        return content

    def _prepare_content(
        self, messages: List[Dict[str, Any]], model_id: str = ""
    ) -> Tuple[List[Union[Dict[str, Any], types.Content]], Optional[str]]:
        """
        Prepare messages content for the API and extract system message if present.

        An assistant message with tool_calls and the tool messages after it
        become a model content with function_call parts followed by one user
        content with function_response parts (see _convert_tool_round).

        Args:
            messages: List of message objects from the request
            model_id: The model ID (decides whether thought signatures are required)

        Returns:
            Tuple of (prepared content list, system message string or None)
        """
        # Extract user-defined system message
        user_system_message = next(
            (msg["content"] for msg in messages if msg.get("role") == "system"),
            None,
        )

        # Combine with default system prompt if configured
        system_message = self._combine_system_prompts(user_system_message)

        # Gemini checks the thought signatures of the function calls in the
        # current turn, which starts at the most recent user message (Open
        # WebUI's message with the images of tool results counts as one).
        turn_start = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            default=-1,
        )
        strict = self._requires_thought_signatures(model_id)

        # Prepare contents for the API
        contents: List[Union[Dict[str, Any], types.Content]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            role = message.get("role")

            tool_calls = message.get("tool_calls") if role == "assistant" else None
            tool_calls = [
                tool_call
                for tool_call in (tool_calls if isinstance(tool_calls, list) else [])
                if isinstance(tool_call, dict)
            ]
            if tool_calls or role == "tool":
                # The run of tool messages that answers this message
                start = index + 1 if tool_calls else index
                end = start
                while end < len(messages) and messages[end].get("role") == "tool":
                    end += 1
                current_turn = index > turn_start
                contents.extend(
                    self._convert_tool_round(
                        message if tool_calls else None,
                        tool_calls,
                        messages[start:end],
                        placeholder=strict and current_turn,
                        # Stored server-side parts only within their own turn:
                        # a later request may have no Search grounding (web
                        # search switched off), and Vertex AI cannot send them.
                        restore_stored=current_turn and not self.valves.USE_VERTEX_AI,
                    )
                )
                index = end
                continue

            index += 1
            if role == "system":
                continue  # Skip system messages, handled separately

            content = message.get("content", "")
            parts = []

            # Map roles: 'assistant' -> 'model', 'user' -> 'user'
            api_role = "model" if role == "assistant" else "user"

            # Never replay our own thinking summary back to the model (issue #176)
            strip_thinking = (
                api_role == "model" and self.valves.STRIP_THINKING_FROM_HISTORY
            )
            if strip_thinking:
                content = self._strip_thinking_from_content(content)

            # Handle different content types
            if isinstance(content, list):  # Multimodal content
                parts.extend(self._process_multimodal_content(content))
            elif isinstance(content, str):  # Plain text content
                parts.append({"text": content})
            else:
                self.log.warning(f"Unsupported message content type: {type(content)}")
                continue  # Skip unsupported content

            # A turn that held nothing but a thinking summary is empty now
            if strip_thinking:
                parts = [
                    p
                    for p in parts
                    if not isinstance(p.get("text"), str) or p["text"].strip()
                ]

            if parts:  # Only add if there are parts
                contents.append({"role": api_role, "parts": parts})

        return contents, system_message

    @staticmethod
    def _requires_thought_signatures(model_id: str) -> bool:
        """Whether Gemini checks the thought signatures of the model's function
        calls (Gemini 3 and later; not Gemini 1.x and 2.x)."""
        return not re.match(r"^gemini-(1|2)[.-]", model_id.lower())

    @staticmethod
    def _tool_call_id(value: Any) -> Optional[str]:
        """A tool call id (or tool_call_id) as str; None when it is missing.

        Some OpenAI-compatible clients send numeric ids; Gemini's ids are str.
        """
        if value is None or value == "":
            return None
        return value if isinstance(value, str) else str(value)

    @staticmethod
    def _tool_call_function(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """The "function" object of a tool call ({} when it is not an object)."""
        function = tool_call.get("function")
        return function if isinstance(function, dict) else {}

    @staticmethod
    def _is_synthetic_tool_call_id(call_id: Any) -> bool:
        """Whether a tool call id is missing or was made up by the pipeline."""
        return not call_id or str(call_id).startswith(SYNTHETIC_TOOL_CALL_ID_PREFIX)

    @staticmethod
    def _is_tool_images_message(message: Dict[str, Any]) -> bool:
        """Whether a message is the user message in which Open WebUI passes on
        the images of tool results."""
        content = message.get("content")
        return (
            message.get("role") == "user"
            and isinstance(content, list)
            and bool(content)
            and isinstance(content[0], dict)
            and content[0].get("type") == "text"
            and content[0].get("text") == TOOL_IMAGES_TEXT
        )

    @classmethod
    def _is_tool_continuation(cls, messages: List[Dict[str, Any]]) -> bool:
        """Whether the request continues a turn after tool results, i.e. it is a
        later round of Open WebUI's tool loop (or an API client's)."""
        if not messages:
            return False
        if messages[-1].get("role") == "tool":
            return True
        return (
            len(messages) > 1
            and cls._is_tool_images_message(messages[-1])
            and messages[-2].get("role") == "tool"
        )

    @staticmethod
    def _message_reasoning_details(message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The pipeline's reasoning_details items of an assistant message."""
        details = message.get("reasoning_details")
        if not details:
            fields = message.get("provider_specific_fields")
            details = (
                fields.get("reasoning_details") if isinstance(fields, dict) else None
            )
        if not isinstance(details, list):
            return []
        return [
            item
            for item in details
            if isinstance(item, dict)
            and item.get("format")
            in (REASONING_FORMAT_SIGNATURE, REASONING_FORMAT_CONTENT)
        ]

    def _convert_tool_round(
        self,
        message: Optional[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        tool_messages: List[Dict[str, Any]],
        placeholder: bool,
        restore_stored: bool = False,
    ) -> List[Union[Dict[str, Any], types.Content]]:
        """Convert an assistant message with tool_calls and its tool messages.

        Gemini expects all function calls of a step in one model content and
        all their results in the user content directly after it (FC1, FC2,
        FR1, FR2); interleaving them is rejected.

        Args:
            message: The assistant message, or None for tool messages that do
                not follow one
            tool_calls: The message's tool calls
            tool_messages: The tool messages directly after the message
            placeholder: Put SKIP_THOUGHT_SIGNATURE on the first function call
                if it has no signature (current turn of a model that checks them)
            restore_stored: Replay the stored model content of a round with
                server-side tool parts (current turn on the Gemini API only)
        """
        contents: List[Union[Dict[str, Any], types.Content]] = []
        call_names: Dict[str, str] = {}
        if message is not None:
            answered = {
                call_id
                for call_id in (
                    self._tool_call_id(tool_message.get("tool_call_id"))
                    for tool_message in tool_messages
                )
                if call_id
            }
            model_content, call_names = self._build_tool_call_content(
                message, tool_calls, answered, placeholder, restore_stored
            )
            if model_content is not None:
                contents.append(model_content)

        responses = [
            part
            for part in (
                self._build_function_response_part(tool_message, call_names)
                for tool_message in tool_messages
            )
            if part is not None
        ]
        if responses:
            contents.append(types.Content(role="user", parts=responses))
        return contents

    def _build_tool_call_content(
        self,
        message: Dict[str, Any],
        tool_calls: List[Dict[str, Any]],
        answered: set,
        placeholder: bool,
        restore_stored: bool = False,
    ) -> Tuple[Optional[types.Content], Dict[str, str]]:
        """Build the model content of an assistant message with tool_calls.

        Returns the content (None if nothing is left) and a map of the tool call
        ids that have a result to their Gemini function names.
        """
        details = self._message_reasoning_details(message)
        parts: Optional[List[types.Part]] = None

        # A round with server-side tool parts (Search grounding together with
        # functions on Gemini 3) is replayed exactly as Gemini sent it within
        # its own turn. Older turns are rebuilt from the tool calls and their
        # signatures: Gemini checks only the current turn, and the server-side
        # parts would need the same grounding tools in this request.
        stored = (
            self._restore_stored_content(details, tool_calls)
            if restore_stored
            else None
        )
        if stored is not None:
            parts = list(stored.parts or [])
        else:
            parts = []
            text = self._tool_call_message_text(message.get("content"))
            if text:
                parts.append(types.Part(text=text))
            signatures = {
                self._tool_call_id(item.get("id")): item.get("data")
                for item in details
                if item.get("type") == "reasoning.encrypted"
                and item.get("format") == REASONING_FORMAT_SIGNATURE
                and self._tool_call_id(item.get("id"))
            }
            for tool_call in tool_calls:
                parts.append(
                    self._build_function_call_part(
                        tool_call,
                        signatures.get(self._tool_call_id(tool_call.get("id"))),
                    )
                )

        # The function_call parts correspond 1:1 (in order) to the tool calls.
        # Only calls with a result are kept: Gemini rejects a function call
        # without a function response (e.g. a call Open WebUI did not run).
        call_names: Dict[str, str] = {}
        kept: List[types.Part] = []
        position = 0
        for part in parts:
            if not part.function_call:
                kept.append(part)
                continue
            tool_call = tool_calls[position] if position < len(tool_calls) else {}
            position += 1
            call_id = self._tool_call_id(tool_call.get("id"))
            if call_id and call_id in answered:
                call_names[call_id] = part.function_call.name or ""
                kept.append(part)
            else:
                self.log.warning(
                    f"Dropping unmatched function call '{part.function_call.name}'"
                )

        calls = [part for part in kept if part.function_call]
        if not calls:
            text = "".join(part.text for part in kept if part.text and not part.thought)
            if not text:
                return None, call_names
            return types.Content(role="model", parts=[types.Part(text=text)]), {}

        if placeholder and not calls[0].thought_signature:
            # Applied after the pairing, so a dropped first call cannot leave the
            # step without a signature. As str: google-genai decodes it like a
            # real (base64) signature and sends the literal value.
            index = next(i for i, part in enumerate(kept) if part is calls[0])
            kept[index] = types.Part.model_validate(
                {
                    **calls[0].model_dump(exclude_none=True),
                    "thought_signature": SKIP_THOUGHT_SIGNATURE,
                }
            )
        return types.Content(role="model", parts=kept), call_names

    def _restore_stored_content(
        self, details: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]]
    ) -> Optional[types.Content]:
        """Restore the model content stored for a round with server-side parts."""
        first_id = self._tool_call_id(tool_calls[0].get("id")) if tool_calls else None
        if not first_id:
            return None
        item = next(
            (
                item
                for item in details
                if item.get("format") == REASONING_FORMAT_CONTENT
                and self._tool_call_id(item.get("id")) == first_id
            ),
            None,
        )
        if item is None:
            return None
        try:
            content = types.Content.model_validate_json(
                base64.b64decode(item.get("data") or "")
            )
            calls = [
                part.function_call for part in content.parts or [] if part.function_call
            ]
            matches = len(calls) == len(tool_calls) and all(
                call.id == self._tool_call_id(tool_call.get("id"))
                if call.id
                else call.name
                == _gemini_function_name(
                    str(self._tool_call_function(tool_call).get("name") or "")
                )
                for call, tool_call in zip(calls, tool_calls)
            )
        except Exception as restore_error:
            self.log.debug(f"Stored model content not readable: {restore_error}")
            matches = False
        if not matches:
            self.log.warning(
                f"Could not restore stored model content for tool call '{first_id}'"
            )
            return None
        return content.model_copy(update={"role": "model"})

    def _tool_call_message_text(self, content: Any) -> str:
        """Text of an assistant message with tool_calls ("" if blank)."""
        if isinstance(content, list):
            content = "".join(
                item["text"]
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            )
        if not isinstance(content, str):
            return ""
        # API clients send the rendered thinking summary back as well
        if self.valves.STRIP_THINKING_FROM_HISTORY:
            content = self._THINKING_DETAILS_RE.sub("", content)
        return content if content.strip() else ""

    def _build_function_call_part(
        self, tool_call: Dict[str, Any], signature: Optional[str]
    ) -> types.Part:
        """Build a function_call part from an OpenAI tool call."""
        function = self._tool_call_function(tool_call)
        name = _gemini_function_name(str(function.get("name") or ""))
        arguments = function.get("arguments")
        args: Any = arguments
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except ValueError:
                args = None
        if not isinstance(args, dict):
            self.log.warning(f"Invalid tool call arguments for '{name}'")
            args = {}
        call_id = self._tool_call_id(tool_call.get("id"))
        function_call = types.FunctionCall(
            id=None if self._is_synthetic_tool_call_id(call_id) else call_id,
            name=name,
            args=args,
        )
        if signature:
            try:
                # The base64 str is passed unchanged; google-genai decodes it
                # (standard and url-safe alphabet) and re-encodes it on the wire.
                return types.Part(
                    function_call=function_call, thought_signature=signature
                )
            except Exception:
                self.log.warning(f"Ignoring invalid thought signature of '{name}'")
        return types.Part(function_call=function_call)

    def _build_function_response_part(
        self, tool_message: Dict[str, Any], call_names: Dict[str, str]
    ) -> Optional[types.Part]:
        """Build a function_response part from a tool message."""
        call_id = self._tool_call_id(tool_message.get("tool_call_id")) or ""
        name = call_names.get(call_id) if call_id else None
        if name is None and tool_message.get("name"):
            # API clients may send the function name with the result
            name = _gemini_function_name(str(tool_message["name"]))
        if name is None:
            self.log.warning(f"Dropping unmatched tool result '{call_id}'")
            return None
        text = self._tool_result_text(tool_message.get("content"))
        response = {"error": text} if text.startswith("Error:") else {"output": text}
        return types.Part(
            function_response=types.FunctionResponse(
                id=None if self._is_synthetic_tool_call_id(call_id) else call_id,
                name=name,
                response=response,
            )
        )

    def _tool_result_text(self, content: Any) -> str:
        """Text of a tool message's content.

        Images are not forwarded here: Open WebUI moves the images of tool
        results into a user message after the tool messages.
        """
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, list):
            texts: List[str] = []
            skipped = 0
            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif (
                    isinstance(item, dict)
                    and item.get("type") in ("text", "input_text")
                    and isinstance(item.get("text"), str)
                ):
                    texts.append(item["text"])
                else:
                    skipped += 1
            if skipped:
                self.log.debug(f"Not forwarding {skipped} non-text tool result part(s)")
            return "".join(texts)
        return json.dumps(content, ensure_ascii=False, default=str)

    def _process_multimodal_content(
        self, content_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multimodal content (text and images).

        Args:
            content_list: List of content items

        Returns:
            List of processed parts for the Gemini API
        """
        parts = []

        for item in content_list:
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")

                if image_url.startswith("data:image"):
                    # Handle base64 encoded image data with optimization
                    try:
                        # Optimize the image before processing
                        optimized_image = self._optimize_image_for_api(image_url)
                        header, encoded = optimized_image.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]

                        # Basic validation for image types
                        if mime_type not in [
                            "image/jpeg",
                            "image/png",
                            "image/webp",
                            "image/heic",
                            "image/heif",
                        ]:
                            self.log.warning(
                                f"Unsupported image mime type: {mime_type}"
                            )
                            parts.append(
                                {"text": f"[Image type {mime_type} not supported]"}
                            )
                            continue

                        # Check if the encoded data is too large
                        if len(encoded) > 15 * 1024 * 1024:  # 15MB limit for base64
                            self.log.warning(
                                f"Image data too large: {len(encoded)} characters"
                            )
                            parts.append(
                                {
                                    "text": "[Image too large for processing - please use a smaller image]"
                                }
                            )
                            continue

                        parts.append(
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": encoded,
                                }
                            }
                        )
                    except Exception as img_ex:
                        self.log.exception(f"Could not parse image data URL: {img_ex}")
                        parts.append({"text": "[Image data could not be processed]"})
                else:
                    # Gemini API doesn't directly support image URLs
                    self.log.warning(f"Direct image URLs not supported: {image_url}")
                    parts.append({"text": f"[Image URL not processed: {image_url}]"})

        return parts

    # _find_image removed (was single-image oriented and is superseded by multi-image logic)

    async def _extract_images_from_message(
        self,
        message: Dict[str, Any],
        *,
        stats_list: Optional[List[Dict[str, Any]]] = None,
        __user__: Optional[dict] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract prompt text and ALL images from a single user message.

        This replaces the previous single-image _find_image logic for image-capable
        models so that multi-image prompts are respected. Open WebUI files are
        read only when they belong to ``__user__`` (see _fetch_file_as_base64).

        Returns:
            (prompt_text, image_parts)
                prompt_text: concatenated text content (may be empty)
                image_parts: list of {"inline_data": {mime_type, data}} dicts
        """
        content = message.get("content", "")
        if not isinstance(content, (list, str)):
            self.log.debug(
                f"Unsupported content type for image extraction: {type(content)}"
            )
        text_segments, urls = self._content_image_sources(content)
        image_parts: List[Dict[str, Any]] = []
        for url in urls:
            part = await self._load_image_part(url, stats_list, __user__)
            if part:
                image_parts.append(part)

        prompt_text = " ".join(s.strip() for s in text_segments if s.strip())
        return prompt_text, image_parts

    @staticmethod
    def _content_image_sources(content: Any) -> Tuple[List[str], List[str]]:
        """(text segments, image URLs) of a message content, both in order.

        The image URLs are data: URLs and Open WebUI file URLs, from image_url
        parts and from markdown image links in the text.
        """
        md_pattern = re.compile(
            r"!\[[^\]]*\]\((data:image[^)]+|/files/[^)]+|/api/v1/files/[^)]+)\)"
        )
        text_segments: List[str] = []
        urls: List[str] = []
        if isinstance(content, str):
            text_segments.append(content)
            urls.extend(match.group(1) for match in md_pattern.finditer(content))
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    txt = item.get("text") or ""
                    text_segments.append(txt)
                    urls.extend(match.group(1) for match in md_pattern.finditer(txt))
                elif item.get("type") == "image_url":
                    url = (item.get("image_url") or {}).get("url") or ""
                    if url.startswith("data:") or "/files/" in url:
                        urls.append(url)
        return text_segments, urls

    async def _load_image_part(
        self,
        url: str,
        stats_list: Optional[List[Dict[str, Any]]],
        __user__: Optional[dict],
    ) -> Optional[Dict[str, Any]]:
        """The inline_data part of one image URL (a data: URL, or an Open WebUI
        file the user may read), optimized in a worker thread so that decoding
        large images does not block the event loop; None if it is unreadable."""
        if url.startswith("data:"):
            data_url: Optional[str] = url
        else:
            data_url = await self._fetch_file_as_base64(url, __user__)
        if not data_url:
            return None
        try:
            optimized = await asyncio.to_thread(
                self._optimize_image_for_api, data_url, stats_list
            )
            header, b64 = optimized.split(",", 1)
            mime = header.split(":", 1)[1].split(";", 1)[0]
            return {"inline_data": {"mime_type": mime, "data": b64}}
        except Exception as e:  # pragma: no cover - defensive
            self.log.warning(f"Skipping image (parse failure): {e}")
            return None

    def _optimize_image_for_api(
        self, image_data: str, stats_list: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Optimize image data for Gemini API using configurable parameters.

        Returns:
            Optimized base64 data URL
        """
        # Check if optimization is enabled
        if not self.valves.IMAGE_ENABLE_OPTIMIZATION:
            self.log.debug("Image optimization disabled via configuration")
            return image_data

        max_size_mb = self.valves.IMAGE_MAX_SIZE_MB
        max_dimension = self.valves.IMAGE_MAX_DIMENSION
        base_quality = self.valves.IMAGE_COMPRESSION_QUALITY
        png_threshold = self.valves.IMAGE_PNG_COMPRESSION_THRESHOLD_MB

        self.log.debug(
            f"Image optimization config: max_size={max_size_mb}MB, max_dim={max_dimension}px, quality={base_quality}, png_threshold={png_threshold}MB"
        )
        try:
            # Parse the data URL
            if image_data.startswith("data:"):
                header, encoded = image_data.split(",", 1)
                mime_type = header.split(":")[1].split(";")[0]
            else:
                encoded = image_data
                mime_type = "image/png"

            # Decode and analyze the image
            image_bytes = base64.b64decode(encoded)
            original_size_mb = len(image_bytes) / (1024 * 1024)
            base64_size_mb = len(encoded) / (1024 * 1024)

            self.log.debug(
                f"Original image: {original_size_mb:.2f} MB (decoded), {base64_size_mb:.2f} MB (base64), type: {mime_type}"
            )

            # Determine optimization strategy
            reasons: List[str] = []
            if original_size_mb > max_size_mb:
                reasons.append(f"size > {max_size_mb} MB")
            if base64_size_mb > max_size_mb * 1.4:
                reasons.append("base64 overhead")
            if mime_type == "image/png" and original_size_mb > png_threshold:
                reasons.append(f"PNG > {png_threshold}MB")

            # Always check dimensions
            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
                resized_flag = False
                if width > max_dimension or height > max_dimension:
                    reasons.append(f"dimensions > {max_dimension}px")

                # Early exit: no optimization triggers -> keep original, record stats
                if not reasons:
                    if stats_list is not None:
                        stats_list.append(
                            {
                                "original_size_mb": round(original_size_mb, 4),
                                "final_size_mb": round(original_size_mb, 4),
                                "quality": None,
                                "format": mime_type.split("/")[-1].upper(),
                                "resized": False,
                                "reasons": ["no_optimization_needed"],
                                "final_hash": hashlib.sha256(
                                    encoded.encode()
                                ).hexdigest(),
                            }
                        )
                    self.log.debug(
                        "Skipping optimization: image already within thresholds"
                    )
                    return image_data

                self.log.debug(f"Optimization triggers: {', '.join(reasons)}")

                # Convert to RGB for JPEG compression
                if img.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    background.paste(
                        img,
                        mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None,
                    )
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize if needed
                if width > max_dimension or height > max_dimension:
                    ratio = min(max_dimension / width, max_dimension / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    self.log.debug(
                        f"Resizing from {width}x{height} to {new_size[0]}x{new_size[1]}"
                    )
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    resized_flag = True

                # Determine quality levels based on original size and user configuration
                if original_size_mb > 5.0:
                    quality_levels = [
                        base_quality,
                        base_quality - 10,
                        base_quality - 20,
                        base_quality - 30,
                        base_quality - 40,
                        max(base_quality - 50, 25),
                    ]
                elif original_size_mb > 2.0:
                    quality_levels = [
                        base_quality,
                        base_quality - 5,
                        base_quality - 15,
                        base_quality - 25,
                        max(base_quality - 35, 35),
                    ]
                else:
                    quality_levels = [
                        min(base_quality + 5, 95),
                        base_quality,
                        base_quality - 10,
                        max(base_quality - 20, 50),
                    ]

                # Ensure quality levels are within valid range (1-100)
                quality_levels = [max(1, min(100, q)) for q in quality_levels]

                # Try compression levels
                for quality in quality_levels:
                    output_buffer = io.BytesIO()
                    format_type = (
                        "JPEG"
                        if original_size_mb > png_threshold or "jpeg" in mime_type
                        else "PNG"
                    )
                    output_mime = f"image/{format_type.lower()}"

                    img.save(
                        output_buffer,
                        format=format_type,
                        quality=quality,
                        optimize=True,
                    )
                    output_bytes = output_buffer.getvalue()
                    output_size_mb = len(output_bytes) / (1024 * 1024)

                    if output_size_mb <= max_size_mb:
                        optimized_b64 = base64.b64encode(output_bytes).decode("utf-8")
                        self.log.debug(
                            f"Optimized: {original_size_mb:.2f} MB → {output_size_mb:.2f} MB (Q{quality})"
                        )
                        if stats_list is not None:
                            stats_list.append(
                                {
                                    "original_size_mb": round(original_size_mb, 4),
                                    "final_size_mb": round(output_size_mb, 4),
                                    "quality": quality,
                                    "format": format_type,
                                    "resized": resized_flag,
                                    "reasons": reasons,
                                    "final_hash": hashlib.sha256(
                                        optimized_b64.encode()
                                    ).hexdigest(),
                                }
                            )
                        return f"data:{output_mime};base64,{optimized_b64}"

                # Fallback: minimum quality
                output_buffer = io.BytesIO()
                img.save(output_buffer, format="JPEG", quality=15, optimize=True)
                output_bytes = output_buffer.getvalue()
                output_size_mb = len(output_bytes) / (1024 * 1024)
                optimized_b64 = base64.b64encode(output_bytes).decode("utf-8")

                self.log.warning(
                    f"Aggressive optimization: {output_size_mb:.2f} MB (Q15)"
                )
                if stats_list is not None:
                    stats_list.append(
                        {
                            "original_size_mb": round(original_size_mb, 4),
                            "final_size_mb": round(output_size_mb, 4),
                            "quality": 15,
                            "format": "JPEG",
                            "resized": resized_flag,
                            "reasons": reasons + ["fallback_min_quality"],
                            "final_hash": hashlib.sha256(
                                optimized_b64.encode()
                            ).hexdigest(),
                        }
                    )
                return f"data:image/jpeg;base64,{optimized_b64}"

        except Exception as e:
            self.log.error(f"Image optimization failed: {e}")
            # Return original or safe fallback
            if image_data.startswith("data:"):
                if stats_list is not None:
                    stats_list.append(
                        {
                            "original_size_mb": None,
                            "final_size_mb": None,
                            "quality": None,
                            "format": None,
                            "resized": False,
                            "reasons": ["optimization_failed"],
                            "final_hash": (
                                hashlib.sha256(encoded.encode()).hexdigest()
                                if "encoded" in locals()
                                else None
                            ),
                        }
                    )
                return image_data
            return f"data:image/jpeg;base64,{encoded if 'encoded' in locals() else image_data}"

    async def _fetch_file_as_base64(
        self, file_url: str, __user__: Optional[dict] = None
    ) -> Optional[str]:
        """
        Fetch a file from Open WebUI's file system and convert to base64.

        Only files of the requesting user are read (an admin may read every
        file), like Open WebUI's own image resolver: a message can name any
        file id, for example in a markdown link sent by an API client.

        Args:
            file_url: File URL from Open WebUI
            __user__: The requesting user (without one no file is read)

        Returns:
            Base64 encoded file data or None if file not found or not readable
        """
        try:
            if "/api/v1/files/" in file_url:
                fid = file_url.split("/api/v1/files/")[-1].split("/")[0].split("?")[0]
            else:
                fid = file_url.split("/files/")[-1].split("/")[0].split("?")[0]

            from pathlib import Path
            from open_webui.storage.provider import Storage

            file_obj = await Files.get_file_by_id(fid)
            user = __user__ or {}
            if (
                file_obj
                and file_obj.user_id != user.get("id")
                and user.get("role") != "admin"
            ):
                # the file id of another user stays out of the log
                self.log.warning(
                    f"Not reading a file for user {user.get('id') or '?'}: it "
                    "does not belong to the requesting user"
                )
                return None
            if file_obj and file_obj.path:
                file_path = await asyncio.to_thread(Storage.get_file, file_obj.path)
                file_path = Path(file_path)
                if file_path.is_file():
                    async with aiofiles.open(file_path, "rb") as fp:
                        raw = await fp.read()
                    enc = base64.b64encode(raw).decode()
                    mime = (file_obj.meta or {}).get("content_type") or "image/png"
                    return f"data:{mime};base64,{enc}"
        except Exception as e:
            self.log.warning(f"Could not fetch file {file_url}: {e}")
        return None

    async def _upload_image_with_status(
        self,
        image_data: Any,
        mime_type: str,
        __request__: Request,
        __user__: dict,
        __event_emitter__: Optional[Callable],
    ) -> str:
        """
        Unified image upload method with status updates and fallback handling.

        Returns:
            URL to uploaded image or data URL fallback
        """
        try:
            await self._safe_emit(
                __event_emitter__,
                {
                    "type": "status",
                    "data": {
                        "action": "image_upload",
                        "description": "Uploading generated image to your library...",
                        "done": False,
                    },
                },
            )

            user = await Users.get_user_by_id(__user__["id"])

            # Convert image data to base64 string if needed
            if isinstance(image_data, bytes):
                image_data_b64 = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data_b64 = str(image_data)

            image_url = await self._upload_image(
                __request__=__request__,
                user=user,
                image_data=image_data_b64,
                mime_type=mime_type,
            )

            await self._safe_emit(
                __event_emitter__,
                {
                    "type": "status",
                    "data": {
                        "action": "image_upload",
                        "description": "Image uploaded successfully!",
                        "done": True,
                    },
                },
            )

            return image_url

        except Exception as e:
            self.log.warning(f"File upload failed, falling back to data URL: {e}")

            if isinstance(image_data, bytes):
                image_data_b64 = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data_b64 = str(image_data)

            await self._safe_emit(
                __event_emitter__,
                {
                    "type": "status",
                    "data": {
                        "action": "image_upload",
                        "description": "Using inline image (upload failed)",
                        "done": True,
                    },
                },
            )

            return f"data:{mime_type};base64,{image_data_b64}"

    async def _upload_image(
        self, __request__: Request, user: UserModel, image_data: str, mime_type: str
    ) -> str:
        """
        Upload generated image to Open WebUI's file system.
        Expects base64 encoded string input.

        Args:
            __request__: FastAPI request object
            user: User model object
            image_data: Base64 encoded image data string
            mime_type: MIME type of the image

        Returns:
            URL to the uploaded image or data URL fallback
        """
        try:
            self.log.debug(
                f"Processing image data, type: {type(image_data)}, length: {len(image_data)}"
            )

            # Decode base64 string to bytes
            try:
                decoded_data = base64.b64decode(image_data)
                self.log.debug(
                    f"Successfully decoded image data: {len(decoded_data)} bytes"
                )
            except Exception as decode_error:
                self.log.error(f"Failed to decode base64 data: {decode_error}")
                # Try to add padding if missing
                try:
                    missing_padding = len(image_data) % 4
                    if missing_padding:
                        image_data += "=" * (4 - missing_padding)
                    decoded_data = base64.b64decode(image_data)
                    self.log.debug(
                        f"Successfully decoded with padding: {len(decoded_data)} bytes"
                    )
                except Exception as second_decode_error:
                    self.log.error(f"Still failed to decode: {second_decode_error}")
                    return f"data:{mime_type};base64,{image_data}"

            bio = io.BytesIO(decoded_data)
            bio.seek(0)

            # Determine file extension
            extension = "png"
            if "jpeg" in mime_type or "jpg" in mime_type:
                extension = "jpg"
            elif "webp" in mime_type:
                extension = "webp"
            elif "gif" in mime_type:
                extension = "gif"

            # Create filename
            filename = f"gemini-generated-{uuid.uuid4().hex}.{extension}"

            # Upload with simple approach like reference
            async with get_async_db_context() as db:
                up_obj = await upload_file(
                    request=__request__,
                    background_tasks=BackgroundTasks(),
                    file=UploadFile(
                        file=bio,
                        filename=filename,
                        headers=Headers({"content-type": mime_type}),
                    ),
                    process=False,  # Matching reference - no heavy processing
                    user=user,
                    metadata={
                        "mime_type": mime_type,
                        "source": "gemini_image_generation",
                    },
                    db=db,
                )

            self.log.debug(
                f"Upload completed. File ID: {up_obj.id}, Decoded size: {len(decoded_data)} bytes"
            )

            # Generate URL using reference method
            return __request__.app.url_path_for("get_file_content_by_id", id=up_obj.id)

        except Exception as e:
            self.log.exception(f"Image upload failed, using data URL fallback: {e}")
            # Fallback to data URL if upload fails
            return f"data:{mime_type};base64,{image_data}"

    async def _upload_video(
        self,
        __request__: Request,
        user: UserModel,
        video_data: bytes,
        mime_type: str = "video/mp4",
        chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Upload generated video to Open WebUI's file system.

        Returns:
            Tuple of (content_url, file_entry)
        """
        bio = io.BytesIO(video_data)
        bio.seek(0)

        extension = "mp4"
        if "webm" in mime_type:
            extension = "webm"

        filename = f"veo-generated-{uuid.uuid4().hex}.{extension}"

        async with get_async_db_context() as db:
            up_obj = await upload_file(
                request=__request__,
                background_tasks=BackgroundTasks(),
                file=UploadFile(
                    file=bio,
                    filename=filename,
                    headers=Headers({"content-type": mime_type}),
                ),
                process=False,
                user=user,
                metadata={"mime_type": mime_type, "source": "veo_video_generation"},
                db=db,
            )

            if chat_id and message_id:
                try:
                    await Chats.insert_chat_files(
                        chat_id=chat_id,
                        message_id=message_id,
                        file_ids=[up_obj.id],
                        user_id=user.id,
                        db=db,
                    )
                except Exception as chat_file_error:
                    self.log.warning(
                        f"Failed to link generated video file to chat message: {chat_file_error}"
                    )

        content_url = str(
            __request__.app.url_path_for("get_file_content_by_id", id=up_obj.id)
        )
        self.log.debug(
            f"Video upload completed. File ID: {up_obj.id}, Size: {len(video_data)} bytes"
        )
        return content_url, self._build_generated_video_file(
            file_id=up_obj.id,
            content_url=content_url,
            filename=filename,
            mime_type=mime_type,
            size=len(video_data),
        )

    async def _upload_video_with_status(
        self,
        video_data: bytes,
        mime_type: str,
        __request__: Request,
        __user__: dict,
        __event_emitter__: Optional[Callable],
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Upload video with status updates and data-URL fallback.

        Returns:
            Tuple of (file_entry_or_None, content_url_or_data_url_or_None)
        """
        try:
            await self._safe_emit(
                __event_emitter__,
                {
                    "type": "status",
                    "data": {
                        "action": "video_upload",
                        "description": "Uploading generated video to your library...",
                        "done": False,
                    },
                },
            )

            user = await Users.get_user_by_id(__user__["id"])
            chat_id = __metadata__.get("chat_id") if __metadata__ else None
            message_id = __metadata__.get("message_id") if __metadata__ else None
            video_url, file_entry = await self._upload_video(
                __request__=__request__,
                user=user,
                video_data=video_data,
                mime_type=mime_type,
                chat_id=chat_id,
                message_id=message_id,
            )

            await self._safe_emit(
                __event_emitter__,
                {
                    "type": "status",
                    "data": {
                        "action": "video_upload",
                        "description": "Video uploaded successfully!",
                        "done": True,
                    },
                },
            )
            return file_entry, video_url

        except Exception as e:
            self.log.warning(f"Video upload failed, falling back to data URL: {e}")
            video_data_b64 = base64.b64encode(video_data).decode("utf-8")
            await self._safe_emit(
                __event_emitter__,
                {
                    "type": "status",
                    "data": {
                        "action": "video_upload",
                        "description": "Using inline video (upload failed)",
                        "done": True,
                    },
                },
            )
            return None, f"data:{mime_type};base64,{video_data_b64}"

    def _get_user_valve_value(
        self, __user__: Optional[dict], valve_name: str
    ) -> Optional[str]:
        """Get a user valve value, returning None if not set or set to 'default'"""
        if __user__ and "valves" in __user__:
            value = getattr(__user__["valves"], valve_name, None)
            if value and value != "default":
                return value
        return None

    # JSON Schema keys that google-genai sends verbatim (parameters_json_schema)
    # but that Gemini does not need or may reject.
    _SCHEMA_DROP_KEYS = frozenset(
        {
            "$schema",
            "$id",
            "$comment",
            "examples",
            "deprecated",
            "readOnly",
            "writeOnly",
        }
    )
    # Keywords whose value maps names to schemas: the names are not keywords.
    _SCHEMA_NAME_MAP_KEYS = frozenset(
        {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
    )

    @classmethod
    def _sanitize_parameters_schema(cls, schema: Any) -> Optional[Dict[str, Any]]:
        """Clean up a tool's JSON Schema for parameters_json_schema.

        parameters_json_schema accepts $ref/$defs, anyOf/oneOf/allOf, type
        arrays with "null", enum, format and default, which FunctionDeclaration
        .parameters (Gemini's Schema) rejects, so the schema is kept as it is
        apart from keys that are not needed or reported to fail.

        Returns None for an object schema without properties (a tool without
        parameters), which is then declared without parameters.
        """

        def walk(node: Any) -> Any:
            if isinstance(node, list):
                return [walk(item) for item in node]
            if not isinstance(node, dict):
                return node
            out: Dict[str, Any] = {}
            for key, value in node.items():
                if key in cls._SCHEMA_DROP_KEYS or str(key).startswith("x-"):
                    continue
                if key in ("exclusiveMinimum", "exclusiveMaximum") and isinstance(
                    value, bool
                ):
                    continue
                if key == "items" and isinstance(value, list):
                    out["prefixItems"] = walk(value)  # tuple form
                elif key in cls._SCHEMA_NAME_MAP_KEYS and isinstance(value, dict):
                    out[key] = {name: walk(sub) for name, sub in value.items()}
                else:
                    out[key] = walk(value)
            required, properties = out.get("required"), out.get("properties")
            if isinstance(required, list) and isinstance(properties, dict):
                out["required"] = [
                    name
                    for name in required
                    if isinstance(name, str) and name in properties
                ]
            return out

        root = walk(copy.deepcopy(schema) if isinstance(schema, dict) else {})
        root.setdefault("type", "object")
        if (
            root.get("type") == "object"
            and not root.get("properties")
            and not any(key in root for key in ("$ref", "anyOf", "oneOf", "allOf"))
        ):
            return None
        return root

    def _build_function_declarations(
        self,
        body: Dict[str, Any],
        __metadata__: Optional[Dict[str, Any]],
        image_model: bool,
        model_id: str,
    ) -> Tuple[List[types.FunctionDeclaration], Dict[str, str]]:
        """Declare the tools of the request (body["tools"]) to Gemini.

        Open WebUI puts an OpenAI tool spec into body["tools"] for every tool it
        offers in Native mode (built-in, workspace, MCP, OpenAPI, terminal and
        direct tools) and runs Gemini's function calls itself; API clients send
        their own tools and run the calls themselves.

        Returns:
            (declarations, {gemini_name: open_webui_name}), rebuilt per request
        """
        tools = body.get("tools")
        if not isinstance(tools, list) or not tools:
            return [], {}
        metadata = __metadata__ or {}
        if (metadata.get("params") or {}).get("function_calling") == "legacy":
            return [], {}  # Open WebUI chooses the tools itself
        if metadata.get("task"):
            self.log.debug(
                f"No function declarations for background task '{metadata.get('task')}'"
            )
            return [], {}
        if image_model:
            # Gemini image models do not support function calling. In Native
            # mode Open WebUI 0.10+ attaches its built-in tools to every chat,
            # so none are sent; this also keeps Open WebUI's own generate_image
            # and edit_image tools away from native Gemini image generation.
            self.log.debug(
                f"Not sending {len(tools)} tool(s) to image model {model_id}"
            )
            return [], {}

        declarations: List[types.FunctionDeclaration] = []
        name_map: Dict[str, str] = {}
        for entry in tools:
            if (
                not isinstance(entry, dict)
                or entry.get("type", "function") != "function"
            ):
                continue
            spec = entry.get("function")
            name = spec.get("name") if isinstance(spec, dict) else None
            if not isinstance(name, str) or not name:
                continue
            gemini_name = _gemini_function_name(name)
            if gemini_name in name_map:
                # Gemini rejects a request with duplicate declarations
                self.log.warning(f"Skipping duplicate tool declaration '{name}'")
                continue
            name_map[gemini_name] = name
            declaration: Dict[str, Any] = {"name": gemini_name}
            description = spec.get("description")
            if isinstance(description, str) and description.strip():
                declaration["description"] = description.strip()
            schema = self._sanitize_parameters_schema(spec.get("parameters"))
            if schema is not None:
                declaration["parameters_json_schema"] = schema
            declarations.append(types.FunctionDeclaration(**declaration))
        return declarations, name_map

    def _map_tool_choice(
        self, tool_choice: Any, name_map: Dict[str, str]
    ) -> Optional[types.FunctionCallingConfig]:
        """Map an OpenAI tool_choice (sent by API clients) to Gemini's config.

        "auto", None and unknown shapes leave Gemini's default in place.
        parallel_tool_calls has no Gemini equivalent and is ignored.
        """
        if tool_choice == "none":
            # The declarations stay, so function calls in the history remain valid
            return types.FunctionCallingConfig(mode="NONE")
        if tool_choice == "required":
            return types.FunctionCallingConfig(mode="ANY")
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function = tool_choice.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(name, str) and name:
                gemini_name = _gemini_function_name(name)
                if name_map.get(gemini_name) == name:
                    return types.FunctionCallingConfig(
                        mode="ANY", allowed_function_names=[gemini_name]
                    )
            self.log.debug(f"Ignoring tool_choice for undeclared function '{name}'")
        return None

    @staticmethod
    def _owui_function_name(name: str, name_map: Dict[str, str]) -> str:
        """Map the name of a Gemini function call back to the Open WebUI tool.

        Unknown names are passed on unchanged; Open WebUI then answers with a
        "Tool not found" result.
        """
        for prefix in ("default_api.", "default_api:"):
            if name.startswith(prefix) and name[len(prefix) :] in name_map:
                name = name[len(prefix) :]
                break
        return name_map.get(name, name)

    @staticmethod
    def _is_server_side_part(part: Any) -> bool:
        """Whether a part holds a tool call Gemini ran itself (e.g. Search)."""
        return any(
            getattr(part, field, None) is not None
            for field in (
                "tool_call",
                "tool_response",
                "executable_code",
                "code_execution_result",
            )
        )

    def _function_calls_to_openai(
        self, parts: List[types.Part], name_map: Dict[str, str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert Gemini function_call parts to OpenAI tool_calls.

        Returns (tool_calls, reasoning_details). Each thought signature becomes
        a reasoning_details item with the id of its tool call; Open WebUI (and
        API clients) send it back with the tool call in the next request.
        """
        tool_calls: List[Dict[str, Any]] = []
        details: List[Dict[str, Any]] = []
        for index, part in enumerate(parts):
            call = part.function_call
            # Vertex AI and Gemini 2.5 may send no id
            call_id = (
                call.id or f"{SYNTHETIC_TOOL_CALL_ID_PREFIX}{uuid.uuid4().hex[:24]}"
            )
            tool_calls.append(
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": self._owui_function_name(call.name or "", name_map),
                        "arguments": json.dumps(
                            call.args or {}, ensure_ascii=False, default=str
                        ),
                    },
                }
            )
            if part.thought_signature:
                details.append(
                    {
                        "type": "reasoning.encrypted",
                        "format": REASONING_FORMAT_SIGNATURE,
                        "id": call_id,
                        "index": index,
                        "data": base64.b64encode(part.thought_signature).decode(
                            "ascii"
                        ),
                    }
                )
        return tool_calls, details

    @staticmethod
    def _stored_content_item(
        parts: List[types.Part], tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """A reasoning_details item that stores the complete model content.

        Used for rounds with server-side tool parts: Gemini expects them back
        with all their fields. Consecutive unsigned text parts with the same
        thought flag (stream pieces) are merged.
        """
        merged: List[Tuple[types.Part, bool]] = []
        for part in parts:
            plain_text = isinstance(part.text, str) and set(
                part.model_dump(exclude_none=True)
            ) <= {"text", "thought"}
            if (
                plain_text
                and merged
                and merged[-1][1]
                and bool(merged[-1][0].thought) == bool(part.thought)
            ):
                previous = merged[-1][0]
                merged[-1] = (
                    previous.model_copy(update={"text": previous.text + part.text}),
                    True,
                )
            else:
                merged.append((part, plain_text))
        content = types.Content(role="model", parts=[part for part, _ in merged])
        return {
            "type": "reasoning.encrypted",
            "format": REASONING_FORMAT_CONTENT,
            "id": tool_calls[0]["id"],
            "index": len(tool_calls),
            "data": base64.b64encode(
                content.model_dump_json(exclude_none=True).encode("utf-8")
            ).decode("ascii"),
        }

    def _tool_round_result(
        self,
        function_call_parts: List[types.Part],
        all_parts: List[types.Part],
        server_side: bool,
        name_map: Dict[str, str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """tool_calls and reasoning_details of a round that ended with calls."""
        tool_calls, details = self._function_calls_to_openai(
            function_call_parts, name_map
        )
        if server_side:
            details.append(self._stored_content_item(all_parts, tool_calls))
        return tool_calls, details

    def _round_chunks(
        self,
        chunk: Callable[..., Dict[str, Any]],
        *,
        reasoning: Optional[str] = None,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        details: Optional[List[Dict[str, Any]]] = None,
        usage: Optional[Dict[str, int]] = None,
        content_first: bool = False,
        stop: bool = False,
    ) -> List[Dict[str, Any]]:
        """The chunks that end a round.

        With tool calls: reasoning_content (R), reasoning_details (D), content
        (C, before D with content_first), tool_calls (T), finish_reason
        "tool_calls" (F) and usage (U). Without tool calls the same minus D/T,
        with a finish_reason "stop" chunk only if `stop` (own SSE framing).
        """
        chunks: List[Dict[str, Any]] = []
        if reasoning:
            chunks.append(chunk({"reasoning_content": reasoning}))
        if content and content_first:
            chunks.append(chunk({"content": content}))
        if details:
            chunks.append(chunk({"reasoning_details": details}))
        if content and not content_first:
            chunks.append(chunk({"content": content}))
        if tool_calls:
            chunks.append(
                chunk({"role": "assistant", "content": None, "tool_calls": tool_calls})
            )
            chunks.append(chunk({}, "tool_calls"))
        elif stop:
            chunks.append(chunk({}, "stop"))
        if usage:
            chunks.append(self._usage_chunk(usage))
        return chunks

    @staticmethod
    def _finish_reason_name(finish_reason: Any) -> str:
        """Name of a finish reason; compared as str because newer SDKs add
        enum members."""
        if finish_reason is None:
            return ""
        return str(getattr(finish_reason, "name", None) or finish_reason)

    @classmethod
    def _tool_call_error_text(cls, finish_reason: Any) -> Optional[str]:
        """Answer text for a round in which Gemini's tool call failed."""
        name = cls._finish_reason_name(finish_reason)
        if name == "MALFORMED_FUNCTION_CALL":
            return (
                "Error: Gemini returned a malformed tool call "
                "(MALFORMED_FUNCTION_CALL). Please try again."
            )
        if name == "UNEXPECTED_TOOL_CALL":
            return (
                "Error: Gemini called a tool that is not available "
                "(UNEXPECTED_TOOL_CALL)."
            )
        return None

    @staticmethod
    def _thinking_details_block(
        thought_text: str, duration_s: int, done: bool = True
    ) -> str:
        """The <details> thinking summary of an answer without tool calls.

        type="reasoning" makes Open WebUI render it as its own reasoning item,
        titled in the user's language ("Thought for 5 seconds", or "Thinking..."
        with a spinner while done="false"); it ignores the <summary> then. The
        summary stays "Thought (Ns)" for API clients and other renderers, and
        _THINKING_DETAILS_RE finds the block by it.
        """
        quoted_content = "\n".join(
            f"> {line}" for line in thought_text.strip().split("\n")
        )
        attributes = (
            f'type="reasoning" done="true" duration="{duration_s}"'
            if done
            else 'type="reasoning" done="false"'
        )
        return f"""<details {attributes}>
<summary>Thought ({duration_s}s)</summary>

{quoted_content}

</details>""".strip()

    def _configure_generation(
        self,
        body: Dict[str, Any],
        system_instruction: Optional[str],
        __metadata__: Dict[str, Any],
        __user__: Optional[dict] = None,
        enable_image_generation: bool = False,
        model_id: str = "",
        function_declarations: Optional[List[types.FunctionDeclaration]] = None,
        name_map: Optional[Dict[str, str]] = None,
    ) -> types.GenerateContentConfig:
        """
        Configure generation parameters and safety settings.

        Args:
            body: The request body containing generation parameters
            system_instruction: Optional system instruction string
            enable_image_generation: Whether to enable image generation
            model_id: The model ID being used (for feature support checks)
            function_declarations: The request's tools (_build_function_declarations)
            name_map: Gemini function name -> Open WebUI tool name

        Returns:
            types.GenerateContentConfig
        """
        gen_config_params = {
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "max_output_tokens": body.get("max_tokens"),
            "stop_sequences": body.get("stop") or None,
            "system_instruction": system_instruction,
        }

        # Enable image generation if requested
        if enable_image_generation:
            gen_config_params["response_modalities"] = ["TEXT", "IMAGE"]

            # Configure image generation parameters (aspect ratio and resolution)
            # ImageConfig is only supported by Gemini 3 models
            if self._check_image_config_support(model_id):
                # Body parameters override valve defaults for per-request customization
                # Get aspect_ratio: body > user_valves (if not "default") > system valves
                user_aspect_ratio = self._get_user_valve_value(
                    __user__, "IMAGE_GENERATION_ASPECT_RATIO"
                )
                aspect_ratio = body.get(
                    "aspect_ratio",
                    user_aspect_ratio or self.valves.IMAGE_GENERATION_ASPECT_RATIO,
                )

                # Get resolution: body > user_valves (if not "default") > system valves
                user_resolution = self._get_user_valve_value(
                    __user__, "IMAGE_GENERATION_RESOLUTION"
                )
                resolution = body.get(
                    "resolution",
                    user_resolution or self.valves.IMAGE_GENERATION_RESOLUTION,
                )

                # Validate and normalize the values
                validated_aspect_ratio = self._validate_aspect_ratio(aspect_ratio)
                validated_resolution = self._validate_resolution(resolution)
                supported_sizes = self._get_supported_image_sizes(model_id)
                if (
                    validated_resolution
                    and supported_sizes is not None
                    and validated_resolution not in supported_sizes
                ):
                    self.log.warning(
                        f"Resolution '{validated_resolution}' is not supported by {model_id} "
                        f"(supported: {', '.join(supported_sizes)}). Using the model default."
                    )
                    validated_resolution = None

                # Create image config if we have at least one valid value
                if validated_aspect_ratio or validated_resolution:
                    try:
                        image_config_params = {}
                        if validated_aspect_ratio:
                            image_config_params["aspect_ratio"] = validated_aspect_ratio
                        if validated_resolution:
                            image_config_params["image_size"] = validated_resolution
                        gen_config_params["image_config"] = types.ImageConfig(
                            **image_config_params
                        )
                        self.log.debug(
                            f"Image generation config: aspect_ratio={validated_aspect_ratio}, resolution={validated_resolution}"
                        )
                    except (AttributeError, TypeError) as e:
                        # Fall back if SDK does not support ImageConfig
                        self.log.warning(
                            f"ImageConfig not supported by SDK version: {e}. Image generation will use default settings."
                        )
                    except Exception as e:
                        # Log unexpected errors but continue without image config
                        self.log.warning(
                            f"Unexpected error configuring ImageConfig: {e}"
                        )
            else:
                self.log.debug(
                    f"Model {model_id} does not support ImageConfig (aspect_ratio/resolution). "
                    "ImageConfig is only available for Gemini 3 image models."
                )

        # Configure Gemini thinking/reasoning for models that support it
        # This is independent of include_thoughts - thinking config controls HOW the model reasons,
        # while include_thoughts controls whether the reasoning is shown in the output
        if self._check_thinking_support(model_id):
            try:
                thinking_config_params: Dict[str, Any] = {}

                # Determine include_thoughts setting
                include_thoughts = body.get("include_thoughts", True)
                if not self.valves.INCLUDE_THOUGHTS:
                    include_thoughts = False
                    self.log.debug(
                        "Thoughts output disabled via GOOGLE_INCLUDE_THOUGHTS"
                    )
                thinking_config_params["include_thoughts"] = include_thoughts

                # Check if model supports thinking_level (Gemini 3 models)
                if self._check_thinking_level_support(model_id):
                    # For Gemini 3 models, use thinking_level (not thinking_budget)
                    # Per-chat reasoning_effort overrides environment-level THINKING_LEVEL
                    reasoning_effort = body.get("reasoning_effort")
                    validated_level = None
                    source = None

                    if reasoning_effort:
                        validated_level = self._validate_thinking_level(
                            reasoning_effort, model_id
                        )
                        if validated_level:
                            source = "per-chat reasoning_effort"
                        else:
                            self.log.debug(
                                f"Invalid reasoning_effort '{reasoning_effort}', falling back to THINKING_LEVEL"
                            )

                    # Fall back to environment-level THINKING_LEVEL if no valid reasoning_effort
                    if not validated_level:
                        validated_level = self._validate_thinking_level(
                            self.valves.THINKING_LEVEL, model_id
                        )
                        if validated_level:
                            source = "THINKING_LEVEL"

                    if validated_level:
                        thinking_config_params["thinking_level"] = validated_level
                        self.log.debug(
                            f"Using thinking_level='{validated_level}' from {source} for model {model_id}"
                        )
                    else:
                        self.log.debug(
                            f"Using default thinking level for model {model_id}"
                        )
                else:
                    # For non-Gemini 3 models (e.g., Gemini 2.5), use thinking_budget
                    # Body-level thinking_budget overrides environment-level THINKING_BUDGET
                    body_thinking_budget = body.get("thinking_budget")
                    validated_budget = None
                    source = None

                    if body_thinking_budget is not None:
                        validated_budget = self._validate_thinking_budget(
                            body_thinking_budget
                        )
                        if validated_budget is not None:
                            source = "body thinking_budget"
                        else:
                            self.log.debug(
                                f"Invalid body thinking_budget '{body_thinking_budget}', falling back to THINKING_BUDGET"
                            )

                    # Fall back to environment-level THINKING_BUDGET
                    if validated_budget is None:
                        validated_budget = self._validate_thinking_budget(
                            self.valves.THINKING_BUDGET
                        )
                        if validated_budget is not None:
                            source = "THINKING_BUDGET"

                    if validated_budget == 0:
                        # Disable thinking if budget is 0
                        thinking_config_params["thinking_budget"] = 0
                        self.log.debug(
                            f"Thinking disabled via thinking_budget=0 from {source} for model {model_id}"
                        )
                    elif validated_budget is not None and validated_budget > 0:
                        thinking_config_params["thinking_budget"] = validated_budget
                        self.log.debug(
                            f"Using thinking_budget={validated_budget} from {source} for model {model_id}"
                        )
                    else:
                        # -1 or None means dynamic thinking
                        thinking_config_params["thinking_budget"] = -1
                        self.log.debug(
                            f"Using dynamic thinking (model decides) for model {model_id}"
                        )

                gen_config_params["thinking_config"] = types.ThinkingConfig(
                    **thinking_config_params
                )
            except (AttributeError, TypeError) as e:
                # Fall back if SDK/model does not support ThinkingConfig
                self.log.debug(f"ThinkingConfig not supported: {e}")
            except Exception as e:
                # Log unexpected errors but continue without thinking config
                self.log.warning(f"Unexpected error configuring ThinkingConfig: {e}")

        # Configure safety settings
        if self.valves.USE_PERMISSIVE_SAFETY:
            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
                ),
            ]
            gen_config_params |= {"safety_settings": safety_settings}

        # Add various tools to Gemini as required
        features = __metadata__.get("features", {})
        params = __metadata__.get("params", {})
        tools = []

        # Background tasks (title, tags, follow-ups, queries) inherit the chat's
        # request metadata, including the grounding flags that the search filters
        # set there. A task only works on the chat it is given, so it gets no
        # grounding tools and does not run Google or Vertex AI searches.
        is_task = bool(__metadata__.get("task"))
        if is_task and (
            features.get("google_search_tool") or features.get("vertex_ai_search")
        ):
            self.log.debug(
                f"Grounding disabled for background task '{__metadata__.get('task')}'"
            )

        if features.get("google_search_tool", False) and not is_task:
            if not self._supports_search_grounding(model_id):
                self.log.debug(
                    f"Search grounding is not supported by {model_id}; "
                    "not sending the search tool"
                )
            elif self.valves.USE_ENTERPRISE_WEB_SEARCH:
                self.log.debug("Enabling Enterprise Web Search grounding")
                tools.append(
                    types.Tool(enterprise_web_search=types.EnterpriseWebSearch())
                )
            else:
                self.log.debug("Enabling Google search grounding")
                tools.append(types.Tool(google_search=types.GoogleSearch()))
            # Gemini image models do not support URL context (most of them
            # support Search grounding, see _supports_search_grounding).
            if enable_image_generation:
                self.log.debug("URL context is not supported by image models")
            else:
                self.log.debug("Enabling URL context grounding")
                tools.append(types.Tool(url_context=types.UrlContext()))

        if not is_task and (
            features.get("vertex_ai_search", False)
            or (
                self.valves.USE_VERTEX_AI
                and (
                    self.valves.VERTEX_AI_RAG_STORE or os.getenv("VERTEX_AI_RAG_STORE")
                )
            )
        ):
            vertex_rag_store = (
                params.get("vertex_rag_store")
                or self.valves.VERTEX_AI_RAG_STORE
                or os.getenv("VERTEX_AI_RAG_STORE")
            )
            if vertex_rag_store:
                self.log.debug(
                    f"Enabling Vertex AI Search grounding: {vertex_rag_store}"
                )
                tools.append(
                    types.Tool(
                        retrieval=types.Retrieval(
                            vertex_ai_search=types.VertexAISearch(
                                datastore=vertex_rag_store
                            )
                        )
                    )
                )
            else:
                self.log.warning(
                    "Vertex AI Search requested but vertex_rag_store not provided in params, valves, or env"
                )

        # Function declarations go after the grounding tools. Gemini only returns
        # the calls; Open WebUI (or the API client) runs the tools.
        tool_config: Optional[types.ToolConfig] = None
        if function_declarations:
            grounding_kinds = {
                kind
                for tool in tools
                for kind in (
                    "google_search",
                    "enterprise_web_search",
                    "url_context",
                    "retrieval",
                )
                if getattr(tool, kind, None) is not None
            }
            if not grounding_kinds:
                tools.append(types.Tool(function_declarations=function_declarations))
                function_calling_config = self._map_tool_choice(
                    body.get("tool_choice"), name_map or {}
                )
                if function_calling_config:
                    tool_config = types.ToolConfig(
                        function_calling_config=function_calling_config
                    )
            elif (
                # Built-in tools together with functions ("tool combination")
                # work on Gemini 3 only and need server-side tool invocations
                # (google-genai >= 1.68.0); Vertex AI does not allow Search with
                # function calling, and the SDK rejects the flag there.
                not self.valves.USE_VERTEX_AI
                and self._is_gemini_3_family_model(model_id)
                and grounding_kinds <= {"google_search", "url_context"}
                and "include_server_side_tool_invocations"
                in types.ToolConfig.model_fields
            ):
                tools.append(types.Tool(function_declarations=function_declarations))
                tool_config_params: Dict[str, Any] = {
                    "include_server_side_tool_invocations": True
                }
                function_calling_config = self._map_tool_choice(
                    body.get("tool_choice"), name_map or {}
                )
                if function_calling_config:
                    tool_config_params["function_calling_config"] = (
                        function_calling_config
                    )
                tool_config = types.ToolConfig(**tool_config_params)
            else:
                self.log.debug(
                    "Grounding tools take precedence: not declaring "
                    f"{len(function_declarations)} function(s) for {model_id}"
                )

        if tools:
            gen_config_params["tools"] = tools
        if tool_config:
            gen_config_params["tool_config"] = tool_config

        # Never let google-genai run tools itself (automatic function calling);
        # no Python callable is passed to it anyway.
        gen_config_params["automatic_function_calling"] = (
            types.AutomaticFunctionCallingConfig(disable=True)
        )

        # Filter out None values for generation config
        filtered_params = {k: v for k, v in gen_config_params.items() if v is not None}
        return types.GenerateContentConfig(**filtered_params)

    @staticmethod
    def _format_grounding_chunks_as_sources(
        grounding_chunks: list[types.GroundingChunk],
    ):
        formatted_sources = []
        for chunk in grounding_chunks:
            if hasattr(chunk, "retrieved_context") and chunk.retrieved_context:
                context = chunk.retrieved_context
                # The SDK field is "text"; "chunk_text" is kept as a fallback.
                chunk_text = (
                    getattr(context, "text", None)
                    or getattr(context, "chunk_text", None)
                    or ""
                )
                formatted_sources.append(
                    {
                        "source": {
                            "name": getattr(context, "title", None) or "Document",
                            "type": "vertex_ai_search",
                            "uri": getattr(context, "uri", None),
                        },
                        "document": [chunk_text],
                        "metadata": [
                            {"source": getattr(context, "title", None) or "Document"}
                        ],
                    }
                )
            elif hasattr(chunk, "web") and chunk.web:
                context = chunk.web
                uri = context.uri
                title = context.title or "Source"

                formatted_sources.append(
                    {
                        "source": {
                            "name": title,
                            "type": "web_search_results",
                            "url": uri,
                        },
                        "document": ["Click the link to view the content."],
                        "metadata": [{"source": title}],
                    }
                )
        return formatted_sources

    async def _process_grounding_metadata(
        self,
        grounding_metadata_list: List[types.GroundingMetadata],
        text: str,
        __event_emitter__: Optional[Callable],
    ):
        """Process and emit grounding metadata events."""
        grounding_chunks = []
        web_search_queries = []
        grounding_supports = []

        for metadata in grounding_metadata_list:
            if metadata.grounding_chunks:
                grounding_chunks.extend(metadata.grounding_chunks)
            if metadata.web_search_queries:
                web_search_queries.extend(metadata.web_search_queries)
            if metadata.grounding_supports:
                grounding_supports.extend(metadata.grounding_supports)

        # Add sources to the response.
        # Emit each source individually via the "source" event type so that
        # citations are persisted by Open WebUI across page refreshes.
        # A "chat:completion" event with a "sources" payload renders citations
        # in the live response but is not stored with the message.
        if grounding_chunks:
            sources = self._format_grounding_chunks_as_sources(grounding_chunks)
            for source in sources:
                await self._safe_emit(
                    __event_emitter__, {"type": "source", "data": source}
                )

        # Search statuses in the shape of Open WebUI's own web search, which it
        # renders in the user's language: the queries ("Searching" + chips), then
        # the sites found ("Searched {{count}} sites" + list). "items" only, no
        # "urls": Open WebUI counts (urls || items). Each query once: Open WebUI
        # keys the query chips by their text, and a duplicate key throws.
        if web_search_queries:
            await self._safe_emit(
                __event_emitter__,
                {
                    "type": "status",
                    "data": {
                        "action": "web_search_queries_generated",
                        "queries": list(dict.fromkeys(web_search_queries)),
                        "done": True,
                    },
                },
            )
            items: List[Dict[str, str]] = []
            for chunk in grounding_chunks:
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None) if web else None
                if uri and all(item["link"] != uri for item in items):
                    items.append({"title": web.title or uri, "link": uri})
            if items:
                await self._safe_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "action": "web_search",
                            "description": "Searched {{count}} sites",
                            "items": items,
                            "done": True,
                        },
                    },
                )

        # Add citations in the text body
        replaced_text: Optional[str] = None
        if grounding_supports:
            # Citation indexes are in bytes
            ENCODING = "utf-8"
            text_bytes = text.encode(ENCODING)
            last_byte_index = 0
            cited_chunks = []

            for support in grounding_supports:
                cited_chunks.append(
                    text_bytes[last_byte_index : support.segment.end_index].decode(
                        ENCODING
                    )
                )

                # Generate and append citations (e.g., "[1][2]")
                footnotes = "".join(
                    [f"[{i + 1}]" for i in support.grounding_chunk_indices]
                )
                cited_chunks.append(f" {footnotes}")

                # Update index for the next segment
                last_byte_index = support.segment.end_index

            # Append any remaining text after the last citation
            if last_byte_index < len(text_bytes):
                cited_chunks.append(text_bytes[last_byte_index:].decode(ENCODING))

            replaced_text = "".join(cited_chunks)

        return replaced_text if replaced_text is not None else text

    async def _start_stream(
        self, open_stream: Callable[[], Awaitable[AsyncIterator[Any]]]
    ) -> AsyncIterator[Any]:
        """Open a Gemini stream, retrying temporary errors up to its first chunk.

        generate_content_stream returns a lazy iterator: the HTTP request, and a
        ServerError (5xx) with it, only happens when the first chunk is read.
        Opening the stream and reading that chunk are therefore retried together
        (RETRY_COUNT), before anything has been sent to Open WebUI. Errors after
        the first chunk are not retried, because the answer has already started.
        """

        async def open_and_read_first() -> Tuple[AsyncIterator[Any], List[Any]]:
            stream = (await open_stream()).__aiter__()
            try:
                return stream, [await stream.__anext__()]
            except StopAsyncIteration:
                return stream, []

        stream, first = await self._retry_with_backoff(open_and_read_first)

        async def chunks() -> AsyncIterator[Any]:
            for chunk in first:
                yield chunk
            async for chunk in stream:
                yield chunk

        return chunks()

    async def _handle_streaming_response(
        self,
        open_stream: Callable[[], Awaitable[AsyncIterator[Any]]],
        __event_emitter__: Optional[Callable],
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
        client: Optional[genai.Client] = None,
        model: str = "",
        __metadata__: Optional[Dict[str, Any]] = None,
        tool_mode: bool = False,
        name_map: Optional[Dict[str, str]] = None,
        continuation: bool = False,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Handle streaming response from Gemini API.

        Args:
            open_stream: Coroutine function that calls generate_content_stream.
                It is called (and retried on temporary errors, see _start_stream)
                when Open WebUI starts reading this generator.
            __event_emitter__: Event emitter for status updates
            client: The per-request genai client used by open_stream. The stream
                sends its HTTP request lazily through this client's transport, so
                the client must stay referenced until the stream ends. If it were
                garbage collected, google-genai would close the transport and the
                stream would fail. It is closed at the end.
            model: Model ID of the request (Open WebUI's, with function prefix)
                for the chat.completion.chunk dicts.
            __metadata__: Request metadata (decides how generated images are
                attached and whether the request belongs to a chat message).
            tool_mode: Functions were declared. A round that ends with function
                calls returns them as tool_calls; on the API path the generator
                also yields its own finish chunks (pipe() wraps it in _sse).
            name_map: Gemini function name -> Open WebUI tool name.
            continuation: The request continues a turn after tool results.

        Returns:
            Generator yielding chat.completion.chunk dicts (answer, error
            messages and tool calls) and a usage chunk
        """
        # Remember statuses that are still running, to close them if the request
        # is stopped or fails (see finally).
        running_statuses: Dict[str, Dict[str, Any]] = {}
        __event_emitter__ = self._track_statuses(__event_emitter__, running_statuses)
        cancelled = False

        is_chat = self._is_chat_message(__metadata__)
        # A later round of Open WebUI's tool loop ("live" mode): from the first
        # tool round on, the chat shows the message's output items, which only
        # chunks yielded by the pipe reach (chat:* events and the text preview
        # change nothing visible). Thoughts and text are yielded as they arrive.
        live = is_chat and continuation
        # API path in tool mode: own finish chunks, so the last finish_reason a
        # client sees is "tool_calls" when the round ended with tool calls.
        own_finish = tool_mode and not is_chat
        response_id = f"{model}-{uuid.uuid4()}"
        created = int(time.time())

        def chunk(
            delta: Dict[str, Any], finish_reason: Optional[str] = None
        ) -> Dict[str, Any]:
            return self._chunk(delta, model, finish_reason, response_id, created)

        def text_chunk(content: str) -> Dict[str, Any]:
            return chunk({"role": "assistant", "content": content})

        async def emit_chat_event(event_type: str, data: Dict[str, Any]) -> None:
            if not __event_emitter__ or live:
                return
            try:
                await __event_emitter__({"type": event_type, "data": data})
            except Exception as emit_error:  # pragma: no cover - defensive
                self.log.warning(f"Failed to emit {event_type} event: {emit_error}")

        await emit_chat_event("chat:start", {"role": "assistant"})

        grounding_metadata_list = []
        # Accumulate content separately for answer and thoughts
        answer_chunks: list[str] = []
        thought_chunks: list[str] = []
        # The thinking lasts from the first thought to the first answer part
        # after it (or to the end of the stream if no answer follows)
        thinking_started_at: Optional[float] = None
        thinking_ended_at: Optional[float] = None
        # In a chat (not live), the thoughts so far are shown as a live
        # <details type="reasoning" done="false"> block through replace events,
        # which Open WebUI renders as its own "Thinking..." item
        show_live_thinking = is_chat and not live and bool(__event_emitter__)
        last_live_thinking = 0.0
        stream_usage_metadata = None
        generated_images: list[str] = []
        generated_image_files: List[Dict[str, Any]] = []
        seen_generated_image_hashes: set[str] = set()
        last_thought_image: Any = None
        last_finish_reason: Any = None
        # Tool mode: function calls of this round (sent when the stream has
        # ended), all its parts and whether Gemini ran server-side tools
        function_call_parts: List[types.Part] = []
        all_parts: List[types.Part] = []
        server_side = False

        def thinking_block(done: bool) -> str:
            started = thinking_started_at or time.time()
            ended = thinking_ended_at or time.time()
            return self._thinking_details_block(
                "".join(thought_chunks), int(max(0, ended - started)), done=done
            )

        async def emit_live_thinking() -> None:
            """Replace the message with the thinking block (done once the answer
            has started) and the answer so far."""
            nonlocal last_live_thinking
            last_live_thinking = time.monotonic()
            block = thinking_block(done=thinking_ended_at is not None)
            await emit_chat_event(
                "replace",
                {"role": "assistant", "content": block + "".join(answer_chunks)},
            )

        try:
            response_iterator = await self._start_stream(open_stream)
            async for response_chunk in response_iterator:
                # Capture usage metadata (final chunk has complete data)
                if getattr(response_chunk, "usage_metadata", None):
                    stream_usage_metadata = response_chunk.usage_metadata
                if response_chunk.candidates and getattr(
                    response_chunk.candidates[0], "finish_reason", None
                ):
                    last_finish_reason = response_chunk.candidates[0].finish_reason

                # Check for safety feedback or empty chunks
                if not response_chunk.candidates:
                    # Check prompt feedback
                    feedback = response_chunk.prompt_feedback
                    if feedback and feedback.block_reason:
                        block_reason = feedback.block_reason.name
                        message = f"[Blocked due to Prompt Safety: {block_reason}]"
                    else:
                        message = "[Blocked by safety settings]"
                    await emit_chat_event(
                        "chat:finish",
                        {
                            "role": "assistant",
                            "content": message,
                            "done": True,
                            "error": True,
                        },
                    )
                    yield text_chunk(message)
                    if own_finish:
                        yield chunk({}, "stop")
                    return  # Stop generation

                candidate = response_chunk.candidates[0]
                if candidate.grounding_metadata:
                    grounding_metadata_list.append(candidate.grounding_metadata)
                # Prefer fine-grained parts to split thoughts vs. normal text.
                # A candidate without content (e.g. a finish reason only) has
                # no parts.
                parts = []
                try:
                    content = getattr(candidate, "content", None)
                    parts = (content.parts or []) if content is not None else []
                except Exception as parts_error:
                    # Fallback: use aggregated text if parts aren't accessible
                    self.log.warning(f"Failed to access content parts: {parts_error}")
                    if hasattr(response_chunk, "text") and response_chunk.text:
                        answer_chunks.append(response_chunk.text)
                        if live:
                            yield chunk({"content": response_chunk.text})
                        else:
                            await self._safe_emit(
                                __event_emitter__,
                                {
                                    "type": "chat:message:delta",
                                    "data": {
                                        "role": "assistant",
                                        "content": response_chunk.text,
                                    },
                                },
                            )
                    continue

                for part in parts:
                    try:
                        if tool_mode:
                            all_parts.append(part)
                            if getattr(part, "function_call", None):
                                function_call_parts.append(part)
                                continue
                            if self._is_server_side_part(part):
                                server_side = True
                                continue

                        is_thought = bool(getattr(part, "thought", False))
                        # Thought parts (internal reasoning)
                        if is_thought and getattr(part, "text", None):
                            if thinking_started_at is None:
                                thinking_started_at = time.time()
                            thought_chunks.append(part.text)
                            if live:
                                yield chunk({"reasoning_content": part.text})
                                continue
                            # Show the thoughts so far (throttled; the first
                            # answer part always sends the finished block)
                            if (
                                show_live_thinking
                                and time.monotonic() - last_live_thinking
                                >= LIVE_THINKING_INTERVAL
                            ):
                                await emit_live_thinking()

                        # Interim image from the thinking process: the final image
                        # follows as a regular part, so this one is not uploaded
                        # (only kept as fallback if no final image arrives).
                        elif is_thought and getattr(part, "inline_data", None):
                            self.log.debug("Skipping interim thought image")
                            last_thought_image = part.inline_data

                        # Regular answer text
                        elif getattr(part, "text", None):
                            answer_chunks.append(part.text)
                            thinking_done = bool(thought_chunks) and (
                                thinking_ended_at is None
                            )
                            if thinking_done:
                                thinking_ended_at = time.time()
                            if live:
                                yield chunk({"content": part.text})
                            elif thinking_done and show_live_thinking:
                                # The finished block plus this first answer part
                                await emit_live_thinking()
                            else:
                                await self._safe_emit(
                                    __event_emitter__,
                                    {
                                        "type": "chat:message:delta",
                                        "data": {
                                            "role": "assistant",
                                            "content": part.text,
                                        },
                                    },
                                )

                        # Generated images: normally image models use the
                        # non-streaming path, but a model that is not detected
                        # as an image model can still return inline images.
                        elif getattr(part, "inline_data", None):
                            self.log.info(
                                "Gemini returned an image while streaming; "
                                "attaching it to the final message"
                            )
                            await self._collect_generated_image(
                                part.inline_data,
                                seen_generated_image_hashes,
                                generated_images,
                                generated_image_files,
                                __request__,
                                __user__,
                                __event_emitter__,
                            )
                    except Exception as part_error:
                        # Log part processing errors but continue with the stream
                        self.log.warning(f"Error processing content part: {part_error}")
                        continue

            # seen_generated_image_hashes records every final (non-thought) image
            # part. If the response had thought images only, attach the last one.
            if (
                last_thought_image is not None
                and not seen_generated_image_hashes
                and self._allows_thought_image_fallback(last_finish_reason)
            ):
                self.log.warning(
                    "Gemini returned thought images but no final image; "
                    "attaching the last thought image instead"
                )
                await self._collect_generated_image(
                    last_thought_image,
                    seen_generated_image_hashes,
                    generated_images,
                    generated_image_files,
                    __request__,
                    __user__,
                    __event_emitter__,
                )

            final_answer_text = "".join(answer_chunks)

            # A failed tool call without any answer: tell the user instead of
            # leaving the message empty.
            error_text = (
                None
                if function_call_parts or final_answer_text
                else self._tool_call_error_text(last_finish_reason)
            )
            if error_text:
                final_answer_text = error_text
                if live:
                    yield chunk({"content": error_text})

            usage = self._build_usage_dict(stream_usage_metadata)

            if live:
                # Sources and the search status only: the text has already been
                # sent, so no [n] citation markers can be added to it.
                if grounding_metadata_list and __event_emitter__:
                    await self._process_grounding_metadata(
                        grounding_metadata_list,
                        final_answer_text,
                        __event_emitter__,
                    )
                if generated_images or generated_image_files:
                    links = await self._append_generated_images(
                        "",
                        final_answer_text,
                        generated_images,
                        generated_image_files,
                        __event_emitter__,
                        __metadata__,
                    )
                    if links:
                        yield chunk(
                            {"content": f"\n\n{links}" if final_answer_text else links}
                        )
                tool_calls, details = (
                    self._tool_round_result(
                        function_call_parts, all_parts, server_side, name_map or {}
                    )
                    if function_call_parts
                    else ([], [])
                )
                for round_chunk in self._round_chunks(
                    chunk, tool_calls=tool_calls, details=details, usage=usage
                ):
                    yield round_chunk
                return

            # After processing all chunks, handle grounding data
            if grounding_metadata_list and __event_emitter__:
                cited = await self._process_grounding_metadata(
                    grounding_metadata_list,
                    final_answer_text,
                    __event_emitter__,
                )
                final_answer_text = cited or final_answer_text

            details_block: Optional[str] = None
            if thought_chunks:
                details_block = thinking_block(done=True)

            if function_call_parts:
                # The round ends with tool calls: the turn is not over, so no
                # chat:message / chat:finish. In the chat the thoughts go to
                # Open WebUI's reasoning item (reasoning_content), which it
                # shows anyway once a thought signature arrives; API clients
                # get the <details> summary as before.
                if show_live_thinking and thought_chunks:
                    # The output items take over: do not leave the live block
                    # in the saved content
                    await emit_chat_event(
                        "replace", {"role": "assistant", "content": ""}
                    )
                content = final_answer_text
                if own_finish and details_block:
                    content = f"{details_block}{content}"
                if generated_images or generated_image_files:
                    content = await self._append_generated_images(
                        content,
                        final_answer_text,
                        generated_images,
                        generated_image_files,
                        __event_emitter__,
                        __metadata__,
                    )
                tool_calls, details = self._tool_round_result(
                    function_call_parts, all_parts, server_side, name_map or {}
                )
                reasoning = (
                    "".join(thought_chunks).strip()
                    if is_chat and thought_chunks
                    else None
                )
                for round_chunk in self._round_chunks(
                    chunk,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_calls,
                    details=details,
                    usage=usage,
                    content_first=own_finish,
                ):
                    yield round_chunk
                return

            final_content = final_answer_text
            if details_block:
                final_content = f"{details_block}{final_answer_text}"

            if not final_content:
                final_content = ""

            if generated_images or generated_image_files:
                final_content = await self._append_generated_images(
                    final_content,
                    final_answer_text,
                    generated_images,
                    generated_image_files,
                    __event_emitter__,
                    __metadata__,
                )

            # Ensure downstream consumers (UI, TTS) receive the complete response once streaming ends.
            await emit_chat_event(
                "replace", {"role": "assistant", "content": final_content}
            )
            await emit_chat_event(
                "chat:message",
                {"role": "assistant", "content": final_content, "done": True},
            )

            if own_finish:
                # API path in tool mode: answer, own finish chunk, then usage
                await emit_chat_event(
                    "chat:finish",
                    {"role": "assistant", "content": final_content, "done": True},
                )
                yield text_chunk(final_content)
                yield chunk({}, "stop")
                if usage:
                    yield self._usage_chunk(usage)
                return

            # Yield usage data as dict so the middleware can extract and save it to DB
            if usage:
                yield self._usage_chunk(usage)

            await emit_chat_event(
                "chat:finish",
                {"role": "assistant", "content": final_content, "done": True},
            )

            # Yield final content to ensure the async iterator completes properly.
            # This ensures the response is persisted even if the user navigates away.
            # As a chunk dict, so an answer starting with "data:" stays content.
            yield text_chunk(final_content)

        except (asyncio.CancelledError, GeneratorExit):
            # Stopped by the user or the client went away
            cancelled = True
            raise

        except Exception as e:
            self.log.exception(f"Error during streaming: {e}")
            # Check if it's a chunk size error and provide specific guidance
            error_msg = str(e).lower()
            if "chunk too big" in error_msg or "chunk size" in error_msg:
                message = "Error: Image too large for processing. Please try with a smaller image (max 15 MB recommended) or reduce image quality."
            elif "quota" in error_msg or "rate limit" in error_msg:
                message = "Error: API quota exceeded. Please try again later."
            else:
                message = f"Error during streaming: {e}"
            await emit_chat_event(
                "chat:finish",
                {
                    "role": "assistant",
                    "content": message,
                    "done": True,
                    "error": True,
                },
            )
            yield text_chunk(message)
            if own_finish:
                yield chunk({}, "stop")

        finally:
            await self._finish_running_statuses(
                __event_emitter__, running_statuses, cancelled
            )
            await self._close_client(client)

    @staticmethod
    def _usage_chunk(usage: Dict[str, int]) -> Dict[str, Any]:
        """OpenAI-style final stream chunk that carries only token usage."""
        return {"choices": [], "usage": usage}

    def _build_non_stream_result(
        self,
        content: str,
        usage: Optional[Dict[str, int]],
        model: str,
        stream_requested: bool,
    ) -> Union[Dict[str, Any], AsyncIterator[Union[str, Dict[str, Any]]]]:
        """
        Return a completed (non-streamed) answer in the shape Open WebUI expects.

        - Request with stream=false (API clients, background tasks such as title,
          tag and follow-up generation): an OpenAI chat.completion dict, so the
          token usage is returned to the caller and saved by Open WebUI.
        - Request with stream=true answered without streaming (image models, or
          GOOGLE_STREAMING_ENABLED=false): an async generator that yields one
          content chunk and then a usage chunk. A chat.completion dict would be
          forwarded as a single chunk without a delta and the message would
          stay empty.
        """
        response_id = f"{model}-{uuid.uuid4()}"
        created = int(time.time())

        if stream_requested:

            async def single_chunk_stream():
                # A dict chunk (not a str) so content starting with "data:" is not
                # mistaken for a raw SSE line by Open WebUI.
                yield {
                    "id": response_id,
                    "created": created,
                    "model": model,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                            "delta": {"role": "assistant", "content": content},
                        }
                    ],
                }
                if usage:
                    yield self._usage_chunk(usage)

            return single_chunk_stream()

        result: Dict[str, Any] = {
            "id": response_id,
            "created": created,
            "model": model,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
        }
        if usage:
            result["usage"] = usage
        return result

    @staticmethod
    async def _iterate_chunks(
        chunks: List[Dict[str, Any]],
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield prepared chunks (a stream answer built without streaming)."""
        for chunk in chunks:
            yield chunk

    async def _build_tool_round_response(
        self,
        *,
        final_answer: str,
        details_block: str,
        reasoning: str,
        function_call_parts: List[types.Part],
        all_parts: List[types.Part],
        server_side: bool,
        name_map: Dict[str, str],
        usage: Optional[Dict[str, int]],
        model: str,
        is_chat: bool,
        continuation: bool,
        stream_requested: bool,
        generated_images: List[str],
        generated_image_files: List[Dict[str, Any]],
        __event_emitter__: Optional[Callable],
        __metadata__: Optional[Dict[str, Any]],
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]], StreamingResponse]:
        """
        Return a round answered without streaming in the shape its path needs.

        - Browser path: reasoning_content, reasoning_details, the answer, the
          tool calls and their finish chunk, then usage (in a later round of the
          tool loop the answer comes right after the thoughts). With stream=true
          as a generator; with stream=false as a StreamingResponse, the only
          shape for which Open WebUI runs its tool loop.
        - API path: the answer as before (<details> summary + text). With
          stream=true as SSE with own finish chunks; with stream=false as a
          chat.completion with message.tool_calls.
        """
        response_id = f"{model}-{uuid.uuid4()}"
        created = int(time.time())

        def chunk(
            delta: Dict[str, Any], finish_reason: Optional[str] = None
        ) -> Dict[str, Any]:
            return self._chunk(delta, model, finish_reason, response_id, created)

        tool_calls, details = (
            self._tool_round_result(
                function_call_parts, all_parts, server_side, name_map
            )
            if function_call_parts
            else ([], [])
        )

        if is_chat:
            content = await self._append_generated_images(
                final_answer,
                final_answer,
                generated_images,
                generated_image_files,
                __event_emitter__,
                __metadata__,
            )
            chunks = self._round_chunks(
                chunk,
                reasoning=reasoning or None,
                content=content,
                tool_calls=tool_calls,
                details=details,
                usage=usage,
                content_first=continuation,
            )
            if stream_requested:
                return self._iterate_chunks(chunks)
            return self._sse(chunks)

        content = await self._append_generated_images(
            details_block + final_answer,
            final_answer,
            generated_images,
            generated_image_files,
            __event_emitter__,
            __metadata__,
        )
        if not stream_requested:
            return self._tool_calls_completion(
                content, tool_calls, details, usage, model
            )
        if tool_calls:
            return self._sse(
                self._round_chunks(
                    chunk,
                    content=content,
                    tool_calls=tool_calls,
                    details=details,
                    usage=usage,
                    content_first=True,
                )
            )
        chunks = [
            chunk(
                {"role": "assistant", "content": content or "[No content generated]"}
            ),
            chunk({}, "stop"),
        ]
        if usage:
            chunks.append(self._usage_chunk(usage))
        return self._sse(chunks)

    @staticmethod
    def _tool_calls_completion(
        content: str,
        tool_calls: List[Dict[str, Any]],
        details: List[Dict[str, Any]],
        usage: Optional[Dict[str, int]],
        model: str,
    ) -> Dict[str, Any]:
        """A chat.completion whose message carries tool_calls (API clients)."""
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
            # Open WebUI returns this dict unchanged; message tool_calls have
            # no index (only stream deltas do)
            "tool_calls": [
                {key: value for key, value in tool_call.items() if key != "index"}
                for tool_call in tool_calls
            ],
        }
        if details:
            message["reasoning_details"] = details
        result: Dict[str, Any] = {
            "id": f"{model}-{uuid.uuid4()}",
            "created": int(time.time()),
            "model": model,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                    "message": message,
                }
            ],
        }
        if usage:
            result["usage"] = usage
        return result

    @staticmethod
    def _build_usage_dict(usage_metadata: Any) -> Optional[Dict[str, int]]:
        """Extract token usage from Gemini usage_metadata into a standardised dict."""
        if not usage_metadata:
            return None
        usage: Dict[str, int] = {}
        if getattr(usage_metadata, "prompt_token_count", None) is not None:
            usage["prompt_tokens"] = usage_metadata.prompt_token_count
        if getattr(usage_metadata, "candidates_token_count", None) is not None:
            usage["completion_tokens"] = usage_metadata.candidates_token_count
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
                "completion_tokens", 0
            )
            return usage
        return None

    def _get_safety_block_message(self, response: Any) -> Optional[str]:
        """Check for safety blocks and return appropriate message."""
        # Check prompt feedback
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"[Blocked due to Prompt Safety: {response.prompt_feedback.block_reason.name}]"

        # Check candidates
        if not response.candidates:
            return "[Blocked by safety settings or no candidates generated]"

        # Check candidate finish reason
        candidate = response.candidates[0]
        if candidate.finish_reason == types.FinishReason.SAFETY:
            blocking_rating = next(
                (r for r in candidate.safety_ratings if r.blocked), None
            )
            reason = f" ({blocking_rating.category.name})" if blocking_rating else ""
            return f"[Blocked by safety settings{reason}]"
        elif candidate.finish_reason == types.FinishReason.PROHIBITED_CONTENT:
            return "[Content blocked due to prohibited content policy violation]"

        return None

    def _allows_thought_image_fallback(self, finish_reason: Any) -> bool:
        """
        Whether the last thought image may stand in for a missing final image.

        Only for a normal finish: any other reason (IMAGE_SAFETY,
        IMAGE_PROHIBITED_CONTENT, NO_IMAGE, ...) means Gemini withheld the final
        image, so no interim image is shown in its place.
        """
        if finish_reason is None:
            return True
        name = getattr(finish_reason, "name", None) or str(finish_reason)
        return name in ("STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED")

    async def _generate_video(
        self,
        body: Dict[str, Any],
        model_id: str,
        __event_emitter__: Optional[Callable],
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        user: Optional[UserModel] = None,
    ) -> Union[str, Dict[str, Any], AsyncIterator[Union[str, Dict[str, Any]]]]:
        """Generate video using Google Veo models (long-running operation with polling).

        `user` is the requesting user for the forwarded user info headers.
        """

        async def emit_status(description: str, done: bool) -> None:
            if not __event_emitter__:
                return
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "action": "video_generation",
                            "description": description,
                            "done": done,
                        },
                    }
                )
            except Exception as e:
                self.log.warning(f"Failed to emit video status event: {e}")

        messages = body.get("messages", [])
        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if not last_user_msg:
            return "Error: No user message found for video generation"

        prompt, images = await self._extract_images_from_message(
            last_user_msg, __user__=__user__
        )
        if not prompt:
            return "Error: No prompt provided for video generation"

        # Convert first attached image to types.Image for image-to-video
        reference_image = None
        if images:
            first_img = images[0]
            try:
                img_data = first_img.get("inline_data", {})
                raw_data = img_data.get("data", "")
                img_bytes = base64.b64decode(raw_data)
                reference_image = types.Image(
                    image_bytes=img_bytes,
                    mime_type=img_data.get("mime_type", "image/png"),
                )
                self.log.debug("Using attached image for image-to-video generation")
            except Exception as e:
                self.log.warning(f"Failed to convert image for Veo: {e}")

        config = self._build_video_generation_config(body, __user__, model_id=model_id)

        await emit_status(f"Starting video generation with {model_id}...", False)

        client = self._get_client(user)
        try:
            return await self._run_video_generation(
                client=client,
                model_id=model_id,
                prompt=prompt,
                reference_image=reference_image,
                config=config,
                emit_status=emit_status,
                body=body,
                __event_emitter__=__event_emitter__,
                __request__=__request__,
                __user__=__user__,
                __metadata__=__metadata__,
            )
        finally:
            # The client is per request: close it after polling and the
            # downloads (which use its sync transport), also on error or stop.
            await self._close_client(client)

    async def _run_video_generation(
        self,
        client: genai.Client,
        model_id: str,
        prompt: str,
        reference_image: Optional[types.Image],
        config: types.GenerateVideosConfig,
        emit_status: Callable[[str, bool], Awaitable[None]],
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable],
        __request__: Optional[Request],
        __user__: Optional[dict],
        __metadata__: Optional[Dict[str, Any]],
    ) -> Union[str, Dict[str, Any], AsyncIterator[Union[str, Dict[str, Any]]]]:
        """Start a Veo operation, poll it, then upload and attach the videos."""
        try:
            # The prompt/image arguments are deprecated in google-genai; the
            # inputs go into a GenerateVideosSource instead.
            source = types.GenerateVideosSource(prompt=prompt, image=reference_image)
            operation = await client.aio.models.generate_videos(
                model=model_id,
                source=source,
                config=config,
            )
        except Exception as e:
            self.log.exception(f"Video generation request failed: {e}")
            await emit_status(f"Video generation failed: {e}", True)
            return f"Error starting video generation: {e}"

        poll_interval = max(self.valves.VIDEO_POLL_INTERVAL, 5)
        poll_timeout = max(self.valves.VIDEO_POLL_TIMEOUT, 0)
        elapsed = 0
        while not operation.done:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            if poll_timeout > 0 and elapsed >= poll_timeout:
                error_msg = (
                    f"Video generation timed out after {elapsed}s "
                    f"(limit: {poll_timeout}s)"
                )
                self.log.error(error_msg)
                await emit_status(error_msg, True)
                return f"Error: {error_msg}"
            try:
                operation = await client.aio.operations.get(operation)
            except Exception as e:
                self.log.warning(f"Polling error (will retry): {e}")
            await emit_status(f"Generating video... ({elapsed}s elapsed)", False)

        if operation.error:
            error_msg = str(operation.error)
            self.log.error(f"Video generation failed: {error_msg}")
            await emit_status(f"Video generation failed: {error_msg}", True)
            return f"Video generation failed: {error_msg}"

        generated_video_files: List[Dict[str, Any]] = []
        generated_video_links: List[str] = []
        upload_failure_count = 0
        attachment_skipped_count = 0
        response = operation.response
        if not response or not response.generated_videos:
            return "Error: No videos were generated"

        for idx, gen_video in enumerate(response.generated_videos):
            video = gen_video.video
            if not video:
                self.log.warning(f"Video {idx}: no video object in response")
                continue

            self.log.debug(
                f"Video {idx}: uri={getattr(video, 'uri', None)}, "
                f"name={getattr(video, 'name', None)}, "
                f"has_bytes={bool(getattr(video, 'video_bytes', None))}"
            )

            video_bytes = None
            if getattr(video, "video_bytes", None):
                video_bytes = video.video_bytes

            # Download video bytes via SDK (sync version is more reliable)
            if not video_bytes:
                try:
                    await asyncio.to_thread(client.files.download, file=video)
                    video_bytes = getattr(video, "video_bytes", None)
                    self.log.debug(
                        f"Video {idx}: SDK download complete, "
                        f"has_bytes={bool(video_bytes)}"
                    )
                except Exception as dl_err:
                    self.log.warning(f"Video {idx} SDK download failed: {dl_err}")

            # Fallback: save to temp file via SDK
            if not video_bytes:
                tmp_path = None
                try:
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        suffix=".mp4", delete=False
                    ) as tmp:
                        tmp_path = tmp.name
                    await asyncio.to_thread(video.save, tmp_path)
                    async with aiofiles.open(tmp_path, "rb") as f:
                        video_bytes = await f.read()
                    self.log.debug(
                        f"Video {idx}: temp-file download complete, "
                        f"size={len(video_bytes)} bytes"
                    )
                except Exception as save_err:
                    self.log.warning(f"Video {idx} temp-file save failed: {save_err}")
                finally:
                    if tmp_path:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

            if not video_bytes:
                self.log.warning(f"Video {idx}: could not obtain video bytes")
                continue

            mime_type = getattr(video, "mime_type", "video/mp4") or "video/mp4"

            file_entry = None
            video_url = None
            attachment_attempted = False
            if __request__ and __user__:
                attachment_attempted = True
                file_entry, video_url = await self._upload_video_with_status(
                    video_bytes,
                    mime_type,
                    __request__,
                    __user__,
                    __event_emitter__,
                    __metadata__,
                )
            else:
                video_data_b64 = base64.b64encode(video_bytes).decode("utf-8")
                video_url = f"data:{mime_type};base64,{video_data_b64}"

            if file_entry:
                generated_video_files.append(file_entry)
                if video_url:
                    generated_video_links.append(
                        f"[\U0001f3ac Generated Video {idx + 1}]({video_url})"
                    )
                continue

            if attachment_attempted:
                upload_failure_count += 1
            else:
                attachment_skipped_count += 1

            if attachment_attempted and video_url and not video_url.startswith("data:"):
                generated_video_links.append(
                    f"[\U0001f3ac Generated Video {idx + 1}]({video_url})"
                )
            elif attachment_attempted:
                generated_video_links.append(
                    f"Generated video {idx + 1}, but it could not be attached to the chat."
                )
            else:
                generated_video_links.append(f"Generated video {idx + 1}.")

        await emit_status(f"Video generation complete ({elapsed}s)", True)

        # Without a chat message (API clients) a "files" event reaches nobody,
        # so the answer links the uploaded videos instead.
        files_emitted = False
        if self._is_chat_message(__metadata__):
            files_emitted = await self._emit_generated_video_files(
                generated_video_files, __event_emitter__
            )

        content_parts: List[str] = []
        if generated_video_files and files_emitted:
            video_count = len(generated_video_files)
            content_parts.append(
                "Generated video attached."
                if video_count == 1
                else f"Generated {video_count} videos attached."
            )
        else:
            content_parts.extend(generated_video_links)

        if upload_failure_count:
            content_parts.append("Some videos could not be attached directly.")

        if attachment_skipped_count:
            content_parts.append(
                "Video attachments were skipped because chat upload context was unavailable."
            )

        content = (
            "\n\n".join(part for part in content_parts if part)
            if content_parts
            else "[No video content generated]"
        )

        # A bare {"choices": [{"message": ...}]} dict loses the text when Open WebUI
        # requested a stream (the browser default), so match the request shape.
        return self._build_non_stream_result(
            content, None, body.get("model", model_id), bool(body.get("stream", False))
        )

    async def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        Retry a function with exponential backoff.

        Args:
            func: Async function to retry
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Result from the function

        Raises:
            The last exception encountered after all retries
        """
        max_retries = self.valves.RETRY_COUNT
        retry_count = 0
        last_exception = None

        while retry_count <= max_retries:
            try:
                return await func(*args, **kwargs)
            except ServerError as e:
                # These errors might be temporary, so retry
                retry_count += 1
                last_exception = e

                if retry_count <= max_retries:
                    # Calculate backoff time (exponential with jitter)
                    wait_time = min(2**retry_count + (0.1 * retry_count), 10)
                    self.log.warning(
                        f"Temporary error from Google API: {e}. Retrying in {wait_time:.1f}s ({retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception:
                # Don't retry other exceptions
                raise

        # If we get here, we've exhausted retries
        assert last_exception is not None
        raise last_exception

    async def pipe(
        self,
        body: Dict[str, Any],
        __metadata__: dict[str, Any],
        __event_emitter__: Optional[Callable],
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
    ) -> Union[
        str,
        Dict[str, Any],
        AsyncIterator[Union[str, Dict[str, Any]]],
        StreamingResponse,
    ]:
        """
        Main method for sending requests to the Google Gemini endpoint.

        Tools are declared from body["tools"] (the OpenAI tool specs Open WebUI
        builds for every tool type, or an API client's tools). When Gemini
        answers with function calls, the round ends and they are returned as
        OpenAI tool_calls: on the browser path Open WebUI runs the tools and
        calls the pipe again with the results, API clients run them themselves.

        Args:
            body: The request body containing messages and other parameters.
            __metadata__: Request metadata
            __event_emitter__: Event emitter for status updates
            __request__: FastAPI request object (for image upload)
            __user__: User information (for image upload)

        Returns:
            Response from Google Gemini API: a string, a dict, an iterator for
            streaming, or an SSE StreamingResponse (see _sse).
        """
        # Setup logging for this request
        request_id = id(body)
        self.log.debug(f"Processing request {request_id}")
        self.log.debug(f"User request body: {__user__}")
        # The requesting user for the forwarded user info headers. A local, not
        # an attribute: the Pipe instance is shared by concurrent requests.
        user: Optional[UserModel] = None
        if __user__ and self.valves.ENABLE_FORWARD_USER_INFO_HEADERS:
            user = await Users.get_user_by_id(__user__["id"])

        # Remember statuses that are still running, to close them if the request
        # is stopped or fails (see finally).
        running_statuses: Dict[str, Dict[str, Any]] = {}
        __event_emitter__ = self._track_statuses(__event_emitter__, running_statuses)
        cancelled = False

        try:
            # Parse and validate model ID
            model_id = body.get("model", "")
            try:
                model_id = self._prepare_model_id(model_id)
                self.log.debug(f"Using model: {model_id}")
            except ValueError as ve:
                return f"Model Error: {ve}"

            # Route Veo video generation models to dedicated handler
            if self._check_video_generation_support(model_id):
                self.log.debug(f"Routing to video generation for model: {model_id}")
                return await self._generate_video(
                    body,
                    model_id,
                    __event_emitter__,
                    __request__,
                    __user__,
                    __metadata__,
                    user=user,
                )

            # Check if this model supports image generation
            supports_image_generation = self._check_image_generation_support(model_id)

            # Get stream flag. stream_requested is what Open WebUI asked for and
            # decides the return shape; stream is whether we stream from Gemini.
            stream = body.get("stream", False)
            stream_requested = bool(stream)
            if not self.valves.STREAMING_ENABLED:
                if stream:
                    self.log.debug("Streaming disabled via GOOGLE_STREAMING_ENABLED")
                stream = False
            messages = body.get("messages", [])
            model_name = body.get("model", model_id)

            # The browser path (a chat message), where Open WebUI runs the tool
            # loop, and whether this request is a later round of a tool turn
            is_chat = self._is_chat_message(__metadata__)
            continuation = self._is_tool_continuation(messages)

            # For image generation models, gather ALL images from the last user turn
            if supports_image_generation:
                try:
                    (
                        contents,
                        system_instruction,
                    ) = await self._build_image_generation_contents(
                        messages, __event_emitter__, __metadata__, __user__
                    )
                    # For image generation, system_instruction is integrated into the prompt
                    # so it will be None here (this is expected and correct)
                    self.log.debug(
                        "Image generation mode: system instruction integrated into prompt"
                    )
                except ValueError as ve:
                    return f"Error: {ve}"
            else:
                # For non-image generation models, use the full conversation history
                # Prepare content and extract system message normally
                contents, system_instruction = self._prepare_content(messages, model_id)
                if not contents:
                    return "Error: No valid message content found"
                self.log.debug(
                    f"Text generation mode: system instruction separate (value: {system_instruction})"
                )

            # Configure generation parameters and safety settings
            self.log.debug(f"Supports image generation: {supports_image_generation}")
            function_declarations, name_map = self._build_function_declarations(
                body, __metadata__, supports_image_generation, model_id
            )
            generation_config = self._configure_generation(
                body,
                system_instruction,
                __metadata__,
                __user__,
                supports_image_generation,
                model_id,
                function_declarations=function_declarations,
                name_map=name_map,
            )
            # Tool mode: functions are actually declared in this request
            tool_mode = any(
                getattr(tool, "function_declarations", None)
                for tool in generation_config.tools or []
            )

            # Make the API call
            client = self._get_client(user)
            if stream:
                # For image generation models, disable streaming to avoid chunk size issues
                if supports_image_generation:
                    self.log.debug(
                        "Disabling streaming for image generation model to avoid chunk size issues"
                    )
                    stream = False
                else:

                    async def get_streaming_response():
                        return await client.aio.models.generate_content_stream(
                            model=model_id,
                            contents=contents,
                            config=generation_config,
                        )

                    self.log.debug(f"Request {request_id}: Streaming response")
                    # The request (with its retries) only starts when Open WebUI
                    # reads the stream. Hand the client over so it lives (and is
                    # closed) with the stream.
                    stream_response = self._handle_streaming_response(
                        get_streaming_response,
                        __event_emitter__,
                        __request__,
                        __user__,
                        client=client,
                        model=model_name,
                        __metadata__=__metadata__,
                        tool_mode=tool_mode,
                        name_map=name_map,
                        continuation=continuation,
                    )
                    if tool_mode and not is_chat:
                        # API clients in tool mode: own SSE framing (see _sse)
                        return self._sse(stream_response)
                    return stream_response

            # Non-streaming path (now also used for image generation)
            if not stream or supports_image_generation:
                try:

                    async def get_response():
                        return await client.aio.models.generate_content(
                            model=model_id,
                            contents=contents,
                            config=generation_config,
                        )

                    # Measure duration for non-streaming path (no status to avoid false indicators)
                    start_ts = time.time()

                    # Send processing status for image generation
                    if supports_image_generation:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "action": "image_processing",
                                    "description": "Processing image request...",
                                    "done": False,
                                },
                            },
                        )

                    response = await self._retry_with_backoff(get_response)
                    self.log.debug(f"Request {request_id}: Got non-streaming response")

                    # Clear processing status for image generation
                    if supports_image_generation:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "action": "image_processing",
                                    "description": "Processing complete",
                                    "done": True,
                                },
                            },
                        )

                    # Handle "Thinking" and produce final formatted content
                    candidate = response.candidates[0] if response.candidates else None
                    parts = (
                        getattr(getattr(candidate, "content", None), "parts", None)
                        or []
                    )
                    # Tool mode: the round's function calls (they win over a
                    # SAFETY or MAX_TOKENS finish)
                    function_call_parts = [
                        part
                        for part in parts
                        if tool_mode and getattr(part, "function_call", None)
                    ]

                    # Check for safety blocks first
                    if not function_call_parts:
                        safety_message = self._get_safety_block_message(response)
                        if safety_message:
                            return safety_message

                    # A failed tool call without any answer gets an error text
                    tool_call_error = self._tool_call_error_text(
                        getattr(candidate, "finish_reason", None)
                    )
                    if not parts and not tool_call_error:
                        return "[No content generated or unexpected response structure]"

                    answer_segments: list[str] = []
                    thought_segments: list[str] = []
                    generated_images: list[str] = []
                    generated_image_files: List[Dict[str, Any]] = []
                    seen_generated_image_hashes: set[str] = set()
                    last_thought_image: Any = None
                    server_side = False

                    for part in parts:
                        if tool_mode and getattr(part, "function_call", None):
                            continue  # collected above
                        if tool_mode and self._is_server_side_part(part):
                            server_side = True
                            continue
                        is_thought = bool(getattr(part, "thought", False))
                        if is_thought and getattr(part, "text", None):
                            thought_segments.append(part.text)
                        elif is_thought and getattr(part, "inline_data", None):
                            # Gemini 3 image models return up to two interim images
                            # from their thinking process as thought parts when
                            # thoughts are included. The final image follows as a
                            # regular part, so interim images are not uploaded
                            # (only kept as fallback if no final image arrives).
                            self.log.debug("Skipping interim thought image")
                            last_thought_image = part.inline_data
                        elif getattr(part, "text", None):
                            answer_segments.append(part.text)
                        elif getattr(part, "inline_data", None):
                            await self._collect_generated_image(
                                part.inline_data,
                                seen_generated_image_hashes,
                                generated_images,
                                generated_image_files,
                                __request__,
                                __user__,
                                __event_emitter__,
                            )

                    # seen_generated_image_hashes records every final (non-thought)
                    # image part. If the response had thought images only, attach
                    # the last one.
                    if (
                        last_thought_image is not None
                        and not seen_generated_image_hashes
                        and self._allows_thought_image_fallback(
                            getattr(candidate, "finish_reason", None)
                        )
                    ):
                        self.log.warning(
                            "Gemini returned thought images but no final image; "
                            "attaching the last thought image instead"
                        )
                        await self._collect_generated_image(
                            last_thought_image,
                            seen_generated_image_hashes,
                            generated_images,
                            generated_image_files,
                            __request__,
                            __user__,
                            __event_emitter__,
                        )

                    final_answer = "".join(answer_segments)
                    if tool_call_error and not function_call_parts and not final_answer:
                        final_answer = tool_call_error

                    # Apply grounding (if available) and send sources/status as needed
                    grounding_metadata_list = []
                    if getattr(candidate, "grounding_metadata", None):
                        grounding_metadata_list.append(candidate.grounding_metadata)
                    # Like the streaming path: without an emitter (background tasks)
                    # sources cannot be shown, so do not add citation markers either.
                    if grounding_metadata_list and __event_emitter__:
                        cited = await self._process_grounding_metadata(
                            grounding_metadata_list,
                            final_answer,
                            __event_emitter__,
                        )
                        final_answer = cited or final_answer

                    # If we have thoughts, wrap them using <details>. Background tasks
                    # (title, tags, follow-ups, ...) parse JSON out of the answer, so
                    # they get the answer only.
                    is_task = bool((__metadata__ or {}).get("task"))
                    details_block = ""
                    if thought_segments and not is_task:
                        details_block = self._thinking_details_block(
                            "".join(thought_segments),
                            int(max(0, time.time() - start_ts)),
                        )

                    # Build response with usage for middleware to extract and save to DB
                    usage = self._build_usage_dict(
                        getattr(response, "usage_metadata", None)
                    )

                    # Tool rounds (and API requests in tool mode) get their own
                    # shapes, see _build_tool_round_response
                    if (
                        function_call_parts
                        or (tool_mode and not is_chat and stream_requested)
                        or (
                            is_chat
                            and continuation
                            and stream_requested
                            and not supports_image_generation
                        )
                    ):
                        return await self._build_tool_round_response(
                            final_answer=final_answer,
                            details_block=details_block,
                            reasoning="".join(thought_segments).strip(),
                            function_call_parts=function_call_parts,
                            all_parts=list(parts),
                            server_side=server_side,
                            name_map=name_map,
                            usage=usage,
                            model=model_name,
                            is_chat=is_chat,
                            continuation=continuation,
                            stream_requested=stream_requested,
                            generated_images=generated_images,
                            generated_image_files=generated_image_files,
                            __event_emitter__=__event_emitter__,
                            __metadata__=__metadata__,
                        )

                    # Combine all content
                    full_response = details_block + final_answer

                    full_response = await self._append_generated_images(
                        full_response,
                        final_answer,
                        generated_images,
                        generated_image_files,
                        __event_emitter__,
                        __metadata__,
                    )

                    content = (
                        full_response if full_response else "[No content generated]"
                    )

                    # Return content and usage in the shape that matches the request
                    # (dict for stream=false, content + usage chunks for stream=true)
                    # so Open WebUI saves both and passes the content to outlet filters.
                    return self._build_non_stream_result(
                        content, usage, model_name, stream_requested
                    )

                except Exception as e:
                    self.log.exception(
                        f"Error in non-streaming request {request_id}: {e}"
                    )
                    if supports_image_generation:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "action": "image_processing",
                                    "description": "Image request failed",
                                    "done": True,
                                },
                            },
                        )
                    return f"Error generating content: {e}"
                finally:
                    # The client is per request; nothing uses it after this point.
                    await self._close_client(client)

        except asyncio.CancelledError:
            # Stopped by the user
            cancelled = True
            raise

        except (ClientError, ServerError, APIError) as api_error:
            error_type = type(api_error).__name__
            error_msg = f"{error_type}: {api_error}"
            self.log.error(error_msg)
            return error_msg

        except ValueError as ve:
            error_msg = f"Configuration error: {ve}"
            self.log.error(error_msg)
            return error_msg

        except Exception as e:
            # Log the full error with traceback
            import traceback

            error_trace = traceback.format_exc()
            self.log.exception(f"Unexpected error: {e}\n{error_trace}")

            # Return a user-friendly error message
            return f"An error occurred while processing your request: {e}"

        finally:
            # A started action (image_processing, video_generation, ...) gets a
            # final status even if the request was stopped or failed. A returned
            # stream closes its own statuses.
            await self._finish_running_statuses(
                __event_emitter__, running_statuses, cancelled
            )
