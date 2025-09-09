"""
title: Google Gemini Pipeline
author: owndev, olivier-lacroix
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 1.5.2
license: Apache License 2.0
description: Highly optimized Google Gemini pipeline with advanced image generation capabilities, intelligent compression, and streamlined processing workflows.
features:
  - Optimized asynchronous API calls for maximum performance
  - Intelligent model caching with configurable TTL
  - Streamlined dynamic model specification with automatic prefix handling
  - Smart streaming response handling with safety checks
  - Advanced multimodal input support (text and images)
  - Unified image generation and editing with Gemini 2.5 Flash Image Preview
  - Intelligent image optimization with size-aware compression algorithms
  - Automated image upload to Open WebUI with robust fallback support
  - Optimized text-to-image and image-to-image workflows
  - Non-streaming mode for image generation to prevent chunk overflow
  - Progressive status updates for optimal user experience
  - Consolidated error handling and comprehensive logging
  - Seamless Google Generative AI and Vertex AI integration
  - Advanced generation parameters (temperature, max tokens, etc.)
  - Configurable safety settings with environment variable support
  - Military-grade encrypted storage of sensitive API keys
  - Intelligent grounding with Google search integration
  - Native tool calling support with automatic signature management
  - Unified image processing with consolidated helper methods
  - Optimized payload creation for image generation models
  - Configurable image processing parameters (size, quality, compression)
  - Flexible upload fallback options and optimization controls
"""

import os
import inspect
from functools import update_wrapper
import re
import time
import asyncio
import base64
import hashlib
import logging
import io
import uuid
import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError, APIError
from typing import List, Union, Optional, Dict, Any, Tuple, AsyncIterator, Callable
from pydantic_core import core_schema
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
from open_webui.env import SRC_LOG_LEVELS
from fastapi import Request, UploadFile, BackgroundTasks
from open_webui.routers.files import upload_file
from open_webui.models.users import UserModel, Users
from starlette.datastructures import Headers


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

    def get_decrypted(self) -> str:
        """Get the decrypted value"""
        return self.decrypt(self)


class Pipe:
    """
    Pipeline for interacting with Google Gemini models.
    """

    # Configuration valves for the pipeline
    class Valves(BaseModel):
        GOOGLE_API_KEY: EncryptedStr = Field(
            default=os.getenv("GOOGLE_API_KEY", ""),
            description="API key for Google Generative AI (used if USE_VERTEX_AI is false).",
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
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=os.getenv("USE_PERMISSIVE_SAFETY", "false").lower() == "true",
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

        # Image Processing Configuration
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

    def _get_client(self) -> genai.Client:
        """
        Validates API credentials and returns a genai.Client instance.
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
            return genai.Client(api_key=self.valves.GOOGLE_API_KEY.get_decrypted())

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

    def get_google_models(self, force_refresh: bool = False) -> List[Dict[str, str]]:
        """
        Retrieve available Google models suitable for content generation.
        Uses caching to reduce API calls.

        Args:
            force_refresh: Whether to force refreshing the model cache

        Returns:
            List of dictionaries containing model id and name.
        """
        # Check cache first
        current_time = time.time()
        if (
            not force_refresh
            and self._model_cache is not None
            and (current_time - self._model_cache_time) < self.valves.MODEL_CACHE_TTL
        ):
            self.log.debug("Using cached model list")
            return self._model_cache

        try:
            client = self._get_client()
            self.log.debug("Fetching models from Google API")
            models = client.models.list()
            available_models = []
            for model in models:
                actions = model.supported_actions
                if actions is None or "generateContent" in actions:
                    model_id = self.strip_prefix(model.name)
                    model_name = model.display_name or model_id

                    # Check if model supports image generation
                    supports_image_generation = self._check_image_generation_support(
                        model_id
                    )
                    if supports_image_generation:
                        model_name += " 🎨"  # Add image generation indicator

                    available_models.append(
                        {
                            "id": model_id,
                            "name": model_name,
                            "image_generation": supports_image_generation,
                        }
                    )

            model_map = {model["id"]: model for model in available_models}

            # Filter map to only include models starting with 'gemini-'
            filtered_models = {
                k: v for k, v in model_map.items() if k.startswith("gemini-")
            }

            # Update cache
            self._model_cache = list(filtered_models.values())
            self._model_cache_time = current_time
            self.log.debug(f"Found {len(self._model_cache)} Gemini models")
            return self._model_cache

        except Exception as e:
            self.log.exception(f"Could not fetch models from Google: {str(e)}")
            # Return a specific error entry for the UI
            return [{"id": "error", "name": f"Could not fetch models: {str(e)}"}]

    def _check_image_generation_support(self, model_id: str) -> bool:
        """
        Check if a model supports image generation.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model supports image generation, False otherwise
        """
        # Known image generation models
        image_generation_models = [
            "gemini-2.5-flash-image-preview",
        ]

        # Check for exact matches or pattern matches
        for pattern in image_generation_models:
            if model_id == pattern or pattern in model_id:
                return True

        # Additional pattern checking for future models
        if "image" in model_id.lower() and (
            "generation" in model_id.lower() or "preview" in model_id.lower()
        ):
            return True

        return False

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
        ]

        # Check for exact matches
        for pattern in non_thinking_models:
            if model_id == pattern or pattern in model_id:
                return False

        # Additional pattern checking - image generation models typically don't support thinking
        if "image" in model_id.lower() and (
            "generation" in model_id.lower() or "preview" in model_id.lower()
        ):
            return False

        # By default, assume models support thinking
        return True

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

        # If the model ID doesn't look like a Gemini model, try to find it by name
        if not model_id.startswith("gemini-"):
            models_list = self.get_google_models()
            found_model = next(
                (m["id"] for m in models_list if m["name"] == original_model_id), None
            )
            if found_model and found_model.startswith("gemini-"):
                model_id = found_model
                self.log.debug(
                    f"Mapped model name '{original_model_id}' to model ID '{model_id}'"
                )
            else:
                # If we still don't have a valid ID, raise an error
                if not model_id.startswith("gemini-"):
                    self.log.error(
                        f"Invalid or unsupported model ID: '{original_model_id}'"
                    )
                    raise ValueError(
                        f"Invalid or unsupported Google model ID or name: '{original_model_id}'"
                    )

        return model_id

    def _prepare_content(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Prepare messages content for the API and extract system message if present.

        Args:
            messages: List of message objects from the request

        Returns:
            Tuple of (prepared content list, system message string or None)
        """
        # Extract system message
        system_message = next(
            (msg["content"] for msg in messages if msg.get("role") == "system"),
            None,
        )

        # Prepare contents for the API
        contents = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue  # Skip system messages, handled separately

            content = message.get("content", "")
            parts = []

            # Handle different content types
            if isinstance(content, list):  # Multimodal content
                parts.extend(self._process_multimodal_content(content))
            elif isinstance(content, str):  # Plain text content
                parts.append({"text": content})
            else:
                self.log.warning(f"Unsupported message content type: {type(content)}")
                continue  # Skip unsupported content

            # Map roles: 'assistant' -> 'model', 'user' -> 'user'
            api_role = "model" if role == "assistant" else "user"
            if parts:  # Only add if there are parts
                contents.append({"role": api_role, "parts": parts})

        return contents, system_message

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

    async def _find_image(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Find an image in the message history.
        Searches through the last few messages for uploaded images.

        Args:
            messages: List of message objects from the request

        Returns:
            Base64 encoded image data or None if no image found
        """
        max_search = 5
        search_count = 0

        for msg in reversed(messages):
            if search_count >= max_search:
                break
            search_count += 1

            content = msg.get("content", [])
            if isinstance(content, str):
                # Search for image markdown syntax
                match = re.search(
                    r"!\[([^\]]*)\]\((data:image[^;]+;base64,[^)]+|/files/[^)]+|/api/v1/files/[^)]+)\)",
                    content,
                )
                if match:
                    url = match.group(2)
                    if url.startswith("data:"):
                        return self._optimize_image_for_api(url)
                    elif "/files/" in url or "/api/v1/files/" in url:
                        b64 = await self._fetch_file_as_base64(url)
                        if b64:
                            return self._optimize_image_for_api(b64)

            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            return self._optimize_image_for_api(url)
                        elif "/files/" in url or "/api/v1/files/" in url:
                            b64 = await self._fetch_file_as_base64(url)
                            if b64:
                                return self._optimize_image_for_api(b64)
                    elif item.get("type") == "text":
                        text = item.get("text", "")
                        match = re.search(
                            r"!\[([^\]]*)\]\((data:image[^;]+;base64,[^)]+|/files/[^)]+|/api/v1/files/[^)]+)\)",
                            text,
                        )
                        if match:
                            url = match.group(2)
                            if url.startswith("data:"):
                                return self._optimize_image_for_api(url)
                            elif "/files/" in url or "/api/v1/files/" in url:
                                b64 = await self._fetch_file_as_base64(url)
                                if b64:
                                    return self._optimize_image_for_api(b64)
        return None

    def _optimize_image_for_api(self, image_data: str) -> str:
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
            reasons = []
            if original_size_mb > max_size_mb:
                reasons.append(f"size > {max_size_mb} MB")
            if base64_size_mb > max_size_mb * 1.4:
                reasons.append("base64 overhead")
            if mime_type == "image/png" and original_size_mb > png_threshold:
                reasons.append(f"PNG > {png_threshold}MB")

            # Always check dimensions
            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
                if width > max_dimension or height > max_dimension:
                    reasons.append(f"dimensions > {max_dimension}px")

                # Apply optimization logic
                if not reasons:
                    reasons.append("ensuring API compatibility")

                self.log.debug(f"Optimization needed: {', '.join(reasons)}")

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
                return f"data:image/jpeg;base64,{optimized_b64}"

        except Exception as e:
            self.log.error(f"Image optimization failed: {e}")
            # Return original or safe fallback
            if image_data.startswith("data:"):
                return image_data
            return f"data:image/jpeg;base64,{encoded if 'encoded' in locals() else image_data}"

    async def _fetch_file_as_base64(self, file_url: str) -> Optional[str]:
        """
        Fetch a file from Open WebUI's file system and convert to base64.

        Args:
            file_url: File URL from Open WebUI

        Returns:
            Base64 encoded file data or None if file not found
        """
        try:
            if "/api/v1/files/" in file_url:
                fid = file_url.split("/api/v1/files/")[-1].split("/")[0].split("?")[0]
            else:
                fid = file_url.split("/files/")[-1].split("/")[0].split("?")[0]

            from open_webui.models.files import Files

            file_obj = Files.get_file_by_id(fid)
            if file_obj and file_obj.path:
                async with aiofiles.open(file_obj.path, "rb") as fp:
                    raw = await fp.read()
                enc = base64.b64encode(raw).decode()
                mime = file_obj.meta.get("content_type", "image/png")
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
        __event_emitter__: Callable,
    ) -> str:
        """
        Unified image upload method with status updates and fallback handling.

        Returns:
            URL to uploaded image or data URL fallback
        """
        try:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_upload",
                        "description": "Uploading generated image to your library...",
                        "done": False,
                    },
                }
            )

            user = Users.get_user_by_id(__user__["id"])

            # Convert image data to base64 string if needed
            if isinstance(image_data, bytes):
                image_data_b64 = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data_b64 = str(image_data)

            image_url = self._upload_image(
                __request__=__request__,
                user=user,
                image_data=image_data_b64,
                mime_type=mime_type,
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_upload",
                        "description": "Image uploaded successfully!",
                        "done": True,
                    },
                }
            )

            return image_url

        except Exception as e:
            self.log.warning(f"File upload failed, falling back to data URL: {e}")

            if isinstance(image_data, bytes):
                image_data_b64 = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data_b64 = str(image_data)

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_upload",
                        "description": "Using inline image (upload failed)",
                        "done": True,
                    },
                }
            )

            return f"data:{mime_type};base64,{image_data_b64}"

    def _upload_image(
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
            up_obj = upload_file(
                request=__request__,
                background_tasks=BackgroundTasks(),
                file=UploadFile(
                    file=bio,
                    filename=filename,
                    headers=Headers({"content-type": mime_type}),
                ),
                process=False,  # Matching reference - no heavy processing
                user=user,
                metadata={"mime_type": mime_type, "source": "gemini_image_generation"},
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

    @staticmethod
    def _create_tool(tool_def):
        """OpenwebUI tool is a functools.partial coroutine, which genai does not support directly.
        See https://github.com/googleapis/python-genai/issues/907

        This function wraps the tool into a callable that can be used with genai.
        In particular, it sets the signature of the function properly,
        removing any frozen keyword arguments (extra_params).
        """
        bound_callable = tool_def["callable"]

        # Create a wrapper for bound_callable, which is always async
        async def wrapper(*args, **kwargs):
            return await bound_callable(*args, **kwargs)

        # Remove 'frozen' keyword arguments (extra_params) from the signature
        original_sig = inspect.signature(bound_callable)
        frozen_kwargs = {
            "__event_emitter__",
            "__event_call__",
            "__user__",
            "__metadata__",
            "__request__",
            "__model__",
        }
        new_parameters = []

        for name, parameter in original_sig.parameters.items():
            # Exclude keyword arguments that are frozen
            if name in frozen_kwargs and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            # Keep remaining parameters
            new_parameters.append(parameter)

        new_sig = inspect.Signature(
            parameters=new_parameters, return_annotation=original_sig.return_annotation
        )

        # Ensure name, docstring and signature are properly set
        update_wrapper(wrapper, bound_callable)
        wrapper.__signature__ = new_sig

        return wrapper

    def _configure_generation(
        self,
        body: Dict[str, Any],
        system_instruction: Optional[str],
        __metadata__: Dict[str, Any],
        __tools__: dict[str, Any] | None = None,
        enable_image_generation: bool = False,
        model_id: str = "",
    ) -> types.GenerateContentConfig:
        """
        Configure generation parameters and safety settings.

        Args:
            body: The request body containing generation parameters
            system_instruction: Optional system instruction string
            enable_image_generation: Whether to enable image generation
            model_id: The model ID being used (for feature support checks)

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

        # Enable Gemini "Thinking" when requested (default: on) and supported by the model
        include_thoughts = body.get("include_thoughts", True)
        if include_thoughts and self._check_thinking_support(model_id):
            try:
                gen_config_params["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=True
                )
            except Exception:
                # Fall back silently if SDK/model does not support ThinkingConfig
                pass

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

        features = __metadata__.get("features", {})
        if features.get("google_search_tool", False):
            self.log.debug("Enabling Google search grounding")
            gen_config_params.setdefault("tools", []).append(
                types.Tool(google_search=types.GoogleSearch())
            )

        if __tools__ is not None and __metadata__.get("function_calling") == "native":
            for name, tool_def in __tools__.items():
                tool = self._create_tool(tool_def)
                self.log.debug(
                    f"Adding tool '{name}' with signature {tool.__signature__}"
                )

                gen_config_params.setdefault("tools", []).append(tool)

        # Filter out None values for generation config
        filtered_params = {k: v for k, v in gen_config_params.items() if v is not None}
        return types.GenerateContentConfig(**filtered_params)

    @staticmethod
    def _format_grounding_chunks_as_sources(
        grounding_chunks: list[types.GroundingChunk],
    ):
        formatted_sources = []
        for chunk in grounding_chunks:
            context = chunk.web or chunk.retrieved_context
            if not context:
                continue

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
        __event_emitter__: Callable,
        *,
        emit_replace: bool = True,
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

        # Add sources to the response
        if grounding_chunks:
            sources = self._format_grounding_chunks_as_sources(grounding_chunks)
            await __event_emitter__(
                {"type": "chat:completion", "data": {"sources": sources}}
            )

        # Add status specifying google queries used for grounding
        if web_search_queries:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "web_search",
                        "description": "This response was grounded with Google Search",
                        "urls": [
                            f"https://www.google.com/search?q={query}"
                            for query in web_search_queries
                        ],
                    },
                }
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
            if emit_replace:
                await __event_emitter__(
                    {
                        "type": "replace",
                        "data": {"content": replaced_text},
                    }
                )

        # Return the transformed text when requested by caller
        if not emit_replace:
            return replaced_text if replaced_text is not None else text

    async def _handle_streaming_response(
        self,
        response_iterator: Any,
        __event_emitter__: Callable,
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        """
        Handle streaming response from Gemini API.

        Args:
            response_iterator: Iterator from generate_content
            __event_emitter__: Event emitter for status updates

        Returns:
            Generator yielding text chunks
        """
        grounding_metadata_list = []
        # Accumulate content separately for answer and thoughts
        answer_chunks: list[str] = []
        thought_chunks: list[str] = []
        thinking_started_at: Optional[float] = None

        try:
            async for chunk in response_iterator:
                # Check for safety feedback or empty chunks
                if not chunk.candidates:
                    # Check prompt feedback
                    if (
                        response_iterator.prompt_feedback
                        and response_iterator.prompt_feedback.block_reason
                    ):
                        yield f"[Blocked due to Prompt Safety: {response_iterator.prompt_feedback.block_reason.name}]"
                    else:
                        yield "[Blocked by safety settings]"
                    return  # Stop generation

                if chunk.candidates[0].grounding_metadata:
                    grounding_metadata_list.append(
                        chunk.candidates[0].grounding_metadata
                    )
                # Prefer fine-grained parts to split thoughts vs. normal text
                parts = []
                try:
                    parts = chunk.candidates[0].content.parts or []
                except Exception as parts_error:
                    # Fallback: use aggregated text if parts aren't accessible
                    self.log.warning(f"Failed to access content parts: {parts_error}")
                    if hasattr(chunk, "text") and chunk.text:
                        answer_chunks.append(chunk.text)
                        await __event_emitter__(
                            {
                                "type": "chat:message:delta",
                                "data": {"content": chunk.text},
                            }
                        )
                    continue

                for part in parts:
                    try:
                        # Thought parts (internal reasoning)
                        if getattr(part, "thought", False) and getattr(
                            part, "text", None
                        ):
                            if thinking_started_at is None:
                                thinking_started_at = time.time()
                            thought_chunks.append(part.text)
                            # Emit a live preview of what is currently being thought
                            preview = part.text.replace("\n", " ").strip()
                            MAX_PREVIEW = 120
                            if len(preview) > MAX_PREVIEW:
                                preview = preview[:MAX_PREVIEW].rstrip() + "…"
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "action": "thinking",
                                        "description": f"Thinking… {preview}",
                                        "done": False,
                                        "hidden": False,
                                    },
                                }
                            )

                        # Regular answer text
                        elif getattr(part, "text", None):
                            answer_chunks.append(part.text)
                            await __event_emitter__(
                                {
                                    "type": "chat:message:delta",
                                    "data": {"content": part.text},
                                }
                            )
                    except Exception as part_error:
                        # Log part processing errors but continue with the stream
                        self.log.warning(f"Error processing content part: {part_error}")
                        continue

            # After processing all chunks, handle grounding data
            final_answer_text = "".join(answer_chunks)
            if grounding_metadata_list and __event_emitter__:
                # Don't emit replace here; we'll compose final content below
                cited = await self._process_grounding_metadata(
                    grounding_metadata_list,
                    final_answer_text,
                    __event_emitter__,
                    emit_replace=False,
                )
                final_answer_text = cited or final_answer_text

            # If we captured thoughts, wrap them in a collapsible <details> section
            if thought_chunks:
                duration_s = int(
                    max(0, time.time() - (thinking_started_at or time.time()))
                )
                # Format each line with > for blockquote while preserving formatting
                thought_content = ''.join(thought_chunks).strip()
                quoted_lines = []
                for line in thought_content.split('\n'):
                    quoted_lines.append(f"> {line}")
                quoted_content = '\n'.join(quoted_lines)
                
                details_block = f"""<details>
<summary>Thought ({duration_s}s)</summary>

{quoted_content}

</details>""".strip()

                # Combine thoughts and answer (images not processed in streaming mode)
                full_content = details_block + final_answer_text

                await __event_emitter__(
                    {
                        "type": "replace",
                        "data": {"content": full_content},
                    }
                )
                # Clear the thinking status without a summary in the status emitter
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"action": "thinking", "done": True, "hidden": True},
                    }
                )
            elif grounding_metadata_list:
                # If no thoughts but we have grounding, ensure final answer (with citations) is reflected
                await __event_emitter__(
                    {"type": "replace", "data": {"content": final_answer_text}}
                )

        except Exception as e:
            self.log.exception(f"Error during streaming: {e}")
            # Check if it's a chunk size error and provide specific guidance
            error_msg = str(e).lower()
            if "chunk too big" in error_msg or "chunk size" in error_msg:
                yield f"Error: Image too large for processing. Please try with a smaller image (max 15 MB recommended) or reduce image quality."
            elif "quota" in error_msg or "rate limit" in error_msg:
                yield f"Error: API quota exceeded. Please try again later."
            else:
                yield f"Error during streaming: {e}"

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
        __event_emitter__: Callable,
        __tools__: dict[str, Any] | None,
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Main method for sending requests to the Google Gemini endpoint.

        Args:
            body: The request body containing messages and other parameters.
            __metadata__: Request metadata
            __event_emitter__: Event emitter for status updates
            __tools__: Available tools
            __request__: FastAPI request object (for image upload)
            __user__: User information (for image upload)

        Returns:
            Response from Google Gemini API, which could be a string or an iterator for streaming.
        """
        # Setup logging for this request
        request_id = id(body)
        self.log.debug(f"Processing request {request_id}")

        try:
            # Parse and validate model ID
            model_id = body.get("model", "")
            try:
                model_id = self._prepare_model_id(model_id)
                self.log.debug(f"Using model: {model_id}")
            except ValueError as ve:
                return f"Model Error: {ve}"

            # Check if this model supports image generation
            supports_image_generation = self._check_image_generation_support(model_id)

            # Get stream flag
            stream = body.get("stream", False)
            messages = body.get("messages", [])

            # For image generation models, check if we need to include an existing image
            image_base64 = None
            if supports_image_generation:
                # Get the last user message to check if it's a generation request
                last_user_msg = next(
                    (msg for msg in reversed(messages) if msg.get("role") == "user"),
                    None,
                )

                if not last_user_msg:
                    return "Error: No user message found"

                # Extract prompt from the last user message
                content = last_user_msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        item.get("text", "")
                        for item in content
                        if item.get("type") == "text"
                    ]
                    prompt = " ".join(text_parts)
                else:
                    prompt = content

                if not prompt.strip():
                    return "Error: No prompt provided"

                # Look for images in the recent message history
                image_base64 = await self._find_image(messages)

                # Create optimized content structure for image generation models
                parts = [{"text": prompt}]

                if image_base64:
                    # This is an image editing request
                    self.log.debug("Image editing request detected")
                    mime_type = re.search(r"data:([^;]+);base64,", image_base64).group(
                        1
                    )
                    raw_base64 = image_base64.split(";base64,")[-1]

                    # Add image to parts for editing workflow
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": raw_base64,
                            }
                        }
                    )
                else:
                    # This is a pure text-to-image generation request
                    self.log.debug("Text-to-image generation request detected")

                # Single optimized content structure for both text-to-image and image editing
                contents = [{"role": "user", "parts": parts}]

                # Override system instruction for image generation models
                system_instruction = None
            else:
                # For non-image generation models, use the full conversation history
                # Prepare content and extract system message normally
                contents, system_instruction = self._prepare_content(messages)
                if not contents:
                    return "Error: No valid message content found"

            # Configure generation parameters and safety settings
            generation_config = self._configure_generation(
                body,
                system_instruction,
                __metadata__,
                __tools__,
                supports_image_generation,
                model_id,
            )

            # Make the API call
            client = self._get_client()
            if stream:
                # For image generation models, disable streaming to avoid chunk size issues
                if supports_image_generation:
                    self.log.debug(
                        "Disabling streaming for image generation model to avoid chunk size issues"
                    )
                    stream = False
                else:
                    try:

                        async def get_streaming_response():
                            return await client.aio.models.generate_content_stream(
                                model=model_id,
                                contents=contents,
                                config=generation_config,
                            )

                        response_iterator = await self._retry_with_backoff(
                            get_streaming_response
                        )
                        self.log.debug(f"Request {request_id}: Got streaming response")
                        return self._handle_streaming_response(
                            response_iterator, __event_emitter__, __request__, __user__
                        )

                    except Exception as e:
                        self.log.exception(
                            f"Error in streaming request {request_id}: {e}"
                        )
                        return f"Error during streaming: {e}"

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
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "action": "image_processing",
                                    "description": "Processing image request...",
                                    "done": False,
                                },
                            }
                        )

                    response = await self._retry_with_backoff(get_response)
                    self.log.debug(f"Request {request_id}: Got non-streaming response")

                    # Clear processing status for image generation
                    if supports_image_generation:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "action": "image_processing",
                                    "description": "Processing complete",
                                    "done": True,
                                },
                            }
                        )

                    # Handle "Thinking" and produce final formatted content
                    # Check for safety blocks first
                    safety_message = self._get_safety_block_message(response)
                    if safety_message:
                        return safety_message

                    # Get the first candidate (safety checks passed)
                    candidate = response.candidates[0]

                    # Process content parts - use new streamlined approach
                    parts = getattr(getattr(candidate, "content", None), "parts", [])
                    if not parts:
                        return "[No content generated or unexpected response structure]"

                    answer_segments: list[str] = []
                    thought_segments: list[str] = []
                    generated_images: list[str] = []

                    for part in parts:
                        if getattr(part, "thought", False) and getattr(
                            part, "text", None
                        ):
                            thought_segments.append(part.text)
                        elif getattr(part, "text", None):
                            answer_segments.append(part.text)
                        elif (
                            getattr(part, "inline_data", None)
                            and __request__
                            and __user__
                        ):
                            # Handle generated images with unified upload method
                            mime_type = part.inline_data.mime_type
                            image_data = part.inline_data.data

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
                            generated_images.append(f"![Generated Image]({image_url})")

                        elif getattr(part, "inline_data", None):
                            # Fallback: return as base64 data URL if no request/user context
                            mime_type = part.inline_data.mime_type
                            image_data = part.inline_data.data

                            if isinstance(image_data, bytes):
                                image_data_b64 = base64.b64encode(image_data).decode(
                                    "utf-8"
                                )
                            else:
                                image_data_b64 = str(image_data)

                            data_url = f"data:{mime_type};base64,{image_data_b64}"
                            generated_images.append(f"![Generated Image]({data_url})")

                    final_answer = "".join(answer_segments)

                    # Apply grounding (if available) and send sources/status as needed
                    grounding_metadata_list = []
                    if getattr(candidate, "grounding_metadata", None):
                        grounding_metadata_list.append(candidate.grounding_metadata)
                    if grounding_metadata_list:
                        cited = await self._process_grounding_metadata(
                            grounding_metadata_list,
                            final_answer,
                            __event_emitter__,
                            emit_replace=False,
                        )
                        final_answer = cited or final_answer

                    # Combine all content
                    full_response = ""

                    # If we have thoughts, wrap them using <details>
                    if thought_segments:
                        duration_s = int(max(0, time.time() - start_ts))
                        # Format each line with > for blockquote while preserving formatting
                        thought_content = ''.join(thought_segments).strip()
                        quoted_lines = []
                        for line in thought_content.split('\n'):
                            quoted_lines.append(f"> {line}")
                        quoted_content = '\n'.join(quoted_lines)
                        
                        details_block = f"""<details>
<summary>Thought ({duration_s}s)</summary>

{quoted_content}

</details>""".strip()
                        full_response += details_block

                    # Add the main answer
                    full_response += final_answer

                    # Add generated images
                    if generated_images:
                        if full_response:
                            full_response += "\n\n"
                        full_response += "\n\n".join(generated_images)

                    return full_response if full_response else "[No content generated]"

                except Exception as e:
                    self.log.exception(
                        f"Error in non-streaming request {request_id}: {e}"
                    )
                    return f"Error generating content: {e}"

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