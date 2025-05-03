"""
title: Google Gemini Pipeline
author: owndev
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 1.0.0
license: Apache License 2.0
description: A manifold pipeline for interacting with Google Gemini models, including dynamic model specification, streaming responses, and flexible error handling.
features:
  - Asynchronous API calls for better performance
  - Model caching to reduce API calls
  - Dynamic model specification with automatic prefix stripping
  - Streaming response handling with safety checks
  - Support for multimodal input (text and images)
  - Flexible error handling and logging
  - Integration with Google Generative AI API for content generation
  - Support for various generation parameters (temperature, max tokens, etc.)
  - Customizable safety settings based on environment variables
  - Encrypted storage of sensitive API keys
"""

import os
import re
import time
import asyncio
import base64
import hashlib
import logging
import google.generativeai as genai
from typing import List, Union, Iterator, Optional, Dict, Any, Tuple, cast
from pydantic_core import core_schema
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
from open_webui.env import SRC_LOG_LEVELS
from google.generativeai.types import GenerationConfig
from google.api_core.exceptions import InvalidArgument, PermissionDenied, ResourceExhausted, ServiceUnavailable


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
            description="API key for Google Generative AI",
        )

        USE_PERMISSIVE_SAFETY: bool = Field(
            default=os.getenv("USE_PERMISSIVE_SAFETY", False),
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

    def __init__(self):
        """Initializes the Pipe instance and configures the genai library if the API key is available."""
        self.valves = self.Valves()
        self.name: str = "Google Gemini: "
        
        # Setup logging
        self.log = logging.getLogger("google_ai.pipe")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        
        # Model cache
        self._model_cache: Optional[List[Dict[str, str]]] = None
        self._model_cache_time: float = 0
        
        # Configure genai upon initialization if API key is present
        if self.valves.GOOGLE_API_KEY:
            try:
                genai.configure(api_key=self.valves.GOOGLE_API_KEY.get_decrypted())
                self.log.debug("Google Generative AI configured successfully")
            except Exception as e:
                self.log.warning(f"Warning: Error configuring Google Generative AI during init: {e}")
                # Allow initialization to continue, pipe method will re-attempt or handle the error

    def validate_api_key(self) -> None:
        """
        Validates that the Google API key is set.

        Raises:
            ValueError: If the API key is not set.
        """
        if not self.valves.GOOGLE_API_KEY:
            self.log.error("GOOGLE_API_KEY is not set")
            raise ValueError(
                "GOOGLE_API_KEY is not set. Please provide the API key in the environment variables or valves."
            )

    def strip_prefix(self, model_name: str) -> str:
        """
        Strip any prefix from the model name up to and including the first '.' or '/'.
        """
        # Use non-greedy regex to remove everything up to and including the first '.' or '/'
        stripped = re.sub(r"^.*?[./]", "", model_name)
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
        if (not force_refresh and 
            self._model_cache is not None and 
            (current_time - self._model_cache_time) < self.valves.MODEL_CACHE_TTL):
            self.log.debug("Using cached model list")
            return self._model_cache
            
        self.validate_api_key()  # Ensure API key is validated before proceeding
        try:
            # Ensure genai is configured before listing models
            genai.configure(api_key=self.valves.GOOGLE_API_KEY.get_decrypted())
            self.log.debug("Fetching models from Google API")

            models = genai.list_models()
            available_models = [
                {
                    "id": self.strip_prefix(model.name),
                    "name": model.display_name,
                }
                for model in models
                if "generateContent" in model.supported_generation_methods
                and model.name.startswith("models/") # Ensure we only get standard models
            ]

            model_map = {model['id']: model for model in available_models}

            # Filter map to only include models starting with 'gemini-'
            filtered_models = {k: v for k, v in model_map.items() if k.startswith("gemini-")}
            
            # Update cache
            self._model_cache = list(filtered_models.values())
            self._model_cache_time = current_time
            
            self.log.debug(f"Found {len(self._model_cache)} Gemini models")
            return self._model_cache

        except Exception as e:
            self.log.exception(f"Could not fetch models from Google: {str(e)}")
            # Return a specific error entry for the UI
            return [
                {"id": "error", "name": f"Could not fetch models: {str(e)}"}
            ]

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
            self.log.exception(f"An unexpected error occurred during pipes listing: {str(e)}")
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
            found_model = next((m['id'] for m in models_list if m['name'] == original_model_id), None)
            if found_model and found_model.startswith("gemini-"):
                model_id = found_model
                self.log.debug(f"Mapped model name '{original_model_id}' to model ID '{model_id}'")
            else:
                # If we still don't have a valid ID, raise an error
                if not model_id.startswith("gemini-"):
                    self.log.error(f"Invalid or unsupported model ID: '{original_model_id}'")
                    raise ValueError(f"Invalid or unsupported Google model ID or name: '{original_model_id}'")
        
        return model_id

    def _prepare_content(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
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

    def _process_multimodal_content(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                    # Handle base64 encoded image data
                    try:
                        header, encoded = image_url.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                        
                        # Basic validation for image types
                        if mime_type not in ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"]:
                            self.log.warning(f"Unsupported image mime type: {mime_type}")
                            parts.append({"text": f"[Image type {mime_type} not supported]"})
                            continue
                            
                        parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": encoded,
                            }
                        })
                    except Exception as img_ex:
                        self.log.exception(f"Could not parse image data URL: {img_ex}")
                        parts.append({"text": "[Image data could not be processed]"})
                else:
                    # Gemini API doesn't directly support image URLs
                    self.log.warning(f"Direct image URLs not supported: {image_url}")
                    parts.append({"text": f"[Image URL not processed: {image_url}]"})
        
        return parts

    def _configure_generation(self, body: Dict[str, Any]) -> Tuple[GenerationConfig, Optional[Dict[str, Any]]]:
        """
        Configure generation parameters and safety settings.
        
        Args:
            body: The request body containing generation parameters
            
        Returns:
            Tuple of (generation config, safety settings or None)
        """
        # Filter out None values for generation config
        gen_config_params = {
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "max_output_tokens": body.get("max_tokens"),
            "stop_sequences": body.get("stop") or None,
        }
        filtered_params = {k: v for k, v in gen_config_params.items() if v is not None}
        generation_config = GenerationConfig(**filtered_params)

        # Configure safety settings
        if self.valves.USE_PERMISSIVE_SAFETY:
            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
        else:
            safety_settings = None  # Use default settings
        
        return generation_config, safety_settings

    def _handle_streaming_response(self, response_iterator: Any) -> Iterator[str]:
        """
        Handle streaming response from Gemini API.
        
        Args:
            response_iterator: Iterator from generate_content
            
        Returns:
            Generator yielding text chunks
        """
        try:
            for chunk in response_iterator:
                # Check for safety feedback or empty chunks
                if not chunk.candidates:
                    # Check prompt feedback
                    if response_iterator.prompt_feedback and response_iterator.prompt_feedback.block_reason:
                        yield f"[Blocked due to Prompt Safety: {response_iterator.prompt_feedback.block_reason.name}]"
                    else:
                        yield "[Blocked by safety settings]"
                    return  # Stop generation
                
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            self.log.exception(f"Error during streaming: {e}")
            yield f"Error during streaming: {e}"

    def _handle_standard_response(self, response: Any) -> str:
        """
        Handle non-streaming response from Gemini API.
        
        Args:
            response: Response from generate_content
            
        Returns:
            Generated text or error message
        """
        # Check for prompt safety blocks
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"[Blocked due to Prompt Safety: {response.prompt_feedback.block_reason.name}]"
            
        # Check for missing candidates
        if not response.candidates:
            return "[Blocked by safety settings or no candidates generated]"
        
        # Check candidate finish reason
        candidate = response.candidates[0]
        if candidate.finish_reason == genai.types.Candidate.FinishReason.SAFETY:
            # Try to get specific safety rating info
            blocking_rating = next((r for r in candidate.safety_ratings if r.blocked), None)
            reason = f" ({blocking_rating.category.name})" if blocking_rating else ""
            return f"[Blocked by safety settings{reason}]"
        
        # Process content parts
        if candidate.content and candidate.content.parts:
            # Combine text from all parts
            return "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
        else:
            return "[No content generated or unexpected response structure]"

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
            except (ServiceUnavailable, ResourceExhausted) as e:
                # These errors might be temporary, so retry
                retry_count += 1
                last_exception = e
                
                if retry_count <= max_retries:
                    # Calculate backoff time (exponential with jitter)
                    wait_time = min(2 ** retry_count + (0.1 * retry_count), 10)
                    self.log.warning(f"Temporary error from Google API: {e}. Retrying in {wait_time:.1f}s ({retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                # Don't retry other exceptions
                raise

        # If we get here, we've exhausted retries
        assert last_exception is not None
        raise last_exception

    async def pipe(
        self, body: Dict[str, Any]
    ) -> Union[str, Iterator[str]]:
        """
        Main method for sending requests to the Google Gemini endpoint.

        Args:
            body: The request body containing messages and other parameters.

        Returns:
            Response from Google Gemini API, which could be a string or an iterator for streaming.
        """
        # Setup logging for this request
        request_id = id(body)
        self.log.debug(f"Processing request {request_id}")

        # Validate API key
        try:
            self.validate_api_key()
        except ValueError as e:
            return f"Error: {e}"

        try:
            # Configure genai API
            genai.configure(api_key=self.valves.GOOGLE_API_KEY.get_decrypted())
            
            # Parse and validate model ID
            model_id = body.get("model", "")
            try:
                model_id = self._prepare_model_id(model_id)
                self.log.debug(f"Using model: {model_id}")
            except ValueError as ve:
                return f"Model Error: {ve}"

            # Get stream flag
            stream = body.get("stream", False)
            messages = body.get("messages", [])
            
            # Prepare content and extract system message
            contents, system_instruction = self._prepare_content(messages)
            if not contents:
                return "Error: No valid message content found"
                
            # Configure generation parameters and safety settings
            generation_config, safety_settings = self._configure_generation(body)
            
            # Initialize the model with system instruction
            client = genai.GenerativeModel(
                model_name=model_id,
                system_instruction=system_instruction
            )
            
            # Make the API call
            if stream:
                # For streaming response we'll still use the synchronous method but wrap it
                # in an async call using a thread pool executor (handled by retry_with_backoff)
                try:
                    async def get_streaming_response():
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(
                            None,
                            lambda: client.generate_content(
                                contents,
                                generation_config=generation_config,
                                safety_settings=safety_settings,
                                stream=True,
                            )
                        )
                    
                    response_iterator = await self._retry_with_backoff(get_streaming_response)
                    self.log.debug(f"Request {request_id}: Got streaming response")
                    return self._handle_streaming_response(response_iterator)
                    
                except Exception as e:
                    self.log.exception(f"Error in streaming request {request_id}: {e}")
                    return f"Error during streaming: {e}"
            else:
                # For non-streaming, use the async method if available
                try:
                    # Use the async method with retry
                    async def get_response():
                        # If the Google library has async methods, use those instead
                        try:
                            # Try to use async method if available
                            if hasattr(client, 'generate_content_async'):
                                return await client.generate_content_async(
                                    contents,
                                    generation_config=generation_config,
                                    safety_settings=safety_settings
                                )
                            else:
                                # Fall back to synchronous method in a thread pool
                                loop = asyncio.get_event_loop()
                                return await loop.run_in_executor(
                                    None,
                                    lambda: client.generate_content(
                                        contents,
                                        generation_config=generation_config,
                                        safety_settings=safety_settings,
                                        stream=False
                                    )
                                )
                        except AttributeError:
                            # Fall back if generate_content_async doesn't exist
                            loop = asyncio.get_event_loop()
                            return await loop.run_in_executor(
                                None,
                                lambda: client.generate_content(
                                    contents, 
                                    generation_config=generation_config,
                                    safety_settings=safety_settings,
                                    stream=False
                                )
                            )
                    
                    # Get response with retry
                    response = await self._retry_with_backoff(get_response)
                    self.log.debug(f"Request {request_id}: Got non-streaming response")
                    return self._handle_standard_response(response)
                    
                except Exception as e:
                    self.log.exception(f"Error in non-streaming request {request_id}: {e}")
                    return f"Error generating content: {e}"

        except PermissionDenied as pe:
            error_msg = f"Permission denied: {pe}. Please check your API key and permissions."
            self.log.error(f"Permission error: {pe}")
            return error_msg
            
        except InvalidArgument as ia:
            error_msg = f"Invalid argument: {ia}. Please check your request parameters."
            self.log.error(f"Invalid argument error: {ia}")
            return error_msg
            
        except ResourceExhausted as re:
            error_msg = f"Resource exhausted: {re}. You may have exceeded your quota or rate limits."
            self.log.error(f"Resource exhausted error: {re}")
            return error_msg
            
        except ValueError as ve:
            error_msg = f"Configuration error: {ve}"
            self.log.error(f"Value error: {ve}")
            return error_msg
            
        except Exception as e:
            # Log the full error with traceback
            import traceback
            error_trace = traceback.format_exc()
            self.log.exception(f"Unexpected error: {e}\n{error_trace}")
            
            # Return a user-friendly error message
            return f"An error occurred while processing your request: {e}"
