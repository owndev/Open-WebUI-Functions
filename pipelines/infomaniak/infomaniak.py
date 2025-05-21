"""
title: Infomaniak AI Tools Pipeline
author: owndev
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
infomaniak_url: https://own.dev/infomaniak-com-en-hosting-ai-tools
version: 2.0.0
license: Apache License 2.0
description: A manifold pipeline for interacting with Infomaniak AI Tools.
features:
  - Manifold pipeline for Infomaniak AI Tools
  - Lists available models for easy access
  - Robust error handling and logging
  - Handles streaming and non-streaming responses
  - Encrypted storage of sensitive API keys
"""

from typing import List, Union, Generator, Iterator, Optional, Dict, Any
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from starlette.background import BackgroundTask
from open_webui.env import AIOHTTP_CLIENT_TIMEOUT, SRC_LOG_LEVELS
from cryptography.fernet import Fernet, InvalidToken
import aiohttp
import json
import os
import logging
import base64
import hashlib
from pydantic_core import core_schema


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


class Pipe:
    # Environment variables for API key, endpoint, and optional model
    class Valves(BaseModel):
        # API key for Infomaniak - automatically encrypted
        INFOMANIAK_API_KEY: EncryptedStr = Field(
            default=os.getenv("INFOMANIAK_API_KEY", "API_KEY"),
            description="API key for Infomaniak AI TOOLS API",
        )
        # Product ID for Infomaniak
        INFOMANIAK_PRODUCT_ID: int = Field(
            default=os.getenv("INFOMANIAK_PRODUCT_ID", 50070),
            description="Product ID for Infomaniak AI TOOLS API",
        )
        # Base URL for Infomaniak API
        INFOMANIAK_BASE_URL: str = Field(
            default=os.getenv("INFOMANIAK_BASE_URL", "https://api.infomaniak.com"),
            description="Base URL for Infomaniak API",
        )
        # Prefix for model names
        NAME_PREFIX: str = Field(
            default="Infomaniak: ", description="Prefix to be added before model names"
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.name: str = self.valves.NAME_PREFIX

    def validate_environment(self) -> None:
        """
        Validates that required environment variables are set.

        Raises:
            ValueError: If required environment variables are not set.
        """
        # Access the decrypted API key
        api_key = self.valves.INFOMANIAK_API_KEY.get_decrypted()
        if not api_key:
            raise ValueError("INFOMANIAK_API_KEY is not set!")
        if not self.valves.INFOMANIAK_PRODUCT_ID:
            raise ValueError("INFOMANIAK_PRODUCT_ID is not set!")
        if not self.valves.INFOMANIAK_BASE_URL:
            raise ValueError("INFOMANIAK_BASE_URL is not set!")

    def get_headers(self) -> Dict[str, str]:
        """
        Constructs the headers for the API request.

        Returns:
            Dictionary containing the required headers for the API request.
        """
        # Access the decrypted API key
        api_key = self.valves.INFOMANIAK_API_KEY.get_decrypted()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        return headers

    def get_api_url(self, endpoint: str = "chat/completions") -> str:
        """
        Constructs the API URL for Infomaniak requests.

        Args:
            endpoint: The API endpoint to use

        Returns:
            Full API URL
        """
        return f"{self.valves.INFOMANIAK_BASE_URL}/1/ai/{self.valves.INFOMANIAK_PRODUCT_ID}/openai/{endpoint}"

    def validate_body(self, body: Dict[str, Any]) -> None:
        """
        Validates the request body to ensure required fields are present.

        Args:
            body: The request body to validate

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if "messages" not in body or not isinstance(body["messages"], list):
            raise ValueError("The 'messages' field is required and must be a list.")

    async def get_infomaniak_models(self) -> List[Dict[str, str]]:
        """
        Returns a list of Infomaniak AI LLM models.

        Returns:
            List of dictionaries containing model id and name.
        """
        log = logging.getLogger("infomaniak_ai_tools.get_models")
        log.setLevel(SRC_LOG_LEVELS["OPENAI"])

        headers = self.get_headers()
        models = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url=f"{self.valves.INFOMANIAK_BASE_URL}/1/ai/models",
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("result") == "success" and "data" in data:
                            models_data = data["data"]
                            if isinstance(models_data, list):
                                for item in models_data:
                                    if not isinstance(item, dict):
                                        log.error(
                                            f"Expected item to be dict but got: {type(item).__name__}"
                                        )
                                        continue
                                    if (
                                        item.get("type") == "llm"
                                    ):  # only include llm models
                                        models.append(
                                            {
                                                "id": item.get("name", ""),
                                                "name": item.get(
                                                    "description", item.get("name", "")
                                                ),
                                                # Profile image and description are currently not working in Open WebUI
                                                "meta": {
                                                    "profile_image_url": item.get(
                                                        "logo_url", ""
                                                    ),
                                                    "description": item.get(
                                                        "documentation_link", ""
                                                    ),
                                                },
                                            }
                                        )
                                return models
                            else:
                                log.error(
                                    "Expected 'data' to be a list but received a non-list value."
                                )
                    log.error(f"Failed to get Infomaniak models: {await resp.text()}")
        except Exception as e:
            log.exception(f"Error getting Infomaniak models: {str(e)}")

        # Default model if API call fails
        return [
            {
                "id": f"{self.valves.INFOMANIAK_PRODUCT_ID}",
                "name": "Infomaniak: LLM API",
            }
        ]

    async def pipes(self) -> List[Dict[str, str]]:
        """
        Returns a list of available pipes based on configuration.

        Returns:
            List of dictionaries containing pipe id and name.
        """
        self.validate_environment()
        return await self.get_infomaniak_models()

    async def pipe(
        self, body: Dict[str, Any]
    ) -> Union[str, Generator, Iterator, Dict[str, Any], StreamingResponse]:
        """
        Main method for sending requests to the Infomaniak AI endpoint.

        Args:
            body: The request body containing messages and other parameters

        Returns:
            Response from Infomaniak AI API, which could be a string, dictionary or streaming response
        """
        log = logging.getLogger("infomaniak_ai_tools.pipe")
        log.setLevel(SRC_LOG_LEVELS["OPENAI"])

        # Validate the request body
        self.validate_body(body)

        # Construct headers
        headers = self.get_headers()

        # Filter allowed parameters (https://developer.infomaniak.com/docs/api/post/1/ai/%7Bproduct_id%7D/openai/chat/completions)
        allowed_params = {
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "messages",
            "model",
            "n",
            "presence_penalty",
            "profile_type",
            "seed",
            "stop",
            "stream",
            "temperature",
            "top_logprobs",
            "top_p",
        }
        filtered_body = {k: v for k, v in body.items() if k in allowed_params}

        # Handle model extraction for Infomaniak
        if "model" in filtered_body and filtered_body["model"]:
            # Extract model ID
            filtered_body["model"] = (
                filtered_body["model"].split(".", 1)[1]
                if "." in filtered_body["model"]
                else filtered_body["model"]
            )

        # Convert the modified body back to JSON
        payload = json.dumps(filtered_body)

        request = None
        session = None
        streaming = False
        response = None

        try:
            session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
            )

            api_url = self.get_api_url()
            request = await session.request(
                method="POST",
                url=api_url,
                data=payload,
                headers=headers,
            )

            # Check if response is SSE
            if "text/event-stream" in request.headers.get("Content-Type", ""):
                streaming = True
                return StreamingResponse(
                    request.content,
                    status_code=request.status,
                    headers=dict(request.headers),
                    background=BackgroundTask(
                        cleanup_response, response=request, session=session
                    ),
                )
            else:
                try:
                    response = await request.json()
                except Exception as e:
                    log.error(f"Error parsing JSON response: {e}")
                    response = await request.text()

                request.raise_for_status()
                return response

        except Exception as e:
            log.exception(f"Error in Infomaniak AI request: {e}")

            detail = f"Exception: {str(e)}"
            if isinstance(response, dict):
                if "error" in response:
                    detail = f"{response['error']['message'] if 'message' in response['error'] else response['error']}"
            elif isinstance(response, str):
                detail = response

            return f"Error: {detail}"
        finally:
            if not streaming and session:
                if request:
                    request.close()
                await session.close()
