"""
title: Azure AI Foundry Pipeline
author: owndev
author_url: https://github.com/owndev
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/owndev/Open-WebUI-Functions
version: 2.1.0
license: Apache License 2.0
description: A pipeline for interacting with Azure AI services, enabling seamless communication with various AI models via configurable headers and robust error handling. This includes support for Azure OpenAI models as well as other Azure AI models by dynamically managing headers and request configurations.
features:
  - Supports dynamic model specification via headers.
  - Filters valid parameters to ensure clean requests.
  - Handles streaming and non-streaming responses.
  - Provides flexible timeout and error handling mechanisms.
  - Compatible with Azure OpenAI and other Azure AI models.
  - Predefined models for easy access.
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
            return value[len("encrypted:"):]  # Return without prefix
        
        try:
            encrypted_part = value[len("encrypted:"):]
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
        return core_schema.union_schema([
            core_schema.is_instance_schema(cls),
            core_schema.chain_schema([
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(
                    lambda value: cls(cls.encrypt(value) if value else value)
                ),
            ]),
        ],
        serialization=core_schema.plain_serializer_function_ser_schema(lambda instance: str(instance))
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
        # API key for Azure AI
        AZURE_AI_API_KEY: EncryptedStr = Field(
            default=os.getenv("AZURE_AI_API_KEY", "API_KEY"),
            description="API key for Azure AI"
        )

        # Endpoint for Azure AI (e.g. "https://<your-endpoint>/chat/completions?api-version=2024-05-01-preview" or "https://<your-endpoint>/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview")
        AZURE_AI_ENDPOINT: str = Field(
            default=os.getenv(
                "AZURE_AI_ENDPOINT",
                "https://<your-endpoint>/chat/completions?api-version=2024-05-01-preview"
            ),
            description="Endpoint for Azure AI"
        )

        # Optional model name, only necessary if not Azure OpenAI or if model name not in URL (e.g. "https://<your-endpoint>/openai/deployments/<model-name>/chat/completions")
        AZURE_AI_MODEL: str = Field(
            default=os.getenv("AZURE_AI_MODEL", ""),
            description="Optional model name for Azure AI"
        )

        # Switch for sending model name in request body
        AZURE_AI_MODEL_IN_BODY: bool = Field(
            default=os.getenv("AZURE_AI_MODEL_IN_BODY", False),
            description="If True, include the model name in the request body instead of as a header."
        )

        # Flag to indicate if predefined Azure AI models should be used        
        USE_PREDEFINED_AZURE_AI_MODELS: bool = Field(
            default=os.getenv("USE_PREDEFINED_AZURE_AI_MODELS", False),
            description="Flag to indicate if predefined Azure AI models should be used."
        )

        # If True, use Authorization header with Bearer token instead of api-key header.
        USE_AUTHORIZATION_HEADER: bool = Field(
            default=bool(os.getenv("AZURE_AI_USE_AUTHORIZATION_HEADER", False)),
            description="Set to True to use Authorization header with Bearer token instead of api-key header."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name: str = "Azure AI"

    def validate_environment(self) -> None:
        """
        Validates that required environment variables are set.
        
        Raises:
            ValueError: If required environment variables are not set.
        """
        # Access the decrypted API key
        api_key = self.valves.AZURE_AI_API_KEY.get_decrypted()
        if not api_key:
            raise ValueError("AZURE_AI_API_KEY is not set!")
        if not self.valves.AZURE_AI_ENDPOINT:
            raise ValueError("AZURE_AI_ENDPOINT is not set!")

    def get_headers(self) -> Dict[str, str]:
        """
        Constructs the headers for the API request, including the model name if defined.
        
        Returns:
            Dictionary containing the required headers for the API request.
        """
        # Access the decrypted API key
        api_key = self.valves.AZURE_AI_API_KEY.get_decrypted()
        if self.valves.USE_AUTHORIZATION_HEADER:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        else:
            headers = {
                "api-key": api_key,
                "Content-Type": "application/json"
            }

        # If the valve indicates that the model name should be in the body,
        # add it to the filtered body.
        if self.valves.AZURE_AI_MODEL and not self.valves.AZURE_AI_MODEL_IN_BODY:
            headers["x-ms-model-mesh-model-name"] = self.valves.AZURE_AI_MODEL
        return headers

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

    def get_azure_models(self) -> List[Dict[str, str]]:
        """
        Returns a list of predefined Azure AI models.
        
        Returns:
            List of dictionaries containing model id and name.
        """
        return [
            {"id": "AI21-Jamba-1.5-Large", "name": "AI21 Jamba 1.5 Large"},
            {"id": "AI21-Jamba-1.5-Mini", "name": "AI21 Jamba 1.5 Mini"},
            {"id": "Codestral-2501", "name": "Codestral 25.01"},
            {"id": "Cohere-command-r", "name": "Cohere Command R"},
            {"id": "Cohere-command-r-08-2024", "name": "Cohere Command R 08-2024"},
            {"id": "Cohere-command-r-plus", "name": "Cohere Command R+"},
            {"id": "Cohere-command-r-plus-08-2024", "name": "Cohere Command R+ 08-2024"},
            {"id": "DeepSeek-R1", "name": "DeepSeek-R1"},
            {"id": "DeepSeek-V3", "name": "DeepSeek-V3"},
            {"id": "jais-30b-chat", "name": "JAIS 30b Chat"},
            {"id": "Llama-3.2-11B-Vision-Instruct", "name": "Llama-3.2-11B-Vision-Instruct"},
            {"id": "Llama-3.2-90B-Vision-Instruct", "name": "Llama-3.2-90B-Vision-Instruct"},
            {"id": "Llama-3.3-70B-Instruct", "name": "Llama-3.3-70B-Instruct"},
            {"id": "Meta-Llama-3-70B-Instruct", "name": "Meta-Llama-3-70B-Instruct"},
            {"id": "Meta-Llama-3-8B-Instruct", "name": "Meta-Llama-3-8B-Instruct"},
            {"id": "Meta-Llama-3.1-405B-Instruct", "name": "Meta-Llama-3.1-405B-Instruct"},
            {"id": "Meta-Llama-3.1-70B-Instruct", "name": "Meta-Llama-3.1-70B-Instruct"},
            {"id": "Meta-Llama-3.1-8B-Instruct", "name": "Meta-Llama-3.1-8B-Instruct"},
            {"id": "Ministral-3B", "name": "Ministral 3B"},
            {"id": "Mistral-large", "name": "Mistral Large"},
            {"id": "Mistral-large-2407", "name": "Mistral Large (2407)"},
            {"id": "Mistral-Large-2411", "name": "Mistral Large 24.11"},
            {"id": "Mistral-Nemo", "name": "Mistral Nemo"},
            {"id": "Mistral-small", "name": "Mistral Small"},
            {"id": "gpt-4o", "name": "OpenAI GPT-4o"},
            {"id": "gpt-4o-mini", "name": "OpenAI GPT-4o mini"},
            {"id": "o1", "name": "OpenAI o1"},
            {"id": "o1-mini", "name": "OpenAI o1-mini"},
            {"id": "o1-preview", "name": "OpenAI o1-preview"},
            {"id": "o3-mini", "name": "OpenAI o3-mini"},
            {"id": "Phi-3-medium-128k-instruct", "name": "Phi-3-medium instruct (128k)"},
            {"id": "Phi-3-medium-4k-instruct", "name": "Phi-3-medium instruct (4k)"},
            {"id": "Phi-3-mini-128k-instruct", "name": "Phi-3-mini instruct (128k)"},
            {"id": "Phi-3-mini-4k-instruct", "name": "Phi-3-mini instruct (4k)"},
            {"id": "Phi-3-small-128k-instruct", "name": "Phi-3-small instruct (128k)"},
            {"id": "Phi-3-small-8k-instruct", "name": "Phi-3-small instruct (8k)"},
            {"id": "Phi-3.5-mini-instruct", "name": "Phi-3.5-mini instruct (128k)"},
            {"id": "Phi-3.5-MoE-instruct", "name": "Phi-3.5-MoE instruct (128k)"},
            {"id": "Phi-3.5-vision-instruct", "name": "Phi-3.5-vision instruct (128k)"},
            {"id": "Phi-4", "name": "Phi-4"}
        ]

    def pipes(self) -> List[Dict[str, str]]:
        """
        Returns a list of available pipes based on configuration.
        
        Returns:
            List of dictionaries containing pipe id and name.
        """
        self.validate_environment()
    
        # If a custom model is provided, use it exclusively.
        if self.valves.AZURE_AI_MODEL:
            self.name = "Azure AI: "
            return [{"id": self.valves.AZURE_AI_MODEL, "name": self.valves.AZURE_AI_MODEL}]
        
        # If custom model is not provided but predefined models are enabled, return those.
        if self.valves.USE_PREDEFINED_AZURE_AI_MODELS:
            self.name = "Azure AI: "
            return self.get_azure_models()
        
        # Otherwise, use a default name.
        return [{"id": "Azure AI", "name": "Azure AI"}]

    async def pipe(self, body: Dict[str, Any]) -> Union[str, Generator, Iterator, Dict[str, Any], StreamingResponse]:
        """
        Main method for sending requests to the Azure AI endpoint.
        The model name is passed as a header if defined.
        
        Args:
            body: The request body containing messages and other parameters
            
        Returns:
            Response from Azure AI API, which could be a string, dictionary or streaming response
        """
        log = logging.getLogger("azure_ai.pipe")
        log.setLevel(SRC_LOG_LEVELS["OPENAI"])

        # Validate the request body
        self.validate_body(body)

        # Construct headers
        headers = self.get_headers()

        # Filter allowed parameters
        allowed_params = {
            "model",
            "messages",
            "frequency_penalty",
            "max_tokens",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "stream",
            "temperature",
            "tool_choice",
            "tools",
            "top_p"
        }
        filtered_body = {k: v for k, v in body.items() if k in allowed_params}

        # If the valve indicates that the model name should be in the body,
        # add it to the filtered body.
        if self.valves.AZURE_AI_MODEL and self.valves.AZURE_AI_MODEL_IN_BODY:
            filtered_body["model"] = self.valves.AZURE_AI_MODEL
        elif "model" in filtered_body and filtered_body["model"]:
            # Safer model extraction with split
            filtered_body["model"] = filtered_body["model"].split(".", 1)[1] if "." in filtered_body["model"] else filtered_body["model"]

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

            request = await session.request(
                method="POST",
                url=self.valves.AZURE_AI_ENDPOINT,
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
            log.exception(f"Error in Azure AI request: {e}")

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