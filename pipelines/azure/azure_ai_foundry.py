"""
title: Azure AI Foundry Pipeline
author: owndev
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 2.4.1
license: Apache License 2.0
description: A pipeline for interacting with Azure AI services, enabling seamless communication with various AI models via configurable headers and robust error handling. This includes support for Azure OpenAI models as well as other Azure AI models by dynamically managing headers and request configurations. Azure AI Search (RAG) integration is only supported with Azure OpenAI endpoints.
features:
  - Supports dynamic model specification via headers.
  - Filters valid parameters to ensure clean requests.
  - Handles streaming and non-streaming responses.
  - Provides flexible timeout and error handling mechanisms.
  - Compatible with Azure OpenAI and other Azure AI models.
  - Predefined models for easy access.
  - Encrypted storage of sensitive API keys
  - Azure AI Search / RAG integration with enhanced citation display (Azure OpenAI only)
"""

from typing import List, Union, Generator, Iterator, Optional, Dict, Any, AsyncIterator
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, GetCoreSchemaHandler
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
        # API key for Azure AI
        AZURE_AI_API_KEY: EncryptedStr = Field(
            default=os.getenv("AZURE_AI_API_KEY", "API_KEY"),
            description="API key for Azure AI",
        )

        # Endpoint for Azure AI (e.g. "https://<your-endpoint>/chat/completions?api-version=2024-05-01-preview" or "https://<your-endpoint>/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview")
        AZURE_AI_ENDPOINT: str = Field(
            default=os.getenv(
                "AZURE_AI_ENDPOINT",
                "https://<your-endpoint>/chat/completions?api-version=2024-05-01-preview",
            ),
            description="Endpoint for Azure AI",
        )

        # Optional model name, only necessary if not Azure OpenAI or if model name not in URL (e.g. "https://<your-endpoint>/openai/deployments/<model-name>/chat/completions")
        # Multiple models can be specified as a semicolon-separated list (e.g. "gpt-4o;gpt-4o-mini")
        # or a comma-separated list (e.g. "gpt-4o,gpt-4o-mini").
        AZURE_AI_MODEL: str = Field(
            default=os.getenv("AZURE_AI_MODEL", ""),
            description="Optional model names for Azure AI (e.g. gpt-4o, gpt-4o-mini)",
        )

        # Switch for sending model name in request body
        AZURE_AI_MODEL_IN_BODY: bool = Field(
            default=os.getenv("AZURE_AI_MODEL_IN_BODY", False),
            description="If True, include the model name in the request body instead of as a header.",
        )

        # Flag to indicate if predefined Azure AI models should be used
        USE_PREDEFINED_AZURE_AI_MODELS: bool = Field(
            default=os.getenv("USE_PREDEFINED_AZURE_AI_MODELS", False),
            description="Flag to indicate if predefined Azure AI models should be used.",
        )

        # If True, use Authorization header with Bearer token instead of api-key header.
        USE_AUTHORIZATION_HEADER: bool = Field(
            default=bool(os.getenv("AZURE_AI_USE_AUTHORIZATION_HEADER", False)),
            description="Set to True to use Authorization header with Bearer token instead of api-key header.",
        )

        # Azure AI Data Sources Configuration (for Azure AI Search / RAG)
        # Only works with Azure OpenAI endpoints: https://<deployment>.openai.azure.com/openai/deployments/<model>/chat/completions?api-version=2025-01-01-preview
        AZURE_AI_DATA_SOURCES: str = Field(
            default=os.getenv("AZURE_AI_DATA_SOURCES", ""),
            description='JSON configuration for data_sources field (for Azure AI Search / RAG). Example: \'[{"type":"azure_search","parameters":{"endpoint":"https://xxx.search.windows.net","index_name":"your-index","authentication":{"type":"api_key","key":"your-key"}}}]\'',
        )

        # Enable enhanced citation display for Azure AI Search responses
        AZURE_AI_ENHANCE_CITATIONS: bool = Field(
            default=bool(os.getenv("AZURE_AI_ENHANCE_CITATIONS", True)),
            description="If True, enhance Azure AI Search responses with better citation formatting and source content display.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name: str = "Azure AI"
        # Extract model name from Azure OpenAI URL if available
        self._extracted_model_name = self._extract_model_from_url()

    def _extract_model_from_url(self) -> Optional[str]:
        """
        Extract model name from Azure OpenAI URL format.
        Expected format: https://<deployment>.openai.azure.com/openai/deployments/<model>/chat/completions

        Returns:
            Model name if found in URL, None otherwise
        """
        if not self.valves.AZURE_AI_ENDPOINT:
            return None

        try:
            # Check if this is an Azure OpenAI URL
            if (
                ".openai.azure.com" in self.valves.AZURE_AI_ENDPOINT
                and "/deployments/" in self.valves.AZURE_AI_ENDPOINT
            ):
                # Extract model name from URL pattern
                # Pattern: .../deployments/{model}/chat/completions...
                parts = self.valves.AZURE_AI_ENDPOINT.split("/deployments/")
                if len(parts) > 1:
                    model_part = parts[1].split("/")[
                        0
                    ]  # Get first segment after deployments/
                    if model_part:
                        # Log for debugging
                        log = logging.getLogger("azure_ai._extract_model_from_url")
                        log.debug(
                            f"Extracted model name '{model_part}' from URL: {self.valves.AZURE_AI_ENDPOINT}"
                        )
                        return model_part
        except Exception as e:
            # Log parsing errors
            log = logging.getLogger("azure_ai._extract_model_from_url")
            log.warning(
                f"Error extracting model from URL {self.valves.AZURE_AI_ENDPOINT}: {e}"
            )

        return None

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

    def get_headers(self, model_name: str = None) -> Dict[str, str]:
        """
        Constructs the headers for the API request, including the model name if defined.

        Args:
            model_name: Optional model name to use instead of the default one

        Returns:
            Dictionary containing the required headers for the API request.
        """
        # Access the decrypted API key
        api_key = self.valves.AZURE_AI_API_KEY.get_decrypted()
        if self.valves.USE_AUTHORIZATION_HEADER:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        else:
            headers = {"api-key": api_key, "Content-Type": "application/json"}

        # If we have a model name and it shouldn't be in the body, add it to headers
        if not self.valves.AZURE_AI_MODEL_IN_BODY:
            # If specific model name provided, use it
            if model_name:
                headers["x-ms-model-mesh-model-name"] = model_name
            # Otherwise, if AZURE_AI_MODEL has a single value, use that
            elif (
                self.valves.AZURE_AI_MODEL
                and ";" not in self.valves.AZURE_AI_MODEL
                and "," not in self.valves.AZURE_AI_MODEL
                and " " not in self.valves.AZURE_AI_MODEL
            ):
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

    def get_azure_ai_data_sources(self) -> Optional[List[Dict[str, Any]]]:
        """
        Builds Azure AI data sources configuration from the AZURE_AI_DATA_SOURCES environment variable.
        Only works with Azure OpenAI endpoints: https://<deployment>.openai.azure.com/openai/deployments/<model>/chat/completions?api-version=2025-01-01-preview

        Returns:
            List containing Azure AI data source configuration, or None if not configured.
        """
        if not self.valves.AZURE_AI_DATA_SOURCES:
            return None

        try:
            data_sources = json.loads(self.valves.AZURE_AI_DATA_SOURCES)
            if isinstance(data_sources, list):
                return data_sources
            else:
                # If it's a single object, wrap it in a list
                return [data_sources]
        except json.JSONDecodeError as e:
            # Log error and return None if JSON parsing fails
            log = logging.getLogger("azure_ai.get_azure_ai_data_sources")
            log.error(f"Error parsing AZURE_AI_DATA_SOURCES: {e}")
            return None

    def enhance_azure_search_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhances Azure AI Search responses by improving citation display and adding source content.

        Args:
            response: The original response from Azure AI

        Returns:
            Enhanced response with better citation formatting
        """
        if not isinstance(response, dict):
            return response

        # Check if this is an Azure AI Search response with citations
        if (
            "choices" not in response
            or not response["choices"]
            or "message" not in response["choices"][0]
            or "context" not in response["choices"][0]["message"]
            or "citations" not in response["choices"][0]["message"]["context"]
        ):
            return response

        try:
            choice = response["choices"][0]
            message = choice["message"]
            context = message["context"]
            citations = context["citations"]
            content = message["content"]

            # Create citation mappings
            citation_details = {}
            for i, citation in enumerate(citations, 1):
                if not isinstance(citation, dict):
                    continue

                doc_ref = f"[doc{i}]"
                citation_details[doc_ref] = {
                    "title": citation.get("title", "Unknown Document"),
                    "content": citation.get("content", ""),
                    "url": citation.get("url"),
                    "filepath": citation.get("filepath"),
                    "chunk_id": citation.get("chunk_id", "0"),
                }

            # Enhance the content with better citation display
            enhanced_content = content

            # Add citation section at the end
            if citation_details:
                citation_section = self._format_citation_section(
                    citations, for_streaming=False
                )
                enhanced_content += citation_section

            # Update the message content
            message["content"] = enhanced_content

            # Add enhanced citation info to context for API consumers
            context["enhanced_citations"] = citation_details

            return response

        except Exception as e:
            log = logging.getLogger("azure_ai.enhance_azure_search_response")
            log.warning(f"Failed to enhance Azure Search response: {e}")
            return response

    def parse_models(self, models_str: str) -> List[str]:
        """
        Parses a string of models separated by commas, semicolons, or spaces.

        Args:
            models_str: String containing model names separated by commas, semicolons, or spaces

        Returns:
            List of individual model names
        """
        if not models_str:
            return []

        # Replace semicolons and commas with spaces, then split by spaces and filter empty strings
        models = []
        for model in models_str.replace(";", " ").replace(",", " ").split():
            if model.strip():
                models.append(model.strip())

        return models

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
            {
                "id": "Cohere-command-r-plus-08-2024",
                "name": "Cohere Command R+ 08-2024",
            },
            {"id": "cohere-command-a", "name": "Cohere Command A"},
            {"id": "DeepSeek-R1", "name": "DeepSeek-R1"},
            {"id": "DeepSeek-V3", "name": "DeepSeek-V3"},
            {"id": "DeepSeek-V3-0324", "name": "DeepSeek-V3-0324"},
            {"id": "jais-30b-chat", "name": "JAIS 30b Chat"},
            {
                "id": "Llama-3.2-11B-Vision-Instruct",
                "name": "Llama-3.2-11B-Vision-Instruct",
            },
            {
                "id": "Llama-3.2-90B-Vision-Instruct",
                "name": "Llama-3.2-90B-Vision-Instruct",
            },
            {"id": "Llama-3.3-70B-Instruct", "name": "Llama-3.3-70B-Instruct"},
            {"id": "Meta-Llama-3-70B-Instruct", "name": "Meta-Llama-3-70B-Instruct"},
            {"id": "Meta-Llama-3-8B-Instruct", "name": "Meta-Llama-3-8B-Instruct"},
            {
                "id": "Meta-Llama-3.1-405B-Instruct",
                "name": "Meta-Llama-3.1-405B-Instruct",
            },
            {
                "id": "Meta-Llama-3.1-70B-Instruct",
                "name": "Meta-Llama-3.1-70B-Instruct",
            },
            {"id": "Meta-Llama-3.1-8B-Instruct", "name": "Meta-Llama-3.1-8B-Instruct"},
            {"id": "Ministral-3B", "name": "Ministral 3B"},
            {"id": "Mistral-large", "name": "Mistral Large"},
            {"id": "Mistral-large-2407", "name": "Mistral Large (2407)"},
            {"id": "Mistral-Large-2411", "name": "Mistral Large 24.11"},
            {"id": "Mistral-Nemo", "name": "Mistral Nemo"},
            {"id": "Mistral-small", "name": "Mistral Small"},
            {"id": "mistral-small-2503", "name": "Mistral Small 3.1"},
            {"id": "mistral-medium-2505", "name": "Mistral Medium 3 (25.05)"},
            {"id": "grok-3", "name": "Grok 3"},
            {"id": "grok-3-mini", "name": "Grok 3 Mini"},
            {"id": "gpt-4o", "name": "OpenAI GPT-4o"},
            {"id": "gpt-4o-mini", "name": "OpenAI GPT-4o mini"},
            {"id": "gpt-4.1", "name": "OpenAI GPT-4.1"},
            {"id": "gpt-4.1-mini", "name": "OpenAI GPT-4.1 Mini"},
            {"id": "gpt-4.1-nano", "name": "OpenAI GPT-4.1 Nano"},
            {"id": "gpt-4.5-preview", "name": "OpenAI GPT-4.5 Preview"},
            {"id": "gpt-5", "name": "OpenAI GPT-5"},
            {"id": "gpt-5-mini", "name": "OpenAI GPT-5 Mini"},
            {"id": "gpt-5-nano", "name": "OpenAI GPT-5 Nano"},
            {"id": "gpt-5-chat", "name": "OpenAI GPT-5 Chat"},
            {"id": "gpt-oss-20b", "name": "OpenAI GPT-OSS 20B"},
            {"id": "gpt-oss-120b", "name": "OpenAI GPT-OSS 120B"},
            {"id": "o1", "name": "OpenAI o1"},
            {"id": "o1-mini", "name": "OpenAI o1-mini"},
            {"id": "o1-preview", "name": "OpenAI o1-preview"},
            {"id": "o3", "name": "OpenAI o3"},
            {"id": "o3-pro", "name": "OpenAI o3 Pro"},
            {"id": "o3-mini", "name": "OpenAI o3-mini"},
            {"id": "o4-mini", "name": "OpenAI o4-mini"},
            {
                "id": "Phi-3-medium-128k-instruct",
                "name": "Phi-3-medium instruct (128k)",
            },
            {"id": "Phi-3-medium-4k-instruct", "name": "Phi-3-medium instruct (4k)"},
            {"id": "Phi-3-mini-128k-instruct", "name": "Phi-3-mini instruct (128k)"},
            {"id": "Phi-3-mini-4k-instruct", "name": "Phi-3-mini instruct (4k)"},
            {"id": "Phi-3-small-128k-instruct", "name": "Phi-3-small instruct (128k)"},
            {"id": "Phi-3-small-8k-instruct", "name": "Phi-3-small instruct (8k)"},
            {"id": "Phi-3.5-mini-instruct", "name": "Phi-3.5-mini instruct (128k)"},
            {"id": "Phi-3.5-MoE-instruct", "name": "Phi-3.5-MoE instruct (128k)"},
            {"id": "Phi-3.5-vision-instruct", "name": "Phi-3.5-vision instruct (128k)"},
            {"id": "Phi-4", "name": "Phi-4"},
            {"id": "Phi-4-mini-instruct", "name": "Phi-4 mini instruct"},
            {"id": "Phi-4-multimodal-instruct", "name": "Phi-4 multimodal instruct"},
            {"id": "Phi-4-reasoning", "name": "Phi-4 Reasoning"},
            {"id": "Phi-4-mini-reasoning", "name": "Phi-4 Mini Reasoning"},
            {"id": "MAI-DS-R1", "name": "Microsoft Deepseek R1"},
            {"id": "model-router", "name": "Model Router"},
        ]

    def pipes(self) -> List[Dict[str, str]]:
        """
        Returns a list of available pipes based on configuration.

        Returns:
            List of dictionaries containing pipe id and name.
        """
        self.validate_environment()

        # Re-extract model name in case valves were updated
        self._extracted_model_name = self._extract_model_from_url()

        # If custom models are provided, parse them and return as pipes
        if self.valves.AZURE_AI_MODEL:
            self.name = "Azure AI: "
            models = self.parse_models(self.valves.AZURE_AI_MODEL)
            if models:
                return [{"id": model, "name": model} for model in models]
            else:
                # Fallback for backward compatibility
                return [
                    {
                        "id": self.valves.AZURE_AI_MODEL,
                        "name": self.valves.AZURE_AI_MODEL,
                    }
                ]

        # If custom model is not provided but predefined models are enabled, return those.
        if self.valves.USE_PREDEFINED_AZURE_AI_MODELS:
            self.name = "Azure AI: "
            return self.get_azure_models()

        # Check if we can extract model name from Azure OpenAI URL
        if self._extracted_model_name:
            self.name = "Azure AI: "
            return [
                {"id": self._extracted_model_name, "name": self._extracted_model_name}
            ]

        # Otherwise, use a default name.
        self.name = f"Azure AI: "
        return [{"id": "azure_ai", "name": "Azure AI"}]

    async def stream_processor_with_citations(
        self,
        content: aiohttp.StreamReader,
        __event_emitter__=None,
        response: Optional[aiohttp.ClientResponse] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> AsyncIterator[bytes]:
        """
        Enhanced stream processor that can handle Azure AI Search citations in streaming responses.

        Args:
            content: The streaming content from the response
            __event_emitter__: Optional event emitter for status updates

        Yields:
            Bytes from the streaming content with enhanced citations
        """
        log = logging.getLogger("azure_ai.stream_processor_with_citations")

        try:
            full_response_buffer = ""
            citations_data = None
            citations_added = False
            all_chunks = []

            async for chunk in content:
                chunk_str = chunk.decode("utf-8", errors="ignore")
                full_response_buffer += chunk_str
                all_chunks.append(chunk)

                # Log chunk for debugging (only first 200 chars to avoid spam)
                log.debug(f"Processing chunk: {chunk_str[:200]}...")

                # Look for citations in any part of the response
                if "citations" in chunk_str.lower() and not citations_data:
                    log.debug("Found 'citations' in chunk, attempting to parse...")

                    # Try to extract citation data from the current buffer
                    try:
                        # Look for SSE data lines
                        lines = full_response_buffer.split("\n")
                        for line in lines:
                            if (
                                line.startswith("data: ")
                                and line.strip() != "data: [DONE]"
                            ):
                                json_str = line[6:].strip()  # Remove 'data: ' prefix
                                if json_str and json_str != "[DONE]":
                                    try:
                                        response_data = json.loads(json_str)

                                        # Check multiple possible locations for citations
                                        citations_found = None

                                        if (
                                            isinstance(response_data, dict)
                                            and "choices" in response_data
                                        ):
                                            for choice in response_data["choices"]:
                                                # Check in delta.context.citations
                                                if (
                                                    "delta" in choice
                                                    and isinstance(
                                                        choice["delta"], dict
                                                    )
                                                    and "context" in choice["delta"]
                                                    and "citations"
                                                    in choice["delta"]["context"]
                                                ):
                                                    citations_found = choice["delta"][
                                                        "context"
                                                    ]["citations"]
                                                    log.debug(
                                                        f"Found citations in delta.context: {len(citations_found)} citations"
                                                    )
                                                    break

                                                # Check in message.context.citations
                                                elif (
                                                    "message" in choice
                                                    and isinstance(
                                                        choice["message"], dict
                                                    )
                                                    and "context" in choice["message"]
                                                    and "citations"
                                                    in choice["message"]["context"]
                                                ):
                                                    citations_found = choice["message"][
                                                        "context"
                                                    ]["citations"]
                                                    log.debug(
                                                        f"Found citations in message.context: {len(citations_found)} citations"
                                                    )
                                                    break

                                        # Store the first valid citations we find
                                        if citations_found and not citations_data:
                                            citations_data = citations_found
                                            log.info(
                                                f"Successfully extracted {len(citations_data)} citations from stream"
                                            )

                                    except json.JSONDecodeError:
                                        # Skip invalid JSON
                                        continue

                    except Exception as parse_error:
                        log.debug(f"Error parsing citations from chunk: {parse_error}")

                # Always yield the original chunk first
                yield chunk

                # Check if this is the end of the stream
                if "data: [DONE]" in chunk_str:
                    log.debug("End of stream detected")
                    break

            # After the stream ends, add citations if we found any
            if citations_data and not citations_added:
                log.info("Adding citation summary at end of stream...")

                citation_section = self._format_citation_section(
                    citations_data, for_streaming=True
                )
                if citation_section:
                    # Convert escaped newlines to actual newlines for display
                    display_section = citation_section.replace("\\n", "\n")

                    # Send the citation section in smaller, safer chunks
                    # Split by lines and send each as a separate SSE event
                    lines = display_section.split("\n")

                    for line in lines:
                        # Escape quotes and backslashes for JSON
                        safe_line = line.replace("\\", "\\\\").replace('"', '\\"')
                        # Create a simple SSE event
                        sse_event = f'data: {{"choices":[{{"delta":{{"content":"{safe_line}\\n"}}}}]}}\n\n'
                        yield sse_event.encode("utf-8")

                    citations_added = True
                    log.info("Citation summary successfully added to stream")

            # If we didn't find citations in the stream but detected citation references,
            # try one more time with the full buffer
            elif not citations_data and "[doc" in full_response_buffer:
                log.warning(
                    "Found [doc] references but no citation data - attempting final parse..."
                )
                # This is a fallback for cases where citation detection failed
                fallback_message = "\\n\\n<details>\\n<summary>⚠️ Citations Processing Issue</summary>\\n\\nThe response contains citation references [doc1], [doc2], etc., but the citation details could not be extracted from the streaming response.\\n\\n</details>\\n"
                fallback_sse = f'data: {{"choices":[{{"delta":{{"content":"{fallback_message}"}}}}]}}\n\n'
                yield fallback_sse.encode("utf-8")

            # Send completion status update when streaming is done
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Streaming completed", "done": True},
                    }
                )

        except Exception as e:
            log.error(f"Error processing stream: {e}")

            # Send error status update
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {str(e)}", "done": True},
                    }
                )
        finally:
            # Always attempt to close response and session to avoid resource leaks
            try:
                if response:
                    response.close()
            except Exception:
                pass
            try:
                if session:
                    await session.close()
            except Exception:
                # Suppress close-time errors (e.g., SSL shutdown timeouts)
                pass

    def _format_citation_section(
        self, citations: List[Dict[str, Any]], for_streaming: bool = False
    ) -> str:
        """
        Creates a formatted citation section using collapsible details elements.

        Args:
            citations: List of citation objects
            for_streaming: If True, format for streaming (with escaping), else for regular response

        Returns:
            Formatted citation section with HTML details elements
        """
        if not citations:
            return ""

        # Collect all citation details
        citation_entries = []

        for i, citation in enumerate(citations, 1):
            if not isinstance(citation, dict):
                continue

            doc_ref = f"[doc{i}]"
            title = citation.get("title", "Unknown Document")
            content = citation.get("content", "")
            filepath = citation.get("filepath")
            url = citation.get("url")
            chunk_id = citation.get("chunk_id", "0")

            # Build individual citation details
            citation_info = []

            if filepath:
                citation_info.append(f"📁 **File:** `{filepath}`")
            elif url:
                citation_info.append(f"🔗 **URL:** {url}")

            citation_info.append(f"📄 **Chunk ID:** {chunk_id}")

            # Add full content if available
            if content:
                try:
                    # Clean content for display
                    clean_content = str(content).strip()
                    # Replace problematic characters for HTML display
                    clean_content = clean_content.replace("\n", " ").replace("\r", " ")
                    if for_streaming:
                        # Additional escaping for streaming
                        clean_content = clean_content.replace("\\", "\\\\").replace(
                            '"', '\\"'
                        )

                    citation_info.append(f"**Content:**")
                    citation_info.append(f"> {clean_content}")
                except Exception:
                    citation_info.append("**Content:** [Content unavailable]")

            # Create collapsible details for individual citation
            if for_streaming:
                # For streaming, we need to escape newlines
                citation_content = "\\n".join(citation_info)
                citation_entry = f"<details>\\n<summary>{doc_ref} - {title}</summary>\\n\\n{citation_content}\\n\\n</details>"
            else:
                citation_content = "\n".join(citation_info)
                citation_entry = f"<details>\n<summary>{doc_ref} - {title}</summary>\n\n{citation_content}\n\n</details>"

            citation_entries.append(citation_entry)

        # Combine all citations into main collapsible section
        if for_streaming:
            all_citations = "\\n\\n".join(citation_entries)
            result = f"\\n\\n<details>\\n<summary>📚 Sources and References</summary>\\n\\n{all_citations}\\n\\n</details>\\n"
        else:
            all_citations = "\n\n".join(citation_entries)
            result = f"\n\n<details>\n<summary>📚 Sources and References</summary>\n\n{all_citations}\n\n</details>\n"

        return result

    async def stream_processor(
        self,
        content: aiohttp.StreamReader,
        __event_emitter__=None,
        response: Optional[aiohttp.ClientResponse] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> AsyncIterator[bytes]:
        """
        Process streaming content and properly handle completion status updates.

        Args:
            content: The streaming content from the response
            __event_emitter__: Optional event emitter for status updates

        Yields:
            Bytes from the streaming content
        """
        try:
            async for chunk in content:
                yield chunk

            # Send completion status update when streaming is done
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Streaming completed", "done": True},
                    }
                )
        except Exception as e:
            log = logging.getLogger("azure_ai.stream_processor")
            log.error(f"Error processing stream: {e}")

            # Send error status update
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {str(e)}", "done": True},
                    }
                )
        finally:
            # Always attempt to close response and session to avoid resource leaks
            try:
                if response:
                    response.close()
            except Exception:
                pass
            try:
                if session:
                    await session.close()
            except Exception:
                # Suppress close-time errors (e.g., SSL shutdown timeouts)
                pass

    async def pipe(
        self, body: Dict[str, Any], __event_emitter__=None
    ) -> Union[str, Generator, Iterator, Dict[str, Any], StreamingResponse]:
        """
        Main method for sending requests to the Azure AI endpoint.
        The model name is passed as a header if defined.

        Args:
            body: The request body containing messages and other parameters
            __event_emitter__: Optional event emitter function for status updates

        Returns:
            Response from Azure AI API, which could be a string, dictionary or streaming response
        """
        log = logging.getLogger("azure_ai.pipe")
        log.setLevel(SRC_LOG_LEVELS["OPENAI"])

        # Validate the request body
        self.validate_body(body)
        selected_model = None

        if "model" in body and body["model"]:
            selected_model = body["model"]
            # Safer model extraction with split
            selected_model = (
                selected_model.split(".", 1)[1]
                if "." in selected_model
                else selected_model
            )

        # Construct headers with selected model
        headers = self.get_headers(selected_model)

        # Filter allowed parameters
        allowed_params = {
            "model",
            "messages",
            "deployment",
            "frequency_penalty",
            "max_tokens",
            "max_citations",
            "presence_penalty",
            "reasoning_effort",
            "response_format",
            "seed",
            "stop",
            "stream",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
            "data_sources",
        }
        filtered_body = {k: v for k, v in body.items() if k in allowed_params}

        if self.valves.AZURE_AI_MODEL and self.valves.AZURE_AI_MODEL_IN_BODY:
            # If a model was explicitly selected in the request, use that
            if selected_model:
                filtered_body["model"] = selected_model
            else:
                # Otherwise, if AZURE_AI_MODEL contains multiple models, only use the first one to avoid errors
                models = self.parse_models(self.valves.AZURE_AI_MODEL)
                if models and len(models) > 0:
                    filtered_body["model"] = models[0]
                else:
                    # Fallback to the original value
                    filtered_body["model"] = self.valves.AZURE_AI_MODEL
        elif "model" in filtered_body and filtered_body["model"]:
            # Safer model extraction with split
            filtered_body["model"] = (
                filtered_body["model"].split(".", 1)[1]
                if "." in filtered_body["model"]
                else filtered_body["model"]
            )

        # Add Azure AI data sources if configured and not already present in request
        if "data_sources" not in filtered_body:
            azure_ai_data_sources = self.get_azure_ai_data_sources()
            if azure_ai_data_sources:
                filtered_body["data_sources"] = azure_ai_data_sources

        # Convert the modified body back to JSON
        payload = json.dumps(filtered_body)

        # Send status update via event emitter if available
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Sending request to Azure AI...",
                        "done": False,
                    },
                }
            )

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

            # If the server returned an error status, parse and raise before streaming logic
            if request.status >= 400:
                err_ct = (request.headers.get("Content-Type") or "").lower()
                if "json" in err_ct:
                    try:
                        response = await request.json()
                    except Exception as e:
                        # In error status, provider may mislabel content-type; keep log at debug to avoid noise
                        log.debug(
                            f"Failed to parse JSON error body despite JSON content-type: {e}"
                        )
                        response = await request.text()
                else:
                    response = await request.text()

                request.raise_for_status()

            # Auto-detect streaming: either requested via body or indicated by response headers
            content_type_header = (request.headers.get("Content-Type") or "").lower()
            wants_stream = bool(filtered_body.get("stream", False))
            is_sse_header = "text/event-stream" in content_type_header

            if wants_stream or is_sse_header:
                streaming = True

                # Send status update for successful streaming connection
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Streaming response from Azure AI...",
                                "done": False,
                            },
                        }
                    )

                # Ensure correct SSE headers are set for downstream consumers
                sse_headers = dict(request.headers)
                sse_headers["Content-Type"] = "text/event-stream"
                sse_headers.pop("Content-Length", None)

                # Use enhanced stream processor if Azure AI Search is configured and citations are enabled
                if (
                    self.valves.AZURE_AI_DATA_SOURCES
                    and self.valves.AZURE_AI_ENHANCE_CITATIONS
                ):
                    stream_processor = self.stream_processor_with_citations
                else:
                    stream_processor = self.stream_processor

                return StreamingResponse(
                    stream_processor(
                        request.content,
                        __event_emitter__=__event_emitter__,
                        response=request,
                        session=session,
                    ),
                    status_code=request.status,
                    headers=sse_headers,
                )
            else:
                # Parse non-stream response based on content-type without noisy error logs
                if "json" in content_type_header:
                    try:
                        response = await request.json()
                    except Exception as e:
                        log.debug(
                            f"Failed to parse JSON response despite JSON content-type: {e}"
                        )
                        response = await request.text()
                else:
                    response = await request.text()

                request.raise_for_status()

                # Enhance Azure Search responses with better citation display
                if (
                    isinstance(response, dict)
                    and self.valves.AZURE_AI_DATA_SOURCES
                    and self.valves.AZURE_AI_ENHANCE_CITATIONS
                ):
                    response = self.enhance_azure_search_response(response)

                # Send completion status update
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "Request completed", "done": True},
                        }
                    )

                return response

        except Exception as e:
            log.exception(f"Error in Azure AI request: {e}")

            detail = f"Exception: {str(e)}"
            if isinstance(response, dict):
                if "error" in response:
                    detail = f"{response['error']['message'] if 'message' in response['error'] else response['error']}"
            elif isinstance(response, str):
                detail = response

            # Send error status update
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {detail}", "done": True},
                    }
                )

            return f"Error: {detail}"
        finally:
            if not streaming and session:
                if request:
                    request.close()
                await session.close()
