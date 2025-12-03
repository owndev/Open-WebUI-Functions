"""
title: Azure AI Foundry Pipeline
author: owndev
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 2.6.0
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
  - Native OpenWebUI citations support with structured events and citation cards (Azure OpenAI only)
"""

from typing import (
    List,
    Union,
    Generator,
    Iterator,
    Optional,
    Dict,
    Any,
    AsyncIterator,
    Set,
    Callable,
)
from urllib.parse import urlparse
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
import re
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
    # Regex pattern for matching [docX] citation references
    DOC_REF_PATTERN = r"\[doc(\d+)\]"

    # Environment variables for API key, endpoint, and optional model
    class Valves(BaseModel):
        # Custom prefix for pipeline display name
        AZURE_AI_PIPELINE_PREFIX: str = Field(
            default=os.getenv("AZURE_AI_PIPELINE_PREFIX", "Azure AI"),
            description="Custom prefix for the pipeline display name (e.g., 'Azure AI', 'My Azure', 'Company AI'). The final display will be: '<prefix>: <model_name>'",
        )

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
            default=False,
            description="If True, enhance Azure AI Search responses with better citation formatting and source content display (markdown/HTML).",
        )

        # Enable native OpenWebUI citations (structured events and fields)
        AZURE_AI_OPENWEBUI_CITATIONS: bool = Field(
            default=True,
            description="If True, emit native OpenWebUI citation events for streaming responses and attach openwebui_citations field for non-streaming responses. Enables citation cards and UI in OpenWebUI frontend.",
        )

        # Enable relevance scores from Azure AI Search
        AZURE_AI_INCLUDE_SEARCH_SCORES: bool = Field(
            default=True,
            description="If True, automatically add 'include_contexts' with 'all_retrieved_documents' to Azure AI Search requests to get relevance scores (original_search_score and rerank_score). This enables relevance percentage display in citation cards.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name: str = f"{self.valves.AZURE_AI_PIPELINE_PREFIX}:"
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
            endpoint_host = urlparse(self.valves.AZURE_AI_ENDPOINT).hostname or ""
            if (
                endpoint_host == "openai.azure.com"
                or endpoint_host.endswith(".openai.azure.com")
            ) and "/deployments/" in self.valves.AZURE_AI_ENDPOINT:
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
        api_key = EncryptedStr.decrypt(self.valves.AZURE_AI_API_KEY)
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
        api_key = EncryptedStr.decrypt(self.valves.AZURE_AI_API_KEY)
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

        If AZURE_AI_INCLUDE_SEARCH_SCORES is enabled, automatically adds 'include_contexts'
        with 'all_retrieved_documents' to get relevance scores from Azure AI Search.

        Returns:
            List containing Azure AI data source configuration, or None if not configured.
        """
        if not self.valves.AZURE_AI_DATA_SOURCES:
            return None

        log = logging.getLogger("azure_ai.get_azure_ai_data_sources")

        try:
            data_sources = json.loads(self.valves.AZURE_AI_DATA_SOURCES)
            if not isinstance(data_sources, list):
                # If it's a single object, wrap it in a list
                data_sources = [data_sources]

            # If AZURE_AI_INCLUDE_SEARCH_SCORES is enabled, add include_contexts
            if self.valves.AZURE_AI_INCLUDE_SEARCH_SCORES:
                for source in data_sources:
                    if (
                        isinstance(source, dict)
                        and source.get("type") == "azure_search"
                        and "parameters" in source
                    ):
                        params = source["parameters"]
                        # Get or create include_contexts list
                        include_contexts = params.get("include_contexts", [])
                        if not isinstance(include_contexts, list):
                            include_contexts = [include_contexts]

                        # Add 'citations' and 'all_retrieved_documents' if not present
                        if "citations" not in include_contexts:
                            include_contexts.append("citations")
                        if "all_retrieved_documents" not in include_contexts:
                            include_contexts.append("all_retrieved_documents")

                        params["include_contexts"] = include_contexts
                        log.debug(
                            f"Added include_contexts to Azure Search: {include_contexts}"
                        )

            return data_sources
        except json.JSONDecodeError as e:
            # Log error and return None if JSON parsing fails
            log.error(f"Error parsing AZURE_AI_DATA_SOURCES: {e}")
            return None

    def _extract_citations_from_response(
        self, response_data: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extract citations from an Azure AI response (streaming or non-streaming).

        Supports both 'citations' and 'all_retrieved_documents' response structures.
        When include_contexts includes 'all_retrieved_documents', the response contains
        additional score fields like 'original_search_score' and 'rerank_score'.

        Args:
            response_data: Response data from Azure AI (can be a delta or full message)

        Returns:
            List of citation objects, or None if no citations found
        """
        log = logging.getLogger("azure_ai._extract_citations_from_response")

        if not isinstance(response_data, dict):
            log.debug(f"Response data is not a dict: {type(response_data)}")
            return None

        # Try multiple possible locations for citations
        citations = None

        # Check in choices[0].delta.context or choices[0].message.context
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            context = None

            # Get context from delta (streaming) or message (non-streaming)
            if "delta" in choice and isinstance(choice["delta"], dict):
                context = choice["delta"].get("context")
            elif "message" in choice and isinstance(choice["message"], dict):
                context = choice["message"].get("context")

            if context and isinstance(context, dict):
                # Try citations first
                if "citations" in context:
                    citations = context["citations"]
                    log.info(
                        f"Found {len(citations) if citations else 0} citations in context.citations"
                    )

                # If all_retrieved_documents is present, merge score data into citations
                if "all_retrieved_documents" in context:
                    all_docs = context["all_retrieved_documents"]
                    log.debug(
                        f"Found {len(all_docs) if all_docs else 0} all_retrieved_documents"
                    )

                    # If we have both citations and all_retrieved_documents,
                    # try to merge score data from all_retrieved_documents into citations
                    if citations and all_docs:
                        self._merge_score_data(citations, all_docs, log)
                    elif all_docs and not citations:
                        # Use all_retrieved_documents as citations if no citations found
                        citations = all_docs
                        log.info(
                            f"Using {len(citations)} all_retrieved_documents as citations"
                        )
            else:
                log.debug(
                    f"No context found in response. Choice keys: {choice.keys() if isinstance(choice, dict) else 'not a dict'}"
                )
        else:
            log.debug(f"No choices in response. Response keys: {response_data.keys()}")

        if citations and isinstance(citations, list):
            log.info(f"Extracted {len(citations)} citations from response")
            # Log first citation structure for debugging (only if INFO logging is enabled)
            if citations and log.isEnabledFor(logging.INFO):
                log.info(
                    f"First citation structure: {json.dumps(citations[0], default=str)[:500]}"
                )
            return citations

        log.debug("No valid citations found in response")
        return None

    def _merge_score_data(
        self,
        citations: List[Dict[str, Any]],
        all_docs: List[Dict[str, Any]],
        log: logging.Logger,
    ) -> None:
        """
        Merge score data from all_retrieved_documents into citations.

        When include_contexts includes 'all_retrieved_documents', Azure returns
        additional documents with score fields. This method attempts to match
        them with citations and copy over the score data.

        Copies:
        - original_search_score: BM25/keyword search score
        - rerank_score: Semantic reranker score (if enabled)
        - filter_reason: Indicates which score is relevant ("score" or "rerank")

        Args:
            citations: List of citation objects to update (modified in place)
            all_docs: List of all_retrieved_documents with score data
            log: Logger instance
        """
        # Build multiple lookup maps to maximize matching chances
        # all_retrieved_documents may have different keys than citations
        doc_data_by_title = {}
        doc_data_by_filepath = {}
        doc_data_by_content = {}
        doc_data_by_chunk_id = {}

        for doc in all_docs:
            doc_data = {
                "original_search_score": doc.get("original_search_score"),
                "rerank_score": doc.get("rerank_score"),
                "filter_reason": doc.get("filter_reason"),
            }

            log.debug(
                f"Processing all_retrieved_document: title='{doc.get('title')}', "
                f"chunk_id='{doc.get('chunk_id')}', "
                f"original_search_score={doc_data['original_search_score']}, "
                f"rerank_score={doc_data['rerank_score']}, "
                f"filter_reason={doc_data['filter_reason']}"
            )

            # Only store if we have at least one score
            if (
                doc_data["original_search_score"] is None
                and doc_data["rerank_score"] is None
            ):
                log.debug(f"Skipping doc with no scores: {doc.get('title')}")
                continue

            # Index by title
            if doc.get("title"):
                doc_data_by_title[doc["title"]] = doc_data

            # Index by filepath
            if doc.get("filepath"):
                doc_data_by_filepath[doc["filepath"]] = doc_data

            # Index by chunk_id (may include title as prefix for uniqueness)
            if doc.get("chunk_id") is not None:
                # Store by plain chunk_id
                doc_data_by_chunk_id[str(doc["chunk_id"])] = doc_data
                # Also store by title-prefixed chunk_id for uniqueness
                if doc.get("title"):
                    chunk_key_with_title = f"{doc['title']}_{doc['chunk_id']}"
                    doc_data_by_chunk_id[chunk_key_with_title] = doc_data

            # Index by content prefix (first 100 chars)
            if doc.get("content"):
                content_key = (
                    doc["content"][:100]
                    if len(doc.get("content", "")) > 100
                    else doc.get("content")
                )
                doc_data_by_content[content_key] = doc_data

        log.debug(
            f"Built score lookup: by_title={len(doc_data_by_title)}, "
            f"by_filepath={len(doc_data_by_filepath)}, "
            f"by_chunk_id={len(doc_data_by_chunk_id)}, "
            f"by_content={len(doc_data_by_content)}"
        )

        # Match citations with score data using multiple strategies
        matched = 0
        for citation in citations:
            doc_data = None

            # Try matching by title first (most reliable)
            if not doc_data and citation.get("title"):
                doc_data = doc_data_by_title.get(citation["title"])
                if doc_data:
                    log.debug(f"Matched citation by title: {citation['title']}")

            # Try matching by filepath
            if not doc_data and citation.get("filepath"):
                doc_data = doc_data_by_filepath.get(citation["filepath"])
                if doc_data:
                    log.debug(f"Matched citation by filepath: {citation['filepath']}")

            # Try matching by chunk_id with title prefix
            if not doc_data and citation.get("chunk_id") is not None:
                chunk_key = str(citation["chunk_id"])
                if citation.get("title"):
                    chunk_key_with_title = f"{citation['title']}_{citation['chunk_id']}"
                    doc_data = doc_data_by_chunk_id.get(chunk_key_with_title)
                if not doc_data:
                    doc_data = doc_data_by_chunk_id.get(chunk_key)
                if doc_data:
                    log.debug(f"Matched citation by chunk_id: {citation['chunk_id']}")

            # Try matching by content prefix
            if not doc_data and citation.get("content"):
                content_key = (
                    citation["content"][:100]
                    if len(citation.get("content", "")) > 100
                    else citation.get("content")
                )
                doc_data = doc_data_by_content.get(content_key)
                if doc_data:
                    log.debug("Matched citation by content prefix")

            if doc_data:
                if doc_data.get("original_search_score") is not None:
                    citation["original_search_score"] = doc_data["original_search_score"]
                if doc_data.get("rerank_score") is not None:
                    citation["rerank_score"] = doc_data["rerank_score"]
                if doc_data.get("filter_reason") is not None:
                    citation["filter_reason"] = doc_data["filter_reason"]
                matched += 1
                log.debug(
                    f"Citation scores: original={doc_data.get('original_search_score')}, "
                    f"rerank={doc_data.get('rerank_score')}, "
                    f"filter_reason={doc_data.get('filter_reason')}"
                )

        log.info(f"Merged score data for {matched}/{len(citations)} citations")

    def _normalize_citation_for_openwebui(
        self, citation: Dict[str, Any], index: int
    ) -> Dict[str, Any]:
        """
        Normalize an Azure citation object to OpenWebUI citation event format.

        The format follows OpenWebUI's official citation event structure:
        https://docs.openwebui.com/features/plugin/development/events#source-or-citation-and-code-execution

        Args:
            citation: Azure citation object
            index: Citation index (1-based)

        Returns:
            Complete citation event object with type and data fields
        """
        log = logging.getLogger("azure_ai._normalize_citation_for_openwebui")

        # Get title with fallback chain: title → filepath → url → "Unknown Document"
        # Handle None values explicitly since dict.get() returns None if key exists but value is None
        title_raw = citation.get("title") or ""
        filepath_raw = citation.get("filepath") or ""
        url_raw = citation.get("url") or ""

        base_title = (
            title_raw.strip()
            or filepath_raw.strip()
            or url_raw.strip()
            or "Unknown Document"
        )
        # Include [docX] prefix in OpenWebUI citation card titles for document identification
        title = f"[doc{index}] - {base_title}"

        # Build source URL for metadata
        source_url = url_raw or filepath_raw

        # Build metadata with source information
        # Use title with [docX] prefix as metadata source for OpenWebUI display
        # The UI may extract display name from metadata.source rather than source.name
        metadata_entry = {"source": title, "url": source_url}
        if citation.get("metadata"):
            metadata_entry.update(citation.get("metadata", {}))

        # Get document content (handle None values)
        content = citation.get("content") or ""

        # Build normalized citation data structure matching OpenWebUI format exactly
        citation_data = {
            "document": [content],
            "metadata": [metadata_entry],
            "source": {"name": title},
        }

        # Add URL to source if available
        if source_url:
            citation_data["source"]["url"] = source_url

        # Add distances array for relevance score (OpenWebUI uses this for percentage display)
        # Azure AI Search returns filter_reason to indicate which score type is relevant:
        # - filter_reason not present or "score": use original_search_score (BM25/keyword)
        # - filter_reason "rerank": use rerank_score (semantic reranker)
        # Reference: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/references/on-your-data
        filter_reason = citation.get("filter_reason")
        rerank_score = citation.get("rerank_score")
        original_search_score = citation.get("original_search_score")
        legacy_score = citation.get("score")

        normalized_score = 0.0

        # Select score based on filter_reason as per Azure documentation:
        # - filter_reason="rerank": Document filtered by rerank score threshold, use rerank_score
        # - filter_reason="score" or not present: Document filtered by/passed original search score, use original_search_score
        if filter_reason == "rerank" and rerank_score is not None:
            # Document filtered by rerank score - use rerank_score
            # Azure AI Search semantic rerankers typically return scores in 0-1 range.
            # However, some Cohere rerankers (via Azure AI) may use 0-4 range.
            score_val = float(rerank_score)
            if score_val > 1.0:
                normalized_score = min(score_val / 4.0, 1.0)
            else:
                normalized_score = score_val
            log.debug(
                f"Using rerank_score (filter_reason=rerank): {rerank_score} -> {normalized_score}"
            )
        elif (
            filter_reason is None or filter_reason == "score"
        ) and original_search_score is not None:
            # filter_reason is "score" or not present - use original_search_score
            # BM25/keyword search scores vary based on term frequency and document collection.
            # Typical BM25 scores in Azure AI Search range from ~0 to ~50 but can go higher.
            score_val = float(original_search_score)
            if score_val > 1.0:
                normalized_score = min(score_val / 100.0, 1.0)
            else:
                normalized_score = score_val
            log.debug(
                f"Using original_search_score (filter_reason={filter_reason}): {original_search_score} -> {normalized_score}"
            )
        elif original_search_score is not None:
            # Fallback for unknown filter_reason values - use original_search_score
            score_val = float(original_search_score)
            if score_val > 1.0:
                normalized_score = min(score_val / 100.0, 1.0)
            else:
                normalized_score = score_val
            log.debug(
                f"Using original_search_score (fallback, filter_reason={filter_reason}): {original_search_score} -> {normalized_score}"
            )
        elif rerank_score is not None:
            # Fallback to rerank_score if available but filter_reason doesn't match
            score_val = float(rerank_score)
            if score_val > 1.0:
                normalized_score = min(score_val / 4.0, 1.0)
            else:
                normalized_score = score_val
            log.debug(
                f"Using rerank_score (fallback): {rerank_score} -> {normalized_score}"
            )
        elif legacy_score is not None:
            normalized_score = float(legacy_score)
            log.debug(f"Using legacy score: {legacy_score}")
        else:
            log.debug("No score available, using default 0.0")

        citation_data["distances"] = [normalized_score]

        # Build complete citation event structure
        citation_event = {
            "type": "citation",
            "data": citation_data,
        }

        # Log the normalized citation for debugging (only if INFO logging is enabled)
        if log.isEnabledFor(logging.INFO):
            log.info(
                f"Normalized citation {index}: title='{title}', "
                f"content_length={len(content)}, "
                f"url='{source_url}', "
                f"filter_reason={filter_reason}, "
                f"rerank_score={rerank_score}, original_search_score={original_search_score}, "
                f"distances={citation_data['distances']}, "
                f"event={json.dumps(citation_event, default=str)[:500]}"
            )

        return citation_event

    def _format_source_tag(
        self, doc_num: int, source_name: Optional[str] = None
    ) -> str:
        """
        Format a <source> tag for a [docX] reference.

        Creates an OpenWebUI-compatible source tag that enables citation linking.

        Args:
            doc_num: The document number (1-based)
            source_name: Optional source name/title for the document

        Returns:
            Formatted <source> tag string
        """
        if source_name:
            # Escape special characters in source_name for HTML attribute
            escaped_name = (
                source_name.replace("&", "&amp;")
                .replace('"', "&quot;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            return f'<source id="{doc_num}" name="{escaped_name}">[doc{doc_num}]</source>'
        else:
            return f'<source id="{doc_num}">[doc{doc_num}]</source>'

    def _convert_doc_refs_to_source_tags(
        self, content: str, citations: List[Dict[str, Any]]
    ) -> str:
        """
        Convert [docX] references in content to OpenWebUI <source> tags for citation linking.

        OpenWebUI uses <source id="X" name="title">text</source> tags to create clickable
        citation links in the response. This method converts Azure's [doc1], [doc2], etc.
        references to this format.

        Reference: https://github.com/open-webui/open-webui/blob/main/backend/open_webui/utils/middleware.py#L1518

        Args:
            content: The response content containing [docX] references
            citations: List of citation objects with title, url, etc.

        Returns:
            Content with [docX] references converted to <source> tags
        """
        if not content or not citations:
            return content

        log = logging.getLogger("azure_ai._convert_doc_refs_to_source_tags")

        # Build a mapping of citation index to source name
        citation_names = {}
        for i, citation in enumerate(citations, 1):
            if isinstance(citation, dict):
                # Get title with fallback chain
                title = citation.get("title") or ""
                filepath = citation.get("filepath") or ""
                url = citation.get("url") or ""

                source_name = (
                    title.strip() or filepath.strip() or url.strip() or f"Document {i}"
                )
                citation_names[i] = source_name

        def replace_doc_ref(match):
            """Replace [docX] with <source id="X" name="title">[docX]</source>"""
            doc_num = int(match.group(1))
            source_name = citation_names.get(doc_num)
            return self._format_source_tag(doc_num, source_name)

        # Replace all [docX] references
        converted = re.sub(self.DOC_REF_PATTERN, replace_doc_ref, content)

        # Count conversions for logging
        original_count = len(re.findall(self.DOC_REF_PATTERN, content))
        if original_count > 0:
            log.info(f"Converted {original_count} [docX] references to <source> tags")

        return converted

    async def _emit_openwebui_citation_events(
        self,
        citations: List[Dict[str, Any]],
        __event_emitter__: Optional[Callable[..., Any]],
        content: str = "",
    ) -> None:
        """
        Emit OpenWebUI citation events for citations.

        Emits one citation event per source document, following the OpenWebUI
        citation event format. Each citation is emitted separately to ensure
        all sources appear in the UI.

        Only emits citations that are actually referenced in the content (e.g., [doc1], [doc2]).

        Args:
            citations: List of Azure citation objects
            __event_emitter__: Event emitter callable for sending citation events
            content: The response content (used to filter only referenced citations)
        """
        log = logging.getLogger("azure_ai._emit_openwebui_citation_events")

        if not __event_emitter__:
            log.warning("No __event_emitter__ provided, cannot emit citation events")
            return

        if not citations:
            log.info("No citations to emit")
            return

        # Extract which citations are actually referenced in the content
        referenced_indices = self._extract_referenced_citations(content)

        # If we couldn't find any references, include all citations (backward compatibility)
        if not referenced_indices:
            referenced_indices = set(range(1, len(citations) + 1))
            log.debug(
                f"No [docX] references found in content, including all {len(citations)} citations"
            )
        else:
            log.info(
                f"Found {len(referenced_indices)} referenced citations: {sorted(referenced_indices)}"
            )

        log.info(
            f"Emitting citation events for {len(referenced_indices)} referenced citations via __event_emitter__"
        )

        emitted_count = 0
        for i, citation in enumerate(citations, 1):
            # Skip citations that are not referenced in the content
            if i not in referenced_indices:
                log.debug(f"Skipping citation {i} - not referenced in content")
                continue

            if not isinstance(citation, dict):
                log.warning(f"Citation {i} is not a dict, skipping: {type(citation)}")
                continue

            try:
                normalized = self._normalize_citation_for_openwebui(citation, i)

                # Log the full citation JSON for debugging
                # log.debug(
                #     f"Full citation event JSON for doc{i}: {json.dumps(normalized, default=str)}"
                # )

                # Emit citation event for this individual source
                source_name = (
                    normalized.get("data", {}).get("source", {}).get("name", "unknown")
                )
                log.info(
                    f"Emitting citation event {i}/{len(citations)} with source.name='{source_name}'"
                )
                await __event_emitter__(normalized)
                emitted_count += 1

                log.info(f"Successfully emitted citation event for doc{i}")

            except Exception as e:
                log.exception(f"Failed to emit citation event for citation {i}: {e}")

        log.info(
            f"Finished emitting {emitted_count}/{len(referenced_indices)} citation events"
        )

    def enhance_azure_search_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance Azure AI Search responses by improving citation display and adding source content.
        Modifies the response in-place and returns it.

        If AZURE_AI_ENHANCE_CITATIONS is True, appends a formatted markdown/HTML citation section
        to the response content.

        Args:
            response: The original response from Azure AI (modified in-place)

        Returns:
            The enhanced response with better citation formatting
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

            # Enhance the content with better citation display (if enabled)
            enhanced_content = content

            # Convert [docX] references to <source> tags for OpenWebUI citation linking
            # Reference: https://github.com/open-webui/open-webui/blob/main/backend/open_webui/utils/middleware.py#L1518
            if self.valves.AZURE_AI_OPENWEBUI_CITATIONS:
                enhanced_content = self._convert_doc_refs_to_source_tags(
                    enhanced_content, citations
                )

            # Add citation section at the end (if markdown/HTML citations are enabled)
            if self.valves.AZURE_AI_ENHANCE_CITATIONS and citation_details:
                citation_section = self._format_citation_section(
                    citations, content, for_streaming=False
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
            {"id": "grok-4", "name": "Grok 4"},
            {"id": "grok-4-fast-reasoning", "name": "Grok 4 Fast Reasoning"},
            {"id": "grok-4-fast-non-reasoning", "name": "Grok 4 Fast Non-Reasoning"},
            {"id": "gpt-4o", "name": "OpenAI GPT-4o"},
            {"id": "gpt-4o-mini", "name": "OpenAI GPT-4o mini"},
            {"id": "gpt-4.1", "name": "OpenAI GPT-4.1"},
            {"id": "gpt-4.1-mini", "name": "OpenAI GPT-4.1 Mini"},
            {"id": "gpt-4.1-nano", "name": "OpenAI GPT-4.1 Nano"},
            {"id": "gpt-4.5-preview", "name": "OpenAI GPT-4.5 Preview"},
            {"id": "gpt-5", "name": "OpenAI GPT-5"},
            {"id": "gpt‑5‑codex", "name": "OpenAI GPT-5 Codex"},
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
            self.name = f"{self.valves.AZURE_AI_PIPELINE_PREFIX}: "
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
            self.name = f"{self.valves.AZURE_AI_PIPELINE_PREFIX}: "
            return self.get_azure_models()

        # Check if we can extract model name from Azure OpenAI URL
        if self._extracted_model_name:
            self.name = f"{self.valves.AZURE_AI_PIPELINE_PREFIX}: "
            return [
                {"id": self._extracted_model_name, "name": self._extracted_model_name}
            ]

        # Otherwise, use a default name.
        self.name = f"{self.valves.AZURE_AI_PIPELINE_PREFIX}: "
        return [{"id": "azure_ai", "name": self.valves.AZURE_AI_PIPELINE_PREFIX}]

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
            response_content = ""  # Track the actual response content
            citations_data = None
            citations_added = False
            all_chunks = []

            async for chunk in content:
                chunk_str = chunk.decode("utf-8", errors="ignore")
                full_response_buffer += chunk_str
                all_chunks.append(chunk)

                # Log chunk for debugging (only first 200 chars to avoid spam)
                # log.debug(f"Processing chunk: {chunk_str[:200]}...")

                # Extract content from delta messages to build the full response content
                try:
                    lines = chunk_str.split("\n")
                    for line in lines:
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            json_str = line[6:].strip()
                            if json_str and json_str != "[DONE]":
                                try:
                                    response_data = json.loads(json_str)
                                    if (
                                        isinstance(response_data, dict)
                                        and "choices" in response_data
                                    ):
                                        for choice in response_data["choices"]:
                                            if (
                                                "delta" in choice
                                                and "content" in choice["delta"]
                                            ):
                                                response_content += choice["delta"][
                                                    "content"
                                                ]
                                except json.JSONDecodeError:
                                    # Malformed or incomplete JSON is expected in streamed chunks; safely skip.
                                    pass
                except Exception as e:
                    log.debug(f"Exception while processing chunk: {e}")

                # Look for citations or all_retrieved_documents in any part of the response
                if (
                    "citations" in chunk_str.lower()
                    or "all_retrieved_documents" in chunk_str.lower()
                ) and not citations_data:
                    log.debug(
                        "Found 'citations' or 'all_retrieved_documents' in chunk, attempting to parse..."
                    )

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
                                        all_docs_found = None

                                        if (
                                            isinstance(response_data, dict)
                                            and "choices" in response_data
                                        ):
                                            for choice in response_data["choices"]:
                                                context = None
                                                # Get context from delta or message
                                                if "delta" in choice and isinstance(
                                                    choice["delta"], dict
                                                ):
                                                    context = choice["delta"].get(
                                                        "context"
                                                    )
                                                elif "message" in choice and isinstance(
                                                    choice["message"], dict
                                                ):
                                                    context = choice["message"].get(
                                                        "context"
                                                    )

                                                if context and isinstance(
                                                    context, dict
                                                ):
                                                    # Check for citations
                                                    if "citations" in context:
                                                        citations_found = context[
                                                            "citations"
                                                        ]
                                                        log.debug(
                                                            f"Found citations in context: {len(citations_found)} citations"
                                                        )
                                                    # Check for all_retrieved_documents
                                                    if (
                                                        "all_retrieved_documents"
                                                        in context
                                                    ):
                                                        all_docs_found = context[
                                                            "all_retrieved_documents"
                                                        ]
                                                        log.debug(
                                                            f"Found all_retrieved_documents in context: {len(all_docs_found)} docs"
                                                        )
                                                    break

                                        # Merge score data if we have both
                                        if citations_found and all_docs_found:
                                            self._merge_score_data(
                                                citations_found, all_docs_found, log
                                            )

                                        # Use citations if found, otherwise use all_retrieved_documents
                                        if citations_found and not citations_data:
                                            citations_data = citations_found
                                            log.info(
                                                f"Successfully extracted {len(citations_data)} citations from stream"
                                            )
                                        elif all_docs_found and not citations_data:
                                            citations_data = all_docs_found
                                            log.info(
                                                f"Using {len(citations_data)} all_retrieved_documents as citations"
                                            )
                                            # Note: OpenWebUI citation events are emitted after the stream ends
                                            # to filter only citations referenced in the response content

                                    except json.JSONDecodeError:
                                        # Skip invalid JSON
                                        continue

                    except Exception as parse_error:
                        log.debug(f"Error parsing citations from chunk: {parse_error}")

                # Convert [docX] references to <source> tags in the chunk content
                # This enables OpenWebUI to link citations in streaming responses
                # Reference: https://github.com/open-webui/open-webui/blob/main/backend/open_webui/utils/middleware.py#L1518
                chunk_modified = False
                if self.valves.AZURE_AI_OPENWEBUI_CITATIONS and "[doc" in chunk_str:
                    try:
                        # Build citation names map if we have citations data
                        citation_names = {}
                        if citations_data:
                            for i, cit in enumerate(citations_data, 1):
                                if isinstance(cit, dict):
                                    title = cit.get("title") or ""
                                    filepath = cit.get("filepath") or ""
                                    url = cit.get("url") or ""
                                    citation_names[i] = (
                                        title.strip()
                                        or filepath.strip()
                                        or url.strip()
                                        or None
                                    )

                        # Parse and modify each SSE data line
                        modified_lines = []
                        chunk_lines = chunk_str.split("\n")

                        for line in chunk_lines:
                            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                                json_str = line[6:].strip()
                                if json_str and json_str != "[DONE]":
                                    try:
                                        data = json.loads(json_str)
                                        if (
                                            isinstance(data, dict)
                                            and "choices" in data
                                            and data["choices"]
                                        ):
                                            line_modified = False
                                            for choice in data["choices"]:
                                                if (
                                                    "delta" in choice
                                                    and "content" in choice["delta"]
                                                ):
                                                    content_val = choice["delta"]["content"]
                                                    if "[doc" in content_val:
                                                        # Convert [docX] to <source> tag
                                                        # Use _format_source_tag with name if available

                                                        def replace_ref(m):
                                                            doc_num = int(m.group(1))
                                                            source_name = citation_names.get(
                                                                doc_num
                                                            )
                                                            return self._format_source_tag(
                                                                doc_num, source_name
                                                            )

                                                        choice["delta"]["content"] = re.sub(
                                                            self.DOC_REF_PATTERN,
                                                            replace_ref,
                                                            content_val,
                                                        )
                                                        line_modified = True

                                            if line_modified:
                                                modified_lines.append(f"data: {json.dumps(data)}")
                                                chunk_modified = True
                                            else:
                                                modified_lines.append(line)
                                        else:
                                            modified_lines.append(line)
                                    except json.JSONDecodeError:
                                        modified_lines.append(line)
                                else:
                                    modified_lines.append(line)
                            else:
                                modified_lines.append(line)

                        # Reconstruct the chunk only if something was modified
                        if chunk_modified:
                            modified_chunk_str = "\n".join(modified_lines)
                            log.debug(
                                "Converted [docX] references to <source> tags in streaming chunk"
                            )
                            chunk = modified_chunk_str.encode("utf-8")

                    except Exception as convert_err:
                        log.debug(f"Error converting [docX] to source tags: {convert_err}")
                        # Fall through to yield original chunk

                # Yield the (possibly modified) chunk
                yield chunk

                # Check if this is the end of the stream
                if "data: [DONE]" in chunk_str:
                    log.debug("End of stream detected")
                    break

            # After the stream ends, emit OpenWebUI citation events if enabled
            if (
                citations_data
                and self.valves.AZURE_AI_OPENWEBUI_CITATIONS
                and __event_emitter__
            ):
                log.info("Emitting OpenWebUI citation events at end of stream...")
                # Filter to only citations referenced in the response content
                await self._emit_openwebui_citation_events(
                    citations_data, __event_emitter__, response_content
                )

            # After the stream ends, add markdown/HTML citations if we found any and it's enabled
            if (
                citations_data
                and not citations_added
                and self.valves.AZURE_AI_ENHANCE_CITATIONS
            ):
                log.info("Adding citation summary at end of stream...")

                # Pass the accumulated response content to filter citations
                citation_section = self._format_citation_section(
                    citations_data, response_content, for_streaming=True
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

    def _extract_referenced_citations(self, content: str) -> Set[int]:
        """
        Extract citation references (e.g., [doc1], [doc2]) from the content.

        Args:
            content: The response content containing citation references

        Returns:
            Set of citation indices that are referenced (e.g., {1, 2, 7, 8, 9})
        """
        # Find all [docN] references in the content using class constant
        matches = re.findall(self.DOC_REF_PATTERN, content)

        # Convert to integers and return as a set
        return {int(match) for match in matches}

    def _format_citation_section(
        self,
        citations: List[Dict[str, Any]],
        content: str = "",
        for_streaming: bool = False,
    ) -> str:
        """
        Creates a formatted citation section using collapsible details elements.
        Only includes citations that are actually referenced in the content.

        Args:
            citations: List of citation objects
            content: The response content (used to filter only referenced citations)
            for_streaming: If True, format for streaming (with escaping), else for regular response

        Returns:
            Formatted citation section with HTML details elements
        """
        if not citations:
            return ""

        # Extract which citations are actually referenced in the content
        referenced_indices = self._extract_referenced_citations(content)

        # If we couldn't find any references, include all citations (backward compatibility)
        if not referenced_indices:
            referenced_indices = set(range(1, len(citations) + 1))

        # Collect only referenced citation details
        citation_entries = []

        for i, citation in enumerate(citations, 1):
            # Skip citations that are not referenced in the content
            if i not in referenced_indices:
                continue

            if not isinstance(citation, dict):
                continue

            doc_ref = f"[doc{i}]"

            # Get title with fallback to filepath or url
            title = citation.get("title", "")
            # Check if title is empty (not just None) and use alternatives
            if not title or not title.strip():
                # Try filepath first
                filepath = citation.get("filepath", "")
                if filepath and filepath.strip():
                    title = filepath
                else:
                    # Try url next
                    url = citation.get("url", "")
                    if url and url.strip():
                        title = url
                    else:
                        # Final fallback
                        title = "Unknown Document"

            content_text = citation.get("content", "")
            filepath = citation.get("filepath", "")
            url = citation.get("url", "")
            chunk_id = citation.get("chunk_id", "")

            # Build individual citation details
            citation_info = []

            # Show filepath if available and not empty
            if filepath and filepath.strip():
                citation_info.append(f"📁 **File:** `{filepath}`")
            # Show URL if available, not empty, and no filepath was shown
            elif url and url.strip():
                citation_info.append(f"🔗 **URL:** {url}")

            # Show chunk_id if available and not empty
            if chunk_id is not None and str(chunk_id).strip():
                citation_info.append(f"📄 **Chunk ID:** {chunk_id}")

            # Add full content if available
            if content_text and str(content_text).strip():
                try:
                    # Clean content for display
                    clean_content = str(content_text).strip()
                    if for_streaming:
                        # Additional escaping for streaming
                        clean_content = clean_content.replace("\\", "\\\\").replace(
                            '"', '\\"'
                        )

                    citation_info.append("**Content:**")
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

        # Only create the section if we have citations to show
        if not citation_entries:
            return ""

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
                if self.valves.AZURE_AI_DATA_SOURCES and (
                    self.valves.AZURE_AI_ENHANCE_CITATIONS
                    or self.valves.AZURE_AI_OPENWEBUI_CITATIONS
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
                # Call this when either citation mode is enabled
                if isinstance(response, dict) and self.valves.AZURE_AI_DATA_SOURCES:
                    if (
                        self.valves.AZURE_AI_ENHANCE_CITATIONS
                        or self.valves.AZURE_AI_OPENWEBUI_CITATIONS
                    ):
                        response = self.enhance_azure_search_response(response)

                    # Emit native OpenWebUI citation events for non-streaming responses
                    if self.valves.AZURE_AI_OPENWEBUI_CITATIONS and __event_emitter__:
                        citations = self._extract_citations_from_response(response)
                        if citations:
                            # Get response content for filtering
                            response_content = ""
                            if (
                                isinstance(response, dict)
                                and "choices" in response
                                and response["choices"]
                            ):
                                message = response["choices"][0].get("message", {})
                                response_content = message.get("content", "")
                            await self._emit_openwebui_citation_events(
                                citations, __event_emitter__, response_content
                            )

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
