"""
title: Azure AI Foundry Function
author: owndev
author_url: https://github.com/owndev
project_url: https://github.com/owndev/Open-WebUI
funding_url: btc:bc1qjkcp5acdnvnhyqtezwrqaeyq8542rgdcxx728z | eth:0x5bD03A83dD470568e56E465f1D4B0f0Ff930E49C
version: 1.0.0
license: MIT
description: A Python-based function for interacting with Azure AI services, enabling seamless communication with various AI models via configurable headers and robust error handling. This includes support for Azure OpenAI models as well as other Azure AI models by dynamically managing headers and request configurations.
features:
  - Supports dynamic model specification via headers.
  - Filters valid parameters to ensure clean requests.
  - Handles streaming and non-streaming responses.
  - Provides flexible timeout and error handling mechanisms.
  - Compatible with Azure OpenAI and other Azure AI models.
"""

from typing import Union, Generator, Iterator
from pydantic import BaseModel
import requests
import os


class Pipe:
    # Environment variables for API key, endpoint, and optional model
    class Valves(BaseModel):
        # API key for Azure AI
        AZURE_AI_API_KEY: str = os.getenv("AZURE_AI_API_KEY", "API_KEY")

        # Endpoint for Azure AI (e.g. "https://<your-endpoint>/chat/completions?api-version=2024-05-01-preview" or "https://<your-endpoint>/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview")
        AZURE_AI_ENDPOINT: str = os.getenv(
            "AZURE_AI_ENDPOINT",
            "https://<your-endpoint>/chat/completions?api-version=2024-05-01-preview"
        )

        # Optional model name, only necessary if not Azure OpenAI or if model name not in URL (e.g. "https://<your-endpoint>/openai/deployments/<model-name>/chat/completions")
        AZURE_AI_MODEL: str = os.getenv("AZURE_AI_MODEL", "")

    def __init__(self):
        self.name = "Azure AI"
        self.valves = self.Valves()
        self.validate_environment()

    def validate_environment(self):
        """
        Validates that required environment variables are set.
        """
        if not self.valves.AZURE_AI_API_KEY:
            raise ValueError("AZURE_AI_API_KEY is not set!")
        if not self.valves.AZURE_AI_ENDPOINT:
            raise ValueError("AZURE_AI_ENDPOINT is not set!")

    def get_headers(self) -> dict:
        """
        Constructs the headers for the API request, including the model name if defined.
        """
        headers = {
            "api-key": self.valves.AZURE_AI_API_KEY,
            "Content-Type": "application/json",
        }
        if self.valves.AZURE_AI_MODEL:
            headers["x-ms-model-mesh-model-name"] = self.valves.AZURE_AI_MODEL
        return headers

    def validate_body(self, body: dict):
        """
        Validates the request body to ensure required fields are present.
        """
        if "messages" not in body or not isinstance(body["messages"], list):
            raise ValueError("The 'messages' field is required and must be a list.")

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        """
        Main method for sending requests to the Azure AI endpoint.
        The model name is passed as a header if defined.
        """
        # Validate the request body
        self.validate_body(body)

        # Construct headers
        headers = self.get_headers()

        # Filter allowed parameters
        allowed_params = {
            "messages",
            "temperature",
            "role",
            "content",
            "contentPart",
            "contentPartImage",
            "enhancements",
            "dataSources",
            "n",
            "stream",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "function_call",
            "funcions",
            "tools",
            "tool_choice",
            "top_p",
            "log_probs",
            "top_logprobs",
            "response_format",
            "seed",
        }
        filtered_body = {k: v for k, v in body.items() if k in allowed_params}

        response = None
        try:
            # Check for streaming support
            do_stream = filtered_body.get("stream", False)

            # Send POST request to the endpoint
            response = requests.post(
                url=self.valves.AZURE_AI_ENDPOINT,
                json=filtered_body,
                headers=headers,
                stream=do_stream,
                timeout=60 if do_stream else 30,  # Longer timeout for streaming
            )
            response.raise_for_status()

            # Return streaming or full response
            if do_stream:
                return self.stream_response(response)
            else:
                return response.json()

        except requests.exceptions.Timeout:
            return "Error: Request timed out. The server did not respond in time."
        except requests.exceptions.ConnectionError:
            return "Error: Failed to connect to the server. Check your network or endpoint."
        except requests.exceptions.HTTPError as http_err:
            return f"HTTP Error: {http_err.response.status_code} - {http_err.response.text}"
        except Exception as e:
            return f"Unexpected error: {e}"

    def stream_response(self, response):
        """
        Handles streaming responses line by line.
        """
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:  # Skip empty lines
                    yield line
        except Exception as e:
            yield f"Error while streaming: {e}"
