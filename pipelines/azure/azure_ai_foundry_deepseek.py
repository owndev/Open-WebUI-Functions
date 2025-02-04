"""
title: Azure AI Foundry Pipeline for DeepSeek-R1
author: owndev
author_url: https://github.com/owndev
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/owndev/Open-WebUI-Functions
version: 1.0.1
license: MIT
description: A pipeline for interacting with DeepSeek-R1 in Azure AI services.
features:
  - Supports dynamic model specification via headers.
  - Filters valid parameters to ensure clean requests.
  - Handles streaming and non-streaming responses.
  - Provides flexible timeout and error handling mechanisms.
  - Compatible with Azure AI DeepSeek-R1.
"""

from typing import Union, Generator, Iterator
from pydantic import BaseModel, Field
import requests
import os

class Pipe:
    # Environment variables for API key, endpoint, and optional model
    class Valves(BaseModel):
        # API key for Azure AI
        AZURE_AI_API_KEY: str = Field(
            default=os.getenv("AZURE_AI_API_KEY", "API_KEY"),
            description="API key for Azure AI"
        )

        # Endpoint for DeepSeek-R1 in Azure AI (e.g. "https://<your-endpoint>.eastus2.models.ai.azure.com/chat/completions")
        AZURE_AI_ENDPOINT: str = Field(
            default=os.getenv("AZURE_AI_ENDPOINT", "https://<your-endpoint>.eastus2.models.ai.azure.com/chat/completions"),
            description="Endpoint for DeepSeek-R1 in Azure AI"
        )

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
            "Authorization": "Bearer " + self.valves.AZURE_AI_API_KEY,
            "Content-Type": "application/json",
        }
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
            "top_p",
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
                timeout=600 if do_stream else 300,  # Longer timeout for streaming
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
            for line in response.iter_lines():
                if line:  # Skip empty lines
                    yield line
        except Exception as e:
            yield f"Error while streaming: {e}"