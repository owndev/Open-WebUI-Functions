"""
title: Azure AI Foundry Pipeline
author: owndev
author_url: https://github.com/owndev
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/owndev/Open-WebUI-Functions
version: 1.1.0
license: MIT
description: A Python-based pipeline for interacting with Azure AI services, enabling seamless communication with various AI models via configurable headers and robust error handling. This includes support for Azure OpenAI models as well as other Azure AI models by dynamically managing headers and request configurations.
features:
  - Supports dynamic model specification via headers.
  - Filters valid parameters to ensure clean requests.
  - Handles streaming and non-streaming responses.
  - Provides flexible timeout and error handling mechanisms.
  - Compatible with Azure OpenAI and other Azure AI models.
"""

from typing import List, Union, Generator, Iterator
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
            default=False,
            description="If True, include the model name in the request body instead of as a header."
        )

        # Flag to indicate if predefined Azure AI models should be used        
        USE_PREDEFINED_AZURE_AI_MODELS: bool = Field(
            default=True,
            description="Flag to indicate if predefined Azure AI models should be used. (currently does not work with Azure OpenAI models)"
        )

    def __init__(self):
        self.valves = self.Valves()

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

        # If the valve indicates that the model name should be in the body,
        # add it to the filtered body.
        if self.valves.AZURE_AI_MODEL and not self.valves.AZURE_AI_MODEL_IN_BODY:
            headers["x-ms-model-mesh-model-name"] = self.valves.AZURE_AI_MODEL
        return headers

    def validate_body(self, body: dict):
        """
        Validates the request body to ensure required fields are present.
        """
        if "messages" not in body or not isinstance(body["messages"], list):
            raise ValueError("The 'messages' field is required and must be a list.")

    def get_azure_models(self):
        return [
            {"id": "AI21-Jamba-1.5-Large", "name": "AI21 Jamba 1.5 Large"},
            {"id": "AI21-Jamba-1.5-Mini", "name": "AI21 Jamba 1.5 Mini"},
            {"id": "Codestral-2501", "name": "Codestral 25.01"},
            {"id": "Cohere-command-r", "name": "Cohere Command R"},
            {"id": "Cohere-command-r-08-2024", "name": "Cohere Command R 08-2024"},
            {"id": "Cohere-command-r-plus", "name": "Cohere Command R+"},
            {"id": "Cohere-command-r-plus-08-2024", "name": "Cohere Command R+ 08-2024"},
            {"id": "DeepSeek-R1", "name": "DeepSeek-R1"},
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

    def pipes(self) -> List[dict]:
        self.validate_environment()
    
        # If a custom model is provided, use it exclusively.
        if self.valves.AZURE_AI_MODEL:
            self.name = f"Azure AI: {self.valves.AZURE_AI_MODEL}"
            return [{"id": self.valves.AZURE_AI_MODEL, "name": self.valves.AZURE_AI_MODEL}]
        
        # If custom model is not provided but predefined models are enabled, return those.
        if self.valves.USE_PREDEFINED_AZURE_AI_MODELS:
            self.name = "Azure AI: "
            return self.get_azure_models()
        
        # Otherwise, use a default name.
        return [{"id": "Azure AI", "name": "Azure AI"}]

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
            "top_p",
        }
        filtered_body = {k: v for k, v in body.items() if k in allowed_params}

        # If the valve indicates that the model name should be in the body,
        # add it to the filtered body.
        if self.valves.AZURE_AI_MODEL and self.valves.AZURE_AI_MODEL_IN_BODY:
            filtered_body["model"] = self.valves.AZURE_AI_MODEL
        elif filtered_body["model"]:
            filtered_body["model"] = filtered_body["model"].split(".")[-1]
            
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
