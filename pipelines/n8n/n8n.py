"""
title: n8n Pipeline
author: owndev
author_url: https://github.com/owndev
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/owndev/Open-WebUI-Functions
n8n_template: https://github.com/owndev/Open-WebUI-Functions/tree/master/pipelines/n8n
version: 2.0.0
license: Apache License 2.0
description: A pipeline for interacting with N8N workflows, enabling seamless communication with various N8N workflows via configurable headers and robust error handling. This includes support for dynamic message handling and real-time interaction with N8N workflows.
features:
  - Integrates with N8N for seamless communication.
  - Supports dynamic message handling.
  - Enables real-time interaction with N8N workflows.
  - Provides configurable status emissions.
  - Cloudflare Access support for secure communication.
  - Encrypted storage of sensitive API keys
"""

from typing import Optional, Callable, Awaitable, Any, Dict
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
import time
import aiohttp
import os
import base64
import hashlib
import logging
from open_webui.env import AIOHTTP_CLIENT_TIMEOUT, SRC_LOG_LEVELS
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

# Helper function for cleaning up aiohttp resources
async def cleanup_session(session: Optional[aiohttp.ClientSession]) -> None:
    """
    Clean up the aiohttp session.
    
    Args:
        session: The ClientSession object to close
    """
    if session:
        await session.close()
    
class Pipe:
    class Valves(BaseModel):
        N8N_URL: str = Field(
            default="https://<your-endpoint>/webhook/<your-webhook>",
            description="URL for the N8N webhook"
        )
        N8N_BEARER_TOKEN: EncryptedStr = Field(
            default="",
            description="Bearer token for authenticating with the N8N webhook"
        )
        INPUT_FIELD: str = Field(
            default="chatInput",
            description="Field name for the input message in the N8N payload"
        )
        RESPONSE_FIELD: str = Field(
            default="output",
            description="Field name for the response message in the N8N payload"
        )
        EMIT_INTERVAL: float = Field(
            default=2.0,
            description="Interval in seconds between status emissions"
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True,
            description="Enable or disable status indicator emissions"
        )
        CF_ACCESS_CLIENT_ID: EncryptedStr = Field(
            default="",
            description="Only if behind Cloudflare: https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/"
        )
        CF_ACCESS_CLIENT_SECRET: EncryptedStr = Field(
            default="",
            description="Only if behind Cloudflare: https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/"
        )

    def __init__(self):
        self.name = "N8N Agent"
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.log = logging.getLogger("n8n_pipeline")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        level: str,
        message: str,
        done: bool,
    ):
        current_time = time.time()
        if (
            __event_emitter__
            and self.valves.ENABLE_STATUS_INDICATOR
            and (
                current_time - self.last_emit_time >= self.valves.EMIT_INTERVAL or done
            )
        ):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )
            self.last_emit_time = current_time

    def extract_event_info(self, event_emitter):
        if not event_emitter or not event_emitter.__closure__:
            return None, None
        for cell in event_emitter.__closure__:
            if isinstance(request_info := cell.cell_contents, dict):
                chat_id = request_info.get("chat_id")
                message_id = request_info.get("message_id")
                return chat_id, message_id
        return None, None

    def get_headers(self) -> Dict[str, str]:
        """
        Constructs the headers for the API request.
        
        Returns:
            Dictionary containing the required headers for the API request.
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add bearer token if available
        bearer_token = self.valves.N8N_BEARER_TOKEN.get_decrypted()
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        
        # Add Cloudflare Access headers if available
        cf_client_id = self.valves.CF_ACCESS_CLIENT_ID.get_decrypted()
        if cf_client_id:
            headers["CF-Access-Client-Id"] = cf_client_id
            
        cf_client_secret = self.valves.CF_ACCESS_CLIENT_SECRET.get_decrypted()
        if cf_client_secret:
            headers["CF-Access-Client-Secret"] = cf_client_secret
            
        return headers

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Optional[dict]:
        await self.emit_status(
            __event_emitter__, "info", f"Calling {self.name} ...", False
        )
        
        session = None
        n8n_response = None
        messages = body.get("messages", [])

        # Verify a message is available
        if messages:
            question = messages[-1]["content"]
            if "Prompt: " in question:
                question = question.split("Prompt: ")[-1]
            try:
                # Extract chat_id and message_id
                chat_id, message_id = self.extract_event_info(__event_emitter__)
                
                self.log.info(f"Starting N8N workflow request for chat ID: {chat_id}")

                # Prepare payload for N8N workflow
                payload = {
                    "systemPrompt": f"{messages[0]['content'].split('Prompt: ')[-1]}",
                    "user_id": __user__.get("id") if __user__ else None,
                    "user_email": __user__.get("email") if __user__ else None,
                    "user_name": __user__.get("name") if __user__ else None,
                    "user_role": __user__.get("role") if __user__ else None,
                    "chat_id": chat_id,
                    "message_id": message_id,
                }
                payload[self.valves.INPUT_FIELD] = question
                
                # Get headers for the request
                headers = self.get_headers()

                # Invoke N8N workflow with aiohttp
                session = aiohttp.ClientSession(
                    trust_env=True,
                    timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
                )
                
                self.log.debug(f"Sending request to N8N: {self.valves.N8N_URL}")
                async with session.post(
                    self.valves.N8N_URL, 
                    json=payload, 
                    headers=headers
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        self.log.debug(f"N8N response received with status code: {response.status}")
                        n8n_response = response_data[self.valves.RESPONSE_FIELD]
                    else:
                        error_text = await response.text()
                        self.log.error(f"N8N error: Status {response.status} - {error_text}")
                        raise Exception(f"Error: {response.status} - {error_text}")

                # Set assistant message with chain reply
                body["messages"].append({"role": "assistant", "content": n8n_response})
                
            except Exception as e:
                error_msg = f"Error during sequence execution: {str(e)}"
                self.log.exception(error_msg)
                await self.emit_status(
                    __event_emitter__,
                    "error",
                    error_msg,
                    True,
                )
                return {"error": str(e)}
            finally:
                if session:
                    await cleanup_session(session)
            
        # If no message is available alert user
        else:
            error_msg = "No messages found in the request body"
            self.log.warning(error_msg)
            await self.emit_status(
                __event_emitter__,
                "error",
                error_msg,
                True,
            )
            body["messages"].append(
                {
                    "role": "assistant",
                    "content": error_msg,
                }
            )

        await self.emit_status(__event_emitter__, "info", "Complete", True)
        return n8n_response