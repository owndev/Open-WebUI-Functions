"""
title: Time Token Tracker
author: owndev
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 2.7.0
required_open_webui_version: 0.8.0
license: Apache License 2.0
description: A filter for tracking the response time and token usage of a request with Azure Log Analytics integration (Logs Ingestion API or the deprecated HTTP Data Collector API).
features:
  - Tracks the response time of a request.
  - Tracks Token Usage.
  - Calculates the average tokens per message.
  - Calculates the tokens per second.
  - Sends metrics to Azure Log Analytics in the background (10 s timeout), so responses do not wait for it: through the Azure Monitor Logs Ingestion API (data collection rule, Microsoft Entra ID token from a client secret, a managed identity on App Service / Functions / Container Apps / VMs, or AKS workload identity; tokens are cached and refreshed before they expire), or through the deprecated HTTP Data Collector API (shared key) as a fallback. LOG_ANALYTICS_INGESTION_API selects auto, logs_ingestion, data_collector or both.
  - Falls back to a len(text) // 4 token estimate while no tiktoken encoding is loaded (e.g. offline), without holding up requests. Estimates are marked (tokensEstimated in the record and the log line, "~" in the status).
changelog:
  - 2.7.0 - Azure Monitor Logs Ingestion API (#188). New valves LOG_ANALYTICS_INGESTION_API, LOG_ANALYTICS_DCR_ENDPOINT, LOG_ANALYTICS_DCR_IMMUTABLE_ID, LOG_ANALYTICS_DCR_STREAM_NAME, LOG_ANALYTICS_AUTH_MODE, LOG_ANALYTICS_TENANT_ID, LOG_ANALYTICS_CLIENT_ID, LOG_ANALYTICS_CLIENT_SECRET, LOG_ANALYTICS_AUTHORITY_HOST and LOG_ANALYTICS_INGESTION_SCOPE, all with environment variable defaults; LOG_ANALYTICS_LOG_TYPE can now also be set by environment variable (an installation that set that variable before, when it was ignored, and never saved the valve now sends to that log type). With the default "auto", records go to the Logs Ingestion API as soon as its settings are complete, otherwise through the HTTP Data Collector API exactly as in 2.6.2, which now logs a one-time deprecation warning. Tokens (client secret, managed identity, AKS workload identity) are requested without extra packages, cached per identity and refreshed before they expire; concurrent records share one token request, and a failed refresh keeps using the still-valid token. After a failed token request, or when fresh tokens keep being rejected (401), new token requests pause for 30 s. Endpoint and authority must use https. Any 2xx counts as success (the API answers 204). 400, 401, 403, 404, 413 and 429 are logged once each with a hint (scope, role assignment, DCR ID, stream name, 1 MB limit, Retry-After); records are not retried. "both" writes each record to both APIs for a side-by-side migration.
  - 2.6.2 - Open WebUI >= 0.10 compatibility. outlet() no longer raises a TypeError when Open WebUI runs outlet filters without an event emitter (API requests), so the Log Analytics send and later outlet filters are no longer skipped; the Log Analytics send no longer depends on the status event, a missing chat id falls back to a generated one, and messageId is the message id Open WebUI passes to the outlet (generated if absent). Open WebUI awaits outlet() before it returns an API response, so the Log Analytics send now runs in a background task with its own timeout (10 s, 5 s to connect) instead of AIOHTTP_CLIENT_TIMEOUT (no timeout by default); a slow or unreachable Log Analytics endpoint no longer delays or stalls responses. The record timestamp is UTC with a "Z" suffix, and the new boolean record field tokensEstimated tells estimated token counts from exact ones. SEND_TO_LOG_ANALYTICS="false" (or any value other than 1/true/yes/on) now disables the send instead of enabling it. A tiktoken encoding that cannot be loaded (offline, no cache) or a text it cannot encode no longer aborts the chat; token counts fall back to an estimate. The encoding is loaded in a worker thread, one load per encoding at a time; requests that arrive while it loads estimate instead of waiting, only the request that starts the first load waits (at most 5 s), and a failed load is retried in the background at most every 5 minutes. inlet() and outlet() are correlated through the request's __metadata__ (shared by both on Open WebUI 0.11), so the metrics stay correct when Open WebUI changes the last user message after the inlet (RAG context, legacy code interpreter prompt); the message fingerprint remains the fallback.
  - 2.6.1 - Replaced global variables with per-request fingerprinted storage to mitigate concurrency issues. Uses a hash of user ID, model, and the last user message to correlate inlet/outlet calls. Adds TTL-based cleanup for stale entries. Note: Open WebUI does not expose a guaranteed per-request ID in both inlet and outlet, so edge-case collisions remain theoretically possible when identical messages are sent simultaneously by anonymous users.
"""

import time
import json
import uuid
import asyncio
import hmac
import base64
import hashlib
import datetime
import math
import os
import re
import logging
import aiohttp
from typing import Optional, Any
from urllib.parse import quote, quote_plus
from open_webui.env import SRC_LOG_LEVELS
from cryptography.fernet import Fernet, InvalidToken
import tiktoken
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

# Per-request storage keyed by a fingerprint derived from user, model, and
# last user message. Replaces the original global variables to fix incorrect
# stats under concurrent requests.
_request_data: dict[str, dict] = {}

# Entries older than this (seconds) are pruned to prevent unbounded growth
# when outlet() is never reached (e.g. cancelled requests, crashes).
_STALE_ENTRY_TIMEOUT = 600

# inlet() also puts its entry into __metadata__ under this key. Open WebUI
# 0.11 passes the same metadata dict to inlet() and outlet() of a request (API
# and UI chats), so outlet() finds the entry even when Open WebUI changed the
# last user message after the inlet (RAG context, legacy code interpreter
# prompt) and the fingerprint no longer matches. The fingerprint stays the
# fallback, e.g. for the legacy /api/chat/completed endpoint, which builds a
# new metadata dict.
_METADATA_KEY = "time_token_tracker"

# tiktoken downloads an encoding's BPE file on first use, without a timeout
# and while holding a global lock. The load runs in a worker thread, at most
# one per encoding at a time. Requests that arrive while it runs estimate
# token counts as len(text) // 4 instead of queueing behind it. A failed load
# (offline, no TIKTOKEN_CACHE_DIR cache) is retried in the background at most
# once per _ENCODING_RETRY_INTERVAL seconds. Only the request that starts the
# first load of an encoding waits for it, for at most _ENCODING_FIRST_LOAD_WAIT
# seconds.
_ENCODING_RETRY_INTERVAL = 300
_ENCODING_FIRST_LOAD_WAIT = 5
_encodings: dict[str, Any] = {}
_encoding_loads: dict[str, asyncio.Task] = {}
_encoding_failures: dict[str, float] = {}

# Open WebUI awaits outlet() before it returns the response of an API request,
# so outlet() hands the Log Analytics send to a background task instead of
# waiting for the round trip. The event loop keeps only weak references to
# tasks; this set keeps each send alive until it is done.
_log_analytics_sends: set[asyncio.Task] = set()

# Own timeout for the send. Open WebUI's AIOHTTP_CLIENT_TIMEOUT is unset by
# default, which means no timeout at all.
_LOG_ANALYTICS_TIMEOUT = aiohttp.ClientTimeout(total=10, sock_connect=5)

# Azure Monitor Logs Ingestion API (data collection rules, Microsoft Entra ID).
# A token is requested once per identity (cache key), not once per record:
# concurrent records wait on one lock per key and share the result. A token is
# refreshed _TOKEN_REFRESH_MARGIN seconds before it expires (at the latest at
# half its lifetime). When a refresh fails, the still-valid cached token keeps
# being used until _TOKEN_EXPIRY_SKEW seconds (at most a quarter of its
# lifetime) before it expires, so a short Entra ID or proxy outage loses no
# records. A failed token request, or a second 401 in a row, blocks new token
# requests for that key for _TOKEN_RETRY_INTERVAL seconds; records without a
# usable token are dropped meanwhile instead of sending one token request each.
# These dicts hold only access tokens, timestamps and keys: never the client
# secret, the IDENTITY_HEADER or the federated assertion, nor a hash of them.
_LOGS_INGESTION_API_VERSION = "2023-01-01"
_DEFAULT_AUTHORITY_HOST = "https://login.microsoftonline.com"
_DEFAULT_INGESTION_SCOPE = "https://monitor.azure.com/.default"
_IMDS_AUTHORITY = "http://169.254.169.254"
_TOKEN_REFRESH_MARGIN = 300
_TOKEN_EXPIRY_SKEW = 60
_TOKEN_RETRY_INTERVAL = 30
# key -> (token, refresh_at, usable_until), time.monotonic() based
_ingestion_tokens: dict[tuple, tuple[str, float, float]] = {}
_ingestion_token_locks: dict[tuple, asyncio.Lock] = {}
_ingestion_token_failures: dict[tuple, float] = {}  # key -> monotonic failure time
_ingestion_token_rejected: set[tuple] = set()  # last token got a 401, no 2xx since
# One-time warnings (deprecation, incomplete settings): once per process.
_log_analytics_warnings: set[str] = set()

_INGESTION_API_MODES = ("auto", "logs_ingestion", "data_collector", "both")
_AUTH_MODES = ("client_secret", "managed_identity")
_DCR_IMMUTABLE_ID = re.compile(r"dcr-[0-9a-fA-F]{32}")
_JWT_BEARER = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
_DATA_COLLECTOR_DEPRECATION = (
    "Log Analytics: sending through the HTTP Data Collector API, which Microsoft "
    "deprecated (support ended on 2026-09-14). Set up the Logs Ingestion API "
    "(LOG_ANALYTICS_DCR_ENDPOINT, LOG_ANALYTICS_DCR_IMMUTABLE_ID and a Microsoft "
    "Entra ID identity), see docs/setup-azure-log-analytics.md. This warning is "
    "logged once."
)
# Token sources (managed identity is detected from the environment).
_TOKEN_SOURCE_LABELS = {
    "client_secret": "client secret",
    "app_service": "managed identity (App Service)",
    "imds": "managed identity (IMDS)",
    "workload_identity": "workload identity",
    "service_fabric": "managed identity (Service Fabric)",
    "azure_arc": "managed identity (Azure Arc)",
    "cloud_shell": "managed identity (Cloud Shell / Azure ML)",
    "identity_binding": "workload identity (AKS identity binding)",
}
_UNSUPPORTED_TOKEN_SOURCES = {
    "service_fabric": "managed identity on Service Fabric",
    "azure_arc": "managed identity on Azure Arc",
    "cloud_shell": "managed identity on Cloud Shell / Azure ML",
    "identity_binding": "AKS identity bindings (AZURE_KUBERNETES_TOKEN_PROXY)",
}
# Hints for Microsoft Entra ID error codes (error_codes[0], never the text).
_AADSTS_HINTS = {
    7000215: "invalid client secret: use the secret's value, not its ID "
    "(LOG_ANALYTICS_CLIENT_SECRET).",
    7000222: "the client secret has expired: create a new one.",
    700016: "application not found in the tenant: check LOG_ANALYTICS_CLIENT_ID, "
    "LOG_ANALYTICS_TENANT_ID and LOG_ANALYTICS_AUTHORITY_HOST.",
    90002: "tenant not found: check LOG_ANALYTICS_TENANT_ID and "
    "LOG_ANALYTICS_AUTHORITY_HOST (cloud).",
    70011: "check LOG_ANALYTICS_INGESTION_SCOPE (it must match the cloud).",
    500011: "check LOG_ANALYTICS_INGESTION_SCOPE (it must match the cloud).",
    700212: "the federated token has the wrong audience (direct federation needs "
    "api://AzureADTokenExchange).",
}
_IMDS_UNREACHABLE = (
    "no managed identity endpoint reachable (not running on Azure, or no identity "
    "assigned)"
)
# Off Azure, 169.254.169.254 usually hangs until sock_connect instead of refusing.
_CONNECT_ERRORS = (
    aiohttp.ClientConnectorError,
    getattr(aiohttp, "ConnectionTimeoutError", aiohttp.ServerTimeoutError),
)


class _TokenError(Exception):
    """A token request failed; the message is safe to log (no secret, no token)."""


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _normalize_url(value: str) -> str:
    """Strip, drop trailing slashes and add https:// when there is no scheme."""
    value = value.strip().rstrip("/")
    if value and "://" not in value:
        value = "https://" + value
    return value


def _first_line(text: Any) -> str:
    lines = str(text or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def _redact(text: str, secrets) -> str:
    """Replace every non-empty secret (also URL-encoded) with ***."""
    forms = set()
    for secret in secrets:
        if secret:
            forms.update(
                {secret, quote(secret), quote(secret, safe=""), quote_plus(secret)}
            )
    for form in sorted(forms, key=len, reverse=True):
        text = text.replace(form, "***")
    return text


def _token_source(cfg: dict) -> tuple[str, str, str]:
    """(source, effective tenant, effective client id) for this request.

    Managed identity is detected from the environment in the order
    azure-identity uses, on every call (the platform may change it).
    """
    if cfg["auth_mode"] == "client_secret":
        return "client_secret", cfg["tenant"], cfg["client_id"]
    env = os.environ
    if env.get("IDENTITY_ENDPOINT"):
        if env.get("IDENTITY_HEADER"):
            if env.get("IDENTITY_SERVER_THUMBPRINT"):
                return "service_fabric", "", cfg["client_id"]
            return "app_service", "", cfg["client_id"]
        if env.get("IMDS_ENDPOINT"):
            return "azure_arc", "", cfg["client_id"]
    elif env.get("MSI_ENDPOINT"):
        return "cloud_shell", "", cfg["client_id"]
    elif env.get("AZURE_FEDERATED_TOKEN_FILE"):
        tenant = cfg["tenant"] or env.get("AZURE_TENANT_ID", "").strip()
        client_id = cfg["client_id"] or env.get("AZURE_CLIENT_ID", "").strip()
        if env.get("AZURE_KUBERNETES_TOKEN_PROXY"):
            return "identity_binding", tenant, client_id
        return "workload_identity", tenant, client_id
    return "imds", "", cfg["client_id"]


def _token_lifetime(payload: dict) -> float:
    """Seconds until the token expires (expires_in, else expires_on)."""
    try:
        if payload.get("expires_in") is not None:
            lifetime = float(payload["expires_in"])
        else:
            lifetime = float(payload.get("expires_on")) - time.time()
    except (TypeError, ValueError):
        lifetime = math.nan
    if not (math.isfinite(lifetime) and lifetime >= 1):
        raise _TokenError("token response without a usable expiry")
    return lifetime


def _build_request_key(body: dict, user: Optional[dict] = None) -> str:
    """
    Build a storage key that is unique per request and consistent between
    inlet and outlet.

    Open WebUI reconstructs the body dict between inlet and outlet and only
    exposes chat_id in outlet, so we cannot rely on a single ID field.
    Instead we hash (user_id, model, number_of_user_messages,
    last_user_message_content). Open WebUI can change the last user message
    after the inlet (RAG context, legacy code interpreter prompt), so outlet()
    uses this key only when it finds no inlet entry in __metadata__.

    Collisions are only possible if the same user sends the exact same
    message at the exact same conversation depth to the same model
    concurrently, which is not a realistic scenario.
    """
    model = body.get("model", "")
    user_id = user.get("id", "") if user else ""

    messages = body.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == "user"]
    num_user_messages = len(user_messages)

    last_user_content = ""
    if user_messages:
        content = user_messages[-1].get("content", "")
        if isinstance(content, str):
            last_user_content = content
        elif content is not None:
            last_user_content = str(content)

    raw = f"{user_id}:{model}:{num_user_messages}:{last_user_content}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _metadata_key(filter_id: Optional[str]) -> str:
    """Key of the inlet entry in __metadata__, per installed copy of the filter."""
    return f"{_METADATA_KEY}:{filter_id}" if filter_id else _METADATA_KEY


def _prune_stale_entries(self) -> None:
    """Remove entries older than _STALE_ENTRY_TIMEOUT to prevent unbounded growth."""
    now = time.time()
    stale_keys = [
        k
        for k, v in _request_data.items()
        if now - v.get("start_time", now) > _STALE_ENTRY_TIMEOUT
    ]
    if stale_keys:
        self.log.info(f"Pruning {len(stale_keys)} stale entries from _request_data")
    for k in stale_keys:
        _request_data.pop(k, None)


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


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        CALCULATE_ALL_MESSAGES: bool = Field(
            default=True,
            description="If true, calculate tokens for all messages. If false, only use the last user and assistant messages.",
        )
        SHOW_AVERAGE_TOKENS: bool = Field(
            default=True,
            description="Show average tokens per message (only used if CALCULATE_ALL_MESSAGES is true).",
        )
        SHOW_RESPONSE_TIME: bool = Field(
            default=True, description="Show the response time."
        )
        SHOW_TOKEN_COUNT: bool = Field(
            default=True, description="Show the token count."
        )
        SHOW_TOKENS_PER_SECOND: bool = Field(
            default=True, description="Show tokens per second for the response."
        )
        SEND_TO_LOG_ANALYTICS: bool = Field(
            default=os.getenv("SEND_TO_LOG_ANALYTICS", "false").strip().lower()
            in ("1", "true", "yes", "on"),
            description="Send logs to Azure Log Analytics workspace",
        )
        LOG_ANALYTICS_WORKSPACE_ID: str = Field(
            default=os.getenv("LOG_ANALYTICS_WORKSPACE_ID", ""),
            description="Azure Log Analytics Workspace ID",
        )
        LOG_ANALYTICS_SHARED_KEY: EncryptedStr = Field(
            default=os.getenv("LOG_ANALYTICS_SHARED_KEY", ""),
            description="Azure Log Analytics Workspace Shared Key",
            json_schema_extra={"input": {"type": "password"}},
        )
        LOG_ANALYTICS_LOG_TYPE: str = Field(
            default=os.getenv("LOG_ANALYTICS_LOG_TYPE", "OpenWebuiMetrics"),
            description="Log Analytics log type name (HTTP Data Collector API); "
            "also the base of the default stream name Custom-<log type>_CL.",
        )
        LOG_ANALYTICS_INGESTION_API: str = Field(
            default=os.getenv("LOG_ANALYTICS_INGESTION_API", "auto"),
            description="auto (Logs Ingestion API once its settings are complete, "
            "otherwise the deprecated HTTP Data Collector API), logs_ingestion, "
            "data_collector or both (side-by-side migration).",
        )
        LOG_ANALYTICS_DCR_ENDPOINT: str = Field(
            default=os.getenv("LOG_ANALYTICS_DCR_ENDPOINT", ""),
            description="Logs ingestion endpoint of the data collection rule "
            "(https://<dcr>-<xxxx>-<region>.logs.z1.ingest.monitor.azure.com) or "
            "a data collection endpoint (required for private link).",
        )
        LOG_ANALYTICS_DCR_IMMUTABLE_ID: str = Field(
            default=os.getenv("LOG_ANALYTICS_DCR_IMMUTABLE_ID", ""),
            description="Immutable ID of the data collection rule (dcr-...).",
        )
        LOG_ANALYTICS_DCR_STREAM_NAME: str = Field(
            default=os.getenv("LOG_ANALYTICS_DCR_STREAM_NAME", ""),
            description="Stream name in the data collection rule. Empty: "
            "Custom-<LOG_ANALYTICS_LOG_TYPE>_CL (e.g. Custom-OpenWebuiMetrics_CL).",
        )
        LOG_ANALYTICS_AUTH_MODE: str = Field(
            default=os.getenv("LOG_ANALYTICS_AUTH_MODE", "client_secret"),
            description="client_secret (app registration) or managed_identity "
            "(App Service, Functions, Container Apps, VMs, AKS workload identity).",
        )
        LOG_ANALYTICS_TENANT_ID: str = Field(
            default=os.getenv("LOG_ANALYTICS_TENANT_ID", ""),
            description="Microsoft Entra ID tenant (GUID or domain). Workload "
            "identity falls back to AZURE_TENANT_ID.",
        )
        LOG_ANALYTICS_CLIENT_ID: str = Field(
            default=os.getenv("LOG_ANALYTICS_CLIENT_ID", ""),
            description="Application (client) ID of the app registration; with "
            "managed_identity the client ID of a user-assigned identity (empty: "
            "system-assigned). Workload identity falls back to AZURE_CLIENT_ID.",
        )
        LOG_ANALYTICS_CLIENT_SECRET: EncryptedStr = Field(
            default=os.getenv("LOG_ANALYTICS_CLIENT_SECRET", ""),
            description="Client secret value (not the secret ID) of the app "
            "registration.",
            json_schema_extra={"input": {"type": "password"}},
        )
        LOG_ANALYTICS_AUTHORITY_HOST: str = Field(
            default=os.getenv(
                "LOG_ANALYTICS_AUTHORITY_HOST", "https://login.microsoftonline.com"
            ),
            description="Microsoft Entra ID authority: "
            "https://login.microsoftonline.us (US Government), "
            "https://login.partner.microsoftonline.cn (21Vianet).",
        )
        LOG_ANALYTICS_INGESTION_SCOPE: str = Field(
            default=os.getenv(
                "LOG_ANALYTICS_INGESTION_SCOPE", "https://monitor.azure.com/.default"
            ),
            description="Token scope: https://monitor.azure.us/.default (US "
            "Government), https://monitor.azure.cn/.default (21Vianet).",
        )

    def __init__(self):
        self.name = "Time Token Tracker"
        self.valves = self.Valves()
        self.log = logging.getLogger("time_token_tracker")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

    def _build_signature(self, date, content_length, method, content_type, resource):
        """Build the signature for Log Analytics authentication."""
        x_headers = "x-ms-date:" + date
        string_to_hash = (
            method
            + "\n"
            + str(content_length)
            + "\n"
            + content_type
            + "\n"
            + x_headers
            + "\n"
            + resource
        )
        bytes_to_hash = string_to_hash.encode("utf-8")
        decoded_key = base64.b64decode(
            EncryptedStr.decrypt(self.valves.LOG_ANALYTICS_SHARED_KEY)
        )
        encoded_hash = base64.b64encode(
            hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest()
        ).decode("utf-8")
        authorization = (
            f"SharedKey {self.valves.LOG_ANALYTICS_WORKSPACE_ID}:{encoded_hash}"
        )
        return authorization

    def _warn_once(self, text: str, key: Optional[str] = None) -> None:
        """Log a WARNING once per process (per ``key``, default the text)."""
        key = key or text
        if key not in _log_analytics_warnings:
            _log_analytics_warnings.add(key)
            self.log.warning(text)

    def _ingestion_api_mode(self) -> str:
        """LOG_ANALYTICS_INGESTION_API normalized; unknown values mean auto."""
        raw = str(self.valves.LOG_ANALYTICS_INGESTION_API or "").strip()
        mode = raw.lower().replace("-", "_") or "auto"
        if mode not in _INGESTION_API_MODES:
            self._warn_once(
                f"Log Analytics: unknown LOG_ANALYTICS_INGESTION_API value "
                f"{raw[:32]!r} (use auto, logs_ingestion, data_collector or both); "
                f"using auto"
            )
            mode = "auto"
        return mode

    def _logs_ingestion_config(self) -> tuple[dict, list]:
        """
        Snapshot of the Logs Ingestion settings of this request, normalized,
        and the names of the missing (or invalid) valves.

        The background task must not read self.valves (Open WebUI swaps them
        per request). client_secret stays the stored, still encrypted value;
        it is decrypted only for the token request. Without WEBUI_SECRET_KEY,
        or for an environment default, it is plaintext: never log cfg as a
        whole, only its non-secret fields.
        """
        v = self.valves

        def text(value) -> str:
            return str(value or "").strip()

        log_type = text(v.LOG_ANALYTICS_LOG_TYPE)
        auth_raw = text(v.LOG_ANALYTICS_AUTH_MODE)
        cfg = {
            "endpoint": _normalize_url(text(v.LOG_ANALYTICS_DCR_ENDPOINT)),
            "dcr_id": text(v.LOG_ANALYTICS_DCR_IMMUTABLE_ID),
            "stream": text(v.LOG_ANALYTICS_DCR_STREAM_NAME)
            or (f"Custom-{log_type}_CL" if log_type else ""),
            "auth_mode": auth_raw.lower().replace("-", "_") or "client_secret",
            "authority": _normalize_url(text(v.LOG_ANALYTICS_AUTHORITY_HOST))
            or _DEFAULT_AUTHORITY_HOST,
            "tenant": text(v.LOG_ANALYTICS_TENANT_ID),
            "client_id": text(v.LOG_ANALYTICS_CLIENT_ID),
            "client_secret": text(v.LOG_ANALYTICS_CLIENT_SECRET),
            "scope": text(v.LOG_ANALYTICS_INGESTION_SCOPE) or _DEFAULT_INGESTION_SCOPE,
        }
        missing = []
        if not cfg["endpoint"]:
            missing.append("LOG_ANALYTICS_DCR_ENDPOINT")
        elif not cfg["endpoint"].lower().startswith("https://"):
            missing.append("LOG_ANALYTICS_DCR_ENDPOINT (must use https)")
        if not cfg["dcr_id"]:
            missing.append("LOG_ANALYTICS_DCR_IMMUTABLE_ID")
        if not cfg["stream"]:
            missing.append("LOG_ANALYTICS_DCR_STREAM_NAME")
        if cfg["auth_mode"] not in _AUTH_MODES:
            missing.append(f"LOG_ANALYTICS_AUTH_MODE (unknown value {auth_raw[:32]!r})")
        if not cfg["authority"].lower().startswith("https://"):
            missing.append("LOG_ANALYTICS_AUTHORITY_HOST (must use https)")
        if cfg["auth_mode"] == "client_secret":
            for name, field in (
                ("LOG_ANALYTICS_TENANT_ID", "tenant"),
                ("LOG_ANALYTICS_CLIENT_ID", "client_id"),
                ("LOG_ANALYTICS_CLIENT_SECRET", "client_secret"),
            ):
                if not cfg[field]:
                    missing.append(name)
        return cfg, missing

    def _logs_ingestion_partial(self) -> bool:
        """True when any valve that only the Logs Ingestion API uses is set."""
        v = self.valves
        names = (
            "LOG_ANALYTICS_DCR_ENDPOINT",
            "LOG_ANALYTICS_DCR_IMMUTABLE_ID",
            "LOG_ANALYTICS_DCR_STREAM_NAME",
            "LOG_ANALYTICS_TENANT_ID",
            "LOG_ANALYTICS_CLIENT_ID",
            "LOG_ANALYTICS_CLIENT_SECRET",
        )
        auth = str(v.LOG_ANALYTICS_AUTH_MODE or "").strip().lower().replace("-", "_")
        if auth not in ("", "client_secret"):
            return True
        return any(str(getattr(v, name) or "").strip() for name in names)

    def _send_to_log_analytics(self, data, chat_id: str, message_id: str) -> bool:
        """
        Send the record through the API(s) LOG_ANALYTICS_INGESTION_API selects,
        each in a background task. Returns True if at least one send started.

        The settings are read here, before outlet() returns, so the background
        tasks get the valve values of this request.
        """
        if not self.valves.SEND_TO_LOG_ANALYTICS:
            self.log.debug("Log Analytics send skipped: not configured")
            return False

        mode = self._ingestion_api_mode()
        dc_missing = [
            name
            for name in ("LOG_ANALYTICS_WORKSPACE_ID", "LOG_ANALYTICS_SHARED_KEY")
            if not getattr(self.valves, name)
        ]
        li_cfg, li_missing, li_partial = None, [], False
        if mode != "data_collector":
            try:
                li_cfg, li_missing = self._logs_ingestion_config()
                li_partial = self._logs_ingestion_partial()
            except Exception as e:
                if mode != "auto":
                    raise
                # A bug in the new code must not stop a 2.6.2 configuration.
                self._warn_once(
                    f"Log Analytics: could not evaluate the Logs Ingestion API "
                    f"settings ({type(e).__name__}); using the HTTP Data Collector API"
                )
                li_cfg, li_missing, li_partial = None, ["(error)"], False
        li_ready = li_cfg is not None and not li_missing
        dc_ready = not dc_missing

        send_li = send_dc = False
        if mode == "data_collector":
            send_dc = True
        elif mode == "logs_ingestion":
            send_li = li_ready
            if not li_ready:
                self._warn_once(
                    f"Log Analytics: LOG_ANALYTICS_INGESTION_API=logs_ingestion, but "
                    f"the Logs Ingestion API is not fully configured (missing: "
                    f"{', '.join(li_missing)}); records are not sent"
                )
        elif mode == "both":
            send_li, send_dc = li_ready, dc_ready
            if not li_ready and not dc_ready:
                self._warn_once(
                    f"Log Analytics: LOG_ANALYTICS_INGESTION_API=both, but neither API "
                    f"is fully configured (missing: "
                    f"{', '.join(li_missing + dc_missing)}); records are not sent"
                )
            elif not li_ready or not dc_ready:
                api = (
                    "Logs Ingestion API" if not li_ready else "HTTP Data Collector API"
                )
                self._warn_once(
                    f"Log Analytics: LOG_ANALYTICS_INGESTION_API=both, but the {api} "
                    f"is not fully configured (missing: "
                    f"{', '.join(li_missing or dc_missing)}); sending only through "
                    f"the other one"
                )
        else:  # auto
            send_li = li_ready
            send_dc = not li_ready
            if not li_ready and li_partial:
                fallback = (
                    "using the HTTP Data Collector API"
                    if dc_ready
                    else "records are not sent"
                )
                self._warn_once(
                    f"Log Analytics: the Logs Ingestion API is not fully configured "
                    f"(missing: {', '.join(li_missing)}); {fallback}"
                )

        sent = False
        if send_li:
            sent = self._send_to_logs_ingestion(li_cfg, data, chat_id, message_id)
        if send_dc:
            sent = self._send_to_data_collector(data, chat_id, message_id) or sent
        return sent

    def _send_to_data_collector(self, data, chat_id: str, message_id: str) -> bool:
        """
        Sign the request now and send it to the HTTP Data Collector API in a
        background task. Returns False when sending is not configured.

        The request is built here, before outlet() returns, so the background
        task gets the valve values of this request and only the signature,
        never the decrypted key.
        """
        if (
            not self.valves.SEND_TO_LOG_ANALYTICS
            or not self.valves.LOG_ANALYTICS_WORKSPACE_ID
            or not self.valves.LOG_ANALYTICS_SHARED_KEY
        ):
            self.log.debug("Log Analytics send skipped: not configured")
            return False

        self.log.debug(
            f"Sending to Log Analytics (workspace={self.valves.LOG_ANALYTICS_WORKSPACE_ID}, "
            f"log_type={self.valves.LOG_ANALYTICS_LOG_TYPE})"
        )

        method = "POST"
        content_type = "application/json"
        resource = "/api/logs"
        rfc1123date = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        )
        content_length = len(json.dumps(data))

        signature = self._build_signature(
            rfc1123date, content_length, method, content_type, resource
        )

        uri = f"https://{self.valves.LOG_ANALYTICS_WORKSPACE_ID}.ods.opinsights.azure.com{resource}?api-version=2016-04-01"

        headers = {
            "Content-Type": content_type,
            "Authorization": signature,
            "Log-Type": self.valves.LOG_ANALYTICS_LOG_TYPE,
            "x-ms-date": rfc1123date,
            "time-generated-field": "timestamp",
        }

        self._warn_once(_DATA_COLLECTOR_DEPRECATION)
        task = asyncio.create_task(
            self._post_to_log_analytics(uri, headers, data, chat_id, message_id)
        )
        _log_analytics_sends.add(task)
        task.add_done_callback(_log_analytics_sends.discard)
        return True

    async def _post_to_log_analytics(
        self, uri: str, headers: dict, data, chat_id: str, message_id: str
    ) -> bool:
        """POST a signed record to Azure Log Analytics (runs as a background task)."""
        session = None
        response = None

        try:
            session = aiohttp.ClientSession(
                trust_env=True, timeout=_LOG_ANALYTICS_TIMEOUT
            )

            response = await session.request(
                method="POST",
                url=uri,
                json=data,
                headers=headers,
            )

            if response.status == 200:
                self.log.info(
                    f"Log Analytics data sent successfully "
                    f"(chat={chat_id}, message={message_id})"
                )
                return True
            else:
                response_text = await response.text()
                self.log.error(
                    f"Error sending to Log Analytics: {response.status} - {response_text}"
                )

        except Exception as e:
            # str() of a timeout is empty: name the exception type.
            self.log.error(
                f"Exception when sending to Log Analytics asynchronously: "
                f"{type(e).__name__}: {e}"
            )
        finally:
            await cleanup_response(response, session)

        self.log.warning(
            f"Failed to send data to Log Analytics "
            f"(chat={chat_id}, message={message_id})"
        )
        return False

    def _send_to_logs_ingestion(
        self, cfg: dict, data, chat_id: str, message_id: str
    ) -> bool:
        """Send a record to the Logs Ingestion API in a background task."""
        if not _DCR_IMMUTABLE_ID.fullmatch(cfg["dcr_id"]):
            # The value is not logged; once per distinct value.
            self._warn_once(
                "Log Analytics: LOG_ANALYTICS_DCR_IMMUTABLE_ID does not look like an "
                "immutable ID (dcr- followed by 32 hex characters); copy immutableId "
                "from the DCR's JSON view, not the DCR name or resource ID",
                key=f"dcr-immutable-id:{cfg['dcr_id']}",
            )
        self.log.debug(
            f"Sending to the Logs Ingestion API (endpoint={cfg['endpoint']}, "
            f"dcr={cfg['dcr_id']}, stream={cfg['stream']}, auth={cfg['auth_mode']})"
        )
        task = asyncio.create_task(
            self._post_to_logs_ingestion(cfg, data, chat_id, message_id)
        )
        _log_analytics_sends.add(task)
        task.add_done_callback(_log_analytics_sends.discard)
        return True

    @staticmethod
    def _ingestion_hint(status: int, cfg: dict, retry_after: Optional[str]) -> str:
        """Actionable hint for an HTTP error of the Logs Ingestion API."""
        if status == 400:
            return (
                "the record does not match the stream declaration of the DCR "
                "(columns and types)."
            )
        if status == 401:
            return (
                "the token was rejected; check that LOG_ANALYTICS_INGESTION_SCOPE "
                "matches the cloud of LOG_ANALYTICS_DCR_ENDPOINT."
            )
        if status == 403:
            return (
                "the identity needs the Monitoring Metrics Publisher role on the "
                "data collection rule; a new role assignment can take up to 30 "
                "minutes to take effect."
            )
        if status == 404:
            return (
                f"check LOG_ANALYTICS_DCR_ENDPOINT, LOG_ANALYTICS_DCR_IMMUTABLE_ID "
                f"({cfg['dcr_id']}) and LOG_ANALYTICS_DCR_STREAM_NAME ({cfg['stream']})."
            )
        if status == 413:
            return "the record exceeds the limit of 1 MB per call."
        if status == 429:
            # Retry-After is seconds or an HTTP date: shown as it came.
            value = " ".join(str(retry_after or "").split())[:64] or "not given"
            return (
                f"throttled (per DCR: 12,000 requests or 2 GB per minute); "
                f"Retry-After: {value}. The record is dropped, not retried."
            )
        return ""

    async def _post_to_logs_ingestion(
        self, cfg: dict, data, chat_id: str, message_id: str
    ) -> bool:
        """
        POST a record to the Logs Ingestion API (runs as a background task).
        Reads only cfg, never self.valves. Logs one line per failed record.
        """
        got = await self._get_ingestion_token(cfg, chat_id, message_id)
        if got is None:
            return False  # already logged
        token, key = got
        uri = (
            f"{cfg['endpoint']}/dataCollectionRules/{quote(cfg['dcr_id'], safe='')}"
            f"/streams/{quote(cfg['stream'], safe='')}"
            f"?api-version={_LOGS_INGESTION_API_VERSION}"
        )
        request_id = str(uuid.uuid4())
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-ms-client-request-id": request_id,
        }
        session = None
        response = None
        try:
            session = aiohttp.ClientSession(
                trust_env=True, timeout=_LOG_ANALYTICS_TIMEOUT
            )
            response = await session.request(
                method="POST", url=uri, json=data, headers=headers
            )
            status = response.status
            if 200 <= status < 300:
                _ingestion_token_rejected.discard(key)
                self.log.info(
                    f"Log Analytics data sent via the Logs Ingestion API "
                    f"(chat={chat_id}, message={message_id})"
                )
                return True
            body = await response.text()
            code = response.headers.get("x-ms-error-code", "")
            detail = ""
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                code = code or str(error.get("code") or "")
                detail = str(error.get("message") or "")
            detail = " ".join(_redact(detail or body, [token]).split())[:300]
            code = " ".join(_redact(code, [token]).split())[:100]
            hint = self._ingestion_hint(
                status, cfg, response.headers.get("Retry-After")
            )
            if status == 401:
                self._ingestion_token_unauthorized(key, token)
            head = f"{status} {code}" if code else str(status)
            self.log.error(
                f"Error sending to Logs Ingestion API: {head}: "
                f"{hint + ' ' if hint else ''}Response: {detail} "
                f"(request={request_id}, chat={chat_id}, message={message_id})"
            )
        except Exception as e:
            # str() of a timeout is empty: name the exception type.
            reason = _redact(f"{type(e).__name__}: {e}", [token])
            self.log.error(
                f"Exception when sending to Logs Ingestion API: {reason} "
                f"(chat={chat_id}, message={message_id})"
            )
        finally:
            await cleanup_response(response, session)
        return False

    @staticmethod
    def _ingestion_token_unauthorized(key: tuple, token: str) -> None:
        """
        A 401 from the ingestion endpoint: drop the cached token (only once
        for concurrent 401s of the same token). A token fetched right after a
        401 that is rejected too starts the token back-off, so a persistent
        401 (wrong scope or cloud) cannot cause one token request per record.
        """
        cached = _ingestion_tokens.get(key)
        if not cached or cached[0] != token:
            return
        _ingestion_tokens.pop(key, None)
        if key in _ingestion_token_rejected:
            _ingestion_token_failures[key] = time.monotonic()
        else:
            _ingestion_token_rejected.add(key)

    async def _get_ingestion_token(
        self, cfg: dict, chat_id: str, message_id: str
    ) -> Optional[tuple[str, tuple]]:
        """
        (token, cache key) for the identity in cfg, or None (logged once).
        One token request per key at a time; see the module comment.
        """
        source, tenant, client_id = _token_source(cfg)
        key = (
            cfg["auth_mode"],
            source,
            cfg["authority"],
            tenant,
            client_id,
            cfg["scope"],
        )
        label = _TOKEN_SOURCE_LABELS.get(source, source)
        cached = _ingestion_tokens.get(key)
        if cached and time.monotonic() < cached[1]:
            return cached[0], key
        lock = _ingestion_token_locks.setdefault(key, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            cached = _ingestion_tokens.get(key)  # filled while we waited
            if cached and now < cached[1]:
                return cached[0], key
            usable = cached[0] if cached and now < cached[2] else None
            failed_at = _ingestion_token_failures.get(key)
            if failed_at is not None and now - failed_at < _TOKEN_RETRY_INTERVAL:
                if usable:
                    return usable, key
                self.log.warning(
                    f"Log Analytics record dropped: no usable token for the Logs "
                    f"Ingestion API ({label}); the last token request failed or its "
                    f"token was rejected {now - failed_at:.0f}s ago, next token "
                    f"request in {_TOKEN_RETRY_INTERVAL - (now - failed_at):.0f}s "
                    f"(chat={chat_id}, message={message_id})"
                )
                return None
            try:
                token, lifetime = await self._request_token(
                    cfg, source, tenant, client_id
                )
            except Exception as e:  # always a _TokenError, unless there is a bug
                _ingestion_token_failures[key] = time.monotonic()
                safe = str(e) if isinstance(e, _TokenError) else type(e).__name__
                keep = (
                    f", keeping the cached token for "
                    f"{cached[2] - time.monotonic():.0f}s"
                    if usable
                    else ""
                )
                self.log.error(
                    f"Could not get a Microsoft Entra ID token for the Logs Ingestion "
                    f"API ({label}){keep}: {safe} (chat={chat_id}, message={message_id})"
                )
                return (usable, key) if usable else None
            _ingestion_token_failures.pop(key, None)
            now = time.monotonic()
            _ingestion_tokens[key] = (
                token,
                now + max(lifetime - _TOKEN_REFRESH_MARGIN, lifetime / 2),
                now + lifetime - min(_TOKEN_EXPIRY_SKEW, lifetime / 4),
            )
            return token, key

    @staticmethod
    def _token_error_text(
        source: str, status: int, payload: Any, body: str, secrets: list
    ) -> str:
        """Loggable text of a failed token response (redacted, then truncated)."""
        if not isinstance(payload, dict):
            return f"{status} response: {_redact(_first_line(body), secrets)[:200]}"
        error = " ".join(str(payload.get("error") or "error").split())[:100]
        description = _redact(_first_line(payload.get("error_description")), secrets)
        description = description[:200]
        if source not in ("client_secret", "workload_identity"):
            return f"{status} {error} - {description}"
        codes = payload.get("error_codes")
        try:
            code = int(codes[0]) if isinstance(codes, list) and codes else None
        except (TypeError, ValueError):
            code = None
        head = f"{status} {error}" + (f" AADSTS{code}" if code is not None else "")
        hint = _AADSTS_HINTS.get(code) if code is not None else None
        text = f"{head}: " + (f"{hint} " if hint else "") + f"Detail: {description}"
        ids = [
            f"{name}={str(payload[name])[:64]}"
            for name in ("trace_id", "correlation_id")
            if payload.get(name)
        ]
        if ids:
            text += f" ({', '.join(ids)})"
        return text

    async def _request_token(
        self, cfg: dict, source: str, tenant: str, client_id: str
    ) -> tuple[str, float]:
        """
        Request an access token: (token, lifetime in seconds).

        Reads only cfg and the environment, never self.valves. Every failure
        leaves as a _TokenError with a redacted text: only this method knows
        the plaintext client secret, IDENTITY_HEADER and federated assertion,
        which are read here, used once and never stored or logged.
        """
        secrets: list = []
        session = None
        response = None
        try:
            if source in _UNSUPPORTED_TOKEN_SOURCES:
                raise _TokenError(
                    f"{_UNSUPPORTED_TOKEN_SOURCES[source]} is not supported; use "
                    f"LOG_ANALYTICS_AUTH_MODE=client_secret"
                )
            scope = cfg["scope"]
            resource = (
                scope[: -len("/.default")] if scope.endswith("/.default") else scope
            )
            if source in ("client_secret", "workload_identity"):
                form = {
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "scope": scope,
                }
                if source == "client_secret":
                    stored = cfg["client_secret"]
                    secret = EncryptedStr.decrypt(stored)
                    if stored.startswith("encrypted:") and (
                        EncryptedStr._get_encryption_key() is None
                        or secret.startswith("encrypted:")
                    ):
                        raise _TokenError(
                            "LOG_ANALYTICS_CLIENT_SECRET could not be decrypted (was "
                            "WEBUI_SECRET_KEY changed or removed?); enter the client "
                            "secret again"
                        )
                    secrets.append(secret)
                    form["client_secret"] = secret
                else:
                    if not tenant or not client_id:
                        raise _TokenError(
                            "workload identity needs a tenant and client ID "
                            "(LOG_ANALYTICS_TENANT_ID / AZURE_TENANT_ID, "
                            "LOG_ANALYTICS_CLIENT_ID / AZURE_CLIENT_ID)"
                        )
                    token_file = os.environ.get("AZURE_FEDERATED_TOKEN_FILE", "")
                    try:
                        assertion = (
                            await asyncio.to_thread(_read_text, token_file)
                        ).strip()
                    except OSError as e:  # the text names the path, not the content
                        raise _TokenError(
                            f"AZURE_FEDERATED_TOKEN_FILE could not be read "
                            f"({type(e).__name__}: {e})"
                        ) from None
                    except Exception as e:
                        raise _TokenError(
                            f"AZURE_FEDERATED_TOKEN_FILE could not be read "
                            f"({type(e).__name__})"
                        ) from None
                    if not assertion:
                        raise _TokenError("AZURE_FEDERATED_TOKEN_FILE is empty")
                    secrets.append(assertion)
                    form["client_assertion_type"] = _JWT_BEARER
                    form["client_assertion"] = assertion
                url = f"{cfg['authority']}/{quote(tenant, safe='')}/oauth2/v2.0/token"
                session = aiohttp.ClientSession(
                    trust_env=True, timeout=_LOG_ANALYTICS_TIMEOUT
                )
                # data=dict: form-encoded (application/x-www-form-urlencoded)
                response = await session.post(url, data=form)
            else:
                params = {"resource": resource}
                if client_id:
                    params["client_id"] = client_id
                if source == "app_service":
                    # Rotated by the platform: read it now, never cache it.
                    identity_header = os.environ.get("IDENTITY_HEADER", "")
                    secrets.append(identity_header)
                    url = os.environ.get("IDENTITY_ENDPOINT", "")
                    params["api-version"] = "2019-08-01"
                    headers = {"X-IDENTITY-HEADER": identity_header}
                else:  # imds
                    host = os.environ.get("AZURE_POD_IDENTITY_AUTHORITY_HOST") or ""
                    host = host.strip().rstrip("/") or _IMDS_AUTHORITY
                    url = f"{host}/metadata/identity/oauth2/token"
                    params["api-version"] = "2018-02-01"
                    headers = {"Metadata": "true"}
                # Local endpoints: never through a proxy.
                session = aiohttp.ClientSession(
                    trust_env=False, timeout=_LOG_ANALYTICS_TIMEOUT
                )
                response = await session.get(url, params=params, headers=headers)
            status = response.status
            body = await response.text()
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            if not 200 <= status < 300:
                text = self._token_error_text(source, status, payload, body, secrets)
                raise _TokenError(_redact(text, secrets))
            if not isinstance(payload, dict):
                raise _TokenError(f"{status} response is not JSON")
            token = payload.get("access_token")
            if not (
                isinstance(token, str)
                and token
                and all(33 <= ord(c) <= 126 for c in token)
            ):
                raise _TokenError(f"{status} response without a usable access_token")
            return token, _token_lifetime(payload)
        except _TokenError:
            raise
        except Exception as e:
            text = f"{type(e).__name__}: {e}"
            if source == "imds" and isinstance(e, _CONNECT_ERRORS):
                text = f"{_IMDS_UNREACHABLE}: {text}"
            raise _TokenError(_redact(text, secrets)[:500]) from None
        finally:
            await cleanup_response(response, session)

    def _get_message_content(self, message):
        """Extract content from a message, handling different formats."""
        content = message.get("content", "")

        # Handle None content
        if content is None:
            content = ""

        # Handle string content
        if isinstance(content, str):
            return content

        # Handle list content (e.g., for messages with multiple content parts)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                else:
                    # Try to convert other types to string
                    try:
                        text_parts.append(str(part))
                    except:  # noqa: E722
                        pass
            return " ".join(text_parts)

        # Handle function_call in message
        if message.get("function_call"):
            try:
                func_call = message["function_call"]
                func_str = f"function: {func_call.get('name', '')}, arguments: {func_call.get('arguments', '')}"
                return func_str
            except:  # noqa: E722
                return ""

        # If nothing else works, try converting to string or return empty
        try:
            return str(content)
        except:  # noqa: E722
            return ""

    async def _load_encoding(self, name: str):
        """Load a tiktoken encoding in a worker thread (it may download it)."""
        try:
            encoding = await asyncio.to_thread(tiktoken.get_encoding, name)
        except Exception as e:
            _encoding_failures[name] = time.time()
            self.log.warning(
                f"tiktoken encoding '{name}' could not be loaded ({e}); "
                f"estimating token counts as len(text) // 4, "
                f"retrying in {_ENCODING_RETRY_INTERVAL}s"
            )
            return None
        _encoding_failures.pop(name, None)
        _encodings[name] = encoding
        self.log.debug(f"tiktoken encoding '{name}' loaded")
        return encoding

    async def _get_encoding(self, model: str):
        """
        Return the tiktoken encoding for the model (cl100k_base for unknown
        models), or None while it is not loaded. Exceptions raised in inlet()
        abort the chat, so a missing encoding must not raise, and a slow or
        hanging download must not hold up the request.
        """
        try:
            name = tiktoken.encoding_name_for_model(model)
        except Exception:  # KeyError: model unknown to tiktoken
            name = "cl100k_base"

        encoding = _encodings.get(name)
        if encoding is not None:
            return encoding

        load = _encoding_loads.get(name)
        if load is not None and not load.done():
            self.log.debug(f"tiktoken encoding '{name}' is still loading, estimating")
            return None

        failed_at = _encoding_failures.get(name)
        if failed_at and time.time() - failed_at < _ENCODING_RETRY_INTERVAL:
            return None

        load = asyncio.create_task(self._load_encoding(name))
        _encoding_loads[name] = load
        if failed_at:
            # Retry in the background; this request estimates.
            return None

        try:
            # shield(): the load keeps running if the wait times out or the
            # request is cancelled; later requests pick up the result.
            return await asyncio.wait_for(
                asyncio.shield(load), timeout=_ENCODING_FIRST_LOAD_WAIT
            )
        except asyncio.TimeoutError:
            self.log.info(
                f"tiktoken encoding '{name}' not loaded within "
                f"{_ENCODING_FIRST_LOAD_WAIT}s, estimating while it loads"
            )
            return None

    def _count_tokens(self, encoding, text: str) -> tuple[int, bool]:
        """
        Count tokens with tiktoken, or estimate them as len(text) // 4.
        Returns (count, estimated).
        """
        if encoding is not None:
            try:
                # disallowed_special=(): text such as "<|endoftext|>" in a
                # message is counted as plain text instead of raising.
                return len(encoding.encode(text, disallowed_special=())), False
            except Exception as e:
                self.log.warning(f"tiktoken could not encode text ({e}), estimating")
        return len(text) // 4, True

    def _count_messages(self, encoding, messages) -> tuple[int, bool]:
        """Sum the token counts of messages; True if any count is an estimate."""
        counts = [
            self._count_tokens(encoding, self._get_message_content(m))
            for m in messages
            if m
        ]
        return sum(n for n, _ in counts), any(e for _, e in counts)

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
        __metadata__: Optional[dict] = None,
        __id__: Optional[str] = None,
    ) -> dict:
        user_id = __user__.get("id", "unknown") if __user__ else "unknown"
        model = body.get("model", "default-model")
        all_messages = body.get("messages", [])
        self.log.debug(
            f"Inlet called: model={model}, user={user_id}, "
            f"messages={len(all_messages)}, body_keys={list(body.keys())}"
        )

        _prune_stale_entries(self)  # Clean up old entries on each inlet call
        storage_key = _build_request_key(body, __user__)
        self.log.debug(
            f"Request key={storage_key}, active_entries={len(_request_data)}"
        )

        encoding = await self._get_encoding(model)
        self.log.debug(
            f"tiktoken encoding for '{model}': "
            f"{encoding.name if encoding else 'unavailable, estimating'}"
        )

        # If CALCULATE_ALL_MESSAGES is true, use all "user" and "system" messages
        if self.valves.CALCULATE_ALL_MESSAGES:
            request_messages = [
                m for m in all_messages if m.get("role") in ("user", "system")
            ]
        else:
            # If CALCULATE_ALL_MESSAGES is false and there are exactly two messages
            # (one user and one system), sum them both.
            request_user_system = [
                m for m in all_messages if m.get("role") in ("user", "system")
            ]
            if len(request_user_system) == 2:
                request_messages = request_user_system
            else:
                # Otherwise, take only the last "user" or "system" message if any
                reversed_messages = list(reversed(all_messages))
                last_user_system = next(
                    (
                        m
                        for m in reversed_messages
                        if m.get("role") in ("user", "system")
                    ),
                    None,
                )
                request_messages = [last_user_system] if last_user_system else []

        request_token_count, tokens_estimated = self._count_messages(
            encoding, request_messages
        )

        entry = {
            "key": storage_key,
            "start_time": time.time(),
            "request_token_count": request_token_count,
            "tokens_estimated": tokens_estimated,
        }
        _request_data[storage_key] = entry
        if isinstance(__metadata__, dict):
            __metadata__[_metadata_key(__id__)] = entry

        self.log.info(
            f"Inlet complete: key={storage_key}, model={model}, "
            f"request_tokens={request_token_count}, "
            f"counted_messages={len(request_messages)}, "
            f"tokens_estimated={tokens_estimated}"
        )

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
        __metadata__: Optional[dict] = None,
        __id__: Optional[str] = None,
    ) -> dict:
        model = body.get("model", "default-model")
        all_messages = body.get("messages", [])
        user_id = __user__.get("id", "unknown") if __user__ else "unknown"
        self.log.debug(
            f"Outlet called: model={model}, user={user_id}, "
            f"messages={len(all_messages)}, body_keys={list(body.keys())}"
        )

        # Prefer the entry inlet() left in this request's metadata; fall back
        # to the fingerprint when outlet() gets another metadata dict.
        request_data = None
        if isinstance(__metadata__, dict):
            request_data = __metadata__.pop(_metadata_key(__id__), None)
        if isinstance(request_data, dict):
            storage_key = request_data.get("key", "")
            matched_by = "metadata"
            # A concurrent identical request may have replaced the entry
            # under the same fingerprint: only drop our own.
            if _request_data.get(storage_key) is request_data:
                _request_data.pop(storage_key, None)
        else:
            storage_key = _build_request_key(body, __user__)
            matched_by = "fingerprint"
            request_data = _request_data.pop(storage_key, {})

        if not request_data:
            self.log.warning(
                f"No inlet data found for key={storage_key}. "
                f"Metrics will show zero values. "
                f"Remaining entries={len(_request_data)}"
            )
        else:
            self.log.debug(
                f"Matched inlet data for key={storage_key} by {matched_by}, "
                f"remaining_entries={len(_request_data)}"
            )

        end_time = time.time()
        response_time = end_time - request_data.get("start_time", end_time)
        request_token_count = request_data.get("request_token_count", 0)
        tokens_estimated = bool(request_data.get("tokens_estimated", False))

        encoding = await self._get_encoding(model)

        reversed_messages = list(
            reversed(all_messages)
        )  # If CALCULATE_ALL_MESSAGES is true, use all "assistant" messages
        if self.valves.CALCULATE_ALL_MESSAGES:
            assistant_messages = [
                m for m in all_messages if m.get("role") == "assistant"
            ]
        else:
            # Take only the last "assistant" message if any
            last_assistant = next(
                (m for m in reversed_messages if m.get("role") == "assistant"), None
            )
            assistant_messages = [last_assistant] if last_assistant else []

        # response_token_count is a local variable here; unlike the original
        # global, it does not need to persist beyond this method.
        response_token_count, response_estimated = self._count_messages(
            encoding, assistant_messages
        )
        tokens_estimated = tokens_estimated or response_estimated
        # Calculate tokens per second (only for the last assistant response)
        resp_tokens_per_sec = 0
        if self.valves.SHOW_TOKENS_PER_SECOND:
            last_assistant_msg = next(
                (m for m in reversed_messages if m.get("role") == "assistant"), None
            )
            last_assistant_tokens, _ = self._count_messages(
                encoding, [last_assistant_msg]
            )
            resp_tokens_per_sec = (
                0 if response_time == 0 else last_assistant_tokens / response_time
            )

        # Calculate averages only if CALCULATE_ALL_MESSAGES is true
        avg_request_tokens = avg_response_tokens = 0
        if self.valves.SHOW_AVERAGE_TOKENS and self.valves.CALCULATE_ALL_MESSAGES:
            req_count = len(
                [m for m in all_messages if m.get("role") in ("user", "system")]
            )
            resp_count = len([m for m in all_messages if m.get("role") == "assistant"])
            avg_request_tokens = request_token_count / req_count if req_count else 0
            avg_response_tokens = response_token_count / resp_count if resp_count else 0

        # Shorter style, e.g.: "10.90s | Req: 175 (Ø 87.50) | Resp: 439 (Ø 219.50) | 40.18 T/s"
        # Estimated token numbers get a "~", e.g. "Req: ~175 (Ø ~87.50)".
        est = "~" if tokens_estimated else ""
        description_parts = []
        if self.valves.SHOW_RESPONSE_TIME:
            description_parts.append(f"{response_time:.2f}s")
        if self.valves.SHOW_TOKEN_COUNT:
            if self.valves.SHOW_AVERAGE_TOKENS and self.valves.CALCULATE_ALL_MESSAGES:
                # Add averages (Ø) into short output
                short_str = (
                    f"Req: {est}{request_token_count} (Ø {est}{avg_request_tokens:.2f}) | "
                    f"Resp: {est}{response_token_count} (Ø {est}{avg_response_tokens:.2f})"
                )
            else:
                short_str = (
                    f"Req: {est}{request_token_count} | "
                    f"Resp: {est}{response_token_count}"
                )
            description_parts.append(short_str)
        if self.valves.SHOW_TOKENS_PER_SECOND:
            description_parts.append(f"{est}{resp_tokens_per_sec:.2f} T/s")
        description = " | ".join(description_parts)

        self.log.info(
            f"Outlet complete: key={storage_key}, model={model}, "
            f"response_time={response_time:.2f}s, "
            f"req_tokens={request_token_count}, resp_tokens={response_token_count}, "
            f"tokens_per_sec={resp_tokens_per_sec:.2f}, "
            f"tokens_estimated={tokens_estimated}"
        )
        self.log.debug(f"Status event: {description}")

        # Send event with description. Since Open WebUI 0.10 outlet filters also
        # run for API requests, where there is no event emitter (None).
        if __event_emitter__:
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": description, "done": True},
                    }
                )
            except Exception as e:
                self.log.warning(f"Could not emit status event: {e}")
        else:
            self.log.debug("No event emitter (e.g. API request): status skipped")

        # If Log Analytics integration is enabled, send the data
        if self.valves.SEND_TO_LOG_ANALYTICS:
            # Chat and message IDs for tracking; API requests have no chat id
            chat_id = body.get("chat_id") or str(uuid.uuid4())
            message_id = body.get("id") or str(uuid.uuid4())
            # User ID if available
            user_id = __user__.get("id", "unknown") if __user__ else "unknown"

            # Create log data for Log Analytics
            log_data = [
                {
                    # ISO 8601 in UTC with "Z", as time-generated-field expects
                    "timestamp": datetime.datetime.now(datetime.timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "chatId": chat_id,
                    "messageId": message_id,
                    "model": model,
                    "userId": user_id,
                    "responseTime": response_time,
                    "requestTokens": request_token_count,
                    "responseTokens": response_token_count,
                    "tokensPerSecond": resp_tokens_per_sec,
                    "tokensEstimated": tokens_estimated,
                }
            ]

            # Add averages if calculated
            if self.valves.SHOW_AVERAGE_TOKENS and self.valves.CALCULATE_ALL_MESSAGES:
                log_data[0]["avgRequestTokens"] = avg_request_tokens
                log_data[0]["avgResponseTokens"] = avg_response_tokens

            # Sent in a background task: outlet() returns without waiting for
            # Log Analytics, the task logs the outcome.
            try:
                if not self._send_to_log_analytics(log_data, chat_id, message_id):
                    self.log.warning(
                        f"Failed to send data to Log Analytics "
                        f"(chat={chat_id}, message={message_id})"
                    )
            except Exception as e:
                self.log.error(f"Error sending to Log Analytics: {e}")

        return body
