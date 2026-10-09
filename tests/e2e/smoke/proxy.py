"""
Recording pass-through proxy of the real-API smoke test (tests/e2e/smoke).

The Gemini pipe's ``BASE_URL`` points here. Every request is forwarded unchanged
to the target (``https://generativelanguage.googleapis.com`` or, with
``--dry-run-mock``, the e2e Gemini mock) and the answer goes back unchanged (a
stream chunk by chunk), so the pipe talks to the real API. For each request a
summary is recorded, the evidence of the smoke scenarios:

- request: declared function names (and those without parameters), tool kinds,
  ``toolConfig`` (server-side tool invocations flag, function calling mode),
  per content the role and part kinds, every function call / response part
  with id, name and whether the call carries a thought signature (``yes``,
  ``skip`` for the placeholder, ``none``), replayed server-side parts,
  ``maxOutputTokens`` and the thinking config
- answer: HTTP status, error message, finish reasons, function calls (name, id,
  signature flag, chunk, argument names), server-side tool parts, thought /
  text parts with signatures, text lengths plus a short preview, usage

Never recorded: headers (the API key travels in ``x-goog-api-key``), the query
string (a ``key=`` parameter would carry it), signature values, request texts.
"""

import json
import time
from typing import Any, Optional

import aiohttp
from aiohttp import web

GENERATE = ("generateContent", "streamGenerateContent")
SKIP_SIGNATURE = "skip_thought_signature_validator"
# Request headers not forwarded (hop-by-hop, recomputed, or compression: the
# proxy reads the answers, aiohttp asks for and decodes what it supports).
REQUEST_DROP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
}
# Answer headers not passed back (the body is sent decoded and re-framed).
RESPONSE_DROP = {
    "connection",
    "keep-alive",
    "transfer-encoding",
    "content-length",
    "content-encoding",
}
PART_KINDS = (
    ("functionCall", "fc"),
    ("functionResponse", "fr"),
    ("toolCall", "toolCall"),
    ("toolResponse", "toolResponse"),
    ("inlineData", "inline"),
    ("fileData", "file"),
    ("executableCode", "code"),
    ("codeExecutionResult", "codeResult"),
)
PREVIEW = 120


def _camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(word[:1].upper() + word[1:] for word in rest)


def _snake(camel: str) -> str:
    return "".join("_" + c.lower() if c.isupper() else c for c in camel)


def _get(data: Any, camel: str) -> Any:
    """``data[camel]`` or its snake_case spelling (google-genai mixes both)."""
    if not isinstance(data, dict):
        return None
    if camel in data:
        return data[camel]
    return data.get(_snake(camel))


def _parts(content: Any) -> list:
    if not isinstance(content, dict):
        return []
    return [p for p in content.get("parts") or [] if isinstance(p, dict)]


def part_kind(part: dict) -> str:
    for camel, kind in PART_KINDS:
        if _get(part, camel) is not None:
            return kind
    if "text" in part:
        return "thought" if part.get("thought") else "text"
    return "other"


def _sig_kind(sig: Any) -> str:
    if not sig:
        return "none"
    return "skip" if sig == SKIP_SIGNATURE else "yes"


def target(path: str) -> tuple:
    """(model, action) of an API path: ``/v1beta/models/m:generateContent`` ->
    ("m", "generateContent"), ``/v1beta/models`` -> ("", "list")."""
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 3 and parts[1] == "models":
        model, _, action = parts[2].partition(":")
        return model, action or "get"
    if len(parts) == 2 and parts[1] == "models":
        return "", "list"
    return "", parts[-1] if parts else ""


def summarize_request(body: dict) -> dict:
    """What a generate request declares and replays (see the module docstring)."""
    declared, noparams, tool_kinds = [], [], []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tool_kinds.extend(_camel(key) for key in tool)
        for decl in _get(tool, "functionDeclarations") or []:
            if not isinstance(decl, dict):
                continue
            declared.append(decl.get("name"))
            keys = ("parameters", "parametersJsonSchema", "parameters_json_schema")
            if not any(k in decl for k in keys):
                noparams.append(decl.get("name"))
    config = _get(body, "toolConfig") or {}
    calling = _get(config, "functionCallingConfig") or {}
    generation = _get(body, "generationConfig") or {}
    thinking = _get(generation, "thinkingConfig") or {}
    kinds, fc, fr, server = [], [], [], 0
    for index, content in enumerate(body.get("contents") or []):
        if not isinstance(content, dict):
            continue
        role = content.get("role") or "user"
        parts = _parts(content)
        kinds.append(f"{role}:" + "+".join(part_kind(p) for p in parts))
        for part in parts:
            call = _get(part, "functionCall")
            if isinstance(call, dict):
                fc.append(
                    {
                        "i": index,
                        "id": call.get("id"),
                        "name": call.get("name"),
                        "sig": _sig_kind(_get(part, "thoughtSignature")),
                    }
                )
            response = _get(part, "functionResponse")
            if isinstance(response, dict):
                payload = response.get("response")
                fr.append(
                    {
                        "i": index,
                        "id": response.get("id"),
                        "name": response.get("name"),
                        "keys": sorted(payload) if isinstance(payload, dict) else [],
                    }
                )
            if role == "model" and part_kind(part) in ("toolCall", "toolResponse"):
                server += 1
    return {
        "declared": declared,
        "declared_n": len(declared),
        "noparams": noparams,
        "tool_kinds": tool_kinds,
        "include_flag": _get(config, "includeServerSideToolInvocations"),
        "fc_mode": _get(calling, "mode"),
        "allowed": _get(calling, "allowedFunctionNames"),
        "kinds": kinds,
        "fc": fc,
        "fr": fr,
        "server_replayed": server,
        "max_output_tokens": _get(generation, "maxOutputTokens"),
        "thinking": {
            key: _get(thinking, key)
            for key in ("includeThoughts", "thinkingLevel", "thinkingBudget")
            if _get(thinking, key) is not None
        },
    }


def _runs(kinds: list) -> list:
    """Run-length form of part kinds: ["thought", "text x3", "fc*"]."""
    out: list = []
    for kind in kinds:
        if out and out[-1][0] == kind:
            out[-1][1] += 1
        else:
            out.append([kind, 1])
    return [k if n == 1 else f"{k} x{n}" for k, n in out]


def summarize_answer(chunks: list) -> dict:
    """What came back: one dict (generateContent) or the SSE chunks."""
    out: dict = {
        "chunks": len(chunks),
        "finish": [],
        "fc": [],
        "fc_chunks": 0,
        "partial_fc": False,
        "server": 0,
        "thought_sigs": 0,
        "text_sigs": 0,
        "thought_len": 0,
        "text_len": 0,
        "preview": "",
        "usage": None,
        "block": None,
        "error": None,
    }
    text, kinds = "", []
    for k, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        error = chunk.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else error
            out["error"] = str(message)[:300]
        usage = _get(chunk, "usageMetadata")
        if isinstance(usage, dict):
            out["usage"] = {
                key: _get(usage, key)
                for key in (
                    "promptTokenCount",
                    "candidatesTokenCount",
                    "thoughtsTokenCount",
                    "totalTokenCount",
                )
                if _get(usage, key) is not None
            }
        block = _get(_get(chunk, "promptFeedback") or {}, "blockReason")
        if block:
            out["block"] = block
        has_fc = False
        for candidate in chunk.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            finish = _get(candidate, "finishReason")
            if finish:
                out["finish"].append(str(finish))
            for part in _parts(candidate.get("content")):
                kind = part_kind(part)
                signed = bool(_get(part, "thoughtSignature"))
                kinds.append(kind + ("*" if signed else ""))
                if kind == "fc":
                    call = _get(part, "functionCall") or {}
                    args = call.get("args")
                    out["fc"].append(
                        {
                            "name": call.get("name"),
                            "id": call.get("id"),
                            "sig": signed,
                            "chunk": k,
                            "args": sorted(args) if isinstance(args, dict) else [],
                        }
                    )
                    partial = ("partialArgs", "willContinue")
                    if any(_get(call, key) is not None for key in partial):
                        out["partial_fc"] = True
                    has_fc = True
                elif kind in ("toolCall", "toolResponse"):
                    out["server"] += 1
                elif kind == "thought":
                    out["thought_len"] += len(part.get("text") or "")
                    out["thought_sigs"] += int(signed)
                elif kind == "text":
                    text += part.get("text") or ""
                    out["text_sigs"] += int(signed)
        out["fc_chunks"] += int(has_fc)
    out["text_len"] = len(text)
    out["preview"] = " ".join(text.split())[:PREVIEW]
    out["parts"] = _runs(kinds)
    return out


def parse_sse(text: str) -> list:
    chunks = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunks.append(json.loads(payload))
        except ValueError:
            chunks.append({"error": f"unparsable SSE line: {payload[:80]}"})
    return chunks


class RecordingProxy:
    """``await start()``, point the pipe at ``url``, read ``records``."""

    def __init__(self, target_url: str, port: int, host: str = "127.0.0.1"):
        self.target = target_url.rstrip("/")
        self.host = host
        self.port = port
        self.records: list = []
        self._runner: Optional[web.AppRunner] = None
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def mark(self) -> int:
        return len(self.records)

    def since(self, mark: int) -> list:
        return self.records[mark:]

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600, sock_connect=30)
        )
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        if self._session is not None:
            await self._session.close()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        model, action = target(request.path)
        entry: dict = {
            "n": len(self.records) + 1,
            "t": round(time.time(), 1),
            "method": request.method,
            "path": request.path,
            "query": sorted(k for k in request.query if k.lower() != "key"),
            "model": model,
            "action": action,
            "status": None,
            "stream": False,
        }
        self.records.append(entry)
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        if action in GENERATE and isinstance(body, dict):
            entry["req"] = summarize_request(body)
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in REQUEST_DROP
        }
        t0 = time.monotonic()
        try:
            upstream = await self._session.request(
                request.method,
                self.target + request.path_qs,
                headers=headers,
                data=raw or None,
                allow_redirects=False,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, answered as 502
            entry["status"] = 502
            entry["error"] = f"proxy: {type(exc).__name__}: {exc}"[:300]
            return web.json_response(
                {
                    "error": {
                        "code": 502,
                        "message": "the smoke proxy could not reach the upstream",
                        "status": "UNAVAILABLE",
                    }
                },
                status=502,
            )
        try:
            entry["status"] = upstream.status
            out_headers = {
                k: v
                for k, v in upstream.headers.items()
                if k.lower() not in RESPONSE_DROP
            }
            if "text/event-stream" in upstream.headers.get("Content-Type", ""):
                entry["stream"] = True
                resp = web.StreamResponse(status=upstream.status, headers=out_headers)
                await resp.prepare(request)
                data = bytearray()
                try:
                    async for piece in upstream.content.iter_any():
                        data += piece
                        await resp.write(piece)
                    await resp.write_eof()
                except (ConnectionError, aiohttp.ClientError) as exc:
                    entry["error"] = f"stream cut: {type(exc).__name__}"
                chunks = parse_sse(data.decode("utf-8", "replace"))
            else:
                data = await upstream.read()
                resp = web.Response(
                    status=upstream.status, body=data, headers=out_headers
                )
                try:
                    chunks = [json.loads(data)] if data else []
                except ValueError:
                    chunks = [{"error": data[:200].decode("utf-8", "replace")}]
            entry["ms"] = round((time.monotonic() - t0) * 1000)
            if action in GENERATE or upstream.status >= 400:
                entry["resp"] = summarize_answer(chunks)
            return resp
        finally:
            upstream.release()
