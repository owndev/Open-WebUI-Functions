"""
Mock of the Gemini Developer REST API (the subset google-genai uses).

Point the pipe's ``BASE_URL`` valve at ``http://127.0.0.1:<port>/``.

Routes
  GET  /{version}/models                                model list
  POST /{version}/models/{model}:generateContent        JSON answer
  POST /{version}/models/{model}:streamGenerateContent  SSE answer (?alt=sse)
  POST /{version}/models/{model}:predictLongRunning     Veo: start an operation
  GET  /{version}/models/{model}/operations/{op}        Veo: poll the operation
  GET  /{version}/files/{id}:download                   Veo: download the video
  GET  /__requests, POST /__reset                       request record

``{version}`` is whatever the pipe's ``API_VERSION`` valve sends (v1alpha, v1beta).

Answers
  - thought parts only when the request asks for them
    (``generationConfig.thinkingConfig.includeThoughts``; google-genai sends some
    nested keys in snake_case, both spellings are read), never for
    gemini-2.5-flash-image (no thinking)
  - text models: a thought part ("Mock thinking." / "Mock pondering.") +
    "Hello from mock ..."; streaming sends the answer in three chunks, usage on
    the last chunk
  - image models (id contains "image" or "nano-banana"): thought text + two
    interim thought images (``thought: true`` inlineData, THOUGHT_PNG_1/2) + text
    + the final image (FINAL_PNG); every PNG has distinct bytes. Streaming
    (only reached by models the pipe does not detect, e.g. "...-imagegen"):
    thought text, a thought image, the final image, then the text
  - a ``googleSearch`` tool in the request adds ``groundingMetadata`` (one web
    chunk, one support for "Hello"), so citations / source events can be checked
  - Open WebUI task prompts (title, tags, follow-ups) get the JSON they expect
    (plus a thought part when thoughts are requested)
  - Veo: the operation starts pending; the first poll reports ``done: true`` with
    one video whose Files API ``uri`` google-genai downloads from this mock

Tools an image model does not support are rejected like the real API does
(HTTP 400 INVALID_ARGUMENT; an approximation of the real messages):
functionDeclarations and urlContext for every image model, googleSearch for
gemini-2.5-flash-image and gemini-3.1-flash-lite-image. Model ids containing
"imagegen" stand for image models the pipe does not recognise and accept tools.

Triggers (a word in the last user message)
  force-400            HTTP 400 INVALID_ARGUMENT
  force-500            HTTP 500 INTERNAL, every time
  force-500-once       HTTP 500 for the first such request since the last reset
  force-503-once       HTTP 503 UNAVAILABLE for the first such request since reset
  prompt-blocked       no candidates, promptFeedback.blockReason SAFETY
  finish-safety        candidate without content, finishReason SAFETY (blocked
                       safety rating HARM_CATEGORY_HARASSMENT)
  data-prefix          the answer text starts with "data:" ("data: starts like SSE.")
  vertex-context       groundingMetadata with a Vertex AI Search ``retrievedContext``
                       chunk (uri gs://e2e-bucket/doc.pdf, title "Vertex Doc",
                       text "chunk body")
  only-thought-images  image models: thought images only, finishReason STOP
  image-safety         image models: thought images only, finishReason IMAGE_SAFETY
  duplicate-image      image models: the final image twice
  two-final-images     image models: two different final images (FINAL_PNG, PNG_B64)
  slow-video           Veo: the operation stays pending for SLOW_VIDEO_POLLS polls

Each recorded request carries ``model``, ``action`` (generateContent,
streamGenerateContent, predictLongRunning, download), ``tool_kinds`` (keys of
the request's ``tools`` entries, e.g. ``googleSearch``) and, for generate
requests, ``status`` (the HTTP status the mock answered with).

usage: python mock_gemini.py [--port 9101]
"""

import argparse
import asyncio
import base64
import itertools
import json
import struct
import zlib
from typing import Optional

from aiohttp import web

from common import PNG_B64, REQUESTS_KEY, annotate, new_app, record, task_answer

MODELS = [
    ("gemini-2.5-flash", "Gemini 2.5 Flash", ["generateContent", "countTokens"]),
    ("gemini-3-pro-preview", "Gemini 3 Pro Preview", None),
    ("gemini-3.1-flash-image-preview", "Gemini 3.1 Flash Image Preview", None),
    ("gemini-3.1-flash-image", "Gemini 3.1 Flash Image", None),
    ("gemini-3.1-flash-lite-image", "Gemini 3.1 Flash Lite Image", None),
    ("gemini-2.5-flash-image", "Gemini 2.5 Flash Image", None),
    ("gemini-nano-banana-2.1", "Nano Banana 2.1", None),
    ("veo-3.1-generate-preview", "Veo 3.1", ["predictLongRunning"]),
    ("text-embedding-004", "Text Embedding 004", ["embedContent"]),
]
# Tiny fake MP4 payload ("ftyp" box header is enough for a file upload).
VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64
USAGE_TEXT = {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 25}
USAGE_STREAM = {"promptTokenCount": 9, "candidatesTokenCount": 4, "totalTokenCount": 20}
USAGE_IMAGE = {
    "promptTokenCount": 12,
    "candidatesTokenCount": 1290,
    "totalTokenCount": 1302,
}
GOOGLE_FILES = "https://generativelanguage.googleapis.com/v1beta/files"
GENERATE = ("generateContent", "streamGenerateContent")
NO_SEARCH_MODELS = ("gemini-2.5-flash-image", "gemini-3.1-flash-lite-image")
SLOW_VIDEO_POLLS = 6
VERTEX_CHUNK = {
    "retrievedContext": {
        "uri": "gs://e2e-bucket/doc.pdf",
        "title": "Vertex Doc",
        "text": "chunk body",
    }
}
_op_ids = itertools.count(1)
_slow_ops: dict = {}  # operation id -> polls left before it is done


def png_b64(r: int, g: int, b: int) -> str:
    """Base64 of a 1x1 RGB PNG of the given colour (distinct bytes per colour)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00" + bytes((r, g, b))))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


THOUGHT_PNG_1 = png_b64(0, 0, 255)
THOUGHT_PNG_2 = png_b64(0, 255, 0)
FINAL_PNG = png_b64(255, 255, 0)


def _camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(word[:1].upper() + word[1:] for word in rest)


def normalize(value):
    """camelCase every dict key (google-genai sends some nested keys in snake_case)."""
    if isinstance(value, dict):
        return {_camel(k): normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def _is_image_model(model: str) -> bool:
    return "image" in model or "nano-banana" in model


def _thoughts(model: str, body) -> bool:
    """The request asks for thoughts and the model thinks."""
    config = normalize((body or {}).get("generationConfig") or {})
    include = (config.get("thinkingConfig") or {}).get("includeThoughts")
    return bool(include) and not model.startswith("gemini-2.5-flash-image")


def _img(data: str, thought: bool = False) -> dict:
    part = {"inlineData": {"mimeType": "image/png", "data": data}}
    if thought:
        part["thought"] = True
    return part


def _tool_kinds(body) -> list:
    kinds = []
    for tool in (body or {}).get("tools") or []:
        if isinstance(tool, dict):
            kinds.extend(tool.keys())
    return kinds


def _last_user_text(body) -> str:
    if not isinstance(body, dict):
        return ""
    for content in reversed(body.get("contents") or []):
        if content.get("role", "user") == "user":
            return " ".join(p.get("text", "") for p in content.get("parts") or [])
    return ""


def _error(code: int, message: str, status: str) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message, "status": status}}, status=code
    )


def _first_since_reset(request: web.Request, trigger: str) -> bool:
    """This is the first generate request with ``trigger`` since the last reset
    (the current request is already recorded)."""
    seen = [
        entry
        for entry in request.app[REQUESTS_KEY]
        if entry.get("action") in GENERATE
        and trigger in _last_user_text(entry.get("body"))
    ]
    return len(seen) == 1


def _reject(request: web.Request, model: str, body: dict) -> Optional[web.Response]:
    """Errors on purpose (triggers) and tools a model does not support (400)."""
    text = _last_user_text(body)
    if "force-400" in text:
        return _error(400, "mock: force-400", "INVALID_ARGUMENT")
    if "force-500-once" in text:
        if _first_since_reset(request, "force-500-once"):
            return _error(500, "mock: force-500-once", "INTERNAL")
    elif "force-500" in text:
        return _error(500, "mock: force-500", "INTERNAL")
    if "force-503-once" in text and _first_since_reset(request, "force-503-once"):
        return _error(503, "mock: force-503-once", "UNAVAILABLE")
    # "imagegen": stands for an image model the pipe does not recognise
    if _is_image_model(model) and "imagegen" not in model:
        kinds = _tool_kinds(body)
        if "functionDeclarations" in kinds:
            return _error(
                400,
                f"Function calling is not enabled for models/{model}",
                "INVALID_ARGUMENT",
            )
        if "urlContext" in kinds:
            return _error(
                400,
                f"Url context is not supported for models/{model}",
                "INVALID_ARGUMENT",
            )
        if "googleSearch" in kinds and model.startswith(NO_SEARCH_MODELS):
            return _error(
                400,
                f"Search Grounding is not supported for models/{model}",
                "INVALID_ARGUMENT",
            )
    return None


def _grounding(body):
    kinds = _tool_kinds(body)
    if "vertex-context" in _last_user_text(body):
        chunks = [VERTEX_CHUNK]
    elif "googleSearch" in kinds:
        chunks = [{"web": {"uri": "https://example.com/a", "title": "Example A"}}]
    else:
        return None
    metadata = {
        "groundingChunks": chunks,
        "groundingSupports": [
            {
                "segment": {"startIndex": 0, "endIndex": 5, "text": "Hello"},
                "groundingChunkIndices": [0],
            }
        ],
    }
    if "googleSearch" in kinds:
        metadata["webSearchQueries"] = ["mock search query"]
    return metadata


def _chunk(parts, body, final=False, usage=None, finish="STOP") -> dict:
    """One GenerateContentResponse with a single candidate."""
    candidate = {"content": {"role": "model", "parts": parts}, "index": 0}
    if final:
        candidate["finishReason"] = finish
        grounding = _grounding(body)
        if grounding:
            candidate["groundingMetadata"] = grounding
    response = {"candidates": [candidate]}
    if usage:
        response["usageMetadata"] = usage
    return response


def _blocked(text: str) -> Optional[dict]:
    """Answer for the safety triggers (None for other prompts)."""
    if "prompt-blocked" in text:
        return {
            "promptFeedback": {"blockReason": "SAFETY"},
            "usageMetadata": USAGE_TEXT,
        }
    if "finish-safety" in text:
        rating = {
            "category": "HARM_CATEGORY_HARASSMENT",
            "probability": "HIGH",
            "blocked": True,
        }
        return {
            "candidates": [
                {"finishReason": "SAFETY", "index": 0, "safetyRatings": [rating]}
            ],
            "usageMetadata": USAGE_TEXT,
        }
    return None


def _answer(model: str, body) -> dict:
    text = _last_user_text(body)
    blocked = _blocked(text)
    if blocked:
        return blocked
    thoughts = _thoughts(model, body)
    task = task_answer(text)
    if task:
        parts = [{"text": "Task thinking.", "thought": True}] if thoughts else []
        return _chunk(parts + [{"text": task}], body, True, USAGE_TEXT)
    if _is_image_model(model):
        parts = []
        if thoughts:
            parts.append({"text": "Mock image thinking.", "thought": True})
            parts += [_img(THOUGHT_PNG_1, True), _img(THOUGHT_PNG_2, True)]
        if "only-thought-images" in text or "image-safety" in text:
            finish = "IMAGE_SAFETY" if "image-safety" in text else "STOP"
            return _chunk(parts, body, True, USAGE_IMAGE, finish)
        if "duplicate-image" in text:
            parts += [{"text": "Here is your image."}, _img(FINAL_PNG), _img(FINAL_PNG)]
        elif "two-final-images" in text:
            parts += [{"text": "Here are your images."}, _img(FINAL_PNG), _img(PNG_B64)]
        else:
            parts += [{"text": "Here is your image."}, _img(FINAL_PNG)]
        return _chunk(parts, body, True, USAGE_IMAGE)
    parts = [{"text": "Mock thinking.", "thought": True}] if thoughts else []
    answer = "Hello from mock (non-stream)."
    if "data-prefix" in text:
        answer = "data: starts like SSE."
    parts.append({"text": answer})
    return _chunk(parts, body, True, USAGE_TEXT)


def _stream_chunks(model: str, body) -> list:
    text = _last_user_text(body)
    blocked = _blocked(text)
    if blocked:
        return [blocked]
    thoughts = _thoughts(model, body)
    if _is_image_model(model):
        chunks = []
        if thoughts:
            chunks.append(_chunk([{"text": "Pondering.", "thought": True}], body))
            chunks.append(_chunk([_img(THOUGHT_PNG_1, True)], body))
        chunks.append(_chunk([_img(FINAL_PNG)], body))
        chunks.append(
            _chunk([{"text": "Here is your image."}], body, True, USAGE_IMAGE)
        )
        return chunks
    task = task_answer(text)
    pieces = [task] if task else ["Hello ", "from mock ", "(stream)."]
    if "data-prefix" in text:
        pieces = ["data: starts ", "like SSE."]
    chunks = []
    if thoughts:
        chunks.append(_chunk([{"text": "Mock pondering.", "thought": True}], body))
    chunks += [_chunk([{"text": piece}], body) for piece in pieces[:-1]]
    chunks.append(_chunk([{"text": pieces[-1]}], body, True, USAGE_STREAM))
    return chunks


async def list_models(request: web.Request) -> web.Response:
    await record(request)
    models = []
    for model_id, display, methods in MODELS:
        entry = {"name": f"models/{model_id}", "displayName": display}
        if methods:
            entry["supportedGenerationMethods"] = methods
        models.append(entry)
    return web.json_response({"models": models})


async def model_action(request: web.Request) -> web.StreamResponse:
    model, _, action = request.match_info["target"].partition(":")
    body = await record(request, model=model, action=action)
    body = body if isinstance(body, dict) else {}
    annotate(request, tool_kinds=_tool_kinds(body))
    if action in GENERATE:
        rejected = _reject(request, model, body)
        annotate(request, status=rejected.status if rejected is not None else 200)
        if rejected is not None:
            return rejected
    if action == "generateContent":
        return web.json_response(_answer(model, body))
    if action == "streamGenerateContent":
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for chunk in _stream_chunks(model, body):
            await resp.write(f"data: {json.dumps(chunk)}\r\n\r\n".encode())
            await asyncio.sleep(0.05)
        await resp.write_eof()
        return resp
    if action == "predictLongRunning":
        op = f"op-{next(_op_ids)}"
        prompt = str(((body.get("instances") or [{}])[0] or {}).get("prompt") or "")
        if "slow-video" in prompt:
            _slow_ops[op] = SLOW_VIDEO_POLLS
        return web.json_response({"name": f"models/{model}/operations/{op}"})
    return _not_found(request)


async def get_operation(request: web.Request) -> web.Response:
    await record(request)
    model, op = request.match_info["model"], request.match_info["op"]
    name = f"models/{model}/operations/{op}"
    if _slow_ops.get(op, 0) > 0:
        _slow_ops[op] -= 1
        return web.json_response({"name": name, "done": False})
    # Real Veo answers carry a Files API URI; google-genai extracts the file id
    # from it and downloads <BASE_URL>/<version>/files/<id>:download from us.
    file_id = "video" + op.rsplit("-", 1)[-1]
    video = {"uri": f"{GOOGLE_FILES}/{file_id}:download?alt=media"}
    return web.json_response(
        {
            "name": name,
            "done": True,
            "response": {
                "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta."
                "PredictLongRunningResponse",
                "generateVideoResponse": {"generatedSamples": [{"video": video}]},
            },
        }
    )


def _not_found(request: web.Request) -> web.Response:
    return web.json_response(
        {"error": {"code": 404, "message": request.path, "status": "NOT_FOUND"}},
        status=404,
    )


async def download_file(request: web.Request) -> web.Response:
    await record(request, action="download")
    return web.Response(body=VIDEO_BYTES, content_type="video/mp4")


async def fallback(request: web.Request) -> web.Response:
    await record(request)
    return _not_found(request)


def make_app() -> web.Application:
    app = new_app()
    app.router.add_get("/{version}/models", list_models)
    app.router.add_post("/{version}/models/{target}", model_action)
    app.router.add_get("/{version}/models/{model}/operations/{op}", get_operation)
    app.router.add_get("/{version}/files/{target}", download_file)
    app.router.add_get("/download/{version}/files/{target}", download_file)
    app.router.add_route("*", "/{tail:.*}", fallback)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini REST API mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9101)
    args = parser.parse_args()
    web.run_app(make_app(), host=args.host, port=args.port)
