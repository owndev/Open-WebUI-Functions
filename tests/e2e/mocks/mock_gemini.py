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
  - text models: a thought part ("Mock thinking.") + "Hello from mock ...";
    streaming sends the answer in three chunks, usage on the last chunk
  - models whose id contains "image": thought + text + a PNG ``inlineData`` part
  - a ``googleSearch`` tool in the request adds ``groundingMetadata`` (one web
    chunk, one support for "Hello"), so citations / source events can be checked
  - Open WebUI task prompts (title, tags, follow-ups) get the JSON they expect
  - Veo: the operation starts pending; the first poll reports ``done: true`` with
    one video whose Files API ``uri`` google-genai downloads from this mock

Each recorded request carries ``model``, ``action`` (generateContent,
streamGenerateContent, predictLongRunning, download) and ``tool_kinds`` (keys of
the request's ``tools`` entries, e.g. ``googleSearch``).

usage: python mock_gemini.py [--port 9101]
"""

import argparse
import asyncio
import itertools
import json

from aiohttp import web

from common import PNG_B64, annotate, new_app, record, task_answer

MODELS = [
    ("gemini-2.5-flash", "Gemini 2.5 Flash", ["generateContent", "countTokens"]),
    ("gemini-3.1-flash-image-preview", "Gemini 3.1 Flash Image Preview", None),
    ("gemini-3.1-flash-image", "Gemini 3.1 Flash Image", None),
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
IMAGE_PART = {"inlineData": {"mimeType": "image/png", "data": PNG_B64}}
GOOGLE_FILES = "https://generativelanguage.googleapis.com/v1beta/files"
_op_ids = itertools.count(1)


def _tool_kinds(body) -> list:
    kinds = []
    for tool in (body or {}).get("tools") or []:
        if isinstance(tool, dict):
            kinds.extend(tool.keys())
    return kinds


def _last_user_text(body) -> str:
    for content in reversed((body or {}).get("contents") or []):
        if content.get("role", "user") == "user":
            return " ".join(p.get("text", "") for p in content.get("parts") or [])
    return ""


def _grounding(body):
    if "googleSearch" not in _tool_kinds(body):
        return None
    return {
        "webSearchQueries": ["mock search query"],
        "groundingChunks": [
            {"web": {"uri": "https://example.com/a", "title": "Example A"}}
        ],
        "groundingSupports": [
            {
                "segment": {"startIndex": 0, "endIndex": 5, "text": "Hello"},
                "groundingChunkIndices": [0],
            }
        ],
    }


def _chunk(parts, body, final=False, usage=None) -> dict:
    """One GenerateContentResponse with a single candidate."""
    candidate = {"content": {"role": "model", "parts": parts}, "index": 0}
    if final:
        candidate["finishReason"] = "STOP"
        grounding = _grounding(body)
        if grounding:
            candidate["groundingMetadata"] = grounding
    response = {"candidates": [candidate]}
    if usage:
        response["usageMetadata"] = usage
    return response


def _answer(model: str, body) -> dict:
    task = task_answer(_last_user_text(body))
    if task:
        return _chunk([{"text": task}], body, True, USAGE_TEXT)
    if "image" in model:
        parts = [
            {"text": "Mock image thinking.", "thought": True},
            {"text": "Here is your image."},
            IMAGE_PART,
        ]
        return _chunk(parts, body, True, USAGE_IMAGE)
    parts = [
        {"text": "Mock thinking.", "thought": True},
        {"text": "Hello from mock (non-stream)."},
    ]
    return _chunk(parts, body, True, USAGE_TEXT)


def _stream_chunks(model: str, body) -> list:
    if "image" in model:
        return [
            _chunk([{"text": "Pondering.", "thought": True}], body),
            _chunk([IMAGE_PART], body),
            _chunk([{"text": ""}], body, True, USAGE_IMAGE),
        ]
    task = task_answer(_last_user_text(body))
    pieces = [task] if task else ["Hello ", "from mock ", "(stream)."]
    chunks = [_chunk([{"text": "Mock pondering.", "thought": True}], body)]
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
    annotate(request, tool_kinds=_tool_kinds(body if isinstance(body, dict) else {}))
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
        return web.json_response(
            {"name": f"models/{model}/operations/op-{next(_op_ids)}"}
        )
    return _not_found(request)


async def get_operation(request: web.Request) -> web.Response:
    await record(request)
    model, op = request.match_info["model"], request.match_info["op"]
    # Real Veo answers carry a Files API URI; google-genai extracts the file id
    # from it and downloads <BASE_URL>/<version>/files/<id>:download from us.
    file_id = "video" + op.rsplit("-", 1)[-1]
    video = {"uri": f"{GOOGLE_FILES}/{file_id}:download?alt=media"}
    return web.json_response(
        {
            "name": f"models/{model}/operations/{op}",
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
