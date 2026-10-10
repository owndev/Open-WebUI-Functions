"""
Mock of the Gemini Developer REST API (the subset google-genai uses).

Point the pipe's ``BASE_URL`` valve at ``http://127.0.0.1:<port>/``.

Routes
  GET  /{version}/models                                model list
  POST /{version}/models/{model}:generateContent        JSON answer
  POST /{version}/models/{model}:streamGenerateContent  SSE answer (?alt=sse)
  POST /{version}/models/{model}:predict                Imagen (Open WebUI's image engine
                                                        "gemini", endpoint method predict)
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
  - Open WebUI's own image engine "gemini" (built-in tools generate_image /
    edit_image): its generateContent requests to an image model get the image
    answer above (no thoughts: it asks for none); ``:predict`` (Imagen) answers
    ``predictions`` with ``parameters.sampleCount`` (default 1) PNGs, FINAL_PNG
    or the ``final-image-<n>`` PNG of the prompt (``instances`` as an object, as
    Open WebUI sends it, or as a list)

Tools an image model does not support are rejected like the real API does
(HTTP 400 INVALID_ARGUMENT; an approximation of the real messages):
functionDeclarations and urlContext for every image model, googleSearch for
gemini-2.5-flash-image and gemini-3.1-flash-lite-image. Model ids containing
"imagegen" stand for image models the pipe does not recognise and accept tools.

Triggers (a word in the turn's user message: the last user content with a text
part that is not Open WebUI's tool-image message, see ``_turn_text``)
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
  final-image-<n>      image models (non-stream): the final image is the PNG of
                       colour (255, 255, n) instead of FINAL_PNG (n = 1-255), so
                       the images of the turns of a chat differ
  slow-video           Veo: the operation stays pending for SLOW_VIDEO_POLLS polls
  paced-thinking       streaming text models: ten thought parts "Paced thought <i>."
                       0.15 s apart, then the answer "Paced answer done." in three
                       parts 1.2 s apart (the first 0.15 s after the last thought)
  slow-thinking        streaming text models: twelve thought parts "Slow thought <i>."
                       0.5 s apart, then the answer "Slow answer."

Native tool calling: ``MOCKTOOLS:{json}`` in the turn's user message
  {"rounds": [[{"name": "get_current_timestamp", "args": {}}], [...]],
   "text_before": "Let me check.", "split": false, "noid": false, "nosig": false,
   "server_side": false, "malformed": false, "allow_undeclared": false}
  - r = number of user contents with functionResponse parts after the turn's user
    content (at least 1 + the highest round of a response id mock-call-<r>-<k>:
    Open WebUI merges rounds without text or reasoning in between into one
    message). r < len(rounds): answer with the function calls of rounds[r];
    otherwise with "MOCK-FINAL <name>=<json response>; ..." over every
    functionResponse of the turn, in order
  - names are Open WebUI names; the call uses _gemini_function_name(name) (the
    pipe's name mapping, copied) and must be declared, else the answer is the text
    "MOCK-ERROR undeclared function <name>; declared=[...]" (HTTP 200) unless
    ``allow_undeclared``
  - ids "mock-call-<r>-<k>" (none with ``noid``); gemini-3* models get a
    thoughtSignature on the first call of a round (not with ``nosig``):
    base64("mock-sig|<model>|<id or name>")
  - parts: thought (when requested), text_before, the server-side toolCall +
    toolResponse (``server_side`` and the request sets
    toolConfig.includeServerSideToolInvocations), the calls. Streaming: one chunk
    each, the calls in one chunk (one per call with ``split``), then {"text": ""}
    with finishReason STOP and USAGE_TOOL
  - with googleSearch, a round with calls cites its own source
    (https://example.com/tool-round, no supports without text_before); the
    final answer cites https://example.com/a
  - ``malformed``: the first request gets a candidate without content and
    finishReason MALFORMED_FUNCTION_CALL (or the finish reason given as a
    string, e.g. ``"malformed": "UNEXPECTED_TOOL_CALL"``)
  HTTP 400 INVALID_ARGUMENT like the real API (every generate request):
  duplicate declaration names, invalid names, ``parameters`` together with
  ``parametersJsonSchema``, a model content with functionCall parts that is not
  directly followed by a user content with matching functionResponse parts
  (count, names, ids), and for gemini-3* the first functionCall part of each
  model content of the current turn without the issued signature (or
  "skip_thought_signature_validator").

Each recorded request carries ``model``, ``action`` (generateContent,
streamGenerateContent, predict, predictLongRunning, download), ``tool_kinds`` (keys of
the request's ``tools`` entries, e.g. ``googleSearch``) and, for generate
requests, ``status`` (the HTTP status the mock answered with) plus what the
tool scenarios assert on (``_tool_view``): ``declared``, ``decl`` (raw schema per
declaration, recorded before any key normalisation), ``tool_config``,
``fc_mode``, ``allowed_names``, ``include_flag``, ``kinds`` (role and part kinds
per content), ``fc`` / ``fr`` (function call / response parts with content
index, id, name, args / response; ``sig`` of a call is none, skip, issued or
bad), ``round``, ``directive``, ``answer`` (fc, final, text, malformed,
mock-error, http-<status>), ``issued``, ``server_echoed``,
``synthetic_ids_upstream`` and ``orphan_fr``.

usage: python mock_gemini.py [--port 9101]
"""

import argparse
import asyncio
import base64
import hashlib
import itertools
import json
import re
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
# usage of a round that ends with function calls (MOCKTOOLS)
USAGE_TOOL = {"promptTokenCount": 21, "candidatesTokenCount": 5, "totalTokenCount": 26}
USAGE_IMAGE = {
    "promptTokenCount": 12,
    "candidatesTokenCount": 1290,
    "totalTokenCount": 1302,
}
GOOGLE_FILES = "https://generativelanguage.googleapis.com/v1beta/files"
GENERATE = ("generateContent", "streamGenerateContent")
NO_SEARCH_MODELS = ("gemini-2.5-flash-image", "gemini-3.1-flash-lite-image")
SLOW_VIDEO_POLLS = 6
# The source of a tool round (MOCKTOOLS) with Google Search; final answers cite
# https://example.com/a
TOOL_ROUND_CHUNK = {
    "web": {"uri": "https://example.com/tool-round", "title": "Tool round"}
}
VERTEX_CHUNK = {
    "retrievedContext": {
        "uri": "gs://e2e-bucket/doc.pdf",
        "title": "Vertex Doc",
        "text": "chunk body",
    }
}
_op_ids = itertools.count(1)
_slow_ops: dict = {}  # operation id -> polls left before it is done
# Paced streams: trigger -> (thought texts, seconds between thoughts, answer
# parts, seconds between answer parts); the thinking ends where the answer starts
PACED = {
    "paced-thinking": (
        [f"Paced thought {i}." for i in range(1, 11)],
        0.15,
        ["Paced ", "answer ", "done."],
        1.2,
    ),
    "slow-thinking": (
        [f"Slow thought {i}." for i in range(1, 13)],
        0.5,
        ["Slow answer."],
        0.05,
    ),
}
STREAM_PAUSE = 0.05  # seconds after every streamed chunk


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
FINAL_IMAGE_TRIGGER = re.compile(r"\bfinal-image-(\d{1,3})\b")


def _final_png(text: str) -> str:
    """The final image of an image model: FINAL_PNG, or for the trigger
    ``final-image-<n>`` (1-255) the PNG of colour (255, 255, n)."""
    match = FINAL_IMAGE_TRIGGER.search(text)
    n = int(match.group(1)) if match else 0
    return png_b64(255, 255, n) if 0 < n < 256 else FINAL_PNG


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
    """Keys of the request's ``tools`` entries, camelCased (``googleSearch``)."""
    kinds = []
    for tool in (body or {}).get("tools") or []:
        if isinstance(tool, dict):
            kinds.extend(_camel(key) for key in tool.keys())
    return kinds


# ------------------------------------------------------- native tool calling
DIRECTIVE = "MOCKTOOLS:"
CALL_ID = re.compile(r"^mock-call-(\d+)-\d+$")
SKIP_SIGNATURE = "skip_thought_signature_validator"
# Open WebUI moves images of tool results into a user message with this text
TOOL_IMAGE_TEXT = "Here are the images from the tool results above"
# FunctionDeclaration.name accepted by the Gemini API
DECLARATION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")
FR_MISMATCH = (
    "Please ensure that the number of function response parts is equal to the "
    "number of function call parts of the function call turn."
)
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

# The pipe's name mapping (spec section 3.4), copied: a declared Open WebUI
# tool name -> the name Gemini sees.
_GEMINI_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


def _gemini_function_name(name: str) -> str:
    if _GEMINI_FUNCTION_NAME_RE.fullmatch(name):
        return name
    base = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    if not re.match(r"[A-Za-z_]", base[:1] or "0"):
        base = "_" + base
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base[:55]}_{digest}"  # <= 64 chars


def _key(data, camel: str):
    """Value of ``camel`` or its snake_case spelling (google-genai mixes both)."""
    if not isinstance(data, dict):
        return None
    if camel in data:
        return data[camel]
    return data.get(re.sub(r"([A-Z])", lambda m: "_" + m.group(1).lower(), camel))


def _contents(body) -> list:
    if not isinstance(body, dict):
        return []
    return [c for c in body.get("contents") or [] if isinstance(c, dict)]


def _role(content: dict) -> str:
    return content.get("role") or "user"


def _parts(content: dict) -> list:
    return [p for p in content.get("parts") or [] if isinstance(p, dict)]


def _part_kind(part: dict) -> str:
    for camel, kind in PART_KINDS:
        if _key(part, camel) is not None:
            return kind
    if "text" in part:
        return "thought" if part.get("thought") else "text"
    return "other"


def _has(content: dict, camel: str) -> bool:
    return any(_key(p, camel) is not None for p in _parts(content))


def _turn(body) -> tuple:
    """(index, text) of the turn's user content: the last user content with a
    text part that is not Open WebUI's tool-image message ((-1, '') if none)."""
    contents = _contents(body)
    for index in range(len(contents) - 1, -1, -1):
        content = contents[index]
        if _role(content) != "user":
            continue
        texts = [p["text"] for p in _parts(content) if isinstance(p.get("text"), str)]
        text = " ".join(texts)
        if texts and not text.startswith(TOOL_IMAGE_TEXT):
            return index, text
    return -1, ""


def _turn_text(body) -> str:
    """Text of the turn's user message (where triggers and MOCKTOOLS are read)."""
    return _turn(body)[1]


def _sig_text(sig) -> str:
    """Decoded thought signature (standard or URL-safe base64), '' if invalid."""
    if not isinstance(sig, str) or not sig:
        return ""
    data = sig.replace("-", "+").replace("_", "/")
    try:
        return base64.b64decode(data + "=" * (-len(data) % 4)).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return ""


def issued_signature(model: str, ref: str) -> str:
    """The thought signature the mock issues for a call (ref = id or name)."""
    return base64.b64encode(f"mock-sig|{model}|{ref}".encode()).decode()


def _sig_kind(sig, call: dict) -> str:
    """none / skip / issued (a mock signature for this call's id or name) / bad."""
    if sig in (None, ""):
        return "none"
    if sig == SKIP_SIGNATURE:
        return "skip"
    text = _sig_text(sig)
    ref = call.get("id") or call.get("name")
    if text.startswith("mock-sig|") and text.rsplit("|", 1)[-1] == ref:
        return "issued"
    return "bad"


def _directive(text: str) -> Optional[dict]:
    """The MOCKTOOLS directive of the turn (None without one)."""
    pos = text.find(DIRECTIVE)
    if pos < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[pos + len(DIRECTIVE) :].lstrip())
    except ValueError:
        return {"error": "unparsable MOCKTOOLS directive"}
    if not isinstance(value, dict):
        return {"error": "MOCKTOOLS directive is not a JSON object"}
    return value


def _tool_view(model: str, body) -> dict:
    """What a generate request carries for tool calling (recorded per request)."""
    contents = _contents(body)
    turn_index, text = _turn(body)
    declared, decl, duplicates, invalid, both = [], {}, [], [], []
    for tool in (body.get("tools") if isinstance(body, dict) else None) or []:
        for item in _key(tool, "functionDeclarations") or []:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if name in declared:
                duplicates.append(name)
            declared.append(name)
            # fullmatch: "$" alone would accept a trailing newline
            if not isinstance(name, str) or not DECLARATION_NAME.fullmatch(name):
                invalid.append(name)
            schema_key = next(
                (
                    k
                    for k in ("parametersJsonSchema", "parameters_json_schema")
                    if k in item
                ),
                None,
            )
            if schema_key and "parameters" in item:
                both.append(name)
            decl.setdefault(
                str(name),
                {  # raw: property names inside schemas are not normalised
                    "schema": item.get(schema_key) if schema_key else None,
                    "parameters": item.get("parameters"),
                    "has_description": bool(item.get("description")),
                    "keys": sorted(item),
                },
            )
    tool_config = _key(body, "toolConfig") if isinstance(body, dict) else None
    config = normalize(tool_config) if isinstance(tool_config, dict) else {}
    calling = config.get("functionCallingConfig") or {}
    fc, fr, kinds = [], [], []
    for index, content in enumerate(contents):
        kinds.append(
            f"{_role(content)}:" + "+".join(_part_kind(p) for p in _parts(content))
        )
        first = True
        for part in _parts(content):
            call = _key(part, "functionCall")
            if isinstance(call, dict):
                fc.append(
                    {
                        "i": index,
                        "id": call.get("id"),
                        "name": call.get("name"),
                        "args": call.get("args"),
                        "sig": _sig_kind(_key(part, "thoughtSignature"), call),
                        "first": first,
                    }
                )
                first = False
            response = _key(part, "functionResponse")
            if isinstance(response, dict):
                fr.append(
                    {
                        "i": index,
                        "id": response.get("id"),
                        "name": response.get("name"),
                        "response": response.get("response"),
                    }
                )
    after = contents[turn_index + 1 :] if turn_index >= 0 else []
    # Rounds answered: user contents with function responses, or more when the
    # responses carry the mock's call ids (mock-call-<round>-<k>): Open WebUI
    # replays two rounds without text or reasoning in between as ONE assistant
    # message with all calls, followed by all results.
    answered_rounds = [
        int(match.group(1))
        for x in fr
        if x["i"] > turn_index
        for match in [CALL_ID.match(str(x.get("id") or ""))]
        if match
    ]
    rounds_done = max(
        sum(1 for c in after if _role(c) == "user" and _has(c, "functionResponse")),
        max(answered_rounds, default=-1) + 1,
    )
    orphan_fr = [
        i
        for i, c in enumerate(contents)
        if _role(c) == "user"
        and _has(c, "functionResponse")
        and not (
            i > 0
            and _role(contents[i - 1]) == "model"
            and _has(contents[i - 1], "functionCall")
        )
    ]
    directive = _directive(text)
    server_echoed = None
    if directive and directive.get("server_side") and rounds_done:
        server_echoed = True
        round_no = 0
        for content in after:
            if _role(content) != "model" or not _has(content, "functionCall"):
                continue
            tool_call = next(
                (p for p in _parts(content) if _key(p, "toolCall") is not None), None
            )
            tool_response = next(
                (p for p in _parts(content) if _key(p, "toolResponse") is not None),
                None,
            )
            server_echoed = (
                server_echoed
                and tool_call is not None
                and tool_response is not None
                and (_key(tool_call, "toolCall") or {}).get("id")
                == f"mock-srv-{round_no}"
                and _sig_text(_key(tool_call, "thoughtSignature"))
                == f"mock-srv-sig|{round_no}"
            )
            round_no += 1
    return {
        "declared": declared,
        "decl": decl,
        "tool_config": tool_config,
        "fc_mode": calling.get("mode"),
        "allowed_names": calling.get("allowedFunctionNames"),
        "include_flag": config.get("includeServerSideToolInvocations"),
        "kinds": kinds,
        "fc": fc,
        "fr": fr,
        "turn": text[:160],
        "turn_index": turn_index,
        "round": rounds_done,
        "directive": directive,
        "server_echoed": server_echoed,
        "synthetic_ids_upstream": any(
            str(x.get("id") or "").startswith("owui_") for x in fc + fr
        ),
        "orphan_fr": orphan_fr,
        "_duplicates": duplicates,
        "_invalid": invalid,
        "_both": both,
    }


def _validate(model: str, body, view: dict) -> Optional[web.Response]:
    """HTTP 400 for requests the real API rejects (tool calling rules)."""

    def bad(message: str) -> web.Response:
        return _error(400, message, "INVALID_ARGUMENT")

    if view["_duplicates"]:
        return bad(f"Duplicate function declaration found: {view['_duplicates'][0]}")
    if view["_invalid"]:
        return bad(f"Invalid function name: {view['_invalid'][0]}")
    if view["_both"]:
        return bad(
            f"Function declaration '{view['_both'][0]}' sets both parameters and "
            "parametersJsonSchema"
        )
    contents = _contents(body)
    for index, content in enumerate(contents):
        if _role(content) != "model" or not _has(content, "functionCall"):
            continue
        calls = [x for x in view["fc"] if x["i"] == index]
        nxt = contents[index + 1] if index + 1 < len(contents) else {}
        responses = [x for x in view["fr"] if x["i"] == index + 1]
        if (
            not nxt
            or _role(nxt) != "user"
            or len(responses) != len(calls)
            or sorted(str(x["name"]) for x in calls)
            != sorted(str(x["name"]) for x in responses)
            or any(
                call["id"] and call["id"] not in {r["id"] for r in responses}
                for call in calls
            )
        ):
            return bad(FR_MISMATCH)
    if model.startswith("gemini-3"):
        # current turn: the contents after the last user content with a part that
        # is not a function response
        start = max(
            (
                i
                for i, c in enumerate(contents)
                if _role(c) == "user" and any(_part_kind(p) != "fr" for p in _parts(c))
            ),
            default=-1,
        )
        for call in view["fc"]:
            if call["i"] <= start or not call["first"]:
                continue
            if call["sig"] == "none":
                return bad(
                    f"Function call `{call['name']}` in the {call['i']}. content "
                    "block is missing a `thought_signature`."
                )
            if call["sig"] == "bad":
                return bad("Corrupted thought signature.")
    return None


def _tool_plan(model: str, view: dict) -> Optional[dict]:
    """The answer to a MOCKTOOLS turn (None without a directive)."""
    directive = view["directive"]
    if directive is None:
        return None
    if "error" in directive:
        return {"answer": "mock-error", "text": f"MOCK-ERROR {directive['error']}"}
    rounds = directive.get("rounds") or []
    r = view["round"]
    if directive.get("malformed") and r == 0:
        finish = directive["malformed"]
        if not isinstance(finish, str):
            finish = "MALFORMED_FUNCTION_CALL"
        return {"answer": "malformed", "finish": finish}
    if r >= len(rounds):
        responses = [x for x in view["fr"] if x["i"] > view["turn_index"]]
        text = "MOCK-FINAL " + "; ".join(
            f"{x['name']}={json.dumps(x['response'], sort_keys=True)}"
            for x in responses
        )
        return {"answer": "final", "text": text}
    declared = {str(name) for name in view["declared"]}
    calls, issued = [], []
    for k, spec in enumerate(rounds[r] or []):
        name = str((spec or {}).get("name") or "")
        gemini_name = _gemini_function_name(name)
        if gemini_name not in declared:
            if not directive.get("allow_undeclared"):
                return {
                    "answer": "mock-error",
                    "text": f"MOCK-ERROR undeclared function {name}; "
                    f"declared={sorted(declared)}",
                }
            gemini_name = name
        call = {"name": gemini_name, "args": (spec or {}).get("args") or {}}
        if not directive.get("noid"):
            call["id"] = f"mock-call-{r}-{k}"
        part = {"functionCall": call}
        if k == 0 and model.startswith("gemini-3") and not directive.get("nosig"):
            part["thoughtSignature"] = issued_signature(
                model, call.get("id") or gemini_name
            )
        calls.append(part)
        issued.append(
            {
                "id": call.get("id"),
                "name": gemini_name,
                "sig": part.get("thoughtSignature"),
            }
        )
    server = []
    if directive.get("server_side") and view["include_flag"]:
        srv = f"mock-srv-{r}"
        server = [
            {
                "toolCall": {
                    "id": srv,
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "args": {"queries": ["mock query"]},
                },
                "thoughtSignature": base64.b64encode(
                    f"mock-srv-sig|{r}".encode()
                ).decode(),
            },
            {
                "toolResponse": {
                    "id": srv,
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "response": {"results": "mock"},
                }
            },
        ]
    return {
        "answer": "fc",
        "calls": calls,
        "issued": issued,
        "server": server,
        "text_before": str(directive.get("text_before") or ""),
        "split": bool(directive.get("split")),
    }


def _malformed(plan: dict) -> dict:
    """A candidate without content and the directive's finishReason."""
    return {
        "candidates": [{"finishReason": plan["finish"], "index": 0}],
        "usageMetadata": USAGE_TEXT,
    }


def _tool_answer(model: str, body, plan: dict) -> dict:
    """generateContent answer of a MOCKTOOLS turn."""
    if plan["answer"] == "malformed":
        return _malformed(plan)
    parts = (
        [{"text": "Mock thinking.", "thought": True}] if _thoughts(model, body) else []
    )
    if plan["answer"] != "fc":
        return _chunk(parts + [{"text": plan["text"]}], body, True, USAGE_TEXT)
    if plan["text_before"]:
        parts.append({"text": plan["text_before"]})
    end = _chunk(parts + plan["server"] + plan["calls"], body, True, USAGE_TOOL)
    return _round_grounding(end, plan)


def _round_grounding(response: dict, plan: dict) -> dict:
    """Grounding metadata of a tool round: its own source (TOOL_ROUND_CHUNK, so
    the sources of each round can be told apart) and, without text, nothing to
    cite (no supports)."""
    for candidate in response.get("candidates") or []:
        metadata = candidate.get("groundingMetadata") or {}
        if metadata.get("groundingChunks"):
            metadata["groundingChunks"] = [TOOL_ROUND_CHUNK]
        if not plan["text_before"]:
            metadata.pop("groundingSupports", None)
    return response


def _tool_stream_chunks(model: str, body, plan: dict) -> list:
    """streamGenerateContent chunks of a MOCKTOOLS turn."""
    if plan["answer"] == "malformed":
        return [_malformed(plan)]
    chunks = []
    if _thoughts(model, body):
        chunks.append(_chunk([{"text": "Mock pondering.", "thought": True}], body))
    if plan["answer"] != "fc":
        chunks.append(_chunk([{"text": plan["text"]}], body, True, USAGE_STREAM))
        return chunks
    if plan["text_before"]:
        chunks.append(_chunk([{"text": plan["text_before"]}], body))
    if plan["server"]:
        chunks.append(_chunk(plan["server"], body))
    if plan["split"]:
        chunks += [_chunk([call], body) for call in plan["calls"]]
    else:
        chunks.append(_chunk(plan["calls"], body))
    chunks.append(
        _round_grounding(_chunk([{"text": ""}], body, True, USAGE_TOOL), plan)
    )
    return chunks


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
        if entry.get("action") in GENERATE and trigger in _turn_text(entry.get("body"))
    ]
    return len(seen) == 1


def _reject(request: web.Request, model: str, body: dict) -> Optional[web.Response]:
    """Errors on purpose (triggers) and tools a model does not support (400)."""
    text = _turn_text(body)
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
    if "vertex-context" in _turn_text(body):
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
    text = _turn_text(body)
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
            parts += [{"text": "Here is your image."}, _img(_final_png(text))]
        return _chunk(parts, body, True, USAGE_IMAGE)
    parts = [{"text": "Mock thinking.", "thought": True}] if thoughts else []
    answer = "Hello from mock (non-stream)."
    if "data-prefix" in text:
        answer = "data: starts like SSE."
    parts.append({"text": answer})
    return _chunk(parts, body, True, USAGE_TEXT)


def _stream_chunks(model: str, body) -> list:
    text = _turn_text(body)
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
    paced = next((key for key in PACED if key in text), None)
    if paced and not task:
        return _paced_chunks(body, thoughts, *PACED[paced])
    pieces = [task] if task else ["Hello ", "from mock ", "(stream)."]
    if "data-prefix" in text:
        pieces = ["data: starts ", "like SSE."]
    chunks = []
    if thoughts:
        chunks.append(_chunk([{"text": "Mock pondering.", "thought": True}], body))
    chunks += [_chunk([{"text": piece}], body) for piece in pieces[:-1]]
    chunks.append(_chunk([{"text": pieces[-1]}], body, True, USAGE_STREAM))
    return chunks


def _paced_chunks(body, thoughts, texts, thought_gap, pieces, piece_gap) -> list:
    """Chunks of a paced stream; a float is an extra pause (seconds) after the
    previous chunk, on top of STREAM_PAUSE."""
    items = []
    if thoughts:
        for text in texts:
            items += [
                _chunk([{"text": text, "thought": True}], body),
                thought_gap - STREAM_PAUSE,
            ]
    for piece in pieces[:-1]:
        items += [_chunk([{"text": piece}], body), piece_gap - STREAM_PAUSE]
    items.append(_chunk([{"text": pieces[-1]}], body, True, USAGE_STREAM))
    return items


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
    plan = None
    if action in GENERATE:
        view = _tool_view(model, body)
        annotate(request, **{k: v for k, v in view.items() if not k.startswith("_")})
        rejected = _reject(request, model, body) or _validate(model, body, view)
        annotate(request, status=rejected.status if rejected is not None else 200)
        if rejected is not None:
            annotate(request, answer=f"http-{rejected.status}")
            return rejected
        text = _turn_text(body)
        if not (task_answer(text) or _blocked(text)):
            plan = _tool_plan(model, view)
        annotate(
            request,
            answer=plan["answer"] if plan else "text",
            issued=(plan or {}).get("issued"),
        )
    if action == "generateContent":
        if plan:
            return web.json_response(_tool_answer(model, body, plan))
        return web.json_response(_answer(model, body))
    if action == "streamGenerateContent":
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        chunks = (
            _tool_stream_chunks(model, body, plan)
            if plan
            else _stream_chunks(model, body)
        )
        for chunk in chunks:
            if isinstance(chunk, float):  # pause of a paced stream
                await asyncio.sleep(max(0.0, chunk))
                continue
            await resp.write(f"data: {json.dumps(chunk)}\r\n\r\n".encode())
            await asyncio.sleep(STREAM_PAUSE)
        await resp.write_eof()
        return resp
    if action == "predict":
        return web.json_response(_predict(body))
    if action == "predictLongRunning":
        op = f"op-{next(_op_ids)}"
        prompt = str(((body.get("instances") or [{}])[0] or {}).get("prompt") or "")
        if "slow-video" in prompt:
            _slow_ops[op] = SLOW_VIDEO_POLLS
        return web.json_response({"name": f"models/{model}/operations/{op}"})
    return _not_found(request)


def _predict(body: dict) -> dict:
    """Imagen ``:predict`` answer: ``sampleCount`` PNGs for the prompt."""
    instances = body.get("instances")
    if isinstance(instances, list):
        instances = instances[0] if instances else {}
    prompt = str((instances if isinstance(instances, dict) else {}).get("prompt") or "")
    count = (body.get("parameters") or {}).get("sampleCount") or 1
    image = {"bytesBase64Encoded": _final_png(prompt), "mimeType": "image/png"}
    return {"predictions": [dict(image) for _ in range(max(1, min(int(count), 4)))]}


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
