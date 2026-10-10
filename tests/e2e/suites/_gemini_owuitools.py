"""
Group ``owuitools`` of the gemini suite: Open WebUI's own built-in tools
``generate_image`` / ``edit_image`` (Open WebUI's image engine "gemini") and
``execute_code`` (code interpreter), called by a Gemini *text* model through
Open WebUI's tool loop (native function calling, google_gemini.py 1.18.0). Run
by suites/gemini.py; the leading underscore keeps this module out of the suite
discovery.

Setup (restored at the end): Admin Settings -> Images with engine "gemini"
pointing at the Gemini mock (``IMAGES_GEMINI_API_BASE_URL`` = <mock>/v1beta, an
API key of its own for generation and for editing), image editing switched on;
the code interpreter is on by default (engine pyodide). The pipe models carry
no capabilities, so every capability counts as on (Open WebUI's default).

Open WebUI 0.11 puts the tools into ``body["tools"]`` only for a browser
session (``session_id``), with function calling not "legacy" and the model
capability ``builtin_tools``, and per tool when the model's builtinTools
category, the global switch (``image_generation.enable``;
``images.edit.enable`` for edit_image; ``code_interpreter.enable``), the model
capability, the request's ``features`` flag (the chat toggle) and the user
permission ``features.<name>`` (admins pass) are all on
(utils/tools.py get_builtin_tools, utils/middleware.py).

The Gemini mock answers the ``MOCKTOOLS:`` directive with the tool call; Open
WebUI runs the tool: the image engine requests reach the same mock (``:predict``
for Imagen, ``:generateContent`` for an image model, recorded with their
``x-goog-api-key``), ``execute_code`` asks the browser session
(``execute:python``, answered by ``BrowserSession.python_answer`` the way the
web UI's pyodide worker acks) or the Jupyter mock of mocks/mock_tools.py.

Every detail starts with ``mock_ok=<bool>`` (the preflight of
suites/_gemini_tools.py); the tokens follow ``_bdetail`` there, plus
``engine=[action:model:key]`` (the image engine requests), ``file_ok=`` (the
saved image has the bytes the engine answered) and ``sent_ok=`` (the image the
edit engine got is the one the call named).
"""

import base64

from harness import Suite, short
from harness.config import MOCK_HOST, MOCK_PORTS

from ._gemini_tools import (
    CALL_0,
    PRO3_ID,
    Ctx,
    _bdetail,
    _calls,
    _declared,
    _final,
    _frs,
    _json,
    _req_tokens,
    directive,
    preflight,
)
from .gemini import (
    IMAGE_GA,
    PRO3,
    TEXT,
    _body,
    _gen,
    _png,
    _status_closed,
    _strings,
    _upload_png,
)

IMG_KEY = "mock-images-key"  # Open WebUI's image generation key (not the pipe's)
EDIT_KEY = "mock-images-edit-key"
IMG_MODEL = "gemini-2.5-flash-image"  # endpoint method generateContent
IMAGEN = "imagen-4.0-generate-001"  # endpoint method predict
EDIT_MODEL = "gemini-3.1-flash-image"  # also a pipe model: engine requests are
# told apart by their key
ENGINE_KEYS = (IMG_KEY, EDIT_KEY)
CODE_TOOL = "execute_code"
OWUI_TOOLS = ("generate_image", "edit_image", CODE_TOOL)
IMG_ON = {"image_generation": True}
CODE_ON = {"code_interpreter": True}
# Open WebUI's system prompt addition for the pyodide engine (native mode)
PYODIDE_PROMPT = "Pyodide Environment"
JUPYTER_URL = f"http://{MOCK_HOST}:{MOCK_PORTS['tools']}/jupyter"
# mocks/mock_tools.py's answer to an execute_request
JUPYTER_STDOUT = "JUPYTER-MOCK-STDOUT"
JUPYTER_RESULT = "'JUPYTER-MOCK-RESULT'"
CODE = "print(2 + 2)"
PLOT_PNG = _png(1, 2, 3)  # a matplotlib-style image line in the pyodide stdout
UPLOAD_PNG = _png(10, 20, 30)


def _engine_key(entry: dict) -> str:
    return (entry.get("headers") or {}).get("x-goog-api-key") or ""


def _engine(entry: dict) -> bool:
    """A request of Open WebUI's image engine (generation or editing): its
    own API keys, not the pipe's."""
    return _engine_key(entry) in ENGINE_KEYS


def _pipe(model_id: str = PRO3_ID):
    """Match the pipe's generate requests to ``model_id``."""
    return lambda e: _gen(e) and e.get("model") == model_id and not _engine(e)


def _engine_tokens(entries: list) -> str:
    items = [f"{e.get('action')}:{e.get('model')}:{_engine_key(e)}" for e in entries]
    return "[" + ",".join(items) + "]"


def _engine_text(entry: dict) -> str:
    """The prompt the image engine sent (generateContent or predict)."""
    body = entry.get("body") if isinstance(entry.get("body"), dict) else {}
    if entry.get("action") == "predict":
        instances = body.get("instances")
        if isinstance(instances, list):
            instances = instances[0] if instances else {}
        return str((instances if isinstance(instances, dict) else {}).get("prompt"))
    return " ".join(
        p.get("text", "")
        for c in body.get("contents") or []
        for p in c.get("parts") or []
        if isinstance(p, dict) and "text" in p
    )


def _engine_images(entry: dict) -> list:
    """Base64 of the images the edit engine sent (inline_data parts)."""
    body = entry.get("body") if isinstance(entry.get("body"), dict) else {}
    images = []
    for content in body.get("contents") or []:
        for part in content.get("parts") or []:
            data = (part.get("inline_data") or part.get("inlineData") or {}).get("data")
            if data:
                images.append(data)
    return images


def _engine_ok(engine: list, path: str, key: str, prompt: str) -> bool:
    return (
        len(engine) == 1
        and engine[0].get("path") == path
        and _engine_key(engine[0]) == key
        and _engine_text(engine[0]) == prompt
    )


def _owui_tools(names: list) -> list:
    return [n for n in OWUI_TOOLS if n in names]


def _system(entry: dict) -> str:
    return " ".join(_strings(_body(entry).get("systemInstruction")))


async def _config(owui, path: str) -> dict:
    status, data = await owui.api("GET", path)
    if status != 200 or not isinstance(data, dict):
        raise RuntimeError(f"GET {path}: HTTP {status} {short(data)}")
    return data


async def _post(owui, path: str, body: dict) -> None:
    status, data = await owui.api("POST", path, body)
    if status != 200:
        raise RuntimeError(f"POST {path}: HTTP {status} {short(data)}")


async def _set_images(owui, **changes) -> dict:
    """Change Admin Settings -> Images (the full form is sent); returns the
    previous settings."""
    previous = await _config(owui, "/api/v1/images/config")
    await _post(owui, "/api/v1/images/config/update", {**previous, **changes})
    return previous


async def _set_code(owui, **changes) -> dict:
    """Change Admin Settings -> Code Execution; returns the previous settings."""
    previous = await _config(owui, "/api/v1/configs/code_execution")
    await _post(owui, "/api/v1/configs/code_execution", {**previous, **changes})
    return previous


async def _file_b64(owui, url: str) -> str:
    """Base64 of an Open WebUI file URL ('' if there is none)."""
    if not url:
        return ""
    r = await owui.request("GET", url)
    return base64.b64encode(r.content).decode() if r.status_code == 200 else ""


def _first_url(files: list) -> str:
    return (files[0] if files else {}).get("url") or ""


def _tool_call(name: str, **args) -> str:
    return directive([[{"name": name, "args": args}]])


# ================================================================= owuitools
async def owuitools(t: Suite, mock) -> None:
    ctx = await preflight(t, mock)
    images_before = code_before = None
    try:
        images_before = await _set_images(
            t.owui,
            ENABLE_IMAGE_GENERATION=True,
            IMAGE_GENERATION_ENGINE="gemini",
            IMAGE_GENERATION_MODEL=IMG_MODEL,
            IMAGES_GEMINI_API_BASE_URL=mock.url + "/v1beta",
            IMAGES_GEMINI_API_KEY=IMG_KEY,
            IMAGES_GEMINI_ENDPOINT_METHOD="generateContent",
            ENABLE_IMAGE_EDIT=True,
            IMAGE_EDIT_ENGINE="gemini",
            IMAGE_EDIT_MODEL=EDIT_MODEL,
            IMAGES_EDIT_GEMINI_API_BASE_URL=mock.url + "/v1beta",
            IMAGES_EDIT_GEMINI_API_KEY=EDIT_KEY,
        )
        code_before = await _set_code(
            t.owui, ENABLE_CODE_INTERPRETER=True, CODE_INTERPRETER_ENGINE="pyodide"
        )
        await api_path(ctx)  # before any websocket session of this user
        async with t.browser() as b:
            await declared(ctx, b)
            generated = await generate(ctx, b)
            await generate_predict(ctx, b)
            await edit(ctx, b)
            await edit_followup(ctx, b, generated)
            await image_model(ctx, b)
            await code(ctx, b)
            await code_jupyter(ctx, b)
    finally:
        if images_before is not None:
            await _post(t.owui, "/api/v1/images/config/update", images_before)
        if code_before is not None:
            await _post(t.owui, "/api/v1/configs/code_execution", code_before)


async def api_path(ctx: Ctx) -> None:
    """API clients (no session_id) get no hidden built-in tools, toggles or not."""
    await ctx.mock.reset()
    r = await ctx.t.owui.chat(
        TEXT,
        "Hello API with image and code features",
        stream=False,
        features={**IMG_ON, **CODE_ON},
    )
    reqs = await ctx.mock.requests(_pipe(TEXT.split(".", 1)[1]))
    names = _declared(reqs[0]) if reqs else []
    ctx.check(
        "owuitools.api",
        "API path with features image_generation + code_interpreter: no "
        "generate_image / edit_image / execute_code declared (Open WebUI adds its "
        "built-in tools for browser sessions only)",
        r.status == 200
        and "Hello from mock" in r.content
        and len(reqs) == 1
        and not _owui_tools(names),
        f"http={r.status} upstream={[e.get('answer') for e in reqs]} "
        f"declared_n={len(names)} owui_tools={_owui_tools(names)} "
        f"pyodide_note={PYODIDE_PROMPT in _system(reqs[0] if reqs else {})} "
        f"content={short(r.content, 120)}",
    )


async def _plain_turn(ctx: Ctx, b, label: str, features: dict) -> tuple:
    """A turn without tool call; (chat, pipe requests, declared names, pyodide
    note in the system instruction)."""
    await ctx.mock.reset()
    c = await ctx.chat(b, PRO3, f"Hello declared {label}", features=features)
    reqs = await ctx.requests()
    first = reqs[0] if reqs else {}
    return c, reqs, _declared(first), PYODIDE_PROMPT in _system(first)


async def declared(ctx: Ctx, b) -> None:
    off, off_reqs, off_names, off_note = await _plain_turn(ctx, b, "off", {})
    on, reqs, names, note = await _plain_turn(ctx, b, "on", {**IMG_ON, **CODE_ON})
    ctx.check(
        "owuitools.declared",
        "browser path: with the Image and Code Interpreter toggles (features) on, "
        "generate_image, edit_image and execute_code are declared to Gemini and "
        "Open WebUI's pyodide note is in the system instruction; with the toggles "
        "off none of them",
        on.done
        and off.done
        and "Hello from mock" in _final(on)
        and len(reqs) == 1
        and _owui_tools(names) == list(OWUI_TOOLS)
        and note
        and len(off_reqs) == 1
        and not _owui_tools(off_names)
        and not off_note,
        _bdetail(
            on,
            reqs,
            f"owui_tools={_owui_tools(names)} pyodide_note={note} "
            f"off_owui_tools={_owui_tools(off_names)} off_pyodide_note={off_note} "
            f"off_done={off.done}",
        ),
    )
    await _set_images(ctx.t.owui, ENABLE_IMAGE_EDIT=False)
    try:
        c, reqs, names, _ = await _plain_turn(ctx, b, "noedit", IMG_ON)
    finally:
        await _set_images(ctx.t.owui, ENABLE_IMAGE_EDIT=True)
    ctx.check(
        "owuitools.declared-noedit",
        "image editing switched off (ENABLE_IMAGE_EDIT=false): generate_image is "
        "declared, edit_image is not",
        c.done
        and "Hello from mock" in _final(c)
        and _owui_tools(names) == ["generate_image"],
        _bdetail(c, reqs, f"owui_tools={_owui_tools(names)}"),
    )


async def _image_turn(ctx: Ctx, b, text: str, **kwargs) -> tuple:
    """One image tool turn of the Gemini 3 text model; (chat, pipe requests,
    image engine requests, saved image as base64)."""
    await ctx.mock.reset()
    c = await ctx.chat(b, PRO3, text, features=IMG_ON, **kwargs)
    entries = await ctx.mock.requests()
    reqs = [e for e in entries if _pipe()(e)]
    ctx.answered += sum(1 for e in reqs if e.get("status") == 200)
    engine = [e for e in entries if _engine(e)]
    return c, reqs, engine, await _file_b64(ctx.t.owui, _first_url(c.files))


def _image_saved(c, name: str) -> bool:
    """The call ran, its result names the one image saved with the answer
    (message files, chat:message:files event), the final answer is saved."""
    result = _json(c.function_outputs.get(CALL_0))
    urls = [i.get("url") for i in result.get("images") or [] if isinstance(i, dict)]
    files = [f.get("url") for f in c.files if f.get("type") == "image"]
    return (
        c.done
        and not c.error
        and _calls(c) == [(name, "completed", CALL_0)]
        and result.get("status") == "success"
        and len(files) == 1
        and urls == files
        and "chat:message:files" in c.event_types
        and _final(c).startswith(f"MOCK-FINAL {name}=")
        and _status_closed(c)
    )


def _continued(reqs: list, name: str, url: str, turn: str = "user:text") -> bool:
    """The continuation sends the signed call and Open WebUI's tool result (with
    the image URL) back to Gemini."""
    second = reqs[1] if len(reqs) > 1 else {}
    responses = str([x.get("response") for x in second.get("fr") or []])
    return (
        [e.get("answer") for e in reqs] == ["fc", "final"]
        and second.get("kinds") == [turn, "model:fc", "user:fr"]
        and [x.get("sig") for x in second.get("fc") or []] == ["issued"]
        and _frs(second) == [(CALL_0, name, ["output"])]
        and bool(url)
        and url in responses
    )


def _image_detail(c, reqs: list, engine: list, file_ok: bool, extra: str = "") -> str:
    second = reqs[1] if len(reqs) > 1 else {}
    return _bdetail(
        c,
        reqs,
        f"engine={_engine_tokens(engine)} "
        f"engine_prompt={short(_engine_text(engine[0]), 60) if engine else None} "
        f"file_ok={file_ok} files={len(c.files)} "
        f"files_event={'chat:message:files' in c.event_types} "
        f"statuses_closed={_status_closed(c)} {extra} {_req_tokens(second)}",
    )


async def generate(ctx: Ctx, b):
    """generate_image with the engine's generateContent method (image model)."""
    prompt = "a yellow square final-image-7"
    c, reqs, engine, saved = await _image_turn(
        ctx, b, _tool_call("generate_image", prompt=prompt)
    )
    path = f"/v1beta/models/{IMG_MODEL}:generateContent"
    file_ok = saved == _png(255, 255, 7)
    ctx.check(
        "owuitools.generate",
        "generate_image (engine gemini, generateContent): Gemini's call runs Open "
        "WebUI's image engine with its own key and the call's prompt, the image is "
        "saved with the answer (message files, chat:message:files), the tool result "
        "with its URL goes back to Gemini, the final answer is saved",
        _image_saved(c, "generate_image")
        and _engine_ok(engine, path, IMG_KEY, prompt)
        and file_ok
        and _continued(reqs, "generate_image", _first_url(c.files)),
        _image_detail(c, reqs, engine, file_ok),
    )
    return c


async def generate_predict(ctx: Ctx, b) -> None:
    """generate_image with the engine's predict method (Imagen)."""
    await _set_images(
        ctx.t.owui,
        IMAGE_GENERATION_MODEL=IMAGEN,
        IMAGES_GEMINI_ENDPOINT_METHOD="predict",
    )
    prompt = "a yellow square final-image-9"
    try:
        c, reqs, engine, saved = await _image_turn(
            ctx, b, _tool_call("generate_image", prompt=prompt)
        )
    finally:
        await _set_images(
            ctx.t.owui,
            IMAGE_GENERATION_MODEL=IMG_MODEL,
            IMAGES_GEMINI_ENDPOINT_METHOD="generateContent",
        )
    path = f"/v1beta/models/{IMAGEN}:predict"
    file_ok = saved == _png(255, 255, 9)
    ctx.check(
        "owuitools.generate-predict",
        "generate_image (engine gemini, predict = Imagen): the engine's :predict "
        "request carries the call's prompt, the image is saved with the answer, the "
        "tool result goes back to Gemini",
        _image_saved(c, "generate_image")
        and _engine_ok(engine, path, IMG_KEY, prompt)
        and file_ok
        and _continued(reqs, "generate_image", _first_url(c.files)),
        _image_detail(c, reqs, engine, file_ok),
    )


async def edit(ctx: Ctx, b) -> None:
    """edit_image on an uploaded image: Open WebUI tells Gemini the file's URL
    (<attached_files>), Gemini passes it in image_urls, the edit engine gets the
    image."""
    item = await _upload_png(ctx.t.owui, UPLOAD_PNG, "photo.png")
    file_id = item.get("url") or ""
    prompt = "make it blue final-image-11"
    c, reqs, engine, saved = await _image_turn(
        ctx,
        b,
        _tool_call("edit_image", prompt=prompt, image_urls=[file_id]),
        user_files=[item],
    )
    first = reqs[0] if reqs else {}
    attached = bool(file_id) and f'url="{file_id}"' in " ".join(
        _strings(_body(first).get("contents"))
    )
    path = f"/v1beta/models/{EDIT_MODEL}:generateContent"
    sent_ok = bool(engine) and _engine_images(engine[0]) == [UPLOAD_PNG]
    file_ok = saved == _png(255, 255, 11)
    ctx.check(
        "owuitools.edit",
        "edit_image on an uploaded image: the request to Gemini names the file in "
        "<attached_files> (and carries the image), Gemini's call with that URL runs "
        "the edit engine (gemini, own key) with the uploaded image, the edited "
        "image is saved with the answer, the tool result goes back to Gemini",
        _image_saved(c, "edit_image")
        and attached
        and first.get("kinds") == ["user:text+text+inline"]
        and _engine_ok(engine, path, EDIT_KEY, prompt)
        and sent_ok
        and file_ok
        and _continued(
            reqs, "edit_image", _first_url(c.files), "user:text+text+inline"
        ),
        _image_detail(
            c,
            reqs,
            engine,
            file_ok,
            f"attached={attached} first_kinds={first.get('kinds')} sent_ok={sent_ok}",
        ),
    )


async def edit_followup(ctx: Ctx, b, generated) -> None:
    """The next turn edits the image generate_image made: the replayed tool
    result holds its URL, the edit engine gets that image."""
    gen_url = _first_url(generated.files)
    if not generated.chat_id or not gen_url:
        ctx.check(
            "owuitools.edit-followup",
            "edit_image in the next turn on the image generate_image made",
            False,
            f"no generated image to edit: generate done={generated.done} "
            f"files={short(generated.files, 120)}",
        )
        return
    prompt = "add a hat final-image-13"
    c, reqs, engine, saved = await _image_turn(
        ctx,
        b,
        _tool_call("edit_image", prompt=prompt, image_urls=[gen_url]),
        chat_id=generated.chat_id,
        parent_id=generated.message_id,
    )
    first = reqs[0] if reqs else {}
    replayed = gen_url in str([x.get("response") for x in first.get("fr") or []])
    path = f"/v1beta/models/{EDIT_MODEL}:generateContent"
    sent_ok = bool(engine) and _engine_images(engine[0]) == [_png(255, 255, 7)]
    file_ok = saved == _png(255, 255, 13)
    ctx.check(
        "owuitools.edit-followup",
        "edit_image in the next turn on the image generate_image made: the "
        "replayed turn-1 tool result holds its URL, the edit engine gets that "
        "image, the edited image is saved with the new answer",
        _image_saved(c, "edit_image")
        and first.get("kinds")
        == ["user:text", "model:fc", "user:fr", "model:text", "user:text"]
        and replayed
        and _engine_ok(engine, path, EDIT_KEY, prompt)
        and sent_ok
        and file_ok,
        _image_detail(
            c,
            reqs,
            engine,
            file_ok,
            f"first_kinds={first.get('kinds')} url_replayed={replayed} "
            f"sent_ok={sent_ok}",
        ),
    )


async def image_model(ctx: Ctx, b) -> None:
    """A pipe image model with the Image toggle on: the pipe drops Open WebUI's
    tools (image models have no function calling), the model makes the image
    itself and Open WebUI's engine is not called."""
    await ctx.mock.reset()
    c = await ctx.chat(b, IMAGE_GA, "Draw a cat final-image-15", features=IMG_ON)
    entries = await ctx.mock.requests()
    reqs = [e for e in entries if _pipe(IMAGE_GA.split(".", 1)[1])(e)]
    engine = [e for e in entries if _engine(e)]
    kinds = [k for e in reqs for k in e.get("tool_kinds") or []]
    file_ok = await _file_b64(ctx.t.owui, _first_url(c.files)) == _png(255, 255, 15)
    ctx.check(
        "owuitools.image-model",
        "pipe image model (gemini-3.1-flash-image) with the Image toggle on: no "
        "function declarations, the model's own image is saved, Open WebUI's image "
        "engine is not called",
        c.done
        and not c.error
        and len(reqs) == 1
        and reqs[0].get("status") == 200
        and "functionDeclarations" not in kinds
        and not engine
        and len(c.files) == 1
        and file_ok,
        f"http={c.http_status} done={c.done} "
        f"error={short(c.error, 120) if c.error else None} "
        f"upstream={[e.get('answer') for e in reqs]} tool_kinds={kinds} "
        f"engine={_engine_tokens(engine)} file_ok={file_ok} files={len(c.files)}",
    )


async def code(ctx: Ctx, b) -> None:
    """execute_code, engine pyodide: Open WebUI asks the browser to run the code
    (execute:python), the browser's result is the tool result; an image line in
    stdout is uploaded and replaced with a markdown image link."""
    stdout = f"4\ndata:image/png;base64,{PLOT_PNG}"
    b.python_answer = lambda data: {"stdout": stdout, "stderr": None, "result": None}
    before = len(b.python_calls)
    try:
        await ctx.mock.reset()
        c = await ctx.chat(b, PRO3, _tool_call(CODE_TOOL, code=CODE), features=CODE_ON)
        reqs = await ctx.requests()
    finally:
        b.python_answer = lambda data: None
    calls = [(x.get("data") or {}).get("data") or {} for x in b.python_calls[before:]]
    session_ok = len(calls) == 1 and calls[0].get("session_id") == b.sid
    out = str(_json(c.function_outputs.get(CALL_0)).get("stdout") or "")
    marker = "![Output Image]("
    link = out.split(marker, 1)[1].split(")", 1)[0] if marker in out else ""
    plot_ok = bool(link) and await _file_b64(ctx.t.owui, link) == PLOT_PNG
    second = reqs[1] if len(reqs) > 1 else {}
    responses = str([x.get("response") for x in second.get("fr") or []])
    ctx.check(
        "owuitools.code",
        "execute_code (engine pyodide): Open WebUI sends the code to the browser "
        "session (execute:python), the browser's stdout is the tool result Gemini "
        "gets back (an image line uploaded and linked), the final answer is saved",
        c.done
        and not c.error
        and _calls(c) == [(CODE_TOOL, "completed", CALL_0)]
        and session_ok
        and calls[0].get("code") == CODE
        and out.startswith("4\n" + marker + "/api/v1/files/")
        and plot_ok
        and [e.get("answer") for e in reqs] == ["fc", "final"]
        and _frs(second) == [(CALL_0, CODE_TOOL, ["output"])]
        and link in responses
        and _final(c).startswith(f"MOCK-FINAL {CODE_TOOL}="),
        _bdetail(
            c,
            reqs,
            f"python_calls={len(calls)} session_ok={session_ok} "
            f"code={short(calls[0].get('code'), 40) if calls else None} "
            f"stdout={short(out, 90)} plot_ok={plot_ok} {_req_tokens(second)}",
        ),
    )


async def code_jupyter(ctx: Ctx, b) -> None:
    """execute_code, engine jupyter: the code runs on the Jupyter server
    (mocks/mock_tools.py), not in the browser; no pyodide note."""
    tool_mock = ctx.t.mock("tools")
    await _set_code(
        ctx.t.owui,
        CODE_INTERPRETER_ENGINE="jupyter",
        CODE_INTERPRETER_JUPYTER_URL=JUPYTER_URL,
        CODE_INTERPRETER_JUPYTER_AUTH="",
        CODE_INTERPRETER_JUPYTER_TIMEOUT=30,
    )
    before = len(b.python_calls)
    try:
        await ctx.mock.reset()
        await tool_mock.reset()
        c = await ctx.chat(b, PRO3, _tool_call(CODE_TOOL, code=CODE), features=CODE_ON)
        reqs = await ctx.requests()
        jupyter = await tool_mock.requests(
            lambda e: str(e.get("path", "")).startswith(("/jupyter/", "jupyter:"))
        )
    finally:
        await _set_code(ctx.t.owui, CODE_INTERPRETER_ENGINE="pyodide")
    first = reqs[0] if reqs else {}
    note = PYODIDE_PROMPT in _system(first)
    executed = [
        (e.get("body") or {}).get("code")
        for e in jupyter
        if e.get("path") == "jupyter:execute"
    ]
    paths = [f"{e.get('method')}:{e.get('path')}" for e in jupyter]
    result = _json(c.function_outputs.get(CALL_0))
    second = reqs[1] if len(reqs) > 1 else {}
    responses = str([x.get("response") for x in second.get("fr") or []])
    ctx.check(
        "owuitools.code-jupyter",
        "execute_code (engine jupyter): the code runs on the Jupyter server "
        "(kernel started, executed, deleted), not in the browser; its stdout and "
        "result are the tool result Gemini gets back; no pyodide note",
        c.done
        and not c.error
        and CODE_TOOL in _declared(first)
        and not note
        and _calls(c) == [(CODE_TOOL, "completed", CALL_0)]
        and len(b.python_calls) == before
        and executed == [CODE]
        and any(p.startswith("DELETE:") for p in paths)
        and result.get("stdout") == JUPYTER_STDOUT
        and result.get("result") == JUPYTER_RESULT
        and JUPYTER_STDOUT in responses
        and [e.get("answer") for e in reqs] == ["fc", "final"],
        _bdetail(
            c,
            reqs,
            f"jupyter={paths} executed={executed} "
            f"browser_calls={len(b.python_calls) - before} pyodide_note={note} "
            f"result={short(result, 160)}",
        ),
    )
