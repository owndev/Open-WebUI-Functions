# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of **standalone Python functions for Open WebUI**. Each `.py` file under `pipelines/` and `filters/` is a self-contained artifact that a user copy-pastes into Open WebUI *Admin Settings → Functions*. There is no package, no `__init__.py`, no import graph between files.

**Consequence: code duplication across files is intentional.** `EncryptedStr`, `cleanup_response`, and logging setup are copied into every file. Do NOT extract them into a shared module — that would break the paste-one-file installation model. When fixing a bug in one of these copied blocks, grep the other files and fix each copy.

## Commands

```bash
pixi run format   # ruff format
pixi run lint     # ruff format + ruff check (line-length 88)
```

No test suite and no local runtime. The `pixi` env contains only Ruff — `open_webui.*`, `google.genai`, `aiohttp`, etc. are **not installed**, so imports will not resolve locally and nothing here is executable outside an Open WebUI instance.

Manual test path: paste the single file into Open WebUI → Functions, set the env vars from its `Valves`, invoke it from a chat. `WEBUI_SECRET_KEY` must be set in the Open WebUI environment or API-key encryption silently degrades to plaintext.

## File anatomy

Every function file starts with a **YAML-ish docstring header** that Open WebUI parses:

```python
"""
title: ...
author: owndev
version: 2.7.0                      # bump on every behavior change
required_open_webui_version: 0.8.0  # bump when using newer Open WebUI APIs
license: Apache License 2.0
description: ...
features:
  - ...
requirements: ...    # filters only: declares the pipeline they pair with
"""
```

Version in this header is the user-visible version; it is separate from the Git/GitVersion repo version.

### Pipelines (`pipelines/<provider>/*.py`)

Expose a single `Pipe` class:

- `class Valves(BaseModel)` — admin config, every field `default=os.getenv("NAME", fallback)`. Secrets are typed `EncryptedStr` with `json_schema_extra={"input": {"type": "password"}}`.
- `class UserValves(BaseModel)` — optional per-user overrides (see `google_gemini.py`); read via a helper that falls back to the admin valve.
- `__init__` sets `self.type = "manifold"` and `self.valves = self.Valves()`.
- `pipes()` — returns `[{"id": ..., "name": ...}]`. Sync or async.
- `pipe(body, __event_emitter__, __user__, __request__, __metadata__, ...)` — the request path. Returns `str`, generator, dict, or `StreamingResponse`.

### Filters (`filters/*.py`)

Expose a `Filter` class with `inlet(body)` and/or `outlet(body)`, mutating the request/response dict in place.

## Invariants to preserve

- **Valve names are the public API.** Never rename or remove one; add new valves with backward-compatible defaults.
- **Secrets:** assigning to an `EncryptedStr` field encrypts (stored with an `encrypted:` prefix). Call `EncryptedStr.decrypt(...)` only at the point of use — never store the decrypted value on `self`, never log it.
- **Body allow-list:** `pipe()` filters the incoming body through an explicit `allowed_params` set before forwarding upstream. Add new provider params to that set deliberately; do not forward `body` wholesale.
- **Status events:** emit via `__event_emitter__` with `{"type": "status"|"chat:*", "data": {...}}` at start, at streaming start, and on completion *or* error. Never leave a request without a terminal status.
- **Async only:** no blocking I/O in `pipe()`. Use `aiohttp` / `aiofiles`.
- **Network cleanup:** close `aiohttp` `ClientSession` and response in `finally` via `cleanup_response` for non-streaming; hand them to `BackgroundTask`/the stream generator's `finally` for `StreamingResponse`.
- **Model ID normalization:** Open WebUI prefixes model IDs with the function ID (`func_id.model`). Every pipeline strips this early — Azure via `split(".", 1)[1]`, Gemini via `strip_prefix()` / `_prepare_model_id()` which also drop `models/` and `publishers/google/models/`.

## Cross-file coupling

- `filters/google_search_tool.py` converts `features.web_search` → `metadata.features.google_search_tool`; `pipelines/google/google_gemini.py` reads that flag to enable Search grounding. `vertex_ai_search_tool.py` works the same way for Vertex AI Search. Changing the flag name requires editing both sides.
- `pipelines/n8n/*.json` are importable N8N workflows kept in sync with `n8n.py`'s expected request/response shape (notably `intermediateSteps`, which N8N only returns in non-streaming mode).

## Provider quirks worth knowing before editing

- **Azure** (`azure_ai_foundry.py`): model goes in the `x-ms-model-mesh-model-name` header, or in the body when `AZURE_AI_MODEL_IN_BODY=true`. `AZURE_AI_MODEL` accepts semicolon/comma/space-separated lists. Azure AI Search citations are extracted, normalized into Open WebUI `source` events, and `[docX]` references are rewritten into markdown links — the streaming path has its own citation-aware processor (`stream_processor_with_citations`).
- **Gemini** (`google_gemini.py`, ~3.6k lines): streaming is force-disabled for image-generation models; thinking output is wrapped in `<details>` and emitted incrementally; generated images/videos are uploaded through Open WebUI's file API and referenced by `url_path_for("get_file_content_by_id", ...)`.
- **N8N** (`n8n.py`): responses may be a mixed SSE/plain stream — `parse_n8n_streaming_chunk` and `extract_content_from_mixed_stream` handle both. Tool-usage display only works non-streaming.

## Docs and release

- Adding or changing a user-visible feature means updating three places: the file's docstring `features:` list + `version:`, the matching `docs/<provider>-integration.md`, and the feature bullets in `README.md`.
- GitFlow branches (`main` / `dev` / `feature/*` / `release/*` / `hotfix/*`) with GitVersion (`GitVersion.yml`). Commits follow `feat:` / `fix:` prefixes; `+semver: major|minor|patch|none` in a commit message overrides the bump.
