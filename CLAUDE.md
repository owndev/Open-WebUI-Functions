# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of **standalone Python functions for Open WebUI**. Each `.py` file under `pipelines/` and `filters/` is a self-contained artifact that a user copy-pastes into Open WebUI *Admin Settings → Functions*. There is no package, no `__init__.py`, no import graph between files.

**Consequence: code duplication across files is intentional.** `EncryptedStr`, `cleanup_response`, and logging setup are copied into every file. Do NOT extract them into a shared module — that would break the paste-one-file installation model. When fixing a bug in one of these copied blocks, grep the other files and fix each copy.

## Commands

```bash
pixi run format   # ruff format
pixi run lint     # ruff format + ruff check (line-length 88)
# without pixi (e.g. Windows): uvx ruff@0.11.10 format <files>; uvx ruff@0.11.10 check <files>
```

The `pixi` env contains only Ruff — `open_webui.*`, `google.genai`, `aiohttp`, etc. are **not installed** on the host, so the functions cannot be imported or run locally. They are tested inside a real Open WebUI container instead.

## Testing

Docker-based E2E tests live in `tests/e2e/` (human guide: `docs/testing.md`). Host needs only Docker + bash (Git Bash on Windows); mocks and driver run inside the Open WebUI container.

```bash
tests/e2e/run.sh <suite>                     # gemini | azure | n8n | infomaniak | filters | all
tests/e2e/run.sh --image v0.11.3-slim gemini # A/B another Open WebUI release (default v0.11.4-slim)
tests/e2e/run.sh --ref <branch> <suite>      # test files from a git ref; --src DIR / --file PATH=FILE
tests/e2e/run.sh --keep --only 'azure.rag' azure   # keep container; then --reuse --name <name>
tests/e2e/run.sh --strict-known all          # as in CI: a tagged check that passes while its marker applies is FAIL
tests/e2e/check_owui_api.sh [TAG|latest]     # static check of open_webui imports/APIs vs a release
```

- **What to run:** changed `pipelines/google/*` → `gemini`; `pipelines/azure/*` → `azure`; `pipelines/n8n/*` → `n8n`; `pipelines/infomaniak/*` → `infomaniak`; `filters/*` → `filters` (+ `gemini` for `google_search_tool`); `tests/e2e/**` → `all`. New Open WebUI release → `check_owui_api.sh latest`, then `run.sh --image <new tag> all`.
- **Results:** `PASS` / `FAIL` / `KNOWN`. Exit 1 only on FAIL; exit 2 = setup error (Docker/container, leftover `NAME-data` volume without `--force`, an option without value, unknown suite, git ref or `--file` path, `--only` regex matching no group, driver error); 130/143 after Ctrl-C/SIGTERM. KNOWN = fails the way a bug registered in `tests/e2e/harness/known_<area>.py` does (its `evidence` matches the detail, or its log signature shows up with `since=`); a tagged check that fails differently is FAIL. KNOWN does not fail the run; its fixing branch may still be an unmerged/unpublished PR. Markers are **version-gated**: each `KnownIssue` names its `file` and the `fixed_in` version; once the tested file has that version, a failure is FAIL ("regression of known …") and a pass is listed as an obsolete marker ("drop marker"). So when you fix such a bug, bump the file's `version:` and set `fixed_in` to it; drop the markers once the fix is merged. `--only` is a Python regex: `'gemini.(api|image)'`, not `\|`. Output (git-ignored): `tests/e2e/out/<run>/` with `driver.txt`, `results.json`, `summary.md`, `server.log`, `mocks.txt`, staged `functions/` (LF); results are written for interrupted or timed-out runs too (partial).
- **Baseline** (v0.11.4-slim, `--strict-known`): `main` → 0 FAIL, 0 KNOWN (no known bug registered), no obsolete markers. Check counts per suite and the scenario list: `docs/testing.md` → "What is tested" (keep the counts there only, not here). The `filters` suite needs network for the tiktoken encodings, edits the container's `/etc/hosts` and trust store, and its `offline` group needs a fresh container (no `--reuse`).
- **Driver checks:** `<suite>.timeout` (`E2E_SUITE_TIMEOUT`, default 900 s; whole run `E2E_TIMEOUT`, default 1800 s), `<suite>.crash`, `<suite>.no-checks`, `<suite>.interrupted`, `run.server-log` (errors or secrets logged outside every suite's scan window; names the `function_<id>`), `run.mocks-log` (errors in `mocks.txt`). Ctrl-C cleans up within seconds; a killed `run.sh` leaves the container, the volume and an orphan driver in the container, which `--reuse` stops. Suites are discovered from `tests/e2e/suites/` (no registration).
- **Add coverage** for every behaviour change: a `t.check(...)` in `tests/e2e/suites/<suite>.py` (API path `t.owui.chat`, browser path `t.browser().chat`, upstream request via `mock.last()`), mock behaviour in `tests/e2e/mocks/mock_<provider>.py`. Keep `tests/e2e` Ruff-clean. Evidence of a new known bug must not match unrelated failures: the CI meta-test (main's files with `E2E_MOCK_FAULT=500`) allows no KNOWN (`REQUEST_SIDE_KNOWN` in `.github/workflows/e2e.yml`, empty today, is only for bugs that can show nothing but the request sent upstream), so the evidence must include proof that the upstream answered (e.g. `HTTP 200` plus the mock's answer).
- **Gotchas:** `valves/update` REPLACES all valves (use `update_valves`, which merges); refresh `/api/models?refresh=true` after model/filterIds changes; settings in the data volume override env vars; only the browser path (socket.io + saved chat) exercises event emitters, saved content, usage and sources; background tasks (`/api/v1/tasks/*`) call pipes with `__task__` set and `__event_emitter__=None`; the first Gemini install pip-installs `google-genai` (~30-60 s); in Git Bash prefix your own `docker exec`/`docker cp` with `MSYS_NO_PATHCONV=1` per command (do not export it; `run.sh` unsets it because `git -C /c/...` needs path conversion); never remove Docker images, only your containers/volumes.

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

- **Azure** (`azure_ai_foundry.py`): model goes in the `x-ms-model-mesh-model-name` header, or in the body when `AZURE_AI_MODEL_IN_BODY=true`. `AZURE_AI_MODEL` accepts semicolon/comma/space-separated lists. Azure AI Search citations are extracted, normalized into Open WebUI `source` events, and `[docX]` references are rewritten into markdown links — the streaming path has its own citation-aware processor (`stream_processor_with_citations`). Since 3.0.0 the pipe queries Azure AI Search itself (`_retrieve`), injects a `<documents>` block into the current user message, and feeds an On-Your-Data-shaped `context` (synthetic first SSE event / `message.context`) into that same citation code. Azure OpenAI On Your Data is gone: nothing sends `data_sources` upstream, and a request body carrying `data_sources` ends with an `AzureSearchError`.
- **Gemini** (`google_gemini.py`, ~3.6k lines): streaming is force-disabled for image-generation models; thinking output is wrapped in `<details>` and emitted incrementally; generated images/videos are uploaded through Open WebUI's file API and referenced by `url_path_for("get_file_content_by_id", ...)`.
- **N8N** (`n8n.py`): streamed responses may mix SSE, NDJSON / back-to-back JSON and plain text — `N8NStreamParser` handles all of them incrementally (line-based, string-aware brace matching whose state persists across `feed()` calls, so keep it linear). n8n `{"type": "error"}` chunks end up in `parser.errors` and the final status. Tool-usage display only works non-streaming.

## Docs and release

- Adding or changing a user-visible feature means updating three places: the file's docstring `features:` list + `version:`, the matching `docs/<provider>-integration.md`, and the feature bullets in `README.md`.
- GitFlow branches (`main` / `dev` / `feature/*` / `release/*` / `hotfix/*`) with GitVersion (`GitVersion.yml`). Commits follow `feat:` / `fix:` prefixes; `+semver: major|minor|patch|none` in a commit message overrides the bump.
