# Open-WebUI-Functions – Quickstart for AI Coding Agents

This repo extends Open WebUI with pluggable Python pipelines and filters. If you’re adding or modifying code, follow these project-specific rules to be productive immediately.

## Big picture

- Two building blocks:
  - Pipelines in `pipelines/<provider>/` expose a `Pipe` with inner `Valves` (config via env). They list models with `pipes()` and execute requests in `pipe(...)` (streaming and non-streaming).
  - Filters in `filters/` expose a `Filter` with `inlet(body)` and/or `outlet(body)` to mutate requests/responses.
- Secrets are declared as `EncryptedStr` in valves and are auto-encrypted on assignment; decrypt only right before use. Requires `WEBUI_SECRET_KEY` in Open WebUI.
- Status/events: emit via `__event_emitter__` using `{type:"status"|"chat:*", data:{...}}`. Always send a completion or error status.

## Patterns that differ from “common” code

- Model IDs are normalized early to avoid provider mismatches:
  - Azure: set header `x-ms-model-mesh-model-name` (or put model in body when `AZURE_AI_MODEL_IN_BODY=true`). `pipelines/azure/azure_ai_foundry.py` parses semicolon/comma/space model lists.
  - Google Gemini: strip prefixes like `models/` and `publishers/google/models/` via `strip_prefix()` and `_prepare_model_id()`.
- Streaming rules by provider:
  - Azure and N8N stream with SSE (`StreamingResponse`); always close `aiohttp` `ClientSession`/response in `finally` (see `cleanup_response`).
  - Gemini disables streaming for image-generation models; thinking is wrapped in `<details>` and emitted incrementally.
- Cross-component integration:
  - `filters/google_search_tool.py` converts `features.web_search` → `metadata.features.google_search_tool`; Gemini reads this to enable grounding tools.
  - N8N non-streaming responses can append a tool-calls section built by `_format_tool_calls_section` with verbosity and truncation valves.

## Dev workflow (local)

- Use Pixi tasks for quality:
  - Format: `pixi run format`
  - Lint: `pixi run lint` (Ruff config in `ruff.toml`)
- End-to-end tests against a real Open WebUI container (Docker + bash only, see `docs/testing.md`):
  - `tests/e2e/run.sh <suite>` with suite `gemini`, `azure`, `n8n`, `infomaniak`, `filters` or `all`; run the suite of every file you change.
  - `--image <tag>` tests another Open WebUI release, `--ref <branch>` / `--src <dir>` / `--file <path>=<file>` test other function files, `--keep` / `--reuse --name <name>` keep the container for debugging.
  - Results are `PASS` / `FAIL` / `KNOWN`; KNOWN marks a registered known bug (`tests/e2e/harness/known.py`) and does not fail the run. Remove the marker when your change fixes it.
  - Add a scenario (`tests/e2e/suites/<suite>.py`) and, if needed, mock behaviour (`tests/e2e/mocks/`) for behaviour changes.
  - New Open WebUI release: `tests/e2e/check_owui_api.sh latest` checks the `open_webui` imports/APIs statically.
  - Gotchas: `valves/update` replaces all valves; refresh `/api/models?refresh=true` after model changes; only the browser path (socket.io + saved chat) exercises event emitters and saved content; background tasks get `__event_emitter__=None`.
- Fast manual test path: paste a single pipeline/filter into Open WebUI Admin → Functions, set required env (see each `Valves`), and call it from a chat. Encryption works once `WEBUI_SECRET_KEY` is set.
- Useful references when implementing:
  - Azure pipeline: citations handling and SSE in `pipelines/azure/azure_ai_foundry.py`
  - Gemini pipeline: model caching, image optimization/upload, grounding in `pipelines/google/google_gemini.py`
  - N8N pipeline: mixed stream parsing and tool display in `pipelines/n8n/n8n.py`
  - Filters: `filters/time_token_tracker.py`, `filters/google_search_tool.py`

## What to keep consistent

- Validate and filter inputs (keep an explicit allow-list of request fields).
- Emit status at start, on streaming start, and on completion/error.
- Avoid blocking I/O in async paths; prefer `aiohttp`/`aiofiles`.
- Close network resources when not streaming; set SSE headers in streaming responses.
- Keep valve names stable; add new ones with backward-compatible defaults.
