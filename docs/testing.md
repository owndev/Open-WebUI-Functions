# Testing the functions locally (Docker E2E)

The functions in this repo only run inside Open WebUI, so they are tested end-to-end
against a **real Open WebUI container**: `tests/e2e/run.sh` starts a throw-away
container, installs the function files through the Open WebUI API, points them at
**mock provider APIs** (Gemini, Azure OpenAI / AI Foundry, Azure AI Search, n8n,
Infomaniak, Azure Log Analytics) and runs scenario suites that chat with them the way
an API client and the browser UI do.
Container and volume are removed afterwards.

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Options](#options)
- [What is tested](#what-is-tested)
- [Results: PASS, FAIL, KNOWN](#results-pass-fail-known)
- [Testing a fix, another branch or another Open WebUI version](#testing-a-fix-another-branch-or-another-open-webui-version)
- [Debugging a failure](#debugging-a-failure)
- [Adding a scenario or a mock behaviour](#adding-a-scenario-or-a-mock-behaviour)
- [Static Open WebUI API check](#static-open-webui-api-check)
- [Real-API smoke test (manual, needs a key)](#real-api-smoke-test-manual-needs-a-key)
- [CI](#ci)
- [Troubleshooting and gotchas](#troubleshooting-and-gotchas)

## Prerequisites

- **Docker** (Docker Desktop on Windows/macOS, Docker Engine on Linux).
- **bash**: Linux/macOS shell, or **Git Bash** on Windows.
- Network access for the first image pull (`ghcr.io/open-webui/open-webui:v0.11.4-slim`,
  ~1 GB), for `pip install google-genai` when the Gemini pipe is installed and, in the
  `filters` suite, for the tiktoken encodings (`cl100k_base`, `o200k_base`) the driver
  downloads into the container's `TIKTOKEN_CACHE_DIR` (without them the suite fails).

No Python on the host: the mocks, the probe pipe and the test driver run **inside** the
Open WebUI container, whose Python already ships `aiohttp`, `httpx`,
`python-socketio`, `openai` and `mcp` (the last two for the tool calling scenarios).

## Quick start

```bash
tests/e2e/run.sh                 # all suites against ghcr.io/open-webui/open-webui:v0.11.4-slim
tests/e2e/run.sh gemini          # one suite (gemini, azure, n8n, infomaniak, filters)
tests/e2e/run.sh azure n8n       # several suites
tests/e2e/run.sh --image v0.11.3-slim gemini   # another Open WebUI version
pixi run e2e gemini              # same via pixi (Linux)
```

Typical output (abridged):

```text
starting owui-e2e-111759-1234 from ghcr.io/open-webui/open-webui:v0.11.4-slim
Open WebUI healthy after 27s (http://localhost:57765)
Open WebUI 0.11.4, suites: gemini, azure, n8n, infomaniak, filters
function versions: google_gemini.py 1.19.0, azure_ai_foundry.py 3.0.0, n8n.py 2.3.1, ...
=== gemini ===
[PASS ] gemini.load  pipelines/google/google_gemini.py loads (create, import, activate)
[PASS ] gemini.api.stream  API stream without websocket session: answer streamed, thinking in the <details type="reasoning" done="true" duration="N"> block
...
[PASS ] gemini.tools.builtin  built-in tool, browser stream=True: Open WebUI runs it, ...
...
--- gemini: 322s
...
=== infomaniak ===
...
[PASS ] infomaniak.models.name-prefix  NAME_PREFIX valve changes the model names, and changing it back restores them
...
SUMMARY: 547 PASS, 0 FAIL, 0 KNOWN in 1110s
total runtime: 1147s
output: tests/e2e/out/20261009-111759-owui-e2e-111759-1234
```

A full run of all suites took about 19 minutes on a shared 8-CPU Docker host (1147 s:
37 s container start-up, then gemini 322 s, azure 422 s, n8n 71 s, infomaniak 36 s,
filters 257 s). Most of it is waiting: the azure `rag` group adds about 5.5 minutes (query-generation timeouts, the query-generation pause and the
45 s retrieval limit are waited for; in the meta-test, where every mock answers HTTP
500, its requests fail at once), the gemini `tools`, `toolsapi`, `imgedit` and
`owuitools` groups about three minutes and the filters `ingest` group (Logs Ingestion
API) another 2.5 minutes. The browser scenarios of the gemini `thinking` group (paced
streams for the live block and its duration, Stop while thinking, a tool round) take
about 20 s.
Network downloads on first use come on top
(`pip install google-genai` when the Gemini function is created, the tiktoken encodings
the `filters` suite caches before its first scenario). How many checks each suite has
and how many are KNOWN today is listed under [What is tested](#what-is-tested).

The exit code is `0` when there is no FAIL (KNOWN results are fine), `1` when at least
one scenario FAILs and `2` for setup errors: Docker missing, `docker run` failing (port
busy, image missing), container not healthy, a volume `NAME-data` left over from an
earlier run (see `--force`), an option without its value, an unknown suite, `--file`
path or git ref, an `--only` regex that selects no scenario group, a driver error, a
driver that outlived `E2E_TIMEOUT` and had to be stopped. After Ctrl-C (SIGINT) or
SIGTERM the script exits with `130` / `143`; see [Interrupted runs](#interrupted-runs).

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `[suite ...]`, `-s, --suites a,b` | `all` | `gemini`, `azure`, `n8n`, `infomaniak`, `filters` (the modules in `tests/e2e/suites/`), `all` |
| `-i, --image IMAGE` | `$OWUI_IMAGE` or `ghcr.io/open-webui/open-webui:v0.11.4-slim` | image, or just a tag (`v0.11.3-slim`) |
| `--only REGEX` | – | only scenario groups whose `<suite>.<group>` matches (Python regex), e.g. `--only gemini.api`; see below |
| `--ref REF` | – | test the function files as of a git ref (`git show REF:path`): a branch, tag or commit |
| `--src DIR` | – | test the function files from another checkout / worktree (not together with `--ref`) |
| `--file PATH=FILE` | – | replace one function file, e.g. `--file pipelines/azure/azure_ai_foundry.py=/tmp/fix.py` (repeatable); `PATH` must be one of the function files the harness installs, anything else is an error |
| `-n, --name NAME` | `$E2E_NAME` or `owui-e2e-<time>-<random>` | container name; the volume is `NAME-data` |
| `-p, --port PORT` | `$E2E_PORT` or a free port picked by Docker | host port of the UI (bound to 127.0.0.1) |
| `-o, --out DIR` | `$E2E_OUT` or `tests/e2e/out/<timestamp>-<name>` | output directory (git-ignored) |
| `-k, --keep` | off (`E2E_KEEP=1` turns it on) | keep container and volume after the run |
| `--reuse` | off | reuse the running container `--name` (implies `--keep`): skips the start-up, stops drivers left over from an interrupted run, copies the current files, restarts the mocks |
| `--force` | off | delete a volume `NAME-data` left over from an earlier run; without it `run.sh` refuses to start (exit 2) instead of silently deleting data |
| `-v, --verbose` | off | print details of passing scenarios too |
| `--strict-known` | off (`E2E_STRICT_KNOWN=1` turns it on; on in CI) | a check tagged with a known bug that passes while its marker still applies is a FAIL, see [Results](#results-pass-fail-known) |

Environment variables (all optional):

| Variable | Default | Meaning |
| --- | --- | --- |
| `OWUI_IMAGE`, `E2E_NAME`, `E2E_PORT`, `E2E_OUT`, `E2E_KEEP=1`, `E2E_STRICT_KNOWN=1` | – | defaults of `--image`, `--name`, `--port`, `--out`, `--keep`, `--strict-known` |
| `E2E_TIMEOUT` | `1800` | time budget of the driver in seconds. A suite still running when it is used up is a `<suite>.timeout` FAIL, suites that can no longer start get the same FAIL, and `results.json` is written. If the driver itself hangs, `run.sh` stops it 60 s later (SIGTERM, then SIGKILL after another 60 s) and exits with 2 |
| `E2E_SUITE_TIMEOUT` | `900` | time limit of one suite in seconds (`<suite>.timeout` FAIL; the next suite still runs) |
| `E2E_HEALTH_TIMEOUT` | `600` | seconds to wait for Open WebUI's `/health` after the container start |
| `E2E_SECRET_KEY` | a fixed test key | `WEBUI_SECRET_KEY` of the container (encrypts the valves) |
| `E2E_MOCK_FAULT` | – | an HTTP status (e.g. `500`) every provider route of the mocks answers with; used by the CI meta-test ([CI](#ci)) |

`--only` takes a plain Python regex. Write alternatives with an unescaped pipe inside
quotes:

```bash
tests/e2e/run.sh --only 'gemini.(api|image)' gemini
```

The groups of each suite are listed in the `GROUPS` constant (and docstring) of its
module. A regex that matches no group stops the run with exit code 2 and prints the
available groups.

Unique container names, volumes and Docker-assigned ports make parallel runs (several
worktrees, several agents) safe. The harness never removes images.

## What is tested

Every suite installs its function(s) through `POST /api/v1/functions/create`, sets the
valves to the mock, and checks at least: the file **loads** (import + activate), the
**valve names** are still there (they are public API), secrets are **stored encrypted**
and never show up in the **server log** (at any level), the **model list**, the **API
path** (non-stream and stream), the **browser path** (saved answer, usage and status
events), a **background title task** and that the server log has no unexpected
`ERROR` / `Traceback` / `ResourceWarning` block.

| Suite | Function(s) | Mock | Groups (`--only <suite>.<group>`) and file-specific scenarios |
| --- | --- | --- | --- |
| `gemini` | `pipelines/google/google_gemini.py` (+ `google_search_tool`) | `mock_gemini.py`, `mock_tools.py` (tool servers, Jupyter) | `models` (image / video indicators, display names #172), `api` (thinking in the `<details type="reasoning" done="true" duration="N">` block, full usage), `thinking` (summaries not replayed #176: the plain block of chats saved before 1.19.0 and the `type="reasoning"` blocks, also a stopped `done="false"` one; budget, level, include and strip valves; browser path: the live `done="false"` block through throttled `replace` events, no thinking status, the switch to `done="true"` with all thoughts on the first answer part and deltas after it, the duration up to the first answer part (mock trigger `paced-thinking`), Stop while thinking and the next turn without the block (`slow-thinking`), a tool round that clears its live block), `browser`, `tasks` (no `<details>` in task answers, no grounding tools for the tasks of a web_search chat), `image` (image models forced non-stream, exactly one saved file, the thinking block), `images` (thought images skipped, used as fallback, not used after IMAGE_SAFETY; dedup; two final images; image link for API clients; image history, where the limit keeps the current and the newest images and cuts more current images than it allows; optimization), `nano` (`gemini-nano-banana-2.1`), `imgvalve` (`IMAGE_GENERATION_MODELS`), `imgconfig` (ImageConfig valves, user valve, body), `imgtools` (tools per image model with web_search), `nostream` (`GOOGLE_STREAMING_ENABLED=false` with `stream=true`, #170), `video` (Veo: text and video saved, request shape, image-to-video), `grounding` (`google_search_tool` → googleSearch + urlContext, sources, `[1]` citations, Open WebUI's search statuses (the queries, then `Searched {{count}} sites` with `items` and no `urls`); no grounding without web_search), `vertex` (Vertex AI Search sources), `errors` (400, 500 with retry, blocked prompt, SAFETY finish, image error status, a streamed answer starting with `data:`), `retry` (`RETRY_COUNT` for streams), `status` (Stop leaves no running status), `valves` (model cache vs. valve changes, safety, whitelist, additional models, system prompt, user headers, API version, params, valve names and defaults), `concurrency` (forwarded user headers belong to the requesting user), `streamimg` (inline image in a stream), `imgedit` (image editing across turns, #194: in a saved chat the image of turn 1, attached as a file only, is sent with the edit request on `gemini-3.1-flash-image` and `gemini-nano-banana-2.1`; the follow-up task gets no images; uploads between generated turns in the default mode, with another image per turn (the mock's `final-image-<n>` trigger) so their order shows; guided regeneration, also of an image-only message and of a text contained in the guidance; `IMAGE_HISTORY_MAX_REFERENCES` keeps the upload of the edit message, else the newest images; an image attached again is sent once and counts at its newest place (with `IMAGE_DEDUP_HISTORY=false` it is sent each time); older saved forms (a data: URL image file, markdown links to a file and to a data: URL; a text file is not sent); a temporary chat and a database error (injected by a test-only filter) take the history from the request; an image file of another user in a non-admin user's chat, in a markdown link on the API path and on the Veo path is not read (logged without its id), an admin continuing the user's chat reads it), `toolsapi` (native tool calling for API clients: client tools → `tool_calls` + `reasoning_details` streamed and non-streamed with `finish_reason` `tool_calls` (also as the openai SDK reads the stream), a text answer to a request with tools (`finish_reason` `stop`), `GOOGLE_STREAMING_ENABLED=false`, `MALFORMED_FUNCTION_CALL` / `UNEXPECTED_TOOL_CALL` streamed and non-streamed, continuation with and without signatures and with `reasoning_details` under `provider_specific_fields`, older turns with a thinking summary, empty arguments, content parts, a numeric id and a `function` that is not an object, `tool_choice` (also an undeclared name and with Search grounding on Gemini 3), name mapping (also a trailing newline) and duplicates, the `default_api.` prefix, schema clean-up, synthetic `owui_` ids, unchanged answers without tools), `tools` (native tool calling through Open WebUI's tool loop: built-in tool streamed and non-streamed, parallel calls (responses in call order), two rounds with summed usage, merged rounds without signatures, text before a call, thinking on / off, workspace Python tool, a tool result with an image, OpenAPI and MCP tool servers, direct tools of the browser, tool approval (approve, reject, two calls), follow-up turns on Gemini 3 / 2.5 / 3 and after an approved call, unknown tool, `MALFORMED_FUNCTION_CALL`, Search grounding with functions on Gemini 3 (the sources of both rounds, the next turn without web search) and grounding only on 2.5, title task, Legacy mode, `builtin_tools` off, `GOOGLE_STREAMING_ENABLED=false`; see [Native tool calling](#native-tool-calling-geminitools-geminitoolsapi)), `owuitools` (Open WebUI's built-in `generate_image`, `edit_image` and `execute_code` called by a Gemini text model: declared only with the chat toggles and for browser sessions, `edit_image` only with image editing on; the image engine `gemini` with `generateContent` and with `predict` (Imagen), editing an upload and the image of the previous turn, the image saved with the answer and its URL sent back to Gemini; a pipe image model makes its own image; the code interpreter with pyodide (the browser runs the code) and with Jupyter; see [Open WebUI's image and code tools](#open-webuis-image-and-code-tools-geminiowuitools)) |
| `azure` | `pipelines/azure/azure_ai_foundry.py` | `mock_azure.py` (requires `api-version`; answers prompts with a `<documents>` block like a plain model, without `context`; query generation, embeddings, tool calls, a ~300 KB and a > 4 MiB stream event, `content: null`), `mock_search.py` (Azure AI Search and an App Service managed identity token endpoint, see below) | `valves` (from 3.0.0 also the `AZURE_AI_SEARCH_*` valves, their defaults, enums and the password input of the key, and no `AZURE_AI_SEARCH_MODE`), `models` (`AZURE_AI_MODEL` lists separated by `;`, `,` or spaces with exact names, `AZURE_AI_PIPELINE_PREFIX`, model from an `*.openai.azure.com` URL, predefined and fallback models), `api` (api-key and Bearer header, path and api-version, allow-listed body: extra client keys dropped, tools forwarded, `stream_options` only for streams; JSON 400 (exactly Azure's message, nothing appended) and text/plain 500 errors), `dotted` (`gpt-4.1`, `Phi-3.5-mini-instruct` reach upstream intact, model in header or body), `browser` (full status sequence, error status with exactly Azure's message), `tasks` (title task, also with Azure AI Search valves, #123), `rag` (the pipe's own Azure AI Search retrieval, 3.0.0, #187; `suites/_azure_rag.py`: API and browser path, stream and non-stream, Foundry `/models` endpoint; search request body per `query_type`, query text rules, embeddings for `deployment_name` (v1 route, gateway prefix, Bearer) and `endpoint` (api-version, own key or access token, chat key only on the same scheme, host and port), integrated vectorizer; `fields_mapping` with `select`, separator and fallback, list-valued titles; `filter`, `top_n_documents` (2 x candidates), `strictness` (per query), the api-version valve, the document budget (auto, also with a large `top_n_documents`, valve, unlimited, dropping a document); prompt placement, one block and one system message, `in_scope` and the tool-results clause, `role_information`, sanitizing (also nested tags, titles and file names), list and image-only content, Open WebUI's tool-images message, an attached file (`<attached_files>` and RAG template around the prompt); citations, `context` event and `message.context`, scores per score type and with `AZURE_AI_INCLUDE_SEARCH_SCORES` off; `[docX]` → links, also split across stream deltas (API and browser, also without a finish chunk; in API streams all text before `[DONE]` and nothing after it), already linked or with parentheses in the URL (stream and non-stream), with a dotted deployment name; links in the history sent back as `[docX]` (also links saved before 2.8.0 with `)` in the URL; a hostile history line in linear time); only referenced sources saved and the show-all valve (also for an answer that cites only a document the search did not return, `[doc9]`, or one next to `[doc1]`); a context event over 128 KiB, an upstream event of ~300 KB (read and passed on) and one over 4 MiB (an error, API and browser); `content: null`; tool rounds (one search, reused only by a tool round of the same message with unchanged valves, no duplicate sources also when the tool round references a document, a non-stream tool call), tools and `stream_options` forwarded, tasks without retrieval and without `data_sources` (also for a saved chat without a websocket session); query generation (follow-ups, `always` / `off`, transcript with Open WebUI's conversation summary, max queries, `<think>` with draft queries, fallback also for JSON nested too deep, partial and all-failed results, merge order by reciprocal rank fusion and reranker score, model in the body, the pause after 3 timeouts in a row: per model, reset by a success, still on after 16 s); auth: key, key valve, access token, system- and user-assigned managed identity (`mi_res_id`), token errors; fail-closed errors with a terminal status (HTTP 500/403/402/404/400/302/429/503, connection, retries, the 45 s retrieval limit, an answer larger than 16 MB, configuration errors, client `data_sources` (API stream and non-stream and the browser path as from an inlet filter: the error naming the removal of On Your Data in 3.0.0, the pipe's ERROR line and the terminal error status, nothing forwarded; an empty list is ignored), `context_length_exceeded` (exactly Azure's message and the budget hint)); Stop during a search and during query generation; `AZURE_AI_SEARCH_MODE=on_your_data` stored by a 2.9.0 pre-release or set in the environment is ignored; `log.debug`: the staged file run in the driver process with every logger at DEBUG logs no key or token, see below; `notice.none`: no On Your Data retirement notice of 2.8.1 in the server log), `logs` (no API key, no search key or token, no citation text, no generated queries, document titles or URLs in the server log at INFO and above) |
| `n8n` | `pipelines/n8n/n8n.py` | `mock_n8n.py` | `api` (request payload contract, bearer / Cloudflare headers, usage, `intermediateSteps` tool display with verbosity and truncation, `<think>` blocks, history / `INPUT_FIELD` / `RESPONSE_FIELD` valves, plain text, NDJSON, SSE streams in separate and coalesced writes with plain lines and `event:` / `id:` / `retry:` fields, OpenAI-style chunks, UTF-8 characters split across writes, braces inside strings, a large object trickling in as small writes (server CPU), webhook error), `browser` (saved answer, usage and final status for JSON, NDJSON, UTF-8, an n8n error chunk, a broken stream and a webhook error; chat context sent to the workflow for chat turns vs. background tasks; Stop during a stream and a non-stream request), `tasks` (title task without and with a chat id) |
| `infomaniak` | `pipelines/infomaniak/infomaniak.py` | `mock_infomaniak.py` | `models` (llm models only, `NAME_PREFIX`), `api` (product id and bearer key, allow-listed body, SSE stream normal, coalesced into one write and split mid-JSON; OpenAI-style and Infomaniak `error.description` errors with one log line each), `browser` (saved answer, usage and status events for those streams plus no final newline, CRLF, a broken stream and an upstream error; Stop during the stream and while waiting for the response headers), `tasks` |
| `filters` | `filters/*.py` + probe pipe | `mock_la.py` (Azure Log Analytics: HTTP Data Collector API, Logs Ingestion API, Microsoft Entra ID token endpoint, managed identity endpoints; the suite starts it, see below) | `model` / `global` (filters attached per model via `meta.filterIds` and as global filters: `features.web_search` → `__metadata__.features.google_search_tool`, `vertex_ai_search` + `VERTEX_AI_RAG_STORE`, API request without `features`, `time_token_tracker` outlet on the API path with exact token counts, its status in the browser path, background task without `__event_emitter__`), `spec` (`SEND_TO_LOG_ANALYTICS` env parsing, encrypted shared key and client secret, Logs Ingestion valve defaults, valve names), `la` (Log Analytics records through the HTTP Data Collector API: signature, headers, payload and exact counts on the API and browser path, special tokens, multi-turn averages, sending switched off, HTTP errors, a slow and a hanging endpoint do not delay the answer, estimate marker), `ingest` (Logs Ingestion API, #188: request shape and 204, token cache (scope and tenant in its key) / concurrency / refresh, refresh failure with a still-valid token (also inside the back-off), the 30 s token back-off (still on after 20 s) and its end, token and HTTP errors with one log line each including a second revocation, a late 401 for a replaced token and a persistent 401, a secret echoed across the cut points of an error text, undecryptable secret, https-only endpoint and authority and other invalid settings, slow / hanging / unreachable endpoints, mode selection and fallback, `both` (also with one side incomplete), deprecation warning, sovereign cloud valves, App Service (also its `{statusCode, message, correlationId}` error) / IMDS (without proxy) / workload identity, no secret or token in the log, DEBUG lines included), `valves` (compact status), `correlation` (inlet/outlet correlation when Open WebUI rewrites the last user message, concurrent identical requests), `encoding` (model-specific encoding, `gpt-4o` → `o200k_base`), `offline` (the tiktoken download hangs: estimates, one load at a time, retry, server not blocked), `multimodel` (multi-model chat and the features dict the models share), `search` (`google_search_tool` with features `{}`, `null` or without web_search, other feature keys kept, no per-user permission check), `vertex` (per-request data store, store only with the feature, `features: null`) |

The **probe pipe** (`tests/e2e/probe/probe_pipe.py`) answers with a JSON report of what
Open WebUI handed it (body keys, model and messages, `__metadata__` features / params /
model id, `__task__`, whether an `__event_emitter__` exists), so filter → pipe coupling
is tested without a provider. `PROBE_SLEEP=<s>` in the last user message delays its
answer (overlapping requests), `PROBE_SLEEP[<model>]=<s>` only that model's answer
(multi-model chats). `PROBE_ENV=<file>` sets (or, for `null`, removes) the environment
variables of a JSON object in that file in the server process, for an allow-list of
managed identity, workload identity and proxy variables only; the `ingest` group uses
it to switch between App Service, IMDS and workload identity without a restart and
puts the container's own `IDENTITY_ENDPOINT` / `IDENTITY_HEADER` (set by `run.sh` for
the azure suite) back at its end. The values go through the file, never through the
chat text. `PROBE_LOG=<file>` writes
everything the `time_token_tracker` logger logs, DEBUG included, to that file
(`PROBE_LOG=off` stops it): the server log runs at INFO, so the `filters` suite captures
the tracker's DEBUG output while its Log Analytics groups run (`tracker_debug.log` in
the output directory) and checks it for plaintext secrets (`log.no-secrets-debug`).

The `filters` suite sets up the Log Analytics mock itself: it maps the workspace host
`<id>.ods.opinsights.azure.com`, the login hosts `login.microsoftonline.com` /
`login.microsoftonline.us` and two DCR ingestion hosts to 127.0.0.1 and an
unreachable DCR host to 127.0.0.9 in the container's `/etc/hosts` (the login and DCR
lines are removed again at the end), installs a throw-away test CA into the container's
trust store (kept across `--reuse` runs, valid 30 days; the server certificate lists
every mapped host and is reissued when a name is missing) and starts
`mocks/mock_la.py` (HTTPS on 127.0.0.1:443, control routes on :9105, managed identity
endpoints on 127.0.0.1:9107 and 127.0.0.2:9107). The `ingest` group takes about two
and a half minutes, mostly timeout, token-expiry and token back-off waits. It also creates the user
`filters-user@example.com`; the `gemini` suite creates `e2e-gemini-user@example.com`.
The `offline` group needs a fresh container (tiktoken keeps loaded encodings per
process), so do not rerun it with `--reuse`.

The azure `rag` group uses `mocks/mock_search.py` (127.0.0.1:9106, started with the
other mocks): Azure AI Search *Documents - Search Post* with the indexes `x100-docs`,
`x100-custom` (other field names, no vector field) and `client-index` (must never be
queried), Search-like validation (api-version, `api-key` / Bearer, `select`, vector
length, `Content-Type`), scores by rank, and trigger words in the search text
(`no-hits`, `search-500`, `search-503-once`, `search-slow`, `doc-inject`, ...; the
list is in the mock's docstring). It also answers `GET /msi/token`, the App Service
managed identity endpoint: `run.sh` starts the container with
`IDENTITY_ENDPOINT=http://127.0.0.1:9106/msi/token` and
`IDENTITY_HEADER=e2e-identity-header`, so azure-identity's `ManagedIdentityCredential`
and `DefaultAzureCredential` in the Open WebUI process mint their tokens there (a
user-assigned identity is selected with `mi_res_id`, as on App Service; a
resource id containing `mi-fail` gets HTTP 400). A container started by an older
`run.sh` lacks these variables; the managed identity checks then fail with `--reuse`.

The server runs at INFO, so `logs.no-secrets` covers INFO and above. Open WebUI at
`GLOBAL_LOG_LEVEL=DEBUG` logs the parameters of its database queries itself (valve
values among them), so a DEBUG server log cannot show what the pipe logs. Instead
`rag.log.debug` loads the staged Azure file into the driver process (with a stub
`open_webui.env`: `SRC_LOG_LEVELS` OPENAI=DEBUG), sets every logger to DEBUG and calls
`pipe()` against the same mocks (search keys, tokens, embeddings, managed identity,
query generation, errors, client `data_sources` with a key, also streamed); no log record may
contain a key or token.

Checks per suite on Open WebUI v0.11.4-slim in strict known mode (observed 2026-10-10,
`main` 2fc7bcc: Azure pipeline 3.0.0, Time Token Tracker 2.7.0 with the Logs Ingestion
API, native tool calling in `google_gemini.py` 1.18.0 with the #194 fix, the
`NAME_PREFIX` fix in `infomaniak.py` 2.2.2 and the 10 `gemini.owuitools` checks; plus
`google_gemini.py` 1.19.0 with the native reasoning block, 6 more gemini checks):

| Suite | Checks | PASS / KNOWN |
| --- | ---: | ---: |
| `gemini` | 165 | 165 / 0 |
| `azure` | 198 | 198 / 0 |
| `n8n` | 51 | 51 / 0 |
| `infomaniak` | 32 | 32 / 0 |
| `filters` | 101 | 101 / 0 |
| **all** | **547** | **547 / 0** |

No FAIL and no obsolete marker; `v0.11.3-slim` gives the same counts for every suite
(measured there for the gemini suite: 165 / 0). The checks of the 1.19.0 thinking block
and search statuses carry no marker either: with `google_gemini.py` 1.18.0, 12 of the 32
checks of `--only 'gemini.(api|thinking|browser|grounding|image)$'` FAIL
(`api.nonstream` / `.stream`, `browser.stream` / `.nonstream`, `image.browser-preview` /
`-ga`, `thinking.strip-reasoning`, `.live`, `.duration`, `.stop`, `.tool-round` and
`grounding.status`); `toolsapi.unchanged` checks the same block shape.
No known bug is registered (`harness/known_n8n.py` has no entry): the last one,
`infomaniak-name-prefix` (the `NAME_PREFIX` valve was read only once), was fixed in
`infomaniak.py` 2.2.2, and its marker went with the fix. The markers of the 62 bugs fixed by #182-#185 were dropped after the merge; their checks stay and
must pass. The `imgedit` group came with its fix and carries no marker either: with
`google_gemini.py` 1.18.0 before the fix (a7fc191) 15 of its 19 checks FAIL. The four
that pass check what did not change: `imgedit.temporary` (the fallback to the
request), `imgedit.video-file` (reading the user's own file) and the two
`imgedit.guided-*` variants (Open WebUI also puts the upload of the regenerated message
into the request). `images.history` and `images.history-current` FAIL there too (the
image limit kept the oldest images and could drop the current ones).

### Native tool calling (`gemini.tools`, `gemini.toolsapi`)

The two groups live in `tests/e2e/suites/_gemini_tools.py` (run by `suites/gemini.py`;
the leading underscore keeps the module out of the suite discovery); `--only
'gemini.tools'` runs both, `--only 'gemini.tools$'` the browser group only. The Gemini mock
answers with function calls, Open WebUI runs the tools (browser path) or hands the
`tool_calls` to the client (API path), and the scenarios check the saved `output`
items, what the client got and what the pipe sent upstream (the mock's record).

**The `MOCKTOOLS:` directive.** A JSON object after `MOCKTOOLS:` in the turn's user
message (the last user content with text that is not Open WebUI's tool-image message)
makes `mock_gemini.py` answer with function calls:

```text
MOCKTOOLS:{"rounds": [[{"name": "get_current_timestamp", "args": {}}],
                      [{"name": "calculate_timestamp", "args": {"days_ago": 1}}]],
           "text_before": "Let me check.", "split": false, "noid": false, "nosig": false,
           "server_side": false, "malformed": false, "allow_undeclared": false}
```

| Option | Effect |
| --- | --- |
| `rounds` | round `r` (the number of user contents with function responses after the turn's message, at least 1 + the highest `<r>` of a response id `mock-call-<r>-<k>`, because Open WebUI merges rounds without text or reasoning in between) answers with the calls of `rounds[r]`; after the last round the answer is `MOCK-FINAL <name>=<json response>; ...` over every function response of the turn |
| names | Open WebUI names; the call uses the pipe's mapped name (`_gemini_function_name`, copied into the mock) and must be declared, else the answer is `MOCK-ERROR undeclared function <name>; declared=[...]` (HTTP 200) |
| `allow_undeclared` | call an undeclared tool anyway (under its own name) |
| `noid` | calls without `id` (otherwise `mock-call-<r>-<k>`) |
| `nosig` | no thought signature (otherwise `gemini-3*` models sign the first call of a round with `base64("mock-sig\|<model>\|<id or name>")`) |
| `text_before` | a text part before the calls |
| `split` | streaming: one chunk per call (otherwise all calls in one chunk) |
| `server_side` | with `toolConfig.includeServerSideToolInvocations` in the request: a server-side `toolCall` (signed) + `toolResponse` before the calls |
| `malformed` | the first request gets a candidate without content and `finishReason: MALFORMED_FUNCTION_CALL`, or the finish reason given as a string (`"malformed": "UNEXPECTED_TOOL_CALL"`) |

A tool round's usage is `USAGE_TOOL` (21 / 5 / 26 tokens); streaming ends with a
`{"text": ""}` chunk with `finishReason: STOP`. With `googleSearch` in the request a
tool round cites `https://example.com/tool-round` and the final answer
`https://example.com/a`, so the sources of each round can be told apart. Like the real API, every generate
request is rejected with HTTP 400 `INVALID_ARGUMENT` for duplicate declaration names,
invalid names, `parameters` together with `parametersJsonSchema`, a model content with
function calls that is not directly followed by a user content with matching function
responses (count, names, ids) and, for `gemini-3*`, a first function call of a model
content of the current turn without the issued signature or
`skip_thought_signature_validator`. Each generate request records `declared`, `decl`
(the raw schema of every declaration, before any key normalisation), `tool_config`,
`fc_mode`, `allowed_names`, `include_flag`, `kinds` (role and part kinds per content),
`fc` / `fr` (calls with id, name, args and signature state `none` / `skip` / `issued` /
`bad`; responses with id, name and response), `round`, `directive`, `answer` (`fc`,
`final`, `text`, `malformed`, `mock-error`, `http-<status>`), `issued`,
`server_echoed`, `synthetic_ids_upstream` and `orphan_fr`.

**Tool mocks.** `mocks/mock_tools.py`, started by `serve_all.py`, is an OpenAPI tool
server on 127.0.0.1:9111 (`get_weather` with a query parameter, `convert_units` with a
JSON body, and `lookup.v2`, an operationId Gemini does not accept as a function name)
and an optional MCP server (FastMCP streamable HTTP from the image's `mcp` package) on
127.0.0.1:9112 (`mcp_echo`, `mcp_sum`; Open WebUI names them `<server id>_mcp_echo`).
`GET /__requests` on 9111 lists the OpenAPI calls and the MCP tool calls. They are not
provider mocks: `E2E_MOCK_FAULT` does not touch them. `probe/workspace_tool.py` is the
workspace Python tool the `tools` group creates (`add_numbers` reports the types it
got, `whoami` the user, the chat and its event emitter, and emits a status;
`make_image` returns a PNG data URI, which Open WebUI passes on in a user message
after the tool results). The group
deletes the tool, restores the tool server connections and the chat settings and
deletes its model overrides when it ends.

**Browser harness.** `BrowserSession` sends the Socket.IO namespace sid as
`session_id` (`sio.get_sid()`, what the web UI sends as `socket.id`; the engine.io sid
`sio.sid` it sent before is unknown to Open WebUI, so every server → client call
failed with "Client session disconnected.") and a `heartbeat` event every 20 s like
the web UI (Open WebUI reaps a session that sent none for 120 s). It answers Open
WebUI's `execute:tool` calls (direct tools) with `direct_tool_answer(data)` as the
socket.io ack, e.g. `[{"value": "client:k1"}, {"content-type": "application/json"}]`
(result, headers) and records them in `execute_calls`. `chat()` takes `tool_ids`,
`tool_servers`, `extra_body` and `until` (a predicate on the saved message that ends
the wait, e.g. a call waiting for approval); `wait()` waits again after
`owui.resolve_tool_call(...)`. `BrowserChat` adds `output`, `function_calls`
(`[(name, status, call_id, arguments)]`), `function_outputs` (`{call_id: text}`),
`reasoning_items` (`[(text, reasoning_details)]`) and `output_text`; `content` falls
back to the output text because Open WebUI's final save of a tool turn leaves
`content` empty. `owui.py` adds `create_tool` / `delete_tool`, `set_tool_servers`,
`chat_config` / `set_chat_config`, `resolve_tool_call`, and `parse_sse` collects
`tool_calls` (merged by index), `reasoning_details`, `reasoning_content`,
`finish_reasons`, `done_last` and `openai_finish_reason` (the openai SDK's
`ChatCompletionStreamState`, which ships with the image).

**Preflight and `mock_ok`.** Both groups start with a plain API request to
`gemini-2.5-flash`; every detail line starts with `mock_ok=<bool>` (the request got
`Hello from mock`). The evidence of a known bug of these groups must require
`mock_ok=True` plus a token that the request itself got the provider's answer
(`upstream=[fc,...]`, `upstream=[mock-error]`, `answered=<n>`, ...), so that with
`E2E_MOCK_FAULT` it can never be KNOWN. The detail tokens are stable (evidence can
match them): `http=`, `done=`, `calls=[name:status,...]`, `outputs=<n>`,
`rd=[format:id,...]`, `upstream=[answers in order]`, `declared_n=`, `safe_names=`,
`sig_echoed=[id:sig,...]`, `fr=[id:name:keys,...]`, `kinds=`, `tool_kinds=`,
`include_flag=`, `server_echoed=`, `finish=[...]`, `openai_finish=`, `usage=` and
`final=` (last). WARNING lines of the pipe that mean lost tool data (`Skipping
duplicate tool declaration`, `Dropping unmatched function call`, `Dropping unmatched
tool result`, `Could not restore stored model content`, `Invalid tool call arguments`)
fail the `server-log` check unless a scenario provokes one (`t.expect_warnings`);
`tools.log` fails on `'callable'`, `__signature__`, `Duplicate function declaration` or
`AFC is enabled` anywhere in the server log of the group.

The two groups came with native tool calling (`google_gemini.py` 1.18.0) and carry no
known-bug markers, so a `--ref` run of an older `google_gemini.py` reports most of
their checks as FAIL (42 of the 48 with 1.17.0). Open WebUI 0.11.4's approval defects
(an approved call loses its result in the saved message; with two calls only the first
is asked and the second is not run) are only recorded in the detail of
`tools.approval-parallel`, never asserted.

### Open WebUI's image and code tools (`gemini.owuitools`)

The group lives in `tests/e2e/suites/_gemini_owuitools.py` (run by `suites/gemini.py`
after `tools`). It checks that Open WebUI's own built-in tools `generate_image`,
`edit_image` (Open WebUI's image engine `gemini`) and `execute_code` (code interpreter)
work with a Gemini text model through Open WebUI's tool loop. It switches on **Admin
Settings → Images** with the engine `gemini` against the Gemini mock
(`IMAGES_GEMINI_API_BASE_URL=<mock>/v1beta`, keys `mock-images-key` and
`mock-images-edit-key`, by which the mock record tells the engine's requests from the
pipe's), image editing with `gemini-3.1-flash-image` and the code interpreter with the
engine pyodide, and restores both settings at its end. The `MOCKTOOLS:` directive makes
Gemini call the tool:

| Check | What it shows |
| --- | --- |
| `owuitools.api` | API path with both feature flags on: none of the three tools is declared (Open WebUI adds built-in tools for browser sessions only); `pyodide_note=` records, without asserting it, that Open WebUI still adds its pyodide note to the system prompt |
| `owuitools.declared` | browser path: with `features.image_generation` and `features.code_interpreter` (the chat toggles) all three are declared and Open WebUI's pyodide note is in the system instruction; without them none |
| `owuitools.declared-noedit` | `ENABLE_IMAGE_EDIT=false`: `generate_image` without `edit_image` |
| `owuitools.generate` | `generate_image` with the endpoint method `generateContent` (`gemini-2.5-flash-image`): the engine request (path, key, prompt), the image saved with the answer (message `files`, `chat:message:files`) with the bytes the mock returned, the tool result with its URL in the continuation, the final answer |
| `owuitools.generate-predict` | the same with `predict` (`imagen-4.0-generate-001`, the mock's `:predict` route) |
| `owuitools.edit` | `edit_image` on an upload: the request to Gemini names the file's URL in `<attached_files>`, the edit engine gets the uploaded image, the edited image is saved |
| `owuitools.edit-followup` | `edit_image` in the next turn on the image of `owuitools.generate`: the replayed tool result holds its URL, the edit engine gets that image |
| `owuitools.image-model` | a pipe image model with the Image toggle on: no function declarations, its own image is saved, the engine is not called |
| `owuitools.code` | `execute_code`, engine pyodide: the browser session gets `execute:python` (the code, its session id), its stdout is the tool result; an image line in it is uploaded and linked |
| `owuitools.code-jupyter` | engine jupyter: the Jupyter mock starts a kernel, runs the code and deletes the kernel; no `execute:python`, no pyodide note; stdout and result reach Gemini |

The Jupyter mock is part of `mocks/mock_tools.py` (`http://127.0.0.1:9111/jupyter/`:
`POST api/kernels`, the kernel websocket `api/kernels/<id>/channels`, which answers an
`execute_request` with stdout `JUPYTER-MOCK-STDOUT`, the result `'JUPYTER-MOCK-RESULT'`
and status idle, and `DELETE api/kernels/<id>`; the executed code is recorded with the
path `jupyter:execute`). `BrowserSession` acks `execute:python` with
`python_answer(data)` (default `None`, an empty ack; the web UI acks with the result of
its pyodide worker, `{stdout, stderr, result}`) and records the calls in
`python_calls`. `mock_gemini.py` answers the image engine: the `generateContent`
request of an image model like any image request (without thoughts, the engine asks for
none), `:predict` with `predictions` (the `final-image-<n>` trigger in the prompt picks
the PNG). Details start with `mock_ok=` as in the tool groups; the extra tokens are
`engine=[action:model:key]`, `file_ok=` and `sent_ok=`. Against `google_gemini.py`
1.17.0, which handed the tools to google-genai's automatic function calling, 6 of the
10 checks FAIL (`generate`, `generate-predict`, `edit`, `edit-followup`, `code`,
`code-jupyter`).

### API path vs. browser path

- **API path** (`OWUI.chat`): `POST /api/chat/completions` without `chat_id`, like an
  OpenAI-compatible client. The answer comes back directly (JSON or SSE); nothing is
  saved; outlet filters still run.
- **Browser path** (`BrowserSession.chat`): a socket.io connection plus `session_id`,
  message ids and `user_message` in the request, like the web UI. Open WebUI runs the
  pipe in the background with a real `__event_emitter__` and **saves** the answer,
  usage, sources, files and `statusHistory` into the chat, which the driver reads back.
  Only this path shows whether events and saved content work.
- **Background tasks**: title/tags/follow-ups call the pipe again with `__task__` set;
  `POST /api/v1/tasks/title/completions` runs one directly (no `__event_emitter__`),
  `background_tasks` in a browser-path request runs them after the answer.

## Results: PASS, FAIL, KNOWN

- `PASS` – the check held.
- `FAIL` – the check did not hold. The run exits with 1.
- `KNOWN` – the check did not hold because of a **known bug** registered in
  `tests/e2e/harness/known_<area>.py` (today only `known_n8n.py` for n8n + Infomaniak,
  with no entry at the moment; re-exported by `known.py`): key, summary, issue reference, pull request with the
  pending fix, evidence, and the function `file` plus the version `fixed_in` that fixes
  it. Printed with the bug, e.g. `known <key> (found by tests/e2e, no issue
  filed): ...; no fix yet`, or for a bug with a pending fix `...; fix pending in
  PR #<n> (not merged yet), fixed in <file> <version>`. KNOWN does not fail the run.

A tagged check only counts as KNOWN when the failure **looks like that bug**: one of the
bug's `evidence` regexes matches the check's detail text, or (for checks that pass
`since=mark`) the server log since `mark` contains the bug's log signature. A tagged
check that fails in another way (an HTTP 500, a missing model, ...) is a FAIL and says
`tagged known <key>, but the failure does not show it`.

**Version gating.** A known marker only applies while the tested copy of its `file`
(the staged file in `functions/`) has a docstring `version:` lower than `fixed_in`:

| Tested file | Tagged check fails | Tagged check passes |
| --- | --- | --- |
| older than `fixed_in` (e.g. `main` before the fix is merged) | KNOWN | PASS with the reminder `no longer reproduces, but <file> is older than <fixed_in>` (a FAIL in strict mode) |
| `fixed_in` or newer (the fix branch, `main` after the merge, a mutant of the fixed file) | FAIL `regression of known <key> (fixed in <file> <version>)` | PASS, listed under **Obsolete known markers: drop marker** in `summary.md` and `obsolete_markers` in `results.json` (not a failure) |

So `main` stays green while a fix is pending, the fix branch is protected against
regressions of exactly that bug, and the markers switch off by themselves once the fix
(with its version bump) is merged. A bug without a fix has `fixed_in=""` and is never
gated.

**Strict mode** (`--strict-known` or `E2E_STRICT_KNOWN=1`, on in CI): a tagged check that
passes while its marker still applies is a FAIL. It catches a fix without a version bump
and evidence that no longer describes the bug.

The fixing branch named in a KNOWN line may only exist as an open pull request (or not be
published yet) until the fix is merged; the issue link, where there is one, is the
stable reference.

The `server-log` check fails on every ERROR / Traceback / `ResourceWarning` block logged
while the suite ran, on plaintext values of password valves (any log level), and on
WARNING blocks a suite registered with `fail_on_warnings`, except blocks provoked on
purpose (`expect_errors` / `expect_warnings` with the signature of the provoked error or
warning) and ERROR or WARNING blocks that match the narrow log signature of a known bug
that reproduced in this suite (function name plus error, e.g. `Error in outlet filter
time_token_tracker` + `'NoneType' object is not callable`). A different error in the
same function, or the same error in another function, still fails it.

The driver adds a few checks of its own; they only show up when something is wrong:

| Check | FAIL when |
| --- | --- |
| `<suite>.timeout` | the suite ran longer than `E2E_SUITE_TIMEOUT`, or the run's budget `E2E_TIMEOUT` was used up (also for suites that could not start any more); the detail names the last check the suite recorded |
| `<suite>.crash` | the suite raised an exception (also `CancelledError`, `KeyboardInterrupt` or `SystemExit` from suite code); the other suites still run |
| `<suite>.no-checks` | the suite recorded no check at all |
| `<suite>.interrupted` | Ctrl-C / SIGTERM stopped the run during the suite (partial results) |
| `<suite>.server-log` | also recorded by the driver for a suite that ended before its own log scan (crash, timeout, return after a failed install) |
| `run.server-log` | an unexpected error block or a plaintext secret in a part of the server log no suite's `server-log` check read: before the first suite, between suites (the driver waits 1 s after each suite so late lines are not charged to the next one) and after the last one (late background tasks). The detail names the window (`after azure`) and the `function_<id>` that logged it |
| `run.mocks-log` | an error in the output of the provider mocks (`mocks.txt`). A mock answer that was cut short because the client (the pipe) closed the connection is listed as a note in `summary.md`, not a FAIL |

The output directory contains:

| File | Content |
| --- | --- |
| `driver.txt` | console output of the run |
| `results.json` | `meta` (image, Open WebUI version, function sources and their `versions`, `strict_known`, time limits, `completed`, `interrupted` / `setup_error` / `driver_error` when the run was cut short, `mock_notes`), `summary`, `obsolete_markers`, one entry per scenario (`suite`, `id`, `title`, `status`, `detail`, `known`, `marker`) |
| `summary.md` | Markdown summary (used as GitHub Actions job summary), with notes on partial results and cut-short mock answers |
| `server.log` | output of the Open WebUI server (`/tmp/e2e/server.log` in the container, the same text as `docker logs`) |
| `mocks.txt` | output of the mock servers |
| `mock_la.txt`, `tracker_debug.log` | `filters` suite only: output of the Log Analytics mock, and what `time_token_tracker` logged (DEBUG included) while the Log Analytics groups ran |
| `functions/` | the exact function files that were tested (converted to LF line endings), plus `SOURCES.txt` (where they came from) |

`results.json` and `summary.md` are also written when the run is cut short (Ctrl-C,
SIGTERM, `E2E_TIMEOUT`, a driver error); `meta.completed` is then `false`.

## Testing a fix, another branch or another Open WebUI version

```bash
tests/e2e/run.sh --ref my-fix-branch azure               # a branch, tag or commit
tests/e2e/run.sh --ref origin/main azure                 # e.g. compare with main
tests/e2e/run.sh --src ../other-worktree gemini          # files of another checkout
tests/e2e/run.sh --file pipelines/n8n/n8n.py=/tmp/n8n_fix.py n8n
tests/e2e/run.sh --image v0.11.3-slim gemini             # A/B against an older release
OWUI_IMAGE=ghcr.io/open-webui/open-webui:main tests/e2e/run.sh
```

A fix is complete when its KNOWN scenarios turn into PASS on the fix branch. The fix
bumps the file's `version:`, and `fixed_in` of the bug's `KnownIssue` names that version:
from then on the markers are off for every file with that version (a regression is a
FAIL), while `main` keeps reporting KNOWN until the fix is merged. Once it is merged,
drop the markers listed under "Obsolete known markers" in `summary.md` (the `known=`
arguments and the `KnownIssue` entry). To test fixes from several branches together,
export each file (`git show my-fix-branch:pipelines/azure/azure_ai_foundry.py >
/tmp/azure.py`) and pass one `--file` per file. `functions/SOURCES.txt` in the output
directory records where each tested file came from.

Function files are staged with LF line endings whatever their source: a Windows checkout
(`core.autocrlf=true`) has CRLF, `--ref` gives the file as stored in git (`n8n.py`,
`infomaniak.py` and `google_search_tool.py` are stored with CRLF) and the Open WebUI
editor gives LF. `SOURCES.txt` says `CRLF converted to LF` for a converted file.

## Debugging a failure

1. Read the scenario's detail line and `server.log` / `driver.txt` in the output dir.
2. Rerun only the interesting part and keep the container:

   ```bash
   tests/e2e/run.sh --keep --name owui-dbg --only 'azure.rag' azure
   ```

   The script prints the URL (`http://localhost:<port>`, login `admin@example.com` /
   `Passw0rd!e2e`). The mocks keep running, so you can chat with the functions in the
   browser.
3. Iterate without waiting for the start-up again: `tests/e2e/run.sh --reuse --name owui-dbg azure`
   (stops a driver left over from an interrupted run, copies the current files,
   restarts the mocks, reruns; `--only` works here too).
4. Look at what reached a mock (Git Bash: prefix with `MSYS_NO_PATHCONV=1`):

   ```bash
   docker exec owui-dbg curl -s http://127.0.0.1:9102/__requests   # 9101 gemini, 9102 azure, 9103 n8n, 9104 infomaniak, 9106 Azure AI Search, 9111 tools
   docker exec owui-dbg tail -n 100 /tmp/e2e/server.log
   ```

5. Clean up: `docker rm -f owui-dbg && docker volume rm owui-dbg-data`.

### Interrupted runs

Ctrl-C (SIGINT) or SIGTERM during a run is handled at once, also while the driver runs:
`run.sh` asks the driver in the container to stop (it records `<suite>.interrupted`
and writes the partial `results.json` / `summary.md`), copies the output directory,
removes container and volume (unless `--keep` / `--reuse`) and exits with 130 / 143.
That takes a few seconds; a second Ctrl-C does not cut the cleanup short. The browser
session's socket.io client is created with `handle_sigint=False`: otherwise engine.io
replaces the driver's SIGINT handler once a browser-path scenario ran (its handler cancels
every task and stops the event loop), and the stop was not recorded as interrupted.

If `run.sh` itself is killed (SIGKILL, a closed terminal on some systems, a CI job
cancelled without grace), nothing can clean up:

- the container and its volume stay; remove them with
  `docker rm -f <name> && docker volume rm <name>-data`. A later run with the same
  `--name` stops at the existing container, and at a leftover volume unless you pass
  `--force`;
- the **driver keeps running inside the container** (an orphan) and would share the
  mocks and the server log with the next run there. `--reuse` stops such orphans before
  it starts (`stopped 1 driver(s) left over from an earlier run`). To look for them by
  hand (the image has no `ps`):

  ```bash
  MSYS_NO_PATHCONV=1 docker exec owui-dbg bash -c \
    'for p in /proc/[0-9]*; do tr "\0" " " <$p/cmdline 2>/dev/null; echo; done | grep "[e]2e.py"'
  ```

`E2E_TIMEOUT` (default 1800 s) bounds a run that hangs: the suite that is still running
becomes a `<suite>.timeout` FAIL and the driver writes its results. A driver that does
not react is stopped by `timeout` in the container 60 s later (exit code 2).

## Adding a scenario or a mock behaviour

Layout of `tests/e2e/`:

```text
run.sh               host entry point (Docker + bash only)
check_owui_api.sh    static Open WebUI API compatibility check
e2e.py               in-container driver entry point
harness/             driver library: owui.py (REST client, API-path chat, valves, models),
                     browser.py (socket.io + saved chats), logs.py (server log),
                     mocks.py, results.py, known.py + known_<area>.py (known bugs),
                     suite.py, config.py
suites/              one module per suite: GROUPS + async def run(t: Suite); _*.py are
                     helper modules of a suite, not suites (_azure_rag.py: the azure rag
                     group; _gemini_tools.py: gemini.tools / toolsapi;
                     _gemini_owuitools.py: gemini.owuitools)
mocks/               aiohttp provider mocks + serve_all.py (127.0.0.1:9101-9104 and :9106 in
                     the container), tool servers mock_tools.py (OpenAPI and Jupyter :9111,
                     MCP :9112, no E2E_MOCK_FAULT); mock_la.py (Log Analytics) is started
                     by the filters suite (:443, :9105, :9107)
probe/probe_pipe.py  test-only pipe reporting what Open WebUI passes to a pipe
probe/workspace_tool.py  test-only workspace Python tool (gemini.tools)
```

A scenario is a few lines in a suite module:

```python
async def api(t: Suite, mock) -> None:
    await mock.reset()
    r = await t.owui.chat("gemini.gemini-2.5-flash", "Hello", stream=True)
    req = await mock.last()                       # what the pipe sent upstream
    t.check(
        "api.stream",                             # id -> "gemini.api.stream"
        "API stream: answer streamed",
        r.status == 200 and "Hello from mock" in r.content,
        r.brief(),                                # printed for FAIL/KNOWN
        known=known.MY_BUG,                       # KnownIssue from harness/known_*.py,
                                                  # only while a known bug breaks it
    )
```

- Browser path: `async with t.browser() as b: c = await b.chat(model, "text", stream=True)`
  then assert on `c.content`, `c.usage`, `c.sources`, `c.files`, `c.status_history`,
  `c.events`, `c.title`; for tool turns `c.function_calls`, `c.function_outputs`,
  `c.reasoning_items`, `c.output_text` (`b.chat(..., tool_ids=[...],
  tool_servers=[...], until=...)`, see [Native tool
  calling](#native-tool-calling-geminitools-geminitoolsapi)). A follow-up turn in the
  same chat passes `chat_id=c.chat_id, parent_id=c.message_id`: Open WebUI then rebuilds
  the history from the saved chat, like for the web UI. `t.owui.upload_file(name, data,
  content_type)` uploads a file like the web UI and `b.chat(..., user_files=[item])`
  attaches it to the user message (the file item the web UI saves, with the file id as
  `url`; see `_upload_png` in `suites/gemini.py`).
- Valves: use `t.owui.update_valves(fid, NAME=value)` (merges, see gotchas).
- Server log: `mark = t.mark()` before, `t.log.errors(mark)` after;
  `t.expect_errors(mark, signature)` for an error you provoke on purpose (only blocks
  matching the signature, e.g. `("function_azure:pipe", "request: 400")`, are
  ignored; anything else in the window still fails `server-log`).
  `t.fail_on_warnings(signature)` makes matching WARNING blocks fail `server-log`
  (`t.expect_warnings(mark, signature)` for one you provoke on purpose), and
  `t.assert_no_secrets(value)` checks that a secret never shows up in the log. Pass
  `since=mark` to a check tagged `known=` when the bug shows in the server log rather
  than in the detail text (e.g. a failing background task).
- Groups: wrap related scenarios in `if t.selected("group"):` so `--only` can select
  them, and list the group in the module's `GROUPS` (an unlisted group raises).
- Mock behaviour: mocks pick behaviour from the request (model name, webhook path or a
  trigger word in the last user message, e.g. `force-400`, or the Gemini mock's
  `MOCKTOOLS:` directive). Add a branch in `tests/e2e/mocks/mock_<provider>.py`;
  requests are recorded automatically.
- New known bug: add a `KnownIssue` to the area's `harness/known_<area>.py` (a new
  area module needs its own import at the bottom of `known.py`) and pass it as
  `known=`. Give it `evidence` (regexes that match the failing check's detail and
  nothing else: include proof that the request itself worked, e.g. `HTTP 200`, or in the
  tool groups `mock_ok=True` plus the mock's answer, so an unrelated failure stays
  FAIL), `log_patterns` when it logs errors or failing warnings (a string or a tuple
  of strings that must all occur in one block; include the function, e.g.
  `function_gemini:pipe`), `file` and `fixed_in` (the function file and the version that
  fixes it, `""` while no fix exists), the fixing branch (or `""`) and an issue `ref`.
  The CI meta-test fails when the evidence also matches a failing upstream (see
  [CI](#ci)). Only a bug that can show nothing but the request sent upstream goes
  into `REQUEST_SIDE_KNOWN` in `.github/workflows/e2e.yml` (empty today).
- New suite: add `tests/e2e/suites/<name>.py` (with `GROUPS` and `async def run(t)`);
  `run.sh` and the driver discover every module there (names starting with `_` are
  skipped), `all` runs new suites after the existing ones in alphabetical order. For a
  new function file add it to `FUNCTION_FILES` in `run.sh`. Mention the suite in this
  guide, in `CLAUDE.md` ("What to run") and, if it should appear there, in the `suites`
  input description of `.github/workflows/e2e.yml`.

Python under `tests/e2e/` follows the repo's Ruff settings:
`uvx ruff@0.11.10 format tests/e2e && uvx ruff@0.11.10 check tests/e2e` (or `pixi run lint`).

## Static Open WebUI API check

```bash
tests/e2e/check_owui_api.sh            # against v0.11.4
tests/e2e/check_owui_api.sh latest     # against the latest Open WebUI release
tests/e2e/check_owui_api.sh v0.12.0    # any tag or branch
```

It reads every `from open_webui... import ...` and `import open_webui...` in
`pipelines/` and `filters/`, fetches the corresponding Open WebUI backend modules at that
tag (`gh api` when authenticated, otherwise `curl` from raw.githubusercontent.com) and
checks that each imported module and symbol, each method called on it
(`Users.get_user_by_id`, ...) and each injected `__param__` still exists; definitions
marked legacy/deprecated are reported as `WARN` (for example `SRC_LOG_LEVELS` is an empty
legacy dict since 0.10, so per-module log levels have no effect). It also prints the
frontmatter of every function, the event types Open WebUI persists and the versions of
shared dependencies. A fetch that fails for another reason than "not found" is tried
3 times; if it still fails the result is `incomplete` (exit 2) rather than a missing
API. Run it whenever Open WebUI publishes a release, then run the Docker suites with the
new image.

## Real-API smoke test (manual, needs a key)

The suites run against mocks, so they cannot show what the real Gemini API accepts.
`tests/e2e/smoke/gemini-tools.sh` checks the server-side assumptions of the Gemini
pipe's native tool calling (risks R1-R15 of the 1.18.0 design) against
`generativelanguage.googleapis.com`. It is opt-in and never runs in CI: it needs a key
and makes billed requests (about 40 small ones on flash models, with the defaults).

```bash
export GOOGLE_API_KEY=...                       # from the environment only, never as an argument
tests/e2e/smoke/gemini-tools.sh                 # v0.11.4-slim, gemini-3-flash-preview + gemini-2.5-flash
tests/e2e/smoke/gemini-tools.sh --image v0.11.3-slim --only R3,R14   # some risks, another image
tests/e2e/smoke/gemini-tools.sh --dry-run-mock  # no key: the same run against the e2e Gemini mock
```

How it works:

- A fresh Open WebUI container is started with `run.sh`'s helpers (`run.sh` only defines
  its functions when it is sourced). `google_gemini.py` and `google_search_tool.py` come
  from the worktree (`--src DIR` for another checkout). The key goes into the encrypted
  `GOOGLE_API_KEY` valve. The tools are deterministic: Open WebUI's built-in tools, the
  e2e workspace tool (`probe/workspace_tool.py`), the OpenAPI and MCP tool mocks and
  client tools on the API path.
- The pipe's `BASE_URL` points at a recording pass-through proxy (`smoke/proxy.py`,
  127.0.0.1:9120 in the container). It forwards every request unchanged to the real API
  and streams the answer back. For each request it records the declared functions,
  tool kinds, `toolConfig`, the part kinds of every content, whether replayed function
  calls carry a signature (`yes` / `skip` / `none`), the HTTP status, the error message,
  finish reasons, the returned function calls with their signature flags, server-side
  tool parts, text lengths and usage. It never records headers, the query string,
  signature values or request texts. `--dry-run-mock` forwards to the e2e Gemini mock
  instead and adds the `MOCKTOOLS:` directive to the prompts; the key is then a dummy.
- Each scenario tells the model exactly which tool to call. A scenario whose model did
  not do what was asked (no call, one call instead of two, no search in the tool round
  of R3) or that got HTTP 429 / 5xx is repeated, at most `--retries` times (default 2).
  Every request carries `max_tokens` (default 2048; on the browser path Open WebUI
  keeps it for the continuations). The valves are `THINKING_LEVEL=low` (Gemini 3) and
  `THINKING_BUDGET=512` (Gemini 2.5, whose thinking counts against `maxOutputTokens`).
  The runner prints an estimate at the start (about 39 generate requests with the
  defaults, about 117 if every scenario used all its retries) and the hard cap, and the
  actual count and token usage at the end.
- Request budget: `--max-requests N` (default twice the printed maximum, 234 with the
  defaults) caps the generate requests the proxy forwards. Beyond it the proxy answers
  HTTP 429 itself, without forwarding; the scenario is not retried, the remaining risks
  are SKIP, and `summary.md` / `smoke.json` report `budget_hit` plus a `BUDGET` FAIL row
  (exit 1). The `LOG` row fails as well, because the pipe logs the proxy's 429 as an
  ERROR (a bad key likewise fails the server-log check with the model listing's 400s).
  The model listing is not capped (not billed); R9's Vertex requests do not
  pass through the proxy and are not capped either.
- Results: `PASS`, `FAIL` or `SKIP` per risk, with the evidence lines (one `upstream #n`
  line per request), plus `LOG` (the server log scan of the suites, which also fails on the
  pipe's WARNINGs for lost tool data) and `SECRETS`. Exit code 0 = no FAIL, 1 = a FAIL, 2 =
  setup error (install, the model list, or the preflight: one plain request per model,
  which catches an unavailable model). A model that the API does not list is added with
  `MODEL_ADDITIONAL`. A bad key already fails the model listing, so neither model shows
  up: the setup error, in the output and in `summary.md`, then names the failing
  upstream requests, e.g. `gemini: models [...] missing from Open WebUI; failing upstream
  requests: GET /v1alpha/models http=400 x4: API key not valid. Please pass a valid API
  key.`
- The key is never printed. The secrets are the key and, with `--vertex-credentials`,
  the secret values of that file (`refresh_token`, `client_secret`, `private_key_id`
  and each base64 line of `private_key`; values under 16 characters are left out). The
  runner redacts them from its stdout, stderr and output files, and `SECRETS` fails when
  one shows up in the server log, the mocks' output or the runner's output. After the
  run the script scans every output file for the same values, redacts any hit in place
  and exits 1.
- Container and volume are removed on exit, also after Ctrl-C (the runner writes partial
  results). `--keep` keeps them: the volume holds the key (encrypted), and with
  `--vertex-credentials` the container holds the credentials file in plain text; the
  script says so and prints the cleanup command. Images are never removed. Output:
  `tests/e2e/out/<time>-<name>/` with `driver.txt`, `summary.md`, `smoke.json`,
  `upstream.jsonl` (the proxy record), `server.log`, `mocks.txt` and the staged
  `functions/`.

| Risk | Scenario | PASS when |
| --- | --- | --- |
| R1 | Browser turn on both models with the workspace tool and the OpenAPI and MCP servers (27 built-ins + 8 tools declared). API request on both models with client tools in the schema styles of pydantic (`title`, `default`, `anyOf` + `null`), OpenAPI (`format`, integer `enum`) and MCP (`$defs`, `$ref`, `additionalProperties: false`), plus `oneOf` / `const`, type arrays and a tool without parameters | Every request gets HTTP 200. Built-ins that another Open WebUI version lacks are listed |
| R2 | Browser, `stream=false`, Gemini 3: `get_current_timestamp` | The tool is declared without a schema and called, the continuation gets 200, the answer is saved |
| R3 | Gemini 3 with `google_search_tool` and web search: Search and `get_current_timestamp` in one round, continued; the next turn with web search off; API `tool_choice` `required` and `none` with web search | `googleSearch` + functions + `includeServerSideToolInvocations` accepted, the call made, the server-side parts sent back within the turn and not in the next one, mode `ANY` gives a call and `NONE` none. If Gemini still did not search in the tool round after the retries, the result is PASS with a note that the replay was not exercised |
| R4 | API continuation without `reasoning_details` on Gemini 3 (placeholder signature), Gemini 2.5 (no signature) and `--model-latest` (default `gemini-flash-latest`) when the API lists it | 200 and an answer |
| R5 | Two calls in one round | At least 2 calls in one round, replayed as one model content and one user content with as many responses, 200 |
| R6 | `make_image` (a tool result with an image) | The continuation has the function responses followed by Open WebUI's image message, 200 |
| R7 | A tool turn, then a turn with `function_calling=legacy` | The second request declares no functions and gets 200 (the evidence says whether Open WebUI sent the old call) |
| R8 | Gemini 2.5 with thinking: a tool turn, a second tool turn, a plain turn | Every request 200 (the evidence counts the signatures on thought parts that are not sent back) |
| R9 | Vertex AI: a tool turn, a 64-character tool name, web search with tools, a Vertex AI Search data store (`--vertex-rag-store`) | Each turn done without an error. SKIP without credentials |
| R10 | Gemini 2.5 with web search | `googleSearch` without `functionDeclarations`, 200 |
| R11 | Note only: the installed `google-genai` version | SKIP with the note; FAIL when it is older than 1.68.0 after the function was saved |
| R12 | Streamed tool turn on Gemini 3 | No partial function call parts, the final text is not empty |
| R13 | `--turns` API tool turns (default 4), plus every answer of the run | Counts the finish reasons, `MALFORMED_FUNCTION_CALL` / `UNEXPECTED_TOOL_CALL` and `default_api.` names. FAIL only when such an answer did not become the pipe's error text |
| R14 | Tool approval (ask) with two calls, approve | The turn pauses, the continuation gets 200, also when Open WebUI 0.11.4 replays fewer calls than Gemini made |
| R15 | `INCLUDE_THOUGHTS=false` tool turn on Gemini 3 | The call carries a thought signature and the saved reasoning item has it |

Read the evidence, not only the status: it shows which risks were exercised (for example
`server_parts_in_round=0` in R3, `history_fc=False` in R7) and what to change if one fails.
For example, a 400 in R7 means that function-call history has to be rendered as text
when no functions are declared.

**Vertex AI (R9).** `--vertex-credentials FILE` (a service account key or the JSON of
`gcloud auth application-default login`; default `$GOOGLE_APPLICATION_CREDENTIALS`) and
`--vertex-project ID` (default `$GOOGLE_CLOUD_PROJECT`, else the file's `project_id`),
plus `--vertex-location` (default `global`). The file is streamed into the container as
its Application Default Credentials file (never into the output directory; it stays
there in plain text until the container is removed), and a
second copy of the pipe, `gemini_vertex`, runs with `USE_VERTEX_AI=true`. Its requests do
not go through the proxy, so the evidence is what Open WebUI saved plus the server log.
This path has not run yet: no Vertex project was available. A run with a fake service
account file and a fake key (it stops at the model list) checked the secret scan: the
runner and the script both took 10 values from the file and found none in the output.

**Dry run.** `--dry-run-mock` checks the runner itself without a key: every scenario runs
end to end against the e2e Gemini mock, which answers the `MOCKTOOLS:` directive instead
of following the prompt. Observed 2026-10-09 on v0.11.4-slim (and 2026-10-10 with
`google_gemini.py` 1.19.0): 15 PASS (R1-R8, R10, R12-R15,
`LOG`, `SECRETS`), 0 FAIL, 2 SKIP (R9 without Vertex credentials, R11 note), 38
generate requests (the mock received the same 38), 90-160 s including the container
start. The mock always follows the directive, so no retry happens. Its answers prove
only the mechanics: real results can differ. With `--max-requests 5` the proxy forwarded
the two preflight requests and three of R1, answered R1's fourth with HTTP 429 (the mock
received 5), R1 was FAIL, the other risks SKIP, `BUDGET` and `LOG` FAIL (the pipe logs
the 429 as an ERROR), exit 1. A fake key gave
exit 2 with the `API key not valid` setup error above, and the key was in no output.

## CI

`.github/workflows/e2e.yml` runs on pull requests and on pushes to `main` / `dev` that
touch `pipelines/**`, `filters/**`, `tests/**` or the workflow, every Monday, and on
demand (*Actions → E2E → Run workflow*, with an image tag and a suites input). Jobs:

| Job | What it does |
| --- | --- |
| `e2e` | `run.sh` with all suites in **strict known mode** (`E2E_STRICT_KNOWN=1`) against the default image, which is read from the `DEFAULT_IMAGE=` line of `run.sh` (the only place it is defined). The weekly run adds `ghcr.io/open-webui/open-webui:latest-slim`; a manual run uses the image tag input. Output directory as artifact, `summary.md` as job summary |
| `meta` | **Meta-test**: `main`'s function files (`--ref origin/main`) with every provider mock answering HTTP 500 (`E2E_MOCK_FAULT=500`), suites `gemini azure n8n infomaniak`. Nearly everything fails, and it must give **no KNOWN**: a KNOWN means the `evidence` of that known bug also matches an unrelated failure and would hide it. `REQUEST_SIDE_KNOWN` in the workflow may list bugs that can only show in the request the pipe sends upstream (they reproduce whatever the mock answers); it is empty, because the evidence of every registered bug also needs proof that the upstream answered (e.g. `HTTP 200` and the mock's answer). Observed 2026-10-09 (`main` 53b8495): 38 PASS / 197 FAIL / 0 KNOWN; on 2026-10-10 with `main` 3ff6cf9 (Azure pipeline 2.8.1) and the harness of the Azure pipeline 3.0.0 (azure `rag` group): 35 PASS / 327 FAIL / 0 KNOWN (466 s), and with `main` 4b80e24 (Azure pipeline 3.0.0, `google_gemini.py` 1.17.0) and the suites with the `gemini.tools` / `toolsapi` / `imgedit` groups: 57 PASS / 370 FAIL / 0 KNOWN (557 s); with `main` d3184b9 (`google_gemini.py` 1.18.0) and the checks of 1.19.0: 57 PASS / 376 FAIL / 0 KNOWN (518 s), and with `main` ecac3fb (`infomaniak.py` 2.2.2, no known bug registered) the same: 57 PASS / 376 FAIL / 0 KNOWN (526 s). The `filters` suite is left out because it uses no provider mock (its known bugs reproduce for real) |
| `api` | `check_owui_api.sh latest` |

`E2E_TIMEOUT` and the steps' `timeout-minutes` bound every job, so a hanging scenario
ends as a `<suite>.timeout` FAIL with results instead of a cancelled job. The jobs are
**informational**: they are not required checks and do not block merging.

## Troubleshooting and gotchas

- **`valves/update` replaces all valves.** `POST /api/v1/functions/id/{id}/valves/update`
  stores exactly what you send; every valve you leave out falls back to its default
  (secrets included). Always send the full set (`update_valves` merges for you).
- **Refresh the model list after model changes.** `/api/models` is cached; call
  `GET /api/models?refresh=true` after creating workspace models (`filterIds`) or
  changing valves that influence `pipes()`.
- **The data volume beats environment variables.** Settings saved in the database
  (`ENABLE_WEB_SEARCH`, task settings, ...) override `-e` variables on later starts.
  The harness uses a fresh volume per run; with `--reuse` the old settings stay.
- **Events and saved content need the browser path.** On the API path nothing is saved
  and status/source/files events go nowhere; usage/sources/files bugs only show up in
  browser-path scenarios.
- **Background tasks have no event emitter.** `/api/v1/tasks/*/completions` call the
  pipe with `__task__` set and `__event_emitter__=None`; pipes and filters must handle
  `None`.
- **A saved status can get lost in a new chat with a title task (Open WebUI race).**
  For a new saved chat, Open WebUI 0.11.4 runs the title generation as its own task,
  in parallel with the answer, and `Chats.update_chat_title_by_id` writes back the
  whole chat JSON it read before. A status event the pipe saves in between
  (`Chats.add_message_status_to_chat_by_id_and_message_id`) is then overwritten. Both
  writes started together lose the status in about half of the attempts (reproduced
  in the container, 20 of 40). The check `no-session.sources` (then in the azure `oyd`
  group, now `azure.rag.no-session.sources`) failed once that way
  (`statuses=[('Request completed', True, False)]`: the first status missing) and
  passed in the next four runs. If only a status is missing and the server log shows
  a normal request, it is this race; rerun the group to confirm.
- **First Gemini install is slow.** Creating the Gemini function pip-installs
  `google-genai` (~30-60 s); the create request waits for it.
- **Git Bash on Windows:** MSYS rewrites arguments that look like paths (`/e2e` →
  `C:/Program Files/Git/e2e`). `run.sh` runs every `docker` command with
  `MSYS_NO_PATHCONV=1` and streams host files with `tar` instead of passing host paths
  to `docker.exe`. A globally exported `MSYS_NO_PATHCONV` / `MSYS2_ARG_CONV_EXCL` would
  break `git -C /c/...` (`--ref`), so `run.sh` unsets both for itself. Prefix your own
  commands per call instead of exporting it: `MSYS_NO_PATHCONV=1 docker exec ...`. `*.sh` files
  are checked out with LF endings (`.gitattributes`), otherwise bash fails with `$'\r'`.
- **Never remove images** to "clean up" – only the container and its volume are
  throw-away; the image is shared with other runs.
- **Slow or unhealthy start:** the first start of a fresh volume runs all database
  migrations (~60-75 s on Docker Desktop, several minutes on a busy machine). The wait
  is limited to 600 s (`E2E_HEALTH_TIMEOUT=900` to raise it); if the container exits,
  see `server.log` in the output dir.
- **Older images and the embedding model.** Without further settings `v0.11.3-slim`
  downloads `sentence-transformers/all-MiniLM-L6-v2` from huggingface.co before
  `/health` answers (1522 s once on a busy network, more than the 600 s limit). The
  suites do not use RAG, so `run.sh` starts the container with
  `RAG_EMBEDDING_ENGINE=openai`: no download, and `v0.11.3-slim` became healthy after
  158 s on a busy machine, with the same results as `v0.11.4-slim`. A container you
  start by hand needs the same variable.
- **tiktoken encodings:** `time_token_tracker` loads its tiktoken encoding on first use
  (a download from `openaipublic.blob.core.windows.net`). The `filters` suite loads
  `cl100k_base` and `o200k_base` in the driver first, which fills the container's
  `TIKTOKEN_CACHE_DIR`, so the tracker reads them from the cache and its first counts
  are exact. Without network access the suite fails.
- **`WEBUI_SECRET_KEY`** is set by the harness; without it Open WebUI generates one and
  `EncryptedStr` valves are still encrypted, but a manual container without a stable key
  cannot decrypt valves after a restart.
