# Testing the functions locally (Docker E2E)

The functions in this repo only run inside Open WebUI, so they are tested end-to-end
against a **real Open WebUI container**: `tests/e2e/run.sh` starts a throw-away
container, installs the function files through the Open WebUI API, points them at
**mock provider APIs** (Gemini, Azure OpenAI / AI Foundry, n8n, Infomaniak, Azure Log
Analytics) and runs scenario suites that chat with them the way an API client and the
browser UI do.
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
Open WebUI container, whose Python already ships `aiohttp`, `httpx` and
`python-socketio`.

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
starting owui-e2e-004128-1234 from ghcr.io/open-webui/open-webui:v0.11.4-slim
Open WebUI healthy after 25s (http://localhost:49213)
Open WebUI 0.11.4, suites: gemini, azure, n8n, infomaniak, filters
function versions: google_gemini.py 1.16.1, azure_ai_foundry.py 2.7.0, n8n.py 2.3.0, ...
=== gemini ===
[PASS ] gemini.load  pipelines/google/google_gemini.py loads (create, import, activate)
[KNOWN] gemini.api.stream  API stream without websocket session: answer streamed, thinking in <details>
          -> known B1 (no issue filed): Gemini API stream without a websocket session ends
             with 'Error during streaming' ...; fix pending in PR #185 (not merged yet),
             fixed in pipelines/google/google_gemini.py 1.17.0
          HTTP 200 stream=True content=Error during streaming:  usage=None errors=[] upstream=None
...
--- gemini: 129s
...
SUMMARY: 167 PASS, 0 FAIL, 124 KNOWN in 435s
total runtime: 471s
output: tests/e2e/out/20261009-004128-owui-e2e-004128-1234
```

A full run of all suites took 7-9.5 minutes on a shared 8-CPU Docker host (427-565 s,
depending on the load; with `main`'s files 471 s: 25 s container start-up, then
gemini 129 s, azure 55 s, n8n 71 s, infomaniak 36 s, filters 143 s). Network downloads
on first use come on top
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
| `gemini` | `pipelines/google/google_gemini.py` (+ `google_search_tool`) | `mock_gemini.py` | `models` (image / video indicators, display names #172), `api` (thinking in `<details>`, full usage), `thinking` (summaries not replayed #176; budget, level, include and strip valves), `browser`, `tasks` (no `<details>` in task answers, no grounding tools for the tasks of a web_search chat), `image` (image models forced non-stream, exactly one saved file), `images` (thought images skipped, used as fallback, not used after IMAGE_SAFETY; dedup; two final images; image link for API clients; image history; optimization), `nano` (`gemini-nano-banana-2.1`), `imgvalve` (`IMAGE_GENERATION_MODELS`), `imgconfig` (ImageConfig valves, user valve, body), `imgtools` (tools per image model with web_search), `nostream` (`GOOGLE_STREAMING_ENABLED=false` with `stream=true`, #170), `video` (Veo: text and video saved, request shape, image-to-video), `grounding` (`google_search_tool` → googleSearch + urlContext, sources, `[1]` citations; no grounding without web_search), `vertex` (Vertex AI Search sources), `errors` (400, 500 with retry, blocked prompt, SAFETY finish, image error status, a streamed answer starting with `data:`), `retry` (`RETRY_COUNT` for streams), `status` (Stop leaves no running status), `valves` (model cache vs. valve changes, safety, whitelist, additional models, system prompt, user headers, API version, params, valve names and defaults), `concurrency` (forwarded user headers belong to the requesting user), `streamimg` (inline image in a stream) |
| `azure` | `pipelines/azure/azure_ai_foundry.py` | `mock_azure.py` (requires `api-version`; ignores `data_sources` when tools are sent, as Azure does) | `valves`, `models` (`AZURE_AI_MODEL` lists separated by `;`, `,` or spaces with exact names, `AZURE_AI_PIPELINE_PREFIX`, model from an `*.openai.azure.com` URL, predefined and fallback models), `api` (api-key and Bearer header, path and api-version, allow-listed body: extra client keys dropped, tools forwarded, `stream_options` only for streams; JSON 400 and text/plain 500 errors), `dotted` (`gpt-4.1`, `Phi-3.5-mini-instruct` reach upstream intact, model in header or body), `browser` (full status sequence, error status), `tasks` (title task, also with Azure AI Search valves, #123), `oyd` (Azure AI Search "On Your Data": `[docX]` → links, also split across stream deltas, already linked or with parentheses in the URL; links in the history sent back as `[docX]`; only referenced sources saved and the show-all valve; relevance scores; no `data_sources` for background tasks, also without a websocket session; no tools or `stream_options` together with `data_sources`; a 300 KB and a > 4 MiB context event; `content: null`), `logs` (no API key and no citation text in the log) |
| `n8n` | `pipelines/n8n/n8n.py` | `mock_n8n.py` | `api` (request payload contract, bearer / Cloudflare headers, usage, `intermediateSteps` tool display with verbosity and truncation, `<think>` blocks, history / `INPUT_FIELD` / `RESPONSE_FIELD` valves, plain text, NDJSON, SSE streams in separate and coalesced writes with plain lines and `event:` / `id:` / `retry:` fields, OpenAI-style chunks, UTF-8 characters split across writes, braces inside strings, a large object trickling in as small writes (server CPU), webhook error), `browser` (saved answer, usage and final status for JSON, NDJSON, UTF-8, an n8n error chunk, a broken stream and a webhook error; chat context sent to the workflow for chat turns vs. background tasks; Stop during a stream and a non-stream request), `tasks` (title task without and with a chat id) |
| `infomaniak` | `pipelines/infomaniak/infomaniak.py` | `mock_infomaniak.py` | `models` (llm models only, `NAME_PREFIX`), `api` (product id and bearer key, allow-listed body, SSE stream normal, coalesced into one write and split mid-JSON; OpenAI-style and Infomaniak `error.description` errors with one log line each), `browser` (saved answer, usage and status events for those streams plus no final newline, CRLF, a broken stream and an upstream error; Stop during the stream and while waiting for the response headers), `tasks` |
| `filters` | `filters/*.py` + probe pipe | `mock_la.py` (Azure Log Analytics Data Collector API; the suite starts it, see below) | `model` / `global` (filters attached per model via `meta.filterIds` and as global filters: `features.web_search` → `__metadata__.features.google_search_tool`, `vertex_ai_search` + `VERTEX_AI_RAG_STORE`, API request without `features`, `time_token_tracker` outlet on the API path with exact token counts, its status in the browser path, background task without `__event_emitter__`), `spec` (`SEND_TO_LOG_ANALYTICS` env parsing, encrypted shared key, valve names), `la` (Log Analytics records: signature, headers, payload and exact counts on the API and browser path, special tokens, multi-turn averages, sending switched off, HTTP errors, a slow and a hanging endpoint do not delay the answer, estimate marker), `valves` (compact status), `correlation` (inlet/outlet correlation when Open WebUI rewrites the last user message, concurrent identical requests), `encoding` (model-specific encoding, `gpt-4o` → `o200k_base`), `offline` (the tiktoken download hangs: estimates, one load at a time, retry, server not blocked), `multimodel` (multi-model chat and the features dict the models share), `search` (`google_search_tool` with features `{}`, `null` or without web_search, other feature keys kept, no per-user permission check), `vertex` (per-request data store, store only with the feature, `features: null`) |

The **probe pipe** (`tests/e2e/probe/probe_pipe.py`) answers with a JSON report of what
Open WebUI handed it (body keys, model and messages, `__metadata__` features / params /
model id, `__task__`, whether an `__event_emitter__` exists), so filter → pipe coupling
is tested without a provider. `PROBE_SLEEP=<s>` in the last user message delays its
answer (overlapping requests), `PROBE_SLEEP[<model>]=<s>` only that model's answer
(multi-model chats).

The `filters` suite sets up the Log Analytics mock itself: it maps the workspace host
`<id>.ods.opinsights.azure.com` to 127.0.0.1 in the container's `/etc/hosts`, installs
a throw-away test CA into the container's trust store and starts `mocks/mock_la.py`
(HTTPS on 127.0.0.1:443, control routes on :9105). It also creates the user
`filters-user@example.com`; the `gemini` suite creates `e2e-gemini-user@example.com`.
The `offline` group needs a fresh container (tiktoken keeps loaded encodings per
process), so do not rerun it with `--reuse`.

Checks per suite on Open WebUI v0.11.4-slim in strict known mode (observed 2026-10-09):

| Suite | Checks | `main` (0e47f2a): PASS / KNOWN | Known bugs seen on `main` | Fixed files of #182-#185: PASS / KNOWN |
| --- | ---: | ---: | ---: | ---: |
| `gemini` | 81 | 44 / 37 | 17 | 81 / 0 |
| `azure` | 70 | 37 / 33 | 14 | 70 / 0 |
| `n8n` | 51 | 37 / 14 | 10 | 51 / 0 |
| `infomaniak` | 32 | 15 / 17 | 8 | 31 / 1 |
| `filters` | 57 | 34 / 23 | 12 | 57 / 0 |
| **all** | **291** | **167 / 124** | **61** | **290 / 1** |

There is no FAIL in either column. `harness/known_*.py` registers 63 known bugs; two
Gemini bugs (`gemini-task-details`, `gemini-stream-data-prefix`) are hidden on `main`
behind B1 / B5 and only show as FAIL when their fix regresses. With the fixed files,
123 tagged checks are listed as obsolete markers; the one KNOWN left is
`infomaniak-name-prefix` (`NAME_PREFIX` is read only once, no fix yet, `fixed_in=""`).

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
  `tests/e2e/harness/known_<area>.py` (`known_gemini.py`, `known_azure.py`,
  `known_filters.py`, `known_n8n.py` for n8n + Infomaniak; re-exported by `known.py`):
  key, summary, issue reference, pull request with the pending fix, evidence, and the
  function `file` plus the version `fixed_in` that fixes it. Printed with the bug, e.g.
  `known B1 (no issue filed): ...; fix pending in PR #185 (not merged yet), fixed in
  pipelines/google/google_gemini.py 1.17.0`. KNOWN does not fail the run.

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
purpose (`expect_errors` with the signature of the provoked error) and blocks that match
the narrow log signature of a known bug that reproduced in this suite (function name plus
error, e.g. `Error in outlet filter time_token_tracker` + `'NoneType' object is not
callable`). A different error in the same function, or the same error in another
function, still fails it.

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
   tests/e2e/run.sh --keep --name owui-dbg --only 'azure.oyd' azure
   ```

   The script prints the URL (`http://localhost:<port>`, login `admin@example.com` /
   `Passw0rd!e2e`). The mocks keep running, so you can chat with the functions in the
   browser.
3. Iterate without waiting for the start-up again: `tests/e2e/run.sh --reuse --name owui-dbg azure`
   (stops a driver left over from an interrupted run, copies the current files,
   restarts the mocks, reruns; `--only` works here too).
4. Look at what reached a mock (Git Bash: prefix with `MSYS_NO_PATHCONV=1`):

   ```bash
   docker exec owui-dbg curl -s http://127.0.0.1:9102/__requests   # 9101 gemini, 9102 azure, 9103 n8n, 9104 infomaniak
   docker exec owui-dbg tail -n 100 /tmp/e2e/server.log
   ```

5. Clean up: `docker rm -f owui-dbg && docker volume rm owui-dbg-data`.

### Interrupted runs

Ctrl-C (SIGINT) or SIGTERM during a run is handled at once, also while the driver runs:
`run.sh` asks the driver in the container to stop (it records `<suite>.interrupted`
and writes the partial `results.json` / `summary.md`), copies the output directory,
removes container and volume (unless `--keep` / `--reuse`) and exits with 130 / 143.
That takes a few seconds; a second Ctrl-C does not cut the cleanup short.

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
suites/              one module per suite: GROUPS + async def run(t: Suite)
mocks/               aiohttp provider mocks + serve_all.py (127.0.0.1:9101-9104 in the container);
                     mock_la.py (Log Analytics) is started by the filters suite (:443, :9105)
probe/probe_pipe.py  test-only pipe reporting what Open WebUI passes to a pipe
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
        known=known.GEMINI_B1,                    # only while a known bug breaks it
    )
```

- Browser path: `async with t.browser() as b: c = await b.chat(model, "text", stream=True)`
  then assert on `c.content`, `c.usage`, `c.sources`, `c.files`, `c.status_history`,
  `c.events`, `c.title`.
- Valves: use `t.owui.update_valves(fid, NAME=value)` (merges, see gotchas).
- Server log: `mark = t.mark()` before, `t.log.errors(mark)` after;
  `t.expect_errors(mark, signature)` for an error you provoke on purpose (only blocks
  matching the signature, e.g. `("function_azure:pipe", "request: 400")`, are
  ignored; anything else in the window still fails `server-log`).
  `t.fail_on_warnings(signature)` makes matching WARNING blocks fail `server-log`, and
  `t.assert_no_secrets(value)` checks that a secret never shows up in the log. Pass
  `since=mark` to a check tagged `known=` when the bug shows in the server log rather
  than in the detail text (e.g. a failing background task).
- Groups: wrap related scenarios in `if t.selected("group"):` so `--only` can select
  them, and list the group in the module's `GROUPS` (an unlisted group raises).
- Mock behaviour: mocks pick behaviour from the request (model name, webhook path or a
  trigger word in the last user message, e.g. `force-400`). Add a branch in
  `tests/e2e/mocks/mock_<provider>.py`; requests are recorded automatically.
- New known bug: add a `KnownIssue` to the area's `harness/known_<area>.py` and pass it
  as `known=`. Give it `evidence` (regexes that match the failing check's detail and
  nothing else: include proof that the request itself worked, e.g. `HTTP 200`, so an
  unrelated failure stays FAIL), `log_patterns` when it logs errors (a string or a tuple
  of strings that must all occur in one error block; include the function, e.g.
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

## CI

`.github/workflows/e2e.yml` runs on pull requests and on pushes to `main` / `dev` that
touch `pipelines/**`, `filters/**`, `tests/**` or the workflow, every Monday, and on
demand (*Actions → E2E → Run workflow*, with an image tag and a suites input). Jobs:

| Job | What it does |
| --- | --- |
| `e2e` | `run.sh` with all suites in **strict known mode** (`E2E_STRICT_KNOWN=1`) against the default image, which is read from the `DEFAULT_IMAGE=` line of `run.sh` (the only place it is defined). The weekly run adds `ghcr.io/open-webui/open-webui:latest-slim`; a manual run uses the image tag input. Output directory as artifact, `summary.md` as job summary |
| `meta` | **Meta-test**: `main`'s function files (`--ref origin/main`) with every provider mock answering HTTP 500 (`E2E_MOCK_FAULT=500`), suites `gemini azure n8n infomaniak`. Nearly everything fails, and it must give **no KNOWN**: a KNOWN means the `evidence` of that known bug also matches an unrelated failure and would hide it. `REQUEST_SIDE_KNOWN` in the workflow may list bugs that can only show in the request the pipe sends upstream (they reproduce whatever the mock answers); it is empty, because the evidence of every registered bug also needs proof that the upstream answered (e.g. `HTTP 200` and the mock's answer). Observed 2026-10-09: 34 PASS / 200 FAIL / 0 KNOWN. The `filters` suite is left out because it uses no provider mock (its known bugs reproduce for real) |
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
