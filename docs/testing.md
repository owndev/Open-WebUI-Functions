# Testing the functions locally (Docker E2E)

The functions in this repo only run inside Open WebUI, so they are tested end-to-end
against a **real Open WebUI container**: `tests/e2e/run.sh` starts a throw-away
container, installs the function files through the Open WebUI API, points them at
**mock provider APIs** (Gemini, Azure OpenAI / AI Foundry, n8n, Infomaniak) and runs
scenario suites that chat with them the way an API client and the browser UI do.
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
  ~1 GB) and for `pip install google-genai` when the Gemini pipe is installed.

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
starting owui-e2e-142233-1234 from ghcr.io/open-webui/open-webui:v0.11.4-slim
Open WebUI healthy after 67s (http://localhost:49213)
=== gemini ===
[PASS ] gemini.load  pipelines/google/google_gemini.py loads (create, import, activate)
[KNOWN] gemini.api.stream  API stream without websocket session: answer streamed
          -> known B1 (no issue filed): Gemini API stream without a websocket session ends
             with 'Error during streaming' ...; fix pending in branch hotfix/gemini-1.16.2
             (not merged yet)
...
SUMMARY: 98 PASS, 0 FAIL, 27 KNOWN in 258s
total runtime: 368s
output: tests/e2e/out/20261008-142233-owui-e2e-142233-1234
```

A full run took 2.5-6 minutes on a laptop with Docker Desktop: 40-70 s container
start-up, then 1.5-4 minutes of scenarios, most of it network downloads on first use
(`pip install google-genai` when the Gemini function is created, the tiktoken encoding
on the first `time_token_tracker` call). Without those the suites run in seconds.

The exit code is `0` when there is no FAIL (KNOWN results are fine), `1` when at least
one scenario FAILs and `2` for setup errors: Docker missing, `docker run` failing (port
busy, image missing), container not healthy, an option without its value, an unknown
suite, `--file` path or git ref, an `--only` regex that selects no scenario group, a
crashed driver.

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
| `-o, --out DIR` | `tests/e2e/out/<timestamp>-<name>` | output directory (git-ignored) |
| `-k, --keep` | off | keep container and volume after the run |
| `--reuse` | off | reuse the running container `--name` (implies `--keep`), skips the start-up |
| `-v, --verbose` | off | print details of passing scenarios too |

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
valves to the mock, and checks at least: the file **loads** (import + activate),
secrets are **stored encrypted**, the **model list**, the **API path** (non-stream and
stream), the **browser path** (saved answer and usage), a **background title task**
and that the **server log** has no unexpected `ERROR` / `Traceback` lines.

| Suite | Function(s) | Mock | File-specific scenarios |
| --- | --- | --- | --- |
| `gemini` | `pipelines/google/google_gemini.py` (+ `google_search_tool`) | `mock_gemini.py` | thinking wrapped in `<details>` and stripped from replayed history (#176), image models forced non-stream with IMAGE modality and the image saved to the chat (`gemini-3.1-flash-image-preview`, `gemini-3.1-flash-image`), Veo long-running operation with the video saved to the chat, Search grounding through the `google_search_tool` filter (googleSearch tool, sources, `[1]` citations) |
| `azure` | `pipelines/azure/azure_ai_foundry.py` | `mock_azure.py` | `AZURE_AI_MODEL` lists, model header vs. body, dotted model names (`gpt-4.1`, `Phi-3.5-mini-instruct`), allow-listed body (extra client keys such as `user` are dropped, `temperature` is kept), upstream errors, Azure AI Search "On Your Data": `[docX]` → links, only referenced sources saved, no `data_sources` for background tasks, `stream_options` with `data_sources` |
| `n8n` | `pipelines/n8n/n8n.py` | `mock_n8n.py` | bearer/Cloudflare headers, `usage`, `intermediateSteps` tool display (list and dict form), `<think>` blocks, n8n NDJSON streaming, SSE streams (separate and coalesced writes), webhook error |
| `infomaniak` | `pipelines/infomaniak/infomaniak.py` | `mock_infomaniak.py` | llm-only model list, product id / bearer key, allow-listed body, SSE stream normal, coalesced into one write and split mid-JSON, upstream error |
| `filters` | `filters/*.py` + probe pipe | – | per-model (`meta.filterIds`) and global filters, `features.web_search` → `__metadata__.features.google_search_tool`, `vertex_ai_search` + `VERTEX_AI_RAG_STORE`, API request without `features`, `time_token_tracker` outlet on the API path and its status in the browser path, background task sees `__task__` and no `__event_emitter__` |

The **probe pipe** (`tests/e2e/probe/probe_pipe.py`) answers with a JSON report of what
Open WebUI handed it (`__metadata__` features/params, `__task__`, whether an
`__event_emitter__` exists), so filter → pipe coupling is tested without a provider.

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
  `tests/e2e/harness/known.py` (key, summary, issue reference, branch with the pending
  fix, evidence). Printed with the bug, e.g.
  `known B1 (no issue filed): ...; fix pending in branch hotfix/gemini-1.16.2 (not merged yet)`.
  KNOWN does not fail the run. When a KNOWN scenario starts to pass (the fix was merged)
  the driver prints `known B1 no longer reproduces: drop the known= marker`.

A tagged check only counts as KNOWN when the failure **looks like that bug**: one of the
bug's `evidence` regexes matches the check's detail text, or (for checks that pass
`since=mark`) the server log since `mark` contains the bug's log signature. A tagged
check that fails in another way (an HTTP 500, a missing model, ...) is a FAIL and says
`tagged known <key>, but the failure does not show it`.

The fixing branch named in a KNOWN line may only exist as an open pull request (or not be
published yet) until the fix is merged; the issue link, where there is one, is the
stable reference.

The `server-log` check fails on every ERROR / Traceback block logged while the suite ran,
except blocks provoked on purpose (e.g. an upstream HTTP 400/500 test) and blocks that
match the narrow log signature of a known bug that reproduced in this suite (function
name plus error, e.g. `Error in outlet filter time_token_tracker` + `'NoneType' object is
not callable`). A different error in the same function, or the same error in another
function, still fails it.

The output directory contains:

| File | Content |
| --- | --- |
| `driver.txt` | console output of the run |
| `results.json` | `meta` (image, Open WebUI version, function sources, duration), `summary`, one entry per scenario (`suite`, `id`, `title`, `status`, `detail`, `known`) |
| `summary.md` | Markdown summary (used as GitHub Actions job summary) |
| `server.log` | `docker logs` of the Open WebUI container |
| `mocks.txt` | output of the mock servers |
| `functions/` | the exact function files that were tested, plus `SOURCES.txt` (where they came from) |

## Testing a fix, another branch or another Open WebUI version

```bash
tests/e2e/run.sh --ref my-fix-branch azure               # a branch, tag or commit
tests/e2e/run.sh --ref origin/main azure                 # e.g. compare with main
tests/e2e/run.sh --src ../other-worktree gemini          # files of another checkout
tests/e2e/run.sh --file pipelines/n8n/n8n.py=/tmp/n8n_fix.py n8n
tests/e2e/run.sh --image v0.11.3-slim gemini             # A/B against an older release
OWUI_IMAGE=ghcr.io/open-webui/open-webui:main tests/e2e/run.sh
```

A fix is complete when its KNOWN scenarios turn into PASS (and the `known=` markers are
removed in the same PR). To test fixes from several branches together, export each file
(`git show my-fix-branch:pipelines/azure/azure_ai_foundry.py > /tmp/azure.py`) and pass
one `--file` per file. `functions/SOURCES.txt` in the output directory records where each
tested file came from.

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
   (copies the current files, restarts the mocks, reruns; `--only` works here too).
4. Look at what reached a mock (Git Bash: prefix with `MSYS_NO_PATHCONV=1`):

   ```bash
   docker exec owui-dbg curl -s http://127.0.0.1:9102/__requests   # 9101 gemini, 9102 azure, 9103 n8n, 9104 infomaniak
   docker exec owui-dbg tail -n 100 /tmp/e2e/server.log
   ```

5. Clean up: `docker rm -f owui-dbg && docker volume rm owui-dbg-data`.

## Adding a scenario or a mock behaviour

Layout of `tests/e2e/`:

```text
run.sh               host entry point (Docker + bash only)
check_owui_api.sh    static Open WebUI API compatibility check
e2e.py               in-container driver entry point
harness/             driver library: owui.py (REST client, API-path chat, valves, models),
                     browser.py (socket.io + saved chats), logs.py (server log),
                     mocks.py, results.py, known.py (known bugs), suite.py, config.py
suites/              one module per suite: GROUPS + async def run(t: Suite)
mocks/               aiohttp provider mocks + serve_all.py (127.0.0.1:9101-9104 in the container)
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
  `t.expect_errors(mark)` for errors you provoke on purpose. Pass `since=mark` to a
  check tagged `known=` when the bug shows in the server log rather than in the detail
  text (e.g. a failing background task).
- Groups: wrap related scenarios in `if t.selected("group"):` so `--only` can select
  them, and list the group in the module's `GROUPS` (an unlisted group raises).
- Mock behaviour: mocks pick behaviour from the request (model name, webhook path or a
  trigger word in the last user message, e.g. `force-400`). Add a branch in
  `tests/e2e/mocks/mock_<provider>.py`; requests are recorded automatically.
- New known bug: add a `KnownIssue` to `harness/known.py` and pass it as `known=`. Give
  it `evidence` (regexes that match the failing check's detail, so other failures stay
  FAIL), `log_patterns` when it logs errors (a string or a tuple of strings that must
  all occur in one error block; include the function, e.g. `function_gemini:pipe`), the
  fixing branch (or `""` when no fix exists yet) and an issue `ref`.
- New suite: add `tests/e2e/suites/<name>.py` (with `GROUPS` and `async def run(t)`)
  and its name to `SUITES` in `harness/config.py`; `run.sh` accepts every module in
  `tests/e2e/suites/`. For a new function file add it to `FUNCTION_FILES` in `run.sh`.
  Mention the suite in this guide, in `CLAUDE.md` ("What to run") and, if it should
  appear there, in the `suites` input description of `.github/workflows/e2e.yml`.

Python under `tests/e2e/` follows the repo's Ruff settings:
`uvx ruff@0.11.10 format tests/e2e && uvx ruff@0.11.10 check tests/e2e` (or `pixi run lint`).

## Static Open WebUI API check

```bash
tests/e2e/check_owui_api.sh            # against v0.11.4
tests/e2e/check_owui_api.sh latest     # against the latest Open WebUI release
tests/e2e/check_owui_api.sh v0.12.0    # any tag or branch
```

It reads every `from open_webui... import ...` in `pipelines/` and `filters/`, fetches the
corresponding Open WebUI backend modules at that tag (`gh api` when authenticated,
otherwise `curl` from raw.githubusercontent.com) and checks that each imported symbol,
each method called on it (`Users.get_user_by_id`, ...) and each injected `__param__`
still exists; definitions marked legacy/deprecated are reported as `WARN` (for example
`SRC_LOG_LEVELS` is an empty legacy dict since 0.10, so per-module log levels have no
effect). It also prints the frontmatter of every function, the event types Open WebUI
persists and the versions of shared dependencies. Run it whenever Open WebUI publishes a
release, then run the Docker suites with the new image.

## CI

`.github/workflows/e2e.yml` runs `tests/e2e/run.sh all` on pull requests that touch
`pipelines/**`, `filters/**` or `tests/**`, and on demand (*Actions → E2E → Run workflow*,
with an image tag input). It uploads the output directory as an artifact and writes
`summary.md` to the job summary. The job is **informational**: it is not a required
check and does not block merging.

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
  migrations (~75 s on Docker Desktop, several minutes on a busy machine). The wait is
  limited to 600 s (`E2E_HEALTH_TIMEOUT=900` to raise it); if the container exits, see
  `server.log` in the output dir.
- **Older images download an embedding model at start-up.** `v0.11.3-slim` fetches
  `sentence-transformers/all-MiniLM-L6-v2` from huggingface.co before `/health` answers
  (several minutes on a slow or busy network); `v0.11.4-slim` does not. The suites do
  not use RAG, the download only costs time.
- **First `time_token_tracker` call is slow:** tiktoken downloads its `cl100k_base`
  encoding on first use (needs network); the first filters scenario took 50-120 s here.
- **`WEBUI_SECRET_KEY`** is set by the harness; without it Open WebUI generates one and
  `EncryptedStr` valves are still encrypted, but a manual container without a stable key
  cannot decrypt valves after a restart.
