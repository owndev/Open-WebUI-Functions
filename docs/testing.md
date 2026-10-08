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
             with 'Error during streaming' ...; fix pending in PR #185 (not
             merged yet)
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
(`core.autocrlf=true`) has CRLF, `--ref` and the Open WebUI editor give LF.
`SOURCES.txt` says `CRLF converted to LF` for a converted file.

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
  [CI](#ci)); a bug that shows in the request sent upstream goes into
  `REQUEST_SIDE_KNOWN` in `.github/workflows/e2e.yml` instead.
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
| `meta` | **Meta-test**: `main`'s function files (`--ref origin/main`) with every provider mock answering HTTP 500 (`E2E_MOCK_FAULT=500`), suites `gemini azure n8n infomaniak`. Nearly everything fails, and it must give **no KNOWN**: a KNOWN means the `evidence` of that known bug also matches an unrelated failure and would hide it. Exceptions are the bugs listed in `REQUEST_SIDE_KNOWN` in the workflow, which show in the request the pipe sends upstream and therefore reproduce whatever the mock answers (e.g. `azure-double-strip`: the mock records `body.model='1'`). The `filters` suite is left out because it uses no provider mock (its known bugs reproduce for real) |
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
- **First `time_token_tracker` call is slow:** tiktoken downloads its `cl100k_base`
  encoding on first use (needs network); the first filters scenario took 50-120 s here.
- **`WEBUI_SECRET_KEY`** is set by the harness; without it Open WebUI generates one and
  `EncryptedStr` valves are still encrypted, but a manual container without a stable key
  cannot decrypt valves after a restart.
