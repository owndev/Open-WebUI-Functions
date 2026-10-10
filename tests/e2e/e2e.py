"""
In-container E2E driver for the Open WebUI functions in this repo.

Started by tests/e2e/run.sh inside the Open WebUI container (needs no host
Python). Talks to Open WebUI on 127.0.0.1:8080 and to the provider mocks
started from tests/e2e/mocks/serve_all.py.

usage: python3 /e2e/e2e.py [--suites gemini,azure,...|all] [--only REGEX]
                           [--out DIR] [--verbose] [--strict-known]
                           [--timeout S] [--suite-timeout S]

Writes <out>/results.json and <out>/summary.md, also when the run is cut short
(SIGINT / SIGTERM, time budget used up, driver error): then they hold the
partial results. Exit code: 0 = only PASS / KNOWN, 1 = at least one FAIL, 2 =
setup error (bad arguments, --only matches no scenario group, Open WebUI or a
mock unreachable, pip / network failure while a function is created, driver
error), 128 + signal number when SIGINT / SIGTERM stopped the run.

--strict-known (or E2E_STRICT_KNOWN=1): a check tagged with a known bug that
passes while its marker still applies is a FAIL (see harness/known.py).

Checks the driver adds to the suites' own:

- ``<suite>.timeout``: the suite ran longer than --suite-timeout
  (E2E_SUITE_TIMEOUT, default 900 s) or the run's budget --timeout (E2E_TIMEOUT,
  default 1800 s) ran out; suites that cannot start any more get it too.
- ``<suite>.crash``: the suite raised (also CancelledError, KeyboardInterrupt or
  SystemExit raised by suite code).
- ``<suite>.no-checks``: the suite recorded no check at all.
- ``<suite>.interrupted``: SIGINT / SIGTERM stopped the run during the suite.
- ``<suite>.server-log`` for a suite that ended before its own log scan (crash,
  timeout, early return after a failed install).
- ``run.server-log``: unexpected ERROR / Traceback / ResourceWarning blocks or
  plaintext secrets that no suite's ``server-log`` check covered (before the
  first suite, between suites, after the last one: late background tasks),
  with the ``function_<id>`` that logged them.
- ``run.mocks-log``: errors of the provider mocks (``mocks.txt``). An answer a
  mock could not finish because the client (the pipe) closed the connection
  early is listed as a note in summary.md, not a FAIL.

The ``run.*`` checks are only recorded when they find something.
"""

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import re
import signal
import sys
import time
import traceback
from typing import Optional

from harness import OWUI, Results, ServerLog, SetupError, Suite
from harness.config import (
    FUNCTIONS_DIR,
    LOG_SETTLE,
    MOCK_PORTS,
    MOCKS_LOG,
    RUN_TIMEOUT,
    SERVER_LOG,
    SUITE_TIMEOUT,
    SUITES,
)
from harness.known import staged_version
from harness.logs import redact
from harness.mocks import Mock

TRUE = ("1", "true", "yes", "on")

# Function modules are named function_<id> in Open WebUI's log lines (0.12:
# function_<id>_<hex>, which ServerLog normalizes to function_<id>).
_FUNCTION = re.compile(r"\bfunction_\w+")

# mocks.txt: lines that start an error block. The first traceback after a
# header that announces one (and a chained traceback after "During handling of
# the above exception ...") belongs to that block.
_MOCK_ERROR_HEADER = re.compile(
    r"^(Error handling request|Task exception was never retrieved|"
    r"Exception in callback|Unclosed client session|Unclosed connector)"
    r"|ResourceWarning: "
)
_MOCK_TRACEBACK_HEADER = re.compile(
    r"^(Error handling request|Task exception was never retrieved|Exception in callback)"
)
_TRACEBACK = "Traceback (most recent call last):"
_CHAINED = (
    "During handling of the above exception",
    "The above exception was the direct cause",
)
# Regular output of serve_all.py (ends an error block).
_MOCK_OUTPUT = re.compile(r"^(mock \w+ listening on |E2E_MOCK_FAULT: )")
# A mock could not finish its answer because the client (the pipe under test)
# closed the connection early: noted, not a FAIL.
_MOCK_CLIENT_GONE = (
    "Cannot write to closing transport",
    "ConnectionResetError",
    "BrokenPipeError",
)
_MOCK_FRAME = re.compile(r'File "/e2e/mocks/(\w+\.py)", line (\d+)')


def _seconds(allow_zero: bool = False):
    """argparse type: a number of seconds > 0 (``allow_zero``: >= 0)."""

    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            value = float("nan")
        if not (value > 0 or (allow_zero and value == 0)):
            raise argparse.ArgumentTypeError(f"expected seconds, got {text!r}")
        return value

    return parse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--suites", default="all", help="comma list or 'all'")
    parser.add_argument("--only", default=None, help="regex on '<suite>.<group>'")
    parser.add_argument("--out", default="/e2e/out")
    parser.add_argument("--verbose", action="store_true", help="details for PASS too")
    parser.add_argument(
        "--strict-known",
        action="store_true",
        default=os.environ.get("E2E_STRICT_KNOWN", "").lower() in TRUE,
        help="a known-tagged check that passes while its marker applies is a FAIL",
    )
    parser.add_argument(
        "--timeout",
        type=_seconds(),
        default=os.environ.get("E2E_TIMEOUT") or str(RUN_TIMEOUT),
        help="time budget of the whole run in seconds (E2E_TIMEOUT)",
    )
    parser.add_argument(
        "--suite-timeout",
        type=_seconds(),
        default=os.environ.get("E2E_SUITE_TIMEOUT") or str(SUITE_TIMEOUT),
        help="time limit of one suite in seconds (E2E_SUITE_TIMEOUT)",
    )
    parser.add_argument(
        "--log-settle",
        type=_seconds(allow_zero=True),
        default=os.environ.get("E2E_LOG_SETTLE") or str(LOG_SETTLE),
        help="pause after each suite before the next log window (E2E_LOG_SETTLE)",
    )
    args = parser.parse_args()
    names = SUITES if args.suites in ("", "all") else args.suites.split(",")
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        parser.error(f"unknown suite(s) {unknown}; choose from {', '.join(SUITES)}")
    args.suite_names = list(names)
    if args.only:
        try:
            re.compile(args.only)
        except re.error as exc:
            parser.error(f"--only {args.only!r} is not a valid regex: {exc}")
    return args


def suite_groups(name: str) -> tuple:
    """``GROUPS`` of a suite module (empty when it cannot be imported; the
    import error is reported as that suite's crash later)."""
    try:
        return tuple(getattr(importlib.import_module(f"suites.{name}"), "GROUPS"))
    except Exception:
        return ()


def only_error(args: argparse.Namespace) -> str:
    """Message when ``--only`` selects no scenario group at all, else ''."""
    if not args.only:
        return ""
    groups = [f"{n}.{g}" for n in args.suite_names for g in suite_groups(n)]
    if not groups or any(re.search(args.only, g) for g in groups):
        return ""
    hint = ""
    if "\\|" in args.only:
        hint = " ('\\|' matches a literal '|'; write alternatives as (a|b))"
    return (
        f"--only {args.only!r} matches no scenario group{hint}; "
        f"groups: {' '.join(groups)}"
    )


def read_sources() -> dict:
    """Provenance of the staged function files (written by run.sh)."""
    path = os.path.join(FUNCTIONS_DIR, "SOURCES.txt")
    sources = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if "\t" in line:
                    repo_path, origin = line.rstrip("\n").split("\t", 1)
                    sources[repo_path] = origin
    return sources


def read_versions(sources: dict) -> dict:
    """Docstring ``version:`` of every staged function file (known markers are
    gated on these, see harness/known.py)."""
    return {repo_path: staged_version(repo_path) for repo_path in sources}


def file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_from(path: str, start: int, stop: Optional[int] = None) -> str:
    """Bytes ``start`` .. ``stop`` (None = end) of a file, decoded."""
    try:
        with open(path, "rb") as fh:
            fh.seek(start)
            data = fh.read() if stop is None else fh.read(max(0, stop - start))
    except OSError:
        return ""
    return data.decode("utf-8", "replace")


def mock_error_blocks(text: str) -> list:
    """Error blocks (lists of lines) in output of the provider mocks."""
    blocks: list = []
    current: Optional[list] = None
    for line in text.splitlines():
        if line.startswith(_TRACEBACK) and current is not None:
            last = next((ln for ln in reversed(current) if ln.strip()), "")
            first_traceback = _MOCK_TRACEBACK_HEADER.match(current[0]) and not any(
                ln.startswith(_TRACEBACK) for ln in current
            )
            if first_traceback or last.startswith(_CHAINED):
                current.append(line)  # traceback of this block
                continue
        if line.startswith(_TRACEBACK) or _MOCK_ERROR_HEADER.search(line):
            current = [line]
            blocks.append(current)
        elif _MOCK_OUTPUT.match(line):
            current = None
        elif current is not None:
            current.append(line)
    return blocks


def mock_block_summary(block: list) -> str:
    """'<first line> [mock_x.py:N] ... <last line>' of a mocks.txt error block."""
    text = "\n".join(block)
    frames = _MOCK_FRAME.findall(text)
    where = f" [{frames[-1][0]}:{frames[-1][1]}]" if frames else ""
    tail = next((ln for ln in reversed(block) if ln.strip()), "")
    summary = block[0].strip()[:160] + where
    if tail is not block[0]:
        summary += " ... " + tail.strip()[:160]
    return summary


class TrackedSuite(Suite):
    """A ``Suite`` that remembers how far its ``server-log`` check read; the
    driver's final scan covers the rest of the server log."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scanned_to: Optional[int] = None

    def scan_log(self) -> bool:
        self.scanned_to = self.log.mark()
        return super().scan_log()


class Run:
    """One driver run. ``write()`` saves whatever results exist, also after an
    interruption, a timeout or a driver error."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.started = time.time()
        self.deadline = time.monotonic() + args.timeout
        self.results = Results(verbose=args.verbose, strict_known=args.strict_known)
        self.log = ServerLog()
        self.log_start = self.log.mark()
        self.mocks_start = file_size(MOCKS_LOG)
        self.owui = OWUI()
        self.version = "?"
        self.sources = read_sources()
        self.versions = read_versions(self.sources)
        self.suites: list = []  # TrackedSuite objects in run order
        self.signal = ""  # name of the signal that stopped the run
        self.setup_error = ""
        self.driver_error = ""
        self.mock_notes: list = []
        self.finished = False  # every selected suite ran and the final scans ran

    # ------------------------------------------------------------- control
    def on_signal(self, sig: signal.Signals, task: asyncio.Task) -> None:
        """SIGINT / SIGTERM: stop the run (the running suite is cancelled, the
        partial results are written). Repeated signals change nothing."""
        if not self.signal:
            self.signal = sig.name
            print(f"\n{sig.name}: stopping the run", file=sys.stderr, flush=True)
            task.cancel()

    def exit_code(self) -> int:
        if self.signal:
            return 128 + signal.Signals[self.signal].value
        if self.setup_error or self.driver_error:
            return 2
        return 1 if self.results.failed() else 0

    async def execute(self) -> None:
        if not await self.owui.wait_healthy():
            self.setup_error = "Open WebUI is not healthy"
            print(self.setup_error, file=sys.stderr, flush=True)
            return
        for name in MOCK_PORTS:
            mock = Mock(name)
            ready = await mock.wait_ready()
            await mock.close()
            if not ready:
                self.setup_error = f"mock {name} is not reachable"
                print(self.setup_error, file=sys.stderr, flush=True)
                return
        await self.owui.login()
        self.version = await self.owui.version()
        print(
            f"Open WebUI {self.version}, suites: {', '.join(self.args.suite_names)}"
            + (", strict known mode" if self.args.strict_known else ""),
            flush=True,
        )
        print(
            "function versions: "
            + ", ".join(
                f"{os.path.basename(p)} {v or '?'}" for p, v in self.versions.items()
            ),
            flush=True,
        )
        for name in self.args.suite_names:
            await self.run_suite(name)
            if self.signal or self.setup_error:
                return
        self.scan_uncovered_log()
        self.scan_mocks()
        self.finished = True

    async def run_suite(self, name: str) -> None:
        suite = TrackedSuite(
            name, self.owui, self.results, self.log, self.args.only, suite_groups(name)
        )
        self.suites.append(suite)
        print(f"\n=== {name} ===", flush=True)
        before = len(self.results.items)
        t0 = time.monotonic()
        limit = min(self.args.suite_timeout, self.deadline - t0)
        if limit <= 0:
            suite.check(
                "timeout",
                "suite started within the run's time budget",
                False,
                f"not started: the run's time budget of {self.args.timeout:.0f}s "
                "(E2E_TIMEOUT) is used up",
            )
            return
        timer = asyncio.timeout(limit)
        try:
            module = importlib.import_module(f"suites.{name}")
            async with timer:
                await module.run(suite)
        except SetupError as exc:  # environment broken: stop with exit code 2
            self.setup_error = f"{name}: {exc}"
            print(f"setup error: {self.setup_error}", file=sys.stderr, flush=True)
            return
        except asyncio.CancelledError:
            if not self.signal:  # raised by the suite itself
                suite.check(
                    "crash", "suite ran to completion", False, traceback.format_exc()
                )
            else:  # SIGINT / SIGTERM
                suite.check(
                    "interrupted",
                    "suite ran to completion",
                    False,
                    f"{self.signal} stopped the run during this suite after "
                    f"{time.monotonic() - t0:.0f}s (partial results)",
                )
                return
        except BaseException as exc:  # a crashing suite is a FAIL, the others run
            if isinstance(exc, TimeoutError) and timer.expired():
                own = [r.id for r in self.results.items[before:]]
                suite.check(
                    "timeout",
                    f"suite finished within {limit:.0f}s",
                    False,
                    f"still running after {limit:.0f}s (E2E_SUITE_TIMEOUT="
                    f"{self.args.suite_timeout:.0f}s, E2E_TIMEOUT="
                    f"{self.args.timeout:.0f}s); last check: "
                    f"{own[-1] if own else 'none'}",
                )
            else:
                suite.check(
                    "crash", "suite ran to completion", False, traceback.format_exc()
                )
        finally:
            await suite.close()
        if len(self.results.items) == before:
            suite.check(
                "no-checks",
                "suite recorded at least one check",
                False,
                "run() returned without recording any check",
            )
        await self.log.settle(self.args.log_settle)
        if suite.scanned_to is None:  # ended before its own server-log check
            suite.scan_log()
            await self.log.settle(self.args.log_settle)
        print(f"--- {name}: {time.monotonic() - t0:.0f}s", flush=True)

    # --------------------------------------------------------- final scans
    def scan_uncovered_log(self) -> None:
        """``run.server-log``: error blocks and plaintext secrets in the parts
        of the server log no suite's ``server-log`` check read: before the
        first suite, and from each suite's scan to the start of the next suite
        (the last one: to now). Blocks after a suite are judged with that
        suite's expected / known signatures."""
        if not self.log.available:
            return
        end = self.log.mark()
        windows = [
            (
                "before the first suite",
                self.log_start,
                self.suites[0].log_start if self.suites else end,
                (),
                (),
            )
        ]
        for i, suite in enumerate(self.suites):
            start = suite.log_start if suite.scanned_to is None else suite.scanned_to
            stop = self.suites[i + 1].log_start if i + 1 < len(self.suites) else end
            windows.append(
                (
                    f"after {suite.name}",
                    start,
                    stop,
                    tuple(suite.ignored_log_patterns),
                    tuple(suite.ignored_log_ranges),
                )
            )
        problems, count = [], 0
        for label, start, stop, ignore, ranges in windows:
            if stop <= start:
                continue
            # The signature "" matches every block: drops the blocks from stop on.
            beyond = ((stop, float("inf"), ("",)),)
            errors = self.log.errors(start, ignore, ranges + beyond)
            segment = read_from(SERVER_LOG, start, stop)
            leaks = [
                f"{name} logged {segment.count(value)}x"
                for value, name in self.owui.secrets.items()
                if value and value in segment
            ]
            if errors or leaks:
                count += len(errors) + len(leaks)
                names = sorted({m for e in errors for m in _FUNCTION.findall(e)})
                problems.append(
                    f"{label} [{', '.join(names) or 'no function_<id> named'}]: "
                    + " || ".join(leaks + errors[:3])
                )
        if problems:
            detail = f"{count} unexpected: " + " ;; ".join(problems)
            for value in self.owui.secrets:  # never print a secret
                detail = redact(detail, value) if value else detail
            self.results.add(
                "run",
                "run.server-log",
                "no unexpected ERROR / Traceback / secret outside the suites' "
                "server-log checks",
                False,
                detail,
            )
        else:
            print("\nfinal server log scan: nothing outside the suites' checks")

    def scan_mocks(self) -> None:
        """``run.mocks-log``: error blocks in the mocks' output since the run
        started (client disconnects are notes)."""
        text = read_from(MOCKS_LOG, self.mocks_start)
        failures = []
        for block in mock_error_blocks(text):
            joined = "\n".join(block)
            target = (
                self.mock_notes
                if any(sig in joined for sig in _MOCK_CLIENT_GONE)
                else failures
            )
            target.append(mock_block_summary(block))
        if self.mock_notes:
            print(
                f"note: {len(self.mock_notes)} mock answer(s) cut short because the "
                "client closed the connection: " + " || ".join(self.mock_notes[:3]),
                flush=True,
            )
        if failures:
            self.results.add(
                "run",
                "run.mocks-log",
                "provider mocks handled every request without an error",
                False,
                f"{len(failures)} error(s) in mocks.txt: " + " || ".join(failures[:3]),
            )

    # -------------------------------------------------------------- output
    def meta(self) -> dict:
        meta = {
            "image": os.environ.get("E2E_IMAGE", "?"),
            "owui_version": self.version,
            "suites": self.args.suite_names,
            "only": self.args.only,
            "strict_known": self.args.strict_known,
            "sources": self.sources,
            "versions": self.versions,
            "started": self.started,
            "duration_s": round(time.time() - self.started),
            "timeouts": {
                "run_s": self.args.timeout,
                "suite_s": self.args.suite_timeout,
            },
            "completed": self.finished,
            "suites_started": [suite.name for suite in self.suites],
        }
        if self.signal:
            meta["interrupted"] = self.signal
        if self.setup_error:
            meta["setup_error"] = self.setup_error
        if self.driver_error:
            meta["driver_error"] = self.driver_error[-2000:]
        if self.mock_notes:
            meta["mock_notes"] = self.mock_notes
        return meta

    def write(self) -> None:
        meta = self.meta()
        os.makedirs(self.args.out, exist_ok=True)
        path = os.path.join(self.args.out, "results.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.results.to_json(meta), fh, indent=1)
        notes = []
        if self.signal:
            notes.append(f"**Partial results**: {self.signal} stopped the run.")
        elif self.setup_error:
            notes.append(f"**Partial results**: setup error: {self.setup_error}.")
        elif self.driver_error:
            notes.append("**Partial results**: driver error (see driver.txt).")
        notes += [
            f"Mock answer cut short (the client closed the connection): `{note}`"
            for note in self.mock_notes
        ]
        with open(
            os.path.join(self.args.out, "summary.md"), "w", encoding="utf-8"
        ) as fh:
            fh.write(self.results.to_markdown(meta))
            if notes:
                fh.write("\n### Notes\n\n" + "".join(f"- {n}\n" for n in notes))
        counts = self.results.counts()
        obsolete = len(self.results.obsolete_markers())
        print(
            f"\nSUMMARY{'' if self.finished else ' (partial)'}: {counts['PASS']} PASS, "
            f"{counts['FAIL']} FAIL, {counts['KNOWN']} KNOWN in {meta['duration_s']}s"
            + (
                f", {obsolete} obsolete known markers (drop marker)" if obsolete else ""
            ),
            flush=True,
        )

    async def close(self) -> None:
        await self.owui.close()


async def main() -> int:
    args = parse_args()
    error = only_error(args)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    run = Run(args)
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, run.on_signal, sig, task)
    try:
        await run.execute()
    except (Exception, asyncio.CancelledError):
        if not run.signal:  # driver bug or Open WebUI gone: setup error, not a FAIL
            run.driver_error = traceback.format_exc()
            print(run.driver_error, file=sys.stderr, flush=True)
    finally:
        run.write()
        with contextlib.suppress(Exception):
            await run.close()
    return run.exit_code()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except Exception:  # driver bug before the results could be written
        traceback.print_exc()
        sys.exit(2)
