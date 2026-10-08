"""
In-container E2E driver for the Open WebUI functions in this repo.

Started by tests/e2e/run.sh inside the Open WebUI container (needs no host
Python). Talks to Open WebUI on 127.0.0.1:8080 and to the provider mocks
started from tests/e2e/mocks/serve_all.py.

usage: python3 /e2e/e2e.py [--suites gemini,azure,...|all] [--only REGEX]
                           [--out DIR] [--verbose] [--strict-known]

Writes <out>/results.json and <out>/summary.md. Exit code: 0 = only PASS /
KNOWN, 1 = at least one FAIL, 2 = setup error (bad arguments, --only matches no
scenario group, Open WebUI or a mock unreachable, pip / network failure while a
function is created, driver crash).

--strict-known (or E2E_STRICT_KNOWN=1): a check tagged with a known bug that
passes while its marker still applies is a FAIL (see harness/known.py).
"""

import argparse
import asyncio
import importlib
import json
import os
import re
import sys
import time
import traceback

from harness import OWUI, Results, ServerLog, SetupError, Suite
from harness.config import FUNCTIONS_DIR, MOCK_PORTS, SUITES
from harness.known import staged_version
from harness.mocks import Mock

TRUE = ("1", "true", "yes", "on")


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


async def main() -> int:
    args = parse_args()
    error = only_error(args)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    started = time.time()
    results = Results(verbose=args.verbose, strict_known=args.strict_known)
    log = ServerLog()
    owui = OWUI()

    if not await owui.wait_healthy():
        print("Open WebUI is not healthy", file=sys.stderr)
        return 2
    for name in MOCK_PORTS:
        mock = Mock(name)
        ready = await mock.wait_ready()
        await mock.close()
        if not ready:
            print(f"mock {name} is not reachable", file=sys.stderr)
            return 2
    await owui.login()
    version = await owui.version()
    sources = read_sources()
    versions = read_versions(sources)
    print(
        f"Open WebUI {version}, suites: {', '.join(args.suite_names)}"
        + (", strict known mode" if args.strict_known else ""),
        flush=True,
    )
    print(
        "function versions: "
        + ", ".join(f"{os.path.basename(p)} {v or '?'}" for p, v in versions.items()),
        flush=True,
    )

    setup_error = ""
    for name in args.suite_names:
        suite = Suite(name, owui, results, log, args.only, suite_groups(name))
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            module = importlib.import_module(f"suites.{name}")
            await module.run(suite)
        except SetupError as exc:  # environment broken: stop with exit code 2
            setup_error = f"{name}: {exc}"
        except Exception:  # a crashing suite is a FAIL, the others still run
            suite.check(
                "crash", "suite ran to completion", False, traceback.format_exc()
            )
        finally:
            await suite.close()
        print(f"--- {name}: {time.time() - t0:.0f}s", flush=True)
        if setup_error:
            print(f"setup error: {setup_error}", file=sys.stderr, flush=True)
            break

    await owui.close()
    meta = {
        "image": os.environ.get("E2E_IMAGE", "?"),
        "owui_version": version,
        "suites": args.suite_names,
        "only": args.only,
        "strict_known": args.strict_known,
        "sources": sources,
        "versions": versions,
        "started": started,
        "duration_s": round(time.time() - started),
    }
    if setup_error:
        meta["setup_error"] = setup_error
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results.to_json(meta), fh, indent=1)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(results.to_markdown(meta))
    counts = results.counts()
    obsolete = len(results.obsolete_markers())
    print(
        f"\nSUMMARY: {counts['PASS']} PASS, {counts['FAIL']} FAIL, "
        f"{counts['KNOWN']} KNOWN in {meta['duration_s']}s"
        + (f", {obsolete} obsolete known markers (drop marker)" if obsolete else ""),
        flush=True,
    )
    if setup_error:
        return 2
    return 1 if results.failed() else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except Exception:  # driver bug or Open WebUI gone: setup error, not a FAIL
        traceback.print_exc()
        sys.exit(2)
