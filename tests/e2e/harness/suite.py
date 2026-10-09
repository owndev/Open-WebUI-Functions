"""The context object a suite module's ``run(t)`` works with."""

import asyncio
import os
import re
from typing import Optional

import httpx

from .browser import BrowserSession
from .config import FUNCTIONS_DIR, PROBE_FILE
from .known import KnownIssue
from .logs import ServerLog
from .mocks import Mock
from .owui import OWUI
from .results import FIXED, Results, SetupError, short

# A function create/update that failed because pip could not install the
# frontmatter requirements (network, package index): retried once, then a
# setup error. Anything else (bad import, syntax error) stays a FAIL.
PIP_FAILURE = ("Error installing packages", "'pip', 'install'")
# Error blocks of such a failed attempt (ignored once the retry succeeded).
PIP_FAILURE_LOG = (
    "Error installing packages",
    "Failed to create a new function",
    "Error loading module",
    "Could not find a version that satisfies",
    "No matching distribution found",
    "'pip', 'install'",
    "pip._internal",
    "pip._vendor",
)
INSTALL_RETRY_DELAY = 10


class Suite:
    """Scenario helpers bound to one suite (ids are prefixed with its name).

    Server-log bookkeeping: the final ``scan_log()`` check reports ERROR /
    Traceback / ResourceWarning blocks logged while the suite ran, except
    blocks that were provoked on purpose (``expect_errors`` windows with their
    signatures, ``expect_log`` signatures) and blocks matching the
    ``log_patterns`` signature of a known bug that reproduced in this suite (see
    ``harness.known``). It also fails on WARNING blocks registered with
    ``fail_on_warnings`` and on plaintext secrets (password valves written
    through ``owui.update_valves``) anywhere in the log since the suite started.
    """

    def __init__(
        self,
        name: str,
        owui: OWUI,
        results: Results,
        log: ServerLog,
        only: Optional[str] = None,
        groups: tuple = (),
    ):
        self.name = name
        self.owui = owui
        self.results = results
        self.log = log
        self.only = re.compile(only) if only else None
        self.groups = tuple(groups)  # the suite module's GROUPS
        self.log_start = log.mark()
        self.ignored_log_patterns: list = []
        self.ignored_log_ranges: list = []
        self.warning_signatures: list = []
        self._mocks: dict = {}

    # -------------------------------------------------------------- selection
    def selected(self, group: str) -> bool:
        """True when ``--only`` is unset or matches ``<suite>.<group>``.

        ``group`` must be listed in the suite module's ``GROUPS`` (e2e.py
        checks ``--only`` against those before anything runs).
        """
        if self.groups and group not in self.groups:
            raise ValueError(
                f"group {group!r} is missing from GROUPS in suites/{self.name}.py"
            )
        return self.only is None or bool(self.only.search(f"{self.name}.{group}"))

    # ----------------------------------------------------------------- checks
    def check(
        self,
        sid: str,
        title: str,
        ok,
        detail: str = "",
        known: Optional[KnownIssue] = None,
        since: Optional[int] = None,
    ) -> bool:
        """Record one scenario result (see ``harness.results``).

        A check tagged with ``known`` is version-gated: when the staged copy of
        ``known.file`` has the version ``known.fixed_in`` (or newer) the marker
        no longer applies, so a failure is a FAIL ("regression of known ...")
        and a pass is listed as an obsolete marker. Otherwise a failure is KNOWN
        only when it shows that bug: one of its ``evidence`` patterns matches
        ``detail``, or (with ``since``, the log mark taken when the scenario
        started) an error block logged since then matches its ``log_patterns``;
        any other failure is a FAIL. Once the bug reproduced, its log
        signatures are ignored by the suite's ``server-log`` check.
        """
        marker = ""
        if known:
            fixed = known.fixed_version()
            if fixed:
                marker = FIXED
                if not ok:
                    detail = (
                        f"regression of known {known.key} (fixed in {known.file} "
                        f"{known.fixed_in}, tested {fixed}): {detail}"
                    )
            elif not ok:
                in_log = since is not None and any(
                    known.log_matches(block) for block in self.log.error_blocks(since)
                )
                if known.detail_matches(detail) or in_log:
                    self.ignored_log_patterns.extend(known.log_patterns)
                else:
                    detail = (
                        f"tagged known {known.key}, but the failure does not show "
                        f"it (evidence {list(known.evidence)}): {detail}"
                    )
                    known = None
        return self.results.add(
            self.name, f"{self.name}.{sid}", title, ok, detail, known, marker
        )

    def mark(self) -> int:
        """Current end of the server log (pass to ``check(since=)``)."""
        return self.log.mark()

    def expect_errors(self, since: int, *signatures) -> None:
        """Error blocks logged since ``since`` that match one of ``signatures``
        were provoked on purpose (a signature is a string, or a tuple of strings
        that must all occur in the block). Other blocks in the window still
        fail the ``server-log`` check."""
        if not signatures:
            raise ValueError(
                "expect_errors(mark, *signatures) needs the signature of the "
                "provoked error, e.g. ('function_azure:pipe', 'request: 400')"
            )
        self.ignored_log_ranges.append((since, self.log.mark(), tuple(signatures)))

    def expect_log(self, *signatures) -> None:
        """Error blocks matching any of ``signatures`` are expected (a string,
        or a tuple of strings that must all occur in the block)."""
        self.ignored_log_patterns.extend(signatures)

    def fail_on_warnings(self, *signatures) -> None:
        """WARNING blocks matching any of ``signatures`` (logged while the
        suite ran) fail the ``server-log`` check, e.g.
        ``('function_time_token_tracker', 'No inlet data found')``."""
        self.warning_signatures.extend(signatures)

    def secret_leaks(self, *values: str) -> list:
        """Plaintext secrets in the server log since the suite started (any
        level): ``values`` plus every password valve written through
        ``owui.update_valves``."""
        secrets = dict(self.owui.secrets)
        for value in values:
            if value:
                secrets.setdefault(value, "secret")
        return self.log.secrets(self.log_start, secrets)

    def assert_no_secrets(
        self, *values: str, sid: str = "log.no-secrets", title: str = ""
    ) -> bool:
        """Check: none of ``values`` (and no password valve value written
        through ``owui.update_valves``) appears anywhere in the server log
        since the suite started. The detail never contains the secret."""
        leaks = self.secret_leaks(*values)
        return self.check(
            sid,
            title or "no plaintext secrets in the server log (any level)",
            not leaks,
            " || ".join(leaks[:5]) if leaks else "no secret logged",
        )

    def scan_log(self) -> bool:
        """Check: no unexpected ERROR / Traceback / ResourceWarning block, no
        registered WARNING and no plaintext secret in the server log for this
        suite."""
        if not self.log.available:
            return self.check("server-log", "server log scan", True, "no server log")
        total = len(self.log.error_blocks(self.log_start))
        errors = self.log.errors(
            self.log_start,
            tuple(self.ignored_log_patterns),
            tuple(self.ignored_log_ranges),
        )
        warnings = (
            self.log.warnings(self.log_start, *self.warning_signatures)
            if self.warning_signatures
            else []
        )
        leaks = self.secret_leaks()
        problems = leaks + warnings + errors
        return self.check(
            "server-log",
            "no unexpected ERROR / Traceback lines in the server log",
            not problems,
            f"{len(errors)} unexpected, {total - len(errors)} expected or known"
            + (f", {len(warnings)} failing warnings" if warnings else "")
            + (f", {len(leaks)} secrets logged" if leaks else "")
            + ": "
            + " || ".join(problems[:5]),
        )

    # -------------------------------------------------------------- resources
    def mock(self, name: str) -> Mock:
        if name not in self._mocks:
            self._mocks[name] = Mock(name)
        return self._mocks[name]

    def browser(self) -> BrowserSession:
        return BrowserSession(self.owui)

    async def valve_names(self, fid: str, user: bool = False) -> list:
        """Sorted valve names of a function (``user=True``: UserValves), for
        snapshots: valve names are the public API and must never disappear."""
        return await self.owui.valve_names(fid, user)

    @staticmethod
    def source(repo_path: str) -> str:
        """Content of a staged function file (``probe`` = the probe pipe)."""
        if repo_path == "probe":
            path = PROBE_FILE
        else:
            path = os.path.join(FUNCTIONS_DIR, repo_path)
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    async def _install_once(self, fid: str, name: str, content: str) -> tuple:
        """One create/update attempt: (status, data, setup problem or '', log
        mark taken before the attempt)."""
        mark = self.log.mark()
        try:
            status, data = await self.owui.install_function(fid, name, content)
        except httpx.TransportError as exc:  # Open WebUI unreachable / timeout
            return 0, {"raw": repr(exc)}, f"request failed: {exc!r}", mark
        if status == 200:
            return status, data, "", mark
        await self.log.settle(0.5)
        text = short(data, 2000) + "\n" + "\n".join(self.log.error_blocks(mark))
        if any(part in text for part in PIP_FAILURE):
            self.expect_errors(mark, *PIP_FAILURE_LOG)
            return status, data, "pip could not install the requirements", mark
        return status, data, "", mark

    async def install(
        self, fid: str, repo_path: str, name: str, sid: str = "load"
    ) -> bool:
        """Create (or update) + activate a function and check that it loaded.

        A failure caused by pip or the network is retried once; when it
        persists the run stops with a setup error (exit 2), not a FAIL.
        """
        content = self.source(repo_path)
        status, data, problem, mark = await self._install_once(fid, name, content)
        if problem:
            print(
                f"          setup: installing {repo_path} failed ({problem}), "
                f"retrying in {INSTALL_RETRY_DELAY}s",
                flush=True,
            )
            await asyncio.sleep(INSTALL_RETRY_DELAY)
            status, data, problem, mark = await self._install_once(fid, name, content)
            if problem:
                raise SetupError(
                    f"installing {repo_path} as {fid!r} failed twice ({problem}): "
                    f"HTTP {status} {short(data, 300)}"
                )
        active = await self.owui.set_active(fid, True) if status == 200 else None
        data = data if isinstance(data, dict) else {"raw": data}
        await self.log.settle(0.5)
        errors = self.log.errors(mark)
        ok = status == 200 and data.get("id") == fid and active is True and not errors
        version = ((data.get("meta") or {}).get("manifest") or {}).get("version")
        shown = "tests/e2e/probe/probe_pipe.py" if repo_path == "probe" else repo_path
        return self.check(
            sid,
            f"{shown} loads (create, import, activate)",
            ok,
            f"HTTP {status} type={data.get('type')} version={version} active={active} "
            f"log_errors={errors[:3]} body={short(data, 200) if status != 200 else ''}",
        )

    async def close(self) -> None:
        for mock in self._mocks.values():
            await mock.close()
