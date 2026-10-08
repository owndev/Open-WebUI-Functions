"""The context object a suite module's ``run(t)`` works with."""

import os
import re
from typing import Optional

from .browser import BrowserSession
from .config import FUNCTIONS_DIR, PROBE_FILE
from .known import KnownIssue
from .logs import ServerLog
from .mocks import Mock
from .owui import OWUI
from .results import Results, short


class Suite:
    """Scenario helpers bound to one suite (ids are prefixed with its name).

    Server-log bookkeeping: the final ``scan_log()`` check reports ERROR /
    Traceback blocks logged while the suite ran, except blocks that were
    provoked on purpose (``expect_errors`` ranges, ``expect_log`` signatures) and
    blocks matching the ``log_patterns`` signature of a known bug that
    reproduced in this suite (see ``harness.known``).
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

        A failing check tagged with ``known`` is KNOWN only when the failure
        shows that bug: one of its ``evidence`` patterns matches ``detail``, or
        (with ``since``, the log mark taken when the scenario started) an error
        block logged since then matches its ``log_patterns``. Otherwise it is a
        FAIL. Once the bug reproduced, its log signatures are ignored by the
        suite's ``server-log`` check.
        """
        if not ok and known:
            in_log = since is not None and any(
                known.log_matches(block) for block in self.log.error_blocks(since)
            )
            if known.detail_matches(detail) or in_log:
                self.ignored_log_patterns.extend(known.log_patterns)
            else:
                detail = (
                    f"tagged known {known.key}, but the failure does not show it "
                    f"(evidence {list(known.evidence)}): {detail}"
                )
                known = None
        return self.results.add(
            self.name, f"{self.name}.{sid}", title, ok, detail, known
        )

    def mark(self) -> int:
        """Current end of the server log (pass to ``check(since=)``)."""
        return self.log.mark()

    def expect_errors(self, since: int) -> None:
        """Errors logged since ``since`` were provoked on purpose."""
        self.ignored_log_ranges.append((since, self.log.mark()))

    def expect_log(self, *signatures) -> None:
        """Error blocks matching any of ``signatures`` are expected (a string,
        or a tuple of strings that must all occur in the block)."""
        self.ignored_log_patterns.extend(signatures)

    def scan_log(self) -> bool:
        """Check: no unexpected ERROR/Traceback in the server log for this suite."""
        if not self.log.available:
            return self.check("server-log", "server log scan", True, "no server log")
        total = len(self.log.error_blocks(self.log_start))
        errors = self.log.errors(
            self.log_start,
            tuple(self.ignored_log_patterns),
            tuple(self.ignored_log_ranges),
        )
        return self.check(
            "server-log",
            "no unexpected ERROR / Traceback lines in the server log",
            not errors,
            f"{len(errors)} unexpected, {total - len(errors)} expected or known: "
            + " || ".join(errors[:5]),
        )

    # -------------------------------------------------------------- resources
    def mock(self, name: str) -> Mock:
        if name not in self._mocks:
            self._mocks[name] = Mock(name)
        return self._mocks[name]

    def browser(self) -> BrowserSession:
        return BrowserSession(self.owui)

    @staticmethod
    def source(repo_path: str) -> str:
        """Content of a staged function file (``probe`` = the probe pipe)."""
        if repo_path == "probe":
            path = PROBE_FILE
        else:
            path = os.path.join(FUNCTIONS_DIR, repo_path)
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    async def install(
        self, fid: str, repo_path: str, name: str, sid: str = "load"
    ) -> bool:
        """Create (or update) + activate a function and check that it loaded."""
        mark = self.log.mark()
        status, data = await self.owui.install_function(
            fid, name, self.source(repo_path)
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
