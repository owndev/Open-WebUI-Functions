"""
Scenario results: PASS / FAIL / KNOWN bookkeeping, console lines and JSON.

- PASS   the check held
- FAIL   the check did not hold and no known bug explains it -> run fails
- KNOWN  the check did not hold and the failure shows a registered known bug
         (``harness.known``) whose marker still applies to the file under test;
         reported with the bug, its issue and the branch with the pending fix,
         does not fail the run

A check tagged with a known bug carries a marker state:

- ``active``  the staged function file is older than the bug's ``fixed_in``
              (or the bug has no fix): a failure that shows the bug is KNOWN
- ``fixed``   the staged file has the fixing version: a failure is a FAIL
              (regression of a fixed bug), a pass is an obsolete marker that is
              listed as "drop marker" (not a failure)

A tagged check that passes while its marker is ``active`` prints a reminder; in
strict mode (``--strict-known``) it is a FAIL.
"""

import json
import time
from dataclasses import asdict, dataclass
from typing import Optional

from .known import KnownIssue

PASS, FAIL, KNOWN = "PASS", "FAIL", "KNOWN"
ACTIVE, FIXED = "active", "fixed"
DETAIL_LIMIT = 600


class SetupError(Exception):
    """The environment failed, not the code under test (pip or network while a
    function is created, ...): the driver stops with exit code 2."""


def short(value, limit: int = 160) -> str:
    """Compact repr for result details."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + f"...(+{len(text) - limit})"


@dataclass
class Result:
    suite: str
    id: str
    title: str
    status: str
    detail: str
    known: Optional[dict]
    known_fixed: bool  # tagged with a known bug, but the check held
    marker: str  # "" (untagged), "active" or "fixed" (see module docstring)
    at: float

    @property
    def obsolete_marker(self) -> bool:
        """Tagged check that held on a file with the fixing version."""
        return self.status == PASS and self.marker == FIXED


class Results:
    def __init__(self, verbose: bool = False, strict_known: bool = False):
        self.items: list[Result] = []
        self.verbose = verbose
        self.strict_known = strict_known

    def add(
        self,
        suite: str,
        sid: str,
        title: str,
        ok: bool,
        detail: str = "",
        known: Optional[KnownIssue] = None,
        marker: str = "",
    ) -> bool:
        """Record one result; returns whether the check held (``ok``).

        ``known`` with marker ``active`` (the default): a failure is KNOWN (the
        caller checked the evidence), a pass is a reminder or, in strict mode,
        a FAIL. Marker ``fixed``: ``known`` is kept for the report only, the
        result is PASS (obsolete marker) or FAIL (regression).
        """
        ok = bool(ok)
        marker = (marker or ACTIVE) if known else ""
        if ok and marker == ACTIVE and self.strict_known:
            status = FAIL
            detail = (
                f"strict: known {known.key} no longer reproduces while its marker "
                f"is active ({known.file or 'no file'} older than fixed_in "
                f"{known.fixed_in or '(none)'}): fixed without a version bump, or "
                f"wrong evidence. {detail}"
            )
        elif ok:
            status = PASS
        elif marker == ACTIVE:
            status = KNOWN
        else:
            status = FAIL
        detail = (
            detail if len(detail) <= DETAIL_LIMIT else detail[:DETAIL_LIMIT] + "..."
        )
        result = Result(
            suite=suite,
            id=sid,
            title=title,
            status=status,
            detail=detail,
            known=known.as_dict() if known else None,
            known_fixed=bool(ok and known),
            marker=marker,
            at=time.time(),
        )
        self.items.append(result)
        self._print(result, known)
        return ok

    def _print(self, result: Result, known: Optional[KnownIssue]) -> None:
        line = f"[{result.status:<5}] {result.id}  {result.title}"
        if result.status == KNOWN:
            line += f"\n          -> {known.label()}"
        if result.obsolete_marker:
            line += (
                f"\n          -> known {known.key} is fixed in {known.file} "
                f"{known.fixed_in}: drop marker (known= argument and its entry "
                "in tests/e2e/harness/known_*.py)"
            )
        elif result.status == PASS and result.known_fixed:
            line += (
                f"\n          -> known {known.key} no longer reproduces, but "
                f"{known.file or 'the file'} is older than "
                f"{known.fixed_in or 'any fix'}: bump the version or drop the "
                "known= marker (a FAIL with --strict-known)"
            )
        if result.detail and (result.status != PASS or self.verbose):
            line += f"\n          {result.detail}"
        print(line, flush=True)

    def counts(self) -> dict:
        out = {PASS: 0, FAIL: 0, KNOWN: 0}
        for item in self.items:
            out[item.status] += 1
        return out

    def failed(self) -> bool:
        return self.counts()[FAIL] > 0

    def obsolete_markers(self) -> list:
        """Results whose known marker no longer applies (fixed version staged)."""
        return [item for item in self.items if item.obsolete_marker]

    def to_json(self, meta: dict) -> dict:
        return {
            "meta": meta,
            "summary": self.counts(),
            "obsolete_markers": [
                {
                    "id": item.id,
                    "key": item.known["key"],
                    "file": item.known["file"],
                    "fixed_in": item.known["fixed_in"],
                }
                for item in self.obsolete_markers()
            ],
            "results": [asdict(item) for item in self.items],
        }

    def to_markdown(self, meta: dict) -> str:
        counts = self.counts()
        lines = [
            "## Open WebUI functions E2E",
            "",
            f"Image `{meta.get('image')}` (Open WebUI {meta.get('owui_version')}), "
            f"suites: {', '.join(meta.get('suites', []))}, "
            f"{meta.get('duration_s')} s"
            + (", strict known mode" if meta.get("strict_known") else ""),
            "",
            f"**{counts[PASS]} PASS, {counts[FAIL]} FAIL, {counts[KNOWN]} KNOWN**",
            "",
        ]
        flagged = [
            i
            for i in self.items
            if i.status != PASS or (i.known_fixed and not i.obsolete_marker)
        ]
        if flagged:
            lines += ["| Status | Scenario | Detail |", "|---|---|---|"]
            for item in flagged:
                note = item.detail
                if item.known and item.status != FAIL:
                    fixed_by = item.known["fixed_by"]
                    ref = f" ({item.known['ref']})" if item.known.get("ref") else ""
                    note = f"known {item.known['key']}{ref}, " + (
                        f"fix pending in `{fixed_by}`" if fixed_by else "no fix yet"
                    )
                    if item.known_fixed:
                        note += " (no longer reproduces: bump the version or drop it)"
                cell = note.replace("|", "\\|")[:300]
                status = "PASS*" if item.status == PASS else item.status
                lines.append(f"| {status} | `{item.id}` {item.title} | {cell} |")
            lines.append("")
        obsolete = self.obsolete_markers()
        if obsolete:
            lines += [
                f"### Obsolete known markers: drop marker ({len(obsolete)})",
                "",
                "The tested file has the fixing version, so these markers no longer "
                "apply (not a failure). Drop the `known=` argument, and the entry in "
                "`tests/e2e/harness/known_*.py` once nothing references it.",
                "",
                "| Scenario | Known bug | Fixed in |",
                "|---|---|---|",
            ]
            for item in obsolete:
                known = item.known
                lines.append(
                    f"| `{item.id}` | {known['key']} | "
                    f"`{known['file']}` {known['fixed_in']} |"
                )
        return "\n".join(lines) + "\n"
