"""
Scenario results: PASS / FAIL / KNOWN bookkeeping, console lines and JSON.

- PASS   the check held
- FAIL   the check did not hold and no known bug explains it -> run fails
- KNOWN  the check did not hold because of a registered known bug
         (``harness.known``); reported with the bug and its fixing branch, does
         not fail the run
"""

import json
import time
from dataclasses import asdict, dataclass
from typing import Optional

from .known import KnownIssue

PASS, FAIL, KNOWN = "PASS", "FAIL", "KNOWN"
DETAIL_LIMIT = 600


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
    known_fixed: bool
    at: float


class Results:
    def __init__(self, verbose: bool = False):
        self.items: list[Result] = []
        self.verbose = verbose

    def add(
        self,
        suite: str,
        sid: str,
        title: str,
        ok: bool,
        detail: str = "",
        known: Optional[KnownIssue] = None,
    ) -> bool:
        ok = bool(ok)
        status = PASS if ok else (KNOWN if known else FAIL)
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
            at=time.time(),
        )
        self.items.append(result)
        self._print(result, known)
        return ok

    def _print(self, result: Result, known: Optional[KnownIssue]) -> None:
        line = f"[{result.status:<5}] {result.id}  {result.title}"
        if result.status == KNOWN:
            line += f"\n          -> {known.label()}"
        if result.known_fixed:
            line += (
                f"\n          -> known {known.key} no longer reproduces: drop the "
                "known= marker (tests/e2e/harness/known.py)"
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

    def to_json(self, meta: dict) -> dict:
        return {
            "meta": meta,
            "summary": self.counts(),
            "results": [asdict(item) for item in self.items],
        }

    def to_markdown(self, meta: dict) -> str:
        counts = self.counts()
        lines = [
            "## Open WebUI functions E2E",
            "",
            f"Image `{meta.get('image')}` (Open WebUI {meta.get('owui_version')}), "
            f"suites: {', '.join(meta.get('suites', []))}, "
            f"{meta.get('duration_s')} s",
            "",
            f"**{counts[PASS]} PASS, {counts[FAIL]} FAIL, {counts[KNOWN]} KNOWN**",
            "",
        ]
        flagged = [i for i in self.items if i.status != PASS or i.known_fixed]
        if flagged:
            lines += ["| Status | Scenario | Detail |", "|---|---|---|"]
            for item in flagged:
                note = item.detail
                if item.known:
                    fixed_by = item.known["fixed_by"]
                    note = f"known {item.known['key']}, " + (
                        f"fixed by `{fixed_by}`" if fixed_by else "no fix yet"
                    )
                    if item.known_fixed:
                        note += " (no longer reproduces)"
                cell = note.replace("|", "\\|")[:300]
                status = "PASS*" if item.known_fixed else item.status
                lines.append(f"| {status} | `{item.id}` {item.title} | {cell} |")
        return "\n".join(lines) + "\n"
