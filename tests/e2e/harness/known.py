"""
Registry of known bugs that make scenarios fail on ``main`` today.

The entries live in one module per area, ``known_<area>.py`` (today only
``known_n8n.py`` for n8n + Infomaniak, with no entry at the moment), and are
re-exported here, so suites write ``known.<NAME>``. A new area module needs its
own import at the bottom of this file.

A failing scenario that carries a ``KnownIssue`` is reported as KNOWN and does
not fail the run, but only when the failure looks like that bug:

- ``evidence``: regular expressions searched in the check's detail text. A
  check tagged with the bug that fails in a different way (HTTP 500, model
  missing, ...) is reported as FAIL. Empty = every failure counts as the bug.
- ``log_patterns``: server-log signatures of the bug. A signature is a string or
  a tuple of strings that must ALL occur in one ERROR / Traceback block; name the
  function (``function_<id>:``, ``outlet filter <id>``) together with the error
  so that nothing else matches. When the check passes ``since=mark``, a matching
  block logged since ``mark`` also counts as evidence. Once the bug has
  reproduced, the suite's ``server-log`` check ignores ERROR and failing
  WARNING blocks that match these signatures (background tasks and later
  requests hit the same bug outside the check); every other error still fails
  it.

Version gating (``file`` + ``fixed_in``): a marker only applies while the staged
copy of ``file`` (``FUNCTIONS_DIR/<file>``, the file under test) has a docstring
``version:`` lower than ``fixed_in``. From that version on the bug counts as
fixed: a failing tagged check is a FAIL ("regression of known <key>") and a
passing one is listed as an obsolete marker ("drop marker"). Fix branches and
mutants of fixed files are therefore protected, ``main`` stays green, and the
markers switch off by themselves once the fix is merged. ``fixed_in=""`` (no fix
yet) never gates.

Strict mode (``--strict-known`` / ``E2E_STRICT_KNOWN=1``): a tagged check that
passes while its marker still applies (the bug was fixed without a version bump,
or the evidence is wrong) is a FAIL.

``fixed_by`` names the pull request (or branch) that carries the fix; the issue
in ``ref`` (when there is one) is the stable pointer.
"""

import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Optional, Union

from .config import FUNCTIONS_DIR

Signature = Union[str, tuple]

_VERSION_LINE = re.compile(r"^version:\s*([^\s#]+)", re.MULTILINE)


def signature_matches(text: str, signature: Signature) -> bool:
    """True when ``text`` contains the signature (every part of a tuple)."""
    if isinstance(signature, str):
        return signature in text
    return all(part in text for part in signature)


def version_tuple(version: str) -> tuple:
    """``"2.8.0"`` -> ``(2, 8, 0)``; non-numeric parts are ignored."""
    return tuple(int(part) for part in re.findall(r"\d+", version or ""))


@lru_cache(maxsize=None)
def staged_version(repo_path: str) -> str:
    """``version:`` of the staged function file's docstring header ('' when the
    file is not staged or has no version line)."""
    try:
        with open(os.path.join(FUNCTIONS_DIR, repo_path), encoding="utf-8") as fh:
            head = fh.read(16384)
    except OSError:
        return ""
    parts = head.split('"""', 2)
    match = _VERSION_LINE.search(parts[1] if len(parts) == 3 else head)
    return match.group(1) if match else ""


@dataclass(frozen=True)
class KnownIssue:
    key: str
    summary: str
    fixed_by: str
    ref: str = ""
    evidence: tuple = field(default=())
    log_patterns: tuple = field(default=())
    file: str = ""  # repo path of the function file with the bug
    fixed_in: str = ""  # docstring version of ``file`` that fixes it ("" = none)

    def label(self) -> str:
        ref = f" ({self.ref})" if self.ref else ""
        fix = (
            f"fix pending in {self.fixed_by} (not merged yet)"
            if self.fixed_by
            else "no fix yet"
        )
        if self.fixed_in:
            fix += f", fixed in {self.file} {self.fixed_in}"
        return f"known {self.key}{ref}: {self.summary}; {fix}"

    def fixed_version(self) -> Optional[str]:
        """Staged version of ``file`` when it is >= ``fixed_in`` (the marker no
        longer applies), else None."""
        if not (self.file and self.fixed_in):
            return None
        staged = staged_version(self.file)
        if staged and version_tuple(staged) >= version_tuple(self.fixed_in):
            return staged
        return None

    def detail_matches(self, detail: str) -> bool:
        """The failure detail shows this bug (always True without evidence)."""
        if not self.evidence:
            return True
        return any(re.search(rx, detail, re.MULTILINE) for rx in self.evidence)

    def log_matches(self, block: str) -> bool:
        """An error block of the server log is this bug."""
        return any(signature_matches(block, sig) for sig in self.log_patterns)

    def as_dict(self) -> dict:
        return asdict(self)


FOUND_BY_E2E = "found by tests/e2e, no issue filed"

# The entries, one module per area. They import the names above, so these
# imports stay at the bottom; every KnownIssue they define is re-exported.
from .known_n8n import *  # noqa: E402, F403


def registry() -> list:
    """Every registered ``KnownIssue`` (all area modules)."""
    return [value for value in globals().values() if isinstance(value, KnownIssue)]
