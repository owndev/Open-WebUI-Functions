"""
Access to the Open WebUI server log from inside the container.

run.sh starts the server with its output mirrored to ``SERVER_LOG`` (``docker
logs`` keeps working), so scenarios can look at what the server logged while
they ran: ``mark()`` remembers the current end of the log, ``since(mark)``
returns what was appended afterwards, ``errors(mark, ...)`` finds unexpected
ERROR / CRITICAL / Traceback / ResourceWarning blocks, ``warnings(mark, ...)``
finds WARNING blocks and ``secrets(mark, ...)`` plaintext secrets at any level.

run.sh starts the container with ``PYTHONWARNINGS=always::ResourceWarning``, so
unclosed sockets / files show up as ``ResourceWarning:`` lines (counted as
errors) instead of being dropped silently.
"""

import asyncio
import os
import re

from .config import SERVER_LOG
from .known import signature_matches

# Loguru line: "2026-10-08 11:44:20.895 | ERROR    | module:function:line - msg"
_LOGURU = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \| (\w+)\s*\|")
_ERROR_START = re.compile(
    r"\| (ERROR|CRITICAL)\s*\||^Traceback \(most recent call last\)|"
    r"^ERROR:|Exception in ASGI application|ResourceWarning: "
)
# Loguru WARNING lines, "WARNING: ..." lines and Python warnings
# ("file.py:12: UserWarning: ...").
_WARNING_START = re.compile(r"\| WARNING\s*\||^WARNING:|\b\w*Warning: ")

# Error blocks Open WebUI itself produces, not caused by the functions under test.
# Each entry is a tuple of substrings that must ALL occur in the block.
GLOBAL_NOISE = (
    # Open WebUI 0.11.4: run_initial_title_generation passes a ctx without
    # "model" to background_tasks_handler -> review_memory_after_turn raises
    # after the title was already saved.
    ("Error generating initial chat title", "KeyError: 'model'"),
)


def _summary(block: list) -> str:
    """First line ... last non-empty line of a block."""
    tail = next((ln for ln in reversed(block) if ln.strip()), "")
    summary = block[0].strip()[:220]
    if tail is not block[0]:
        summary += " ... " + tail.strip()[:160]
    return summary


def redact(text: str, secret: str) -> str:
    return text.replace(secret, "***")


class ServerLog:
    def __init__(self, path: str = SERVER_LOG):
        self.path = path

    @property
    def available(self) -> bool:
        return os.path.exists(self.path)

    def mark(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def since(self, mark: int) -> str:
        try:
            with open(self.path, "rb") as fh:
                fh.seek(mark)
                return fh.read().decode("utf-8", "replace")
        except OSError:
            return ""

    async def settle(self, seconds: float = 1.0) -> None:
        """Give the server a moment to flush log lines of a finished request."""
        await asyncio.sleep(seconds)

    def has_errors(self, mark: int) -> bool:
        """Any ERROR / Traceback block since ``mark`` (no filtering)."""
        return bool(self._blocks(mark))

    def lines(self, mark: int, *needles: str) -> list:
        """Lines appended since ``mark`` that contain any of ``needles``."""
        return [
            line.strip()[:400]
            for line in self.since(mark).splitlines()
            if any(n in line for n in needles)
        ]

    def _blocks(self, mark: int, start: re.Pattern = _ERROR_START) -> list:
        """Blocks since ``mark`` as (file offset, lines).

        A block starts at a loguru line matching ``start`` (or a non-loguru line
        matching it outside any block, e.g. a traceback) and runs until the next
        loguru line.
        """
        try:
            with open(self.path, "rb") as fh:
                fh.seek(mark)
                data = fh.read()
        except OSError:
            return []
        blocks, current, offset = [], None, mark
        for raw in data.split(b"\n"):
            line = raw.decode("utf-8", "replace").rstrip("\r")
            is_loguru = bool(_LOGURU.match(line))
            if start.search(line) and (is_loguru or current is None):
                current = (offset, [line])
                blocks.append(current)
            elif is_loguru:
                current = None
            elif current is not None:
                current[1].append(line)
            offset += len(raw) + 1
        return blocks

    def error_blocks(self, mark: int) -> list:
        """Full text of every ERROR / Traceback block since ``mark``."""
        return ["\n".join(block) for _, block in self._blocks(mark)]

    def errors(self, mark: int, ignore: tuple = (), ranges: tuple = ()) -> list:
        """Unexpected error blocks since ``mark`` (first line ... last line).

        Dropped: blocks matching an ``ignore`` signature (a string, or a tuple
        of strings that must all occur in the block; see
        ``harness.known.signature_matches``), blocks that start inside one of
        the ``ranges`` ``(start, end, signatures)`` AND match one of its
        signatures (errors a scenario provoked on purpose), and
        ``GLOBAL_NOISE``.
        """
        out = []
        for offset, block in self._blocks(mark):
            text = "\n".join(block)
            if any(signature_matches(text, sig) for sig in ignore):
                continue
            if any(
                start <= offset < end
                and any(signature_matches(text, sig) for sig in signatures)
                for start, end, signatures in ranges
            ):
                continue
            if any(signature_matches(text, noise) for noise in GLOBAL_NOISE):
                continue
            out.append(_summary(block))
        return out

    def warnings(self, mark: int, *patterns) -> list:
        """WARNING blocks since ``mark`` (first line ... last line).

        Loguru ``| WARNING |`` lines, ``WARNING:`` lines and Python warnings
        (``file.py:12: UserWarning: ...``). With ``patterns`` (signatures: a
        string, or a tuple of strings that must all occur in the block) only
        the matching blocks are returned.
        """
        out = []
        for _, block in self._blocks(mark, _WARNING_START):
            text = "\n".join(block)
            if patterns and not any(signature_matches(text, p) for p in patterns):
                continue
            out.append(_summary(block))
        return out

    def secrets(self, mark: int, secrets: dict) -> list:
        """Plaintext secrets in the log since ``mark``, at any level.

        ``secrets`` maps each secret value to a label (e.g. the valve name).
        Returns one ``"<label> logged Nx: <first line, redacted>"`` per secret
        found; the secret value itself is never part of the result.
        """
        text = self.since(mark)
        out = []
        for value, label in secrets.items():
            count = text.count(value) if value else 0
            if not count:
                continue
            line = next(ln for ln in text.splitlines() if value in ln)
            for other in secrets:
                line = redact(line, other) if other else line
            out.append(f"{label} logged {count}x: {line.strip()[:200]}")
        return out
