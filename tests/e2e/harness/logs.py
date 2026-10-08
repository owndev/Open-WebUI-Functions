"""
Access to the Open WebUI server log from inside the container.

run.sh starts the server with its output mirrored to ``SERVER_LOG`` (``docker
logs`` keeps working), so scenarios can look at what the server logged while
they ran: ``mark()`` remembers the current end of the log, ``since(mark)``
returns what was appended afterwards, ``errors(mark, ...)`` finds unexpected
ERROR / CRITICAL / Traceback blocks.
"""

import asyncio
import os
import re

from .config import SERVER_LOG

# Loguru line: "2026-10-08 11:44:20.895 | ERROR    | module:function:line - msg"
_LOGURU = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \| (\w+)\s*\|")
_ERROR_START = re.compile(
    r"\| (ERROR|CRITICAL)\s*\||^Traceback \(most recent call last\)|"
    r"^ERROR:|Exception in ASGI application"
)

# Error blocks Open WebUI itself produces, not caused by the functions under test.
# Each entry is a tuple of substrings that must ALL occur in the block.
GLOBAL_NOISE = (
    # Open WebUI 0.11.4: run_initial_title_generation passes a ctx without
    # "model" to background_tasks_handler -> review_memory_after_turn raises
    # after the title was already saved.
    ("Error generating initial chat title", "KeyError: 'model'"),
)


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

    def _blocks(self, mark: int) -> list:
        """Error blocks since ``mark`` as (file offset, lines).

        A block starts at an ERROR/CRITICAL loguru line (or a traceback outside
        any block) and runs until the next loguru line.
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
            if _ERROR_START.search(line) and (is_loguru or current is None):
                current = (offset, [line])
                blocks.append(current)
            elif is_loguru:
                current = None
            elif current is not None:
                current[1].append(line)
            offset += len(raw) + 1
        return blocks

    def errors(self, mark: int, ignore: tuple = (), ranges: tuple = ()) -> list:
        """Unexpected error blocks since ``mark`` (first line ... last line).

        Dropped: blocks containing any ``ignore`` substring, blocks starting
        inside one of the ``ranges`` (file offsets of scenarios whose errors are
        expected or attributed to a known bug), and ``GLOBAL_NOISE``.
        """
        out = []
        for offset, block in self._blocks(mark):
            text = "\n".join(block)
            if any(p in text for p in ignore):
                continue
            if any(start <= offset < end for start, end in ranges):
                continue
            if any(all(p in text for p in noise) for noise in GLOBAL_NOISE):
                continue
            tail = next((ln for ln in reversed(block) if ln.strip()), "")
            summary = block[0].strip()[:220]
            if tail is not block[0]:
                summary += " ... " + tail.strip()[:160]
            out.append(summary)
        return out
