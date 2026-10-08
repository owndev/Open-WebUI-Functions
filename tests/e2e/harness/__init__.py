"""In-container driver library for the Open WebUI functions E2E tests.

Runs inside the Open WebUI container (python3 with httpx, aiohttp and
python-socketio from the image); see docs/testing.md.
"""

from .browser import BrowserChat, BrowserSession
from .known import KnownIssue
from .logs import ServerLog
from .mocks import Mock
from .owui import OWUI, ChatResult, parse_sse
from .results import FAIL, KNOWN, PASS, Results, SetupError, short
from .suite import Suite

__all__ = [
    "BrowserChat",
    "BrowserSession",
    "ChatResult",
    "FAIL",
    "KNOWN",
    "KnownIssue",
    "Mock",
    "OWUI",
    "PASS",
    "Results",
    "ServerLog",
    "SetupError",
    "Suite",
    "parse_sse",
    "short",
]
