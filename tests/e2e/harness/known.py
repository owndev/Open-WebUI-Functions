"""
Registry of known bugs that make scenarios fail on ``main`` today.

A scenario that fails while it carries a ``KnownIssue`` is reported as KNOWN and
does not fail the run. When the fixing branch is merged the scenario starts to
PASS and the driver prints a reminder to drop the ``known=`` argument (and the
entry here, once nothing references it).

``log_patterns`` are server-log substrings the bug produces; the per-suite scan
for unexpected ERROR / Traceback lines ignores them.
"""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class KnownIssue:
    key: str
    summary: str
    fixed_by: str
    ref: str = ""
    log_patterns: tuple = field(default=())

    def label(self) -> str:
        ref = f" ({self.ref})" if self.ref else ""
        fix = f"fixed by branch {self.fixed_by}" if self.fixed_by else "no fix yet"
        return f"known {self.key}{ref}: {self.summary}; {fix}"

    def as_dict(self) -> dict:
        return asdict(self)


GEMINI_FIX = "hotfix/gemini-1.16.2"
AZURE_FIX = "hotfix/azure-2.7.1"
FILTERS_FIX = "hotfix/filters-owui-0.10-compat"
N8N_INFOMANIAK_FIX = "hotfix/n8n-infomaniak-streaming"

GEMINI_B1 = KnownIssue(
    "B1",
    "Gemini API stream without a websocket session ends with 'Error during "
    "streaming' (the genai client is closed while the stream is consumed)",
    GEMINI_FIX,
    log_patterns=("Error during streaming",),
)
GEMINI_B5 = KnownIssue(
    "B5",
    "Gemini's non-streaming path calls __event_emitter__ without a None check "
    "(usage event, image status), so requests without an emitter (background "
    "title/tags tasks) answer \"Error generating content: 'NoneType' object is "
    'not callable"',
    GEMINI_FIX,
    log_patterns=("'NoneType' object is not callable",),
)
GEMINI_172 = KnownIssue(
    "#172",
    "gemini-3.1-flash-image (non-preview) is not recognised as an image model, "
    "the generated image is dropped",
    GEMINI_FIX,
    ref="https://github.com/owndev/Open-WebUI-Functions/issues/172",
)
GEMINI_NONSTREAM_USAGE = KnownIssue(
    "gemini-nonstream-usage",
    "Gemini non-streaming answers are returned as a plain string and usage only "
    "goes out as a 'usage' event, which Open WebUI 0.11 neither saves nor shows "
    "(docs promise a usage dict); API clients get no usage either",
    GEMINI_FIX,
    ref="found by tests/e2e, no issue filed",
)
AZURE_DOUBLE_STRIP = KnownIssue(
    "azure-double-strip",
    "Azure strips the function prefix twice for non-streaming / data_sources "
    "requests, so dotted model names (gpt-4.1 -> '1') are mangled",
    AZURE_FIX,
)
AZURE_123 = KnownIssue(
    "#123",
    "Azure background tasks (title/tags/follow-ups) are sent with data_sources, "
    "so task answers are grounded and return citations (the 'too many sources' "
    "report)",
    AZURE_FIX,
    ref="https://github.com/owndev/Open-WebUI-Functions/issues/123",
)
AZURE_STREAM_OPTIONS = KnownIssue(
    "azure-stream-options",
    "Azure forwards a client-supplied stream_options together with data_sources, "
    "which Azure 'On Your Data' rejects (HTTP 400)",
    AZURE_FIX,
)
FILTER_TRACKER_NO_EMITTER = KnownIssue(
    "tracker-no-emitter",
    "time_token_tracker outlet calls __event_emitter__ unconditionally; on the "
    "API path Open WebUI passes None and the outlet raises",
    FILTERS_FIX,
    log_patterns=("'NoneType' object is not callable", "Error in outlet"),
)
FILTER_SEARCH_KEYERROR = KnownIssue(
    "search-tool-keyerror",
    "google_search_tool inlet does features.pop('web_search') without a default; "
    "requests without features.web_search fail with KeyError 'web_search'",
    FILTERS_FIX,
    log_patterns=("'web_search'", "KeyError"),
)
N8N_DICT_IN_STREAM = KnownIssue(
    "n8n-dict-in-stream",
    "n8n returns a chat.completion dict (usage) for stream=True; the streaming "
    "middleware ignores choices[].message, the saved answer is empty",
    N8N_INFOMANIAK_FIX,
)
N8N_SSE_CONTROL_LINES = KnownIssue(
    "n8n-sse-control-lines",
    "n8n parses SSE answers per network chunk: comment lines (': ...') and "
    "'data: [DONE]' end up in the answer text",
    N8N_INFOMANIAK_FIX,
    ref="found by tests/e2e, no issue filed",
)
INFOMANIAK_CHUNKING = KnownIssue(
    "infomaniak-chunking",
    "Infomaniak forwards raw network chunks; coalesced or split SSE events are "
    "dropped by the streaming middleware (empty answer / missing usage)",
    N8N_INFOMANIAK_FIX,
)
