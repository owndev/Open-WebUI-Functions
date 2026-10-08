"""
Registry of known bugs that make scenarios fail on ``main`` today.

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
  reproduced, the suite's ``server-log`` check ignores blocks that match these
  signatures (background tasks and later requests hit the same bug outside the
  check); every other error still fails it.

When the fix is merged the scenario starts to PASS and the driver prints a
reminder to drop the ``known=`` argument (and the entry here, once nothing
references it).

``fixed_by`` names the pull request (or branch) that carries the fix. Until it is
merged the scenario stays KNOWN; the issue in ``ref`` (when there is one) is the
stable pointer.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Union

Signature = Union[str, tuple]


def signature_matches(text: str, signature: Signature) -> bool:
    """True when ``text`` contains the signature (every part of a tuple)."""
    if isinstance(signature, str):
        return signature in text
    return all(part in text for part in signature)


@dataclass(frozen=True)
class KnownIssue:
    key: str
    summary: str
    fixed_by: str
    ref: str = ""
    evidence: tuple = field(default=())
    log_patterns: tuple = field(default=())

    def label(self) -> str:
        ref = f" ({self.ref})" if self.ref else ""
        fix = (
            f"fix pending in {self.fixed_by} (not merged yet)"
            if self.fixed_by
            else "no fix yet"
        )
        return f"known {self.key}{ref}: {self.summary}; {fix}"

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


GEMINI_FIX = "PR #185"
AZURE_FIX = "PR #183"
FILTERS_FIX = "PR #184"
N8N_INFOMANIAK_FIX = "PR #182"
NO_ISSUE = "no issue filed"
FOUND_BY_E2E = "found by tests/e2e, no issue filed"

GEMINI_B1 = KnownIssue(
    "B1",
    "Gemini API stream without a websocket session ends with 'Error during "
    "streaming' (the genai client is closed while the stream is consumed)",
    GEMINI_FIX,
    ref=NO_ISSUE,
    evidence=(r"Error during streaming",),
    log_patterns=(
        ("function_gemini:_handle_streaming_response", "Error during streaming"),
    ),
)
GEMINI_B5 = KnownIssue(
    "B5",
    "Gemini's non-streaming path calls __event_emitter__ without a None check "
    "(usage event, image status), so requests without an emitter (background "
    "title/tags tasks) answer \"Error generating content: 'NoneType' object is "
    'not callable"',
    GEMINI_FIX,
    ref=NO_ISSUE,
    evidence=(r"Error generating content: 'NoneType' object is not callable",),
    log_patterns=(("function_gemini:pipe", "'NoneType' object is not callable"),),
)
GEMINI_172 = KnownIssue(
    "#172",
    "gemini-3.1-flash-image (non-preview) is not recognised as an image model, "
    "the generated image is dropped",
    GEMINI_FIX,
    ref="https://github.com/owndev/Open-WebUI-Functions/issues/172",
    # listed without the image marker / streamed instead of forced non-stream
    evidence=(r"^name='[^'🎨]+'$", r"upstream=\['streamGenerateContent'\]"),
)
GEMINI_NONSTREAM_USAGE = KnownIssue(
    "gemini-nonstream-usage",
    "Gemini non-streaming answers are returned as a plain string and usage only "
    "goes out as a 'usage' event, which Open WebUI 0.11 neither saves nor shows "
    "(docs promise a usage dict); API clients get no usage either",
    GEMINI_FIX,
    ref=FOUND_BY_E2E,
    evidence=(r"usage=None",),
)
AZURE_DOUBLE_STRIP = KnownIssue(
    "azure-double-strip",
    "Azure strips the function prefix twice for non-streaming / data_sources "
    "requests, so dotted model names (gpt-4.1 -> '1') are mangled",
    AZURE_FIX,
    ref=NO_ISSUE,
    # what is left of gpt-4.1 / Phi-3.5-mini-instruct after the second strip
    evidence=(r"body\.model='(1|5-mini-instruct)'",),
)
AZURE_123 = KnownIssue(
    "#123",
    "Azure background tasks (title/tags/follow-ups) are sent with data_sources, "
    "so task answers are grounded and return citations (the 'too many sources' "
    "report)",
    AZURE_FIX,
    ref="https://github.com/owndev/Open-WebUI-Functions/issues/123",
    evidence=(r"with data_sources=[1-9]",),
)
AZURE_STREAM_OPTIONS = KnownIssue(
    "azure-stream-options",
    "Azure forwards a client-supplied stream_options together with data_sources, "
    "which Azure 'On Your Data' rejects (HTTP 400)",
    AZURE_FIX,
    ref=NO_ISSUE,
    evidence=(r"upstream stream_options=\{'include_usage': True\}",),
    log_patterns=(
        (
            "function_azure:pipe",
            "Error in Azure AI request: 400",
            "/openai/deployments/",
        ),
    ),
)
FILTER_TRACKER_NO_EMITTER = KnownIssue(
    "tracker-no-emitter",
    "time_token_tracker outlet calls __event_emitter__ unconditionally; on the "
    "API path Open WebUI passes None and the outlet raises",
    FILTERS_FIX,
    ref="https://github.com/owndev/Open-WebUI-Functions/issues/175",
    evidence=(r"outlet filter time_token_tracker.*'NoneType' object is not callable",),
    log_patterns=(
        (
            "Error in outlet filter time_token_tracker",
            "'NoneType' object is not callable",
        ),
    ),
)
FILTER_SEARCH_KEYERROR = KnownIssue(
    "search-tool-keyerror",
    "google_search_tool inlet does features.pop('web_search') without a default; "
    "requests without features.web_search fail with KeyError 'web_search'",
    FILTERS_FIX,
    ref=NO_ISSUE,
    evidence=(r"HTTP 400 .*'web_search'",),
    log_patterns=("Error processing chat payload: 'web_search'",),
)
N8N_DICT_IN_STREAM = KnownIssue(
    "n8n-dict-in-stream",
    "n8n returns a chat.completion dict (usage) for stream=True; the streaming "
    "middleware ignores choices[].message, the saved answer is empty",
    N8N_INFOMANIAK_FIX,
    ref=NO_ISSUE,
    evidence=(r"content= usage=\{",),  # empty answer, usage saved
)
N8N_SSE_CONTROL_LINES = KnownIssue(
    "n8n-sse-control-lines",
    "n8n parses SSE answers per network chunk: comment lines (': ...') and "
    "'data: [DONE]' end up in the answer text",
    N8N_INFOMANIAK_FIX,
    ref=FOUND_BY_E2E,
    evidence=(r"content=.*(: keep-alive|data: \[DONE\])",),
)
INFOMANIAK_CHUNKING = KnownIssue(
    "infomaniak-chunking",
    "Infomaniak forwards raw network chunks; coalesced or split SSE events are "
    "dropped by the streaming middleware (empty answer / missing usage)",
    N8N_INFOMANIAK_FIX,
    ref=NO_ISSUE,
    evidence=(r"content= usage=None", r"^usage=None$"),
)
