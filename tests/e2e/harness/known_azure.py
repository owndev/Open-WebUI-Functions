"""Known bugs of pipelines/azure/azure_ai_foundry.py (see ``harness.known``).

Every entry is fixed in 2.8.0 (PR #183). The evidence patterns need the
symptom together with a healthy rest of the answer (an answer that came back,
the mock's grounded or plain text), so a check that fails for another reason
(an upstream HTTP 500, a missing model) stays a FAIL.
"""

from .known import AZURE_FIX, NO_ISSUE, KnownIssue

AZURE_FILE = "pipelines/azure/azure_ai_foundry.py"
AZURE_FIXED_IN = "2.8.0"  # PR #183

# An unlinked "[docN]" (not part of "[[docN]](url)" or "[docN](url)") in the
# content of an answer: a reference split across stream deltas that was never
# converted.
_UNLINKED_REF = r"content=[^|]*(?<!\[)\[doc\d\](?![(\]])"

AZURE_DOUBLE_STRIP = KnownIssue(
    "azure-double-strip",
    "Azure strips the function prefix twice for non-streaming / data_sources "
    "requests, so dotted model names (gpt-4.1 -> '1') are mangled",
    AZURE_FIX,
    ref=NO_ISSUE,
    # what is left of gpt-4.1 / Phi-3.5-mini-instruct after the second strip,
    # in a request that was answered
    evidence=(
        r"body\.model='(1|5-mini-instruct)' HTTP 200 stream=\w+ "
        r"content=(Hello from mock Azure|The X100 charges)",
    ),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_123 = KnownIssue(
    "#123",
    "Azure background tasks (title/tags/follow-ups) are sent with data_sources, "
    "so task answers are grounded and return citations (the 'too many sources' "
    "report)",
    AZURE_FIX,
    ref="https://github.com/owndev/Open-WebUI-Functions/issues/123",
    evidence=(r"answered=True task requests=\d+ with data_sources=[1-9]",),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_STREAM_OPTIONS = KnownIssue(
    "azure-stream-options",
    "Azure forwards a client-supplied stream_options together with data_sources, "
    "which Azure 'On Your Data' rejects (HTTP 400)",
    AZURE_FIX,
    ref=NO_ISSUE,
    evidence=(
        r"content=Error: Validation error at #/stream_options: Extra inputs .*"
        r"upstream stream_options=\{'include_usage': True\}",
    ),
    log_patterns=(
        (
            "function_azure:pipe",
            "Error in Azure AI request: 400",
            "/openai/deployments/",
        ),
    ),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_NONSTREAM_STREAM_OPTIONS = KnownIssue(
    "azure-nonstream-stream-options",
    "Azure forwards a client-supplied stream_options in non-streaming requests "
    "(only valid with stream=true)",
    AZURE_FIX,
    ref=NO_ISSUE,
    evidence=(r"answered=True upstream stream_options=\{'include_usage': True\}",),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_TOOLS_DATA_SOURCES = KnownIssue(
    "azure-tools-data-sources",
    "Azure forwards tools/tool_choice together with data_sources; Azure then "
    "ignores the data sources, and Open WebUI 0.10+ adds its built-in tools to "
    "every browser chat, so web UI answers are not grounded",
    AZURE_FIX,
    ref=NO_ISSUE,
    # the mock's plain (ungrounded) answer, and the mock dropped data_sources
    # because tools came along
    evidence=(
        r"content=Hello from mock Azure \(gpt-4\.1\)\..*data_sources_ignored=True",
    ),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_HOLDBACK = KnownIssue(
    "azure-holdback",
    "Azure converts [docX] only when a reference arrives in one stream delta: "
    "references split across deltas stay unlinked, already linked ones split "
    "across deltas are broken",
    AZURE_FIX,
    ref=NO_ISSUE,
    evidence=(_UNLINKED_REF,),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_FLUSH_BEFORE_DONE = KnownIssue(
    "azure-flush-before-done",
    "Azure: a stream that ends on a reference (no finish_reason) keeps the "
    "reference unlinked instead of sending it as its own event before [DONE]",
    AZURE_FIX,
    ref=NO_ISSUE,
    evidence=(_UNLINKED_REF,),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_SHOW_ALL_VALVE = KnownIssue(
    "azure-show-all-valve",
    "Azure has no AZURE_AI_SHOW_ALL_CITATIONS_WITHOUT_REFERENCES valve: an "
    "answer without [docX] references always shows all retrieved documents",
    AZURE_FIX,
    ref=NO_ISSUE,
    evidence=(
        r"content=The requested information is not available in the retrieved "
        r"data\. .*sources=\['\[doc1\] - X100 Product Manual', "
        r"'\[doc2\] - Warranty FAQ', '\[doc3\] - Release Notes'\]",
    ),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_PAREN_URL = KnownIssue(
    "azure-paren-url",
    "Azure puts citation URLs with parentheses into markdown links unencoded, "
    "so the link ends at the first ')'",
    AZURE_FIX,
    ref=NO_ISSUE,
    evidence=(r"\(https://docs\.example\.com/x100/manual_\(v2\)\.pdf\)",),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_HISTORY_UNLINK = KnownIssue(
    "azure-history-unlink",
    "Azure sends the [[docX]](url) links of earlier answers back to the model, "
    "which then copies the link syntax",
    AZURE_FIX,
    ref=NO_ISSUE,
    # the request was answered and the history still carries a link
    evidence=(r"answered=True sent='[^']*\[\[doc\d\]\]\(http",),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_CITATION_INFO_LOG = KnownIssue(
    "azure-citation-info-log",
    "Azure logs citation content (document text) at INFO",
    AZURE_FIX,
    ref=NO_ISSUE,
    evidence=(r"citation text logged [1-9]\d*x",),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_LINETOOLONG = KnownIssue(
    "azure-linetoolong",
    "Azure stops a stream whose SSE event (On Your Data context) is longer "
    "than 128 KiB (aiohttp LineTooLong): API clients get an empty answer "
    "without an error",
    AZURE_FIX,
    ref=NO_ISSUE,
    evidence=(r"line_too_long_log=[1-9]",),
    log_patterns=(
        (
            "function_azure:stream_processor_with_citations",
            "Got more than 131072 bytes",
        ),
    ),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_CONTENT_NULL = KnownIssue(
    "azure-content-null",
    "Azure: a non-stream On Your Data answer with content null (e.g. content "
    "filter) makes the pipe answer 'Error: expected string or bytes-like "
    "object'",
    AZURE_FIX,
    ref=NO_ISSUE,
    evidence=(r"expected string or bytes-like object",),
    log_patterns=(("function_azure:pipe", "expected string or bytes-like object"),),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
AZURE_CLIENT_DATA_SOURCES = KnownIssue(
    "azure-client-data-sources",
    "Azure only processes citations when the AZURE_AI_DATA_SOURCES valve is "
    "set: data_sources sent by the client come back with unlinked [docX] and "
    "no sources",
    AZURE_FIX,
    ref=NO_ISSUE,
    # the grounded answer of the mock, references not converted
    evidence=(
        r"content=The X100 charges via USB-C \[doc1\]\. It has a two-year "
        r"warranty \[doc2\]\.",
    ),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
