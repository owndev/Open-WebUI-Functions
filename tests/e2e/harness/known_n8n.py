"""Known bugs of pipelines/n8n/n8n.py and pipelines/infomaniak/infomaniak.py
(see ``harness.known``)."""

from .known import FOUND_BY_E2E, N8N_INFOMANIAK_FIX, NO_ISSUE, KnownIssue

N8N_FILE = "pipelines/n8n/n8n.py"
N8N_FIXED_IN = "2.3.1"  # PR #182
INFOMANIAK_FILE = "pipelines/infomaniak/infomaniak.py"
INFOMANIAK_FIXED_IN = "2.2.1"  # PR #182


def _n8n(key: str, summary: str, *evidence: str, **extra) -> KnownIssue:
    """A bug of n8n.py fixed by PR #182 (n8n 2.3.1)."""
    extra.setdefault("ref", FOUND_BY_E2E)
    return KnownIssue(
        key,
        summary,
        N8N_INFOMANIAK_FIX,
        evidence=evidence,
        file=N8N_FILE,
        fixed_in=N8N_FIXED_IN,
        **extra,
    )


def _infomaniak(key: str, summary: str, *evidence: str, **extra) -> KnownIssue:
    """A bug of infomaniak.py fixed by PR #182 (Infomaniak 2.2.1)."""
    extra.setdefault("ref", FOUND_BY_E2E)
    return KnownIssue(
        key,
        summary,
        N8N_INFOMANIAK_FIX,
        evidence=evidence,
        file=INFOMANIAK_FILE,
        fixed_in=INFOMANIAK_FIXED_IN,
        **extra,
    )


# ------------------------------------------------------------------------ n8n
N8N_DICT_IN_STREAM = _n8n(
    "n8n-dict-in-stream",
    "n8n returns a chat.completion dict (usage) for stream=True; the streaming "
    "middleware ignores choices[].message, the saved answer is empty",
    # API path: one chat.completion chunk instead of delta chunks + [DONE]
    r"stream answered with chat\.completion",
    # browser path: empty answer, but the webhook's usage saved
    r"content= usage=\{'prompt_tokens': 5, 'completion_tokens': 3, "
    r"'total_tokens': 8[,}]",
    ref=NO_ISSUE,
)
N8N_SSE_CONTROL_LINES = _n8n(
    "n8n-sse-control-lines",
    "n8n parses SSE answers per network chunk: comment lines (': ...') and "
    "'data: [DONE]' end up in the answer text",
    r"content=.*(: keep-alive|data: \[DONE\])",
)
N8N_PLAIN_LINE = _n8n(
    "n8n-plain-line",
    "n8n drops a plain-text line that arrives in the same network chunk as SSE "
    "JSON events",
    r"content=SSE part one\. SSE part two\.",
)
N8N_SSE_FIELDS = _n8n(
    "n8n-sse-fields",
    "n8n keeps SSE framing: event:/id:/retry: fields and the 'data:' prefix of "
    "continuation lines end up in the answer",
    r"content=.*data: (multi|line|\[DONE\])",
)
N8N_OPENAI_CHUNKS = _n8n(
    "n8n-openai-chunks",
    "n8n appends OpenAI-style finish / usage chunks or 'data: [DONE]' to the "
    "answer of an OpenAI-style SSE stream",
    r"content=OpenAI style\..*(data: \[DONE\]|\{)",
)
N8N_UTF8_SPLIT = _n8n(
    "n8n-utf8-split",
    "n8n decodes every network chunk on its own (errors ignored): UTF-8 "
    "characters split across chunks are dropped",
    r"content=Gre nave",
)
N8N_BRACES = _n8n(
    "n8n-braces",
    "n8n's JSON object scanner counts braces inside JSON strings: streamed items "
    "with '{' or '}' leak raw JSON into the answer",
    r"content=.*\"(type|metadata)\"",
)
N8N_STOP = _n8n(
    "n8n-stop",
    "Stop leaves an n8n request without a final status (in progress forever), and "
    "a request stopped while waiting for the reply leaks its aiohttp session",
    r"last_status='Sending request to N8N\.\.\.' done=False",
    # the leaked session, logged when it is garbage-collected (no function name)
    log_patterns=("Unclosed client session",),
)
N8N_ERROR_CHUNK = _n8n(
    "n8n-error-chunk",
    "n8n stream error chunks ({'type': 'error'}) are dropped: the answer is "
    "'(Empty response received from N8N)' and the final status 'Streaming "
    "complete'",
    r"content=\(Empty response received from N8N\)",
)
N8N_MIDSTREAM_STATUS = _n8n(
    "n8n-midstream-status",
    "an n8n stream that breaks off mid-way ends with the status 'Streaming "
    "complete' instead of an error",
    r"last_status='Streaming complete' done=True",
)

# ----------------------------------------------------------------- Infomaniak
INFOMANIAK_CHUNKING = _infomaniak(
    "infomaniak-chunking",
    "Infomaniak forwards raw network chunks; coalesced or split SSE events are "
    "dropped by the streaming middleware (empty answer / missing usage)",
    # nothing of the coalesced / split events saved (the request itself worked)
    r"^HTTP 200 done=True content= usage=None ",
    # usage requested from the upstream (include_usage) but not saved, while
    # the answer is the expected one (or lost the same way)
    r"^usage=None include_usage_sent=True answer_ok=True$",
    r"^usage=None include_usage_sent=True content=$",
    ref=NO_ISSUE,
)
INFOMANIAK_STATUS = _infomaniak(
    "infomaniak-status",
    "Infomaniak emits no status events: no 'Sending' / 'Streaming' / "
    "'completed' / error status in the chat",
    r"^status=\[\] answer_ok=True$",  # no status, the answer itself is right
)
INFOMANIAK_REMAINDER = _infomaniak(
    "infomaniak-remainder",
    "Infomaniak drops a final SSE line without a trailing newline (and coalesced "
    "events): the answer / usage of such a stream is lost",
    r"^HTTP 200 done=True content= usage=None ",
)
INFOMANIAK_CRLF = _infomaniak(
    "infomaniak-crlf",
    "Infomaniak forwards CRLF-terminated SSE events in one chunk: the answer is lost",
    r"^HTTP 200 done=True content= usage=None ",
)
INFOMANIAK_MIDFAIL = _infomaniak(
    "infomaniak-midfail",
    "an Infomaniak stream that breaks off mid-way ends without an error status",
    r"^status=\[\] HTTP 200 done=True content=partial ",
)
INFOMANIAK_STOP = _infomaniak(
    "infomaniak-stop",
    "Stop leaves an Infomaniak answer without a final status",
    # no status; the answer is what arrived before Stop (or nothing)
    r"^status=\[\] stopped=\[\[200, .* content=(t0 [^|]*)? usage=None ",
)
INFOMANIAK_ERROR_DETAIL = _infomaniak(
    "infomaniak-error-detail",
    "Infomaniak upstream errors: an error.description body is shown as a raw "
    "dict, and every HTTP 4xx is logged with a full traceback",
    r"content=Error: \{'code': 'validation_failed'",
    r"Error in Infomaniak AI request: 400.*\(with Traceback\)",
)
INFOMANIAK_NAME_PREFIX = KnownIssue(
    "infomaniak-name-prefix",
    "changing the NAME_PREFIX valve does not change the model names (the "
    "prefix is read once in __init__)",
    "",  # no fix yet (follow-up issue)
    ref=FOUND_BY_E2E,
    evidence=(r"^name='Infomaniak: ",),
    file=INFOMANIAK_FILE,
    fixed_in="",
)
