"""Known bugs of pipelines/n8n/n8n.py and pipelines/infomaniak/infomaniak.py
(see ``harness.known``)."""

from .known import FOUND_BY_E2E, N8N_INFOMANIAK_FIX, NO_ISSUE, KnownIssue

N8N_FILE = "pipelines/n8n/n8n.py"
N8N_FIXED_IN = "2.3.1"  # PR #182
INFOMANIAK_FILE = "pipelines/infomaniak/infomaniak.py"
INFOMANIAK_FIXED_IN = "2.2.1"  # PR #182

N8N_DICT_IN_STREAM = KnownIssue(
    "n8n-dict-in-stream",
    "n8n returns a chat.completion dict (usage) for stream=True; the streaming "
    "middleware ignores choices[].message, the saved answer is empty",
    N8N_INFOMANIAK_FIX,
    ref=NO_ISSUE,
    evidence=(r"content= usage=\{",),  # empty answer, usage saved
    file=N8N_FILE,
    fixed_in=N8N_FIXED_IN,
)
N8N_SSE_CONTROL_LINES = KnownIssue(
    "n8n-sse-control-lines",
    "n8n parses SSE answers per network chunk: comment lines (': ...') and "
    "'data: [DONE]' end up in the answer text",
    N8N_INFOMANIAK_FIX,
    ref=FOUND_BY_E2E,
    evidence=(r"content=.*(: keep-alive|data: \[DONE\])",),
    file=N8N_FILE,
    fixed_in=N8N_FIXED_IN,
)
INFOMANIAK_CHUNKING = KnownIssue(
    "infomaniak-chunking",
    "Infomaniak forwards raw network chunks; coalesced or split SSE events are "
    "dropped by the streaming middleware (empty answer / missing usage)",
    N8N_INFOMANIAK_FIX,
    ref=NO_ISSUE,
    evidence=(r"content= usage=None", r"^usage=None$"),
    file=INFOMANIAK_FILE,
    fixed_in=INFOMANIAK_FIXED_IN,
)
