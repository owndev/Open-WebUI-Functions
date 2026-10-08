"""Known bugs of filters/*.py (see ``harness.known``).

Open WebUI logs a failing inlet filter only as ``Error processing chat payload:
<message>`` (the filter id is logged at DEBUG level), so the inlet signatures
below cannot name the filter; the scenarios that carry them attach only the
filter under test to the model.
"""

from .known import FILTERS_FIX, NO_ISSUE, KnownIssue

TRACKER_FILE = "filters/time_token_tracker.py"
TRACKER_FIXED_IN = "2.6.2"  # PR #184
SEARCH_FILE = "filters/google_search_tool.py"
SEARCH_FIXED_IN = "1.0.1"  # PR #184
VERTEX_FILE = "filters/vertex_ai_search_tool.py"
VERTEX_FIXED_IN = "1.0.1"  # PR #184

TRACKER_EMITTER_ISSUE = "https://github.com/owndev/Open-WebUI-Functions/issues/175"

# time_token_tracker outlet on the API path (no event emitter) on Open WebUI >= 0.10
TRACKER_EMITTER_LOG = (
    "Error in outlet filter time_token_tracker",
    "'NoneType' object is not callable",
)

FILTER_TRACKER_NO_EMITTER = KnownIssue(
    "tracker-no-emitter",
    "time_token_tracker outlet calls __event_emitter__ unconditionally; on the "
    "API path Open WebUI passes None and the outlet raises",
    FILTERS_FIX,
    ref=TRACKER_EMITTER_ISSUE,
    evidence=(r"outlet filter time_token_tracker.*'NoneType' object is not callable",),
    log_patterns=(TRACKER_EMITTER_LOG,),
    file=TRACKER_FILE,
    fixed_in=TRACKER_FIXED_IN,
)
FILTER_TRACKER_LA_API = KnownIssue(
    "tracker-la-api",
    "time_token_tracker outlet raises on the API path (no __event_emitter__) "
    "before the Log Analytics send, so API requests are never recorded",
    FILTERS_FIX,
    ref=TRACKER_EMITTER_ISSUE,
    evidence=(
        r"posts=0\b.*outlet filter time_token_tracker.*'NoneType' object is not "
        r"callable",
    ),
    log_patterns=(TRACKER_EMITTER_LOG,),
    file=TRACKER_FILE,
    fixed_in=TRACKER_FIXED_IN,
)
FILTER_TRACKER_MESSAGEID = KnownIssue(
    "tracker-messageid",
    "time_token_tracker records a random messageId instead of the assistant "
    "message id Open WebUI passes to the outlet",
    FILTERS_FIX,
    ref=NO_ISSUE,
    evidence=(r"problems=\['messageId=[0-9a-f-]+ want [0-9a-f-]+'\]",),
    file=TRACKER_FILE,
    fixed_in=TRACKER_FIXED_IN,
)
FILTER_TRACKER_SEND_ENV = KnownIssue(
    "tracker-send-env-parse",
    "SEND_TO_LOG_ANALYTICS defaults to bool(os.getenv(...)), so "
    "SEND_TO_LOG_ANALYTICS=false in the environment turns the send on",
    FILTERS_FIX,
    ref=NO_ISSUE,
    evidence=(r"default=True env='false'",),
    file=TRACKER_FILE,
    fixed_in=TRACKER_FIXED_IN,
)
FILTER_TRACKER_CORRELATION = KnownIssue(
    "tracker-correlation",
    "time_token_tracker finds the inlet entry only by a fingerprint of the last "
    "user message: when Open WebUI changes that message after the inlet (legacy "
    "code interpreter prompt) or two identical requests overlap, the outlet "
    "logs 'No inlet data found' and records zero values",
    FILTERS_FIX,
    ref=NO_ISSUE,
    evidence=(r"No inlet data found",),
    file=TRACKER_FILE,
    fixed_in=TRACKER_FIXED_IN,
)
FILTER_TRACKER_TIKTOKEN_OFFLINE = KnownIssue(
    "tracker-tiktoken-offline",
    "time_token_tracker loads the tiktoken encoding synchronously in the event "
    "loop and raises when it cannot be downloaded: requests fail with HTTP 400 "
    "and the server stalls while the download hangs",
    FILTERS_FIX,
    ref=NO_ISSUE,
    evidence=(r"HTTP \[?400\b.*openaipublic\.blob\.core\.windows\.net",),
    log_patterns=(
        ("Error processing chat payload", "openaipublic.blob.core.windows.net"),
    ),
    file=TRACKER_FILE,
    fixed_in=TRACKER_FIXED_IN,
)
FILTER_TRACKER_SPECIAL_TOKEN = KnownIssue(
    "tracker-special-token",
    "time_token_tracker encodes with tiktoken's default disallowed_special: "
    "'<|endoftext|>' in a message raises and the request fails with HTTP 400",
    FILTERS_FIX,
    ref=NO_ISSUE,
    evidence=(r"HTTP 400 .*disallowed special token",),
    log_patterns=(("Error processing chat payload", "disallowed special token"),),
    file=TRACKER_FILE,
    fixed_in=TRACKER_FIXED_IN,
)
FILTER_TRACKER_ESTIMATE_MARKER = KnownIssue(
    "tracker-estimate-marker",
    "time_token_tracker records and logs len(text) // 4 estimates like exact "
    "token counts (no tokensEstimated flag)",
    FILTERS_FIX,
    ref=NO_ISSUE,
    evidence=(r"tokensEstimated flag missing",),
    file=TRACKER_FILE,
    fixed_in=TRACKER_FIXED_IN,
)
FILTER_SEARCH_KEYERROR = KnownIssue(
    "search-tool-keyerror",
    "google_search_tool inlet does features.pop('web_search') without a default "
    "on body.get('features', {}): requests without features.web_search fail "
    "with KeyError 'web_search', 'features': null with a NoneType error (HTTP 400)",
    FILTERS_FIX,
    ref=NO_ISSUE,
    evidence=(
        r"HTTP 400 .*'web_search'",
        r"HTTP 400 .*'NoneType' object has no attribute 'pop'",
    ),
    log_patterns=(
        "Error processing chat payload: 'web_search'",
        "Error processing chat payload: 'NoneType' object has no attribute 'pop'",
    ),
    file=SEARCH_FILE,
    fixed_in=SEARCH_FIXED_IN,
)
FILTER_SEARCH_MULTIMODEL = KnownIssue(
    "search-multimodel",
    "google_search_tool pops web_search from body['features'], the dict all "
    "models of a multi-model chat share: the next model's inlet fails with "
    "KeyError 'web_search' and models without the filter lose web_search",
    FILTERS_FIX,
    ref=NO_ISSUE,
    evidence=(r"errors=\[.*'web_search'",),
    log_patterns=("Error processing chat payload: 'web_search'",),
    file=SEARCH_FILE,
    fixed_in=SEARCH_FIXED_IN,
)
FILTER_VERTEX_FEATURES_NULL = KnownIssue(
    "vertex-features-null",
    "vertex_ai_search_tool inlet calls .pop() on body.get('features', {}): a "
    "request with 'features': null fails with a NoneType error (HTTP 400)",
    FILTERS_FIX,
    ref=NO_ISSUE,
    evidence=(r"HTTP 400 .*'NoneType' object has no attribute 'pop'",),
    log_patterns=(
        "Error processing chat payload: 'NoneType' object has no attribute 'pop'",
    ),
    file=VERTEX_FILE,
    fixed_in=VERTEX_FIXED_IN,
)
FILTER_VERTEX_REQUEST_STORE = KnownIssue(
    "vertex-request-store",
    "Open WebUI moves params.vertex_rag_store to the top level of the body "
    "before the inlets; vertex_ai_search_tool reads only __metadata__.params, so "
    "the request's data store never reaches the pipe (the env store is used)",
    FILTERS_FIX,
    ref=NO_ISSUE,
    evidence=(r"store='[^']*/dataStores/e2e-store' want '[^']*/per-request'",),
    file=VERTEX_FILE,
    fixed_in=VERTEX_FIXED_IN,
)
