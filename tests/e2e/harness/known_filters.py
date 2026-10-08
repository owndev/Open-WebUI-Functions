"""Known bugs of filters/*.py (see ``harness.known``)."""

from .known import FILTERS_FIX, NO_ISSUE, KnownIssue

TRACKER_FILE = "filters/time_token_tracker.py"
TRACKER_FIXED_IN = "2.6.2"  # PR #184
SEARCH_FILE = "filters/google_search_tool.py"
SEARCH_FIXED_IN = "1.0.1"  # PR #184
VERTEX_FILE = "filters/vertex_ai_search_tool.py"
VERTEX_FIXED_IN = "1.0.1"  # PR #184

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
    file=TRACKER_FILE,
    fixed_in=TRACKER_FIXED_IN,
)
FILTER_SEARCH_KEYERROR = KnownIssue(
    "search-tool-keyerror",
    "google_search_tool inlet does features.pop('web_search') without a default; "
    "requests without features.web_search fail with KeyError 'web_search'",
    FILTERS_FIX,
    ref=NO_ISSUE,
    evidence=(r"HTTP 400 .*'web_search'",),
    log_patterns=("Error processing chat payload: 'web_search'",),
    file=SEARCH_FILE,
    fixed_in=SEARCH_FIXED_IN,
)
