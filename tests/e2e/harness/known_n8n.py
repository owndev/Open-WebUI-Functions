"""Known bugs of pipelines/n8n/n8n.py and pipelines/infomaniak/infomaniak.py
(see ``harness.known``)."""

from .known import FOUND_BY_E2E, KnownIssue

INFOMANIAK_FILE = "pipelines/infomaniak/infomaniak.py"

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
