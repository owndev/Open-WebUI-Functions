"""Known bugs of pipelines/azure/azure_ai_foundry.py (see ``harness.known``)."""

from .known import AZURE_FIX, NO_ISSUE, KnownIssue

AZURE_FILE = "pipelines/azure/azure_ai_foundry.py"
AZURE_FIXED_IN = "2.8.0"  # PR #183

AZURE_DOUBLE_STRIP = KnownIssue(
    "azure-double-strip",
    "Azure strips the function prefix twice for non-streaming / data_sources "
    "requests, so dotted model names (gpt-4.1 -> '1') are mangled",
    AZURE_FIX,
    ref=NO_ISSUE,
    # what is left of gpt-4.1 / Phi-3.5-mini-instruct after the second strip
    evidence=(r"body\.model='(1|5-mini-instruct)'",),
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
    evidence=(r"with data_sources=[1-9]",),
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
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
    file=AZURE_FILE,
    fixed_in=AZURE_FIXED_IN,
)
