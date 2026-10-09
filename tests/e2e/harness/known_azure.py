"""Known bugs of pipelines/azure/azure_ai_foundry.py (see ``harness.known``).

The evidence is text only a working mock_azure produces (its emulated On Your
Data retirement answer), so a check that fails for another reason (a mock
fault, an upstream HTTP 500, a missing model) stays a FAIL.
"""

from .known import KnownIssue

AZURE_FILE = "pipelines/azure/azure_ai_foundry.py"

AZURE_OYD_RETIRED = KnownIssue(
    "azure-oyd-retired",
    "Azure AI Search only works through On Your Data (data_sources), which Azure "
    "retires on 2026-10-14; there is no pipeline-side retrieval yet",
    "feature/azure-search-pipeline-mode",
    ref="#187",
    # the mock's retirement answer for data_sources with gpt-5-mini, which the
    # pipe passes on as "Error: ..." (detail of the rag checks)
    evidence=(r"On Your Data is retired \(mock\)",),
    log_patterns=(
        (
            "function_azure:pipe",
            "Error in Azure AI request: 400",
            "/deployments/gpt-5-mini/",
        ),
    ),
    file=AZURE_FILE,
    fixed_in="2.9.0",
)
