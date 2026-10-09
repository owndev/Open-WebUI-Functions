"""Constants shared by the in-container driver (paths, ports, credentials)."""

import os
import pkgutil

# Open WebUI as seen from inside its own container.
OWUI_URL = os.environ.get("E2E_OWUI_URL", "http://127.0.0.1:8080")

# First account created on a fresh volume becomes the admin.
ADMIN_NAME = "E2E Admin"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "Passw0rd!e2e"

# Provider mocks (tests/e2e/mocks/serve_all.py), reachable from the pipes.
MOCK_HOST = "127.0.0.1"
MOCK_PORTS = {
    "gemini": 9101,
    "azure": 9102,
    "n8n": 9103,
    "infomaniak": 9104,
    "search": 9106,  # Azure AI Search; 9105 is the Log Analytics mock control port
}

# Layout inside the container (run.sh copies tests/e2e/ to E2E_ROOT).
E2E_ROOT = os.environ.get("E2E_ROOT", "/e2e")
FUNCTIONS_DIR = os.path.join(E2E_ROOT, "functions")  # staged function files
PROBE_FILE = os.path.join(E2E_ROOT, "probe", "probe_pipe.py")
SERVER_LOG = os.environ.get("E2E_SERVER_LOG", "/tmp/e2e/server.log")
# Output of the provider mocks (run.sh starts serve_all.py with it).
MOCKS_LOG = os.environ.get("E2E_MOCKS_LOG", os.path.join(E2E_ROOT, "out", "mocks.txt"))

# VERTEX_AI_RAG_STORE the container is started with (vertex_ai_search_tool test).
VERTEX_RAG_STORE = os.environ.get("VERTEX_AI_RAG_STORE", "")

# Default time limits in seconds (E2E_TIMEOUT / E2E_SUITE_TIMEOUT override them,
# see e2e.py): the budget of the whole driver run and the limit of one suite.
RUN_TIMEOUT = 1800
SUITE_TIMEOUT = 900
# Pause after each suite (before the next one starts) and before the final log
# scan, so that log lines of finished requests and background tasks are flushed
# and not charged to the next suite (E2E_LOG_SETTLE overrides it).
LOG_SETTLE = 1.0

# Suites are the modules in tests/e2e/suites/; a new module needs no registration.
# "all" runs the suites named here first, in this order, then every other suite
# alphabetically.
SUITES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "suites"
)
_SUITE_ORDER = ("gemini", "azure", "n8n", "infomaniak", "filters")


def discover_suites() -> tuple:
    """Names of the suite modules in ``SUITES_DIR`` (``_*`` modules excluded)."""
    found = {
        module.name
        for module in pkgutil.iter_modules([SUITES_DIR])
        if not module.name.startswith("_")
    }
    first = tuple(name for name in _SUITE_ORDER if name in found)
    return first + tuple(sorted(found - set(first)))


SUITES = discover_suites()
