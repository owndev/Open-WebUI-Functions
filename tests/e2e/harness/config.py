"""Constants shared by the in-container driver (paths, ports, credentials)."""

import os

# Open WebUI as seen from inside its own container.
OWUI_URL = os.environ.get("E2E_OWUI_URL", "http://127.0.0.1:8080")

# First account created on a fresh volume becomes the admin.
ADMIN_NAME = "E2E Admin"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "Passw0rd!e2e"

# Provider mocks (tests/e2e/mocks/serve_all.py), reachable from the pipes.
MOCK_HOST = "127.0.0.1"
MOCK_PORTS = {"gemini": 9101, "azure": 9102, "n8n": 9103, "infomaniak": 9104}

# Layout inside the container (run.sh copies tests/e2e/ to E2E_ROOT).
E2E_ROOT = os.environ.get("E2E_ROOT", "/e2e")
FUNCTIONS_DIR = os.path.join(E2E_ROOT, "functions")  # staged function files
PROBE_FILE = os.path.join(E2E_ROOT, "probe", "probe_pipe.py")
SERVER_LOG = os.environ.get("E2E_SERVER_LOG", "/tmp/e2e/server.log")

# VERTEX_AI_RAG_STORE the container is started with (vertex_ai_search_tool test).
VERTEX_RAG_STORE = os.environ.get("VERTEX_AI_RAG_STORE", "")

SUITES = ("gemini", "azure", "n8n", "infomaniak", "filters")
