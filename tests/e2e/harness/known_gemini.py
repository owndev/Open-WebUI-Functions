"""Known bugs of pipelines/google/google_gemini.py (see ``harness.known``).

Native tool calling before 1.18.0: the pipe hands Open WebUI's tool callables
(``__tools__``) to google-genai, whose automatic function calling (AFC) runs
them inside the pipe; Open WebUI's tool loop never sees a tool call. Every
evidence needs ``mock_ok=True`` (the group's preflight request got the mock's
answer) plus a token that the request itself worked, so a failing provider
(``E2E_MOCK_FAULT``) is never reported as one of these bugs.
"""

from .known import FOUND_BY_E2E, KnownIssue

GEMINI_FILE = "pipelines/google/google_gemini.py"
GEMINI_TOOLS_FIX = "feature/gemini-native-tool-calling"
GEMINI_TOOLS_FIXED_IN = "1.18.0"

GEMINI_TOOLS_AFC = KnownIssue(
    "gemini-tools-afc",
    "native tool calls are run by google-genai's automatic function calling "
    "inside the pipe: Open WebUI saves no tool call (no approval, no direct "
    "tools, no signatures carried); the SDK's follow-up request (model content "
    "split per chunk, function responses without ids) is rejected",
    GEMINI_TOOLS_FIX,
    ref="#169",
    evidence=(
        # the mock answered with function calls (fc) and the pipe handled them
        # itself (AFC follow-up, rejected with 400, or a KeyError for a tool it
        # does not know): Open WebUI saved no function call
        r"mock_ok=True http=200 done=True .*calls=\[\] outputs=0 .*upstream=\[fc[,\]]",
        # follow-up turn: the turn-1 call never reached the saved history
        r"mock_ok=True http=200 done=True .*upstream=\[text\] .*old_fc=\[\] old_fr=0",
        # server log of the tool group, while the mock answered
        r"mock_ok=True answered=[1-9]\d* found=\[.*AFC is enabled",
    ),
    log_patterns=(
        ("function_gemini", "Please ensure that the number of function response parts"),
        ("function_gemini", "Error during streaming: 'no_such_tool'"),
        # the Gemini mock stream AFC abandoned on that KeyError, collected later
        # (a ResourceWarning and/or aiohttp's "Unclosed connection" error)
        ("Unclosed connection", "port=9101"),
    ),
    file=GEMINI_FILE,
    fixed_in=GEMINI_TOOLS_FIXED_IN,
)

GEMINI_TOOLS_ANNOTATIONS = KnownIssue(
    "gemini-tools-annotations",
    "a workspace tool run by automatic function calling fails on its string "
    "annotations (the function responses carry an isinstance error)",
    GEMINI_TOOLS_FIX,
    ref="#169",
    evidence=(
        r"mock_ok=True http=200 done=True .*calls=\[\] .*upstream=\[fc[,\]].*"
        r"responses=.*isinstance\(\) arg 2 must be a type",
    ),
    log_patterns=GEMINI_TOOLS_AFC.log_patterns,
    file=GEMINI_FILE,
    fixed_in=GEMINI_TOOLS_FIXED_IN,
)

GEMINI_TOOLS_DUPLICATE = KnownIssue(
    "gemini-tools-duplicate",
    "OpenAPI / MCP tool callables are all named tool_function: Gemini rejects "
    "the request (Duplicate function declaration found: tool_function)",
    GEMINI_TOOLS_FIX,
    ref="#169",
    evidence=(
        r"mock_ok=True http=200 .*upstream=\[http-400\].*"
        r"Duplicate function declaration found: tool_function",
    ),
    log_patterns=(
        ("function_gemini", "Duplicate function declaration found: tool_function"),
    ),
    file=GEMINI_FILE,
    fixed_in=GEMINI_TOOLS_FIXED_IN,
)

GEMINI_TOOLS_MCP = KnownIssue(
    "gemini-tools-mcp",
    "an MCP tool breaks every request: its callable has no __signature__ "
    "(AttributeError in the pipe's debug line)",
    GEMINI_TOOLS_FIX,
    ref="#169",
    evidence=(r"mock_ok=True http=200 done=True .*upstream=\[\] .*__signature__",),
    log_patterns=(("function_gemini", "__signature__"),),
    file=GEMINI_FILE,
    fixed_in=GEMINI_TOOLS_FIXED_IN,
)

GEMINI_TOOLS_DIRECT = KnownIssue(
    "gemini-tools-direct",
    "a direct tool (tool server of the browser) breaks every request: it has no "
    "callable (KeyError: 'callable')",
    GEMINI_TOOLS_FIX,
    ref="#169",
    evidence=(r"mock_ok=True http=200 done=True .*upstream=\[\] .*'callable'",),
    log_patterns=(("function_gemini", "'callable'"),),
    file=GEMINI_FILE,
    fixed_in=GEMINI_TOOLS_FIXED_IN,
)

GEMINI_TOOLS_API = KnownIssue(
    "gemini-tools-api",
    "client tools of API requests are ignored (__tools__ is empty there): no "
    "function is declared and no tool_calls are returned",
    GEMINI_TOOLS_FIX,
    ref=FOUND_BY_E2E,
    evidence=(
        r"mock_ok=True http=200 .*upstream=\[(mock-error|text)(,text)*\] "
        r"declared_n=\[?0",
    ),
    file=GEMINI_FILE,
    fixed_in=GEMINI_TOOLS_FIXED_IN,
)

GEMINI_TOOLS_HISTORY = KnownIssue(
    "gemini-tools-history",
    "tool calls and tool results in the history are sent as plain text: no "
    "functionCall / functionResponse parts reach Gemini",
    GEMINI_TOOLS_FIX,
    ref=FOUND_BY_E2E,
    evidence=(r"mock_ok=True http=200 .*upstream=\[text\] .*sig_echoed=\[\] fr=\[\]",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_TOOLS_FIXED_IN,
)

GEMINI_TOOLS_GROUNDING = KnownIssue(
    "gemini-tools-grounding",
    "grounding and functions are combined the wrong way: Gemini 2.5 gets "
    "function declarations next to Search grounding, Gemini 3 gets them without "
    "includeServerSideToolInvocations",
    GEMINI_TOOLS_FIX,
    ref=FOUND_BY_E2E,
    evidence=(
        r"mock_ok=True http=200 done=True .*"
        r"tool_kinds=\[googleSearch,urlContext,functionDeclarations\] "
        r"include_flag=None",
    ),
    file=GEMINI_FILE,
    fixed_in=GEMINI_TOOLS_FIXED_IN,
)

GEMINI_TOOLS_MALFORMED = KnownIssue(
    "gemini-tools-malformed",
    "a MALFORMED_FUNCTION_CALL finish is answered with an empty or generic "
    "answer instead of an error message that names it",
    GEMINI_TOOLS_FIX,
    ref=FOUND_BY_E2E,
    evidence=(r"mock_ok=True http=200 done=True .*upstream=\[malformed\]",),
    log_patterns=(("function_gemini", "Failed to access content parts"),),
    file=GEMINI_FILE,
    fixed_in=GEMINI_TOOLS_FIXED_IN,
)
