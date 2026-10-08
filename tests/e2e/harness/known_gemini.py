"""Known bugs of pipelines/google/google_gemini.py (see ``harness.known``)."""

from .known import FOUND_BY_E2E, GEMINI_FIX, NO_ISSUE, KnownIssue

GEMINI_FILE = "pipelines/google/google_gemini.py"
GEMINI_FIXED_IN = "1.17.0"  # PR #185

GEMINI_B1 = KnownIssue(
    "B1",
    "Gemini API stream without a websocket session ends with 'Error during "
    "streaming' (the genai client is closed while the stream is consumed)",
    GEMINI_FIX,
    ref=NO_ISSUE,
    evidence=(r"Error during streaming",),
    log_patterns=(
        ("function_gemini:_handle_streaming_response", "Error during streaming"),
    ),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_B5 = KnownIssue(
    "B5",
    "Gemini's non-streaming path calls __event_emitter__ without a None check "
    "(usage event, image status), so requests without an emitter (background "
    "title/tags tasks) answer \"Error generating content: 'NoneType' object is "
    'not callable"',
    GEMINI_FIX,
    ref=NO_ISSUE,
    evidence=(r"Error generating content: 'NoneType' object is not callable",),
    log_patterns=(("function_gemini:pipe", "'NoneType' object is not callable"),),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_172 = KnownIssue(
    "#172",
    "gemini-3.1-flash-image (non-preview) is not recognised as an image model, "
    "the generated image is dropped",
    GEMINI_FIX,
    ref="https://github.com/owndev/Open-WebUI-Functions/issues/172",
    # listed without the image marker / streamed instead of forced non-stream
    evidence=(r"^name='[^'🎨]+'$", r"upstream=\['streamGenerateContent'\]"),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_NONSTREAM_USAGE = KnownIssue(
    "gemini-nonstream-usage",
    "Gemini non-streaming answers are returned as a plain string and usage only "
    "goes out as a 'usage' event, which Open WebUI 0.11 neither saves nor shows "
    "(docs promise a usage dict); API clients get no usage either",
    GEMINI_FIX,
    ref=FOUND_BY_E2E,
    evidence=(r"usage=None",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
