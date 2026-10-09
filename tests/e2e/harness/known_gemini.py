"""Known bugs of pipelines/google/google_gemini.py (see ``harness.known``).

Every entry is fixed in google_gemini.py 1.17.0 (PR #185): on that version a
failing tagged check is a FAIL ("regression of known ..."). The evidence
patterns match tokens the gemini suite writes into its check details (e.g.
``interim_thought_images=2``), so a failure of another kind stays a FAIL.
"""

from .known import FOUND_BY_E2E, GEMINI_FIX, NO_ISSUE, KnownIssue

GEMINI_FILE = "pipelines/google/google_gemini.py"
GEMINI_FIXED_IN = "1.17.0"  # PR #185
ISSUE_181 = "https://github.com/owndev/Open-WebUI-Functions/issues/181"
FOUND_BY_REVIEW = "found in the review of PR #185, no issue filed"

GEMINI_B1 = KnownIssue(
    "B1",
    "Gemini API stream without a websocket session ends with 'Error during "
    "streaming' (the genai client is closed while the stream is consumed)",
    GEMINI_FIX,
    ref=NO_ISSUE,
    # the answer is only the error with an empty message
    evidence=(r"content=Error during streaming: +usage=",),
    log_patterns=(
        (
            "function_gemini:_handle_streaming_response",
            "Error during streaming",
            "assert self._connector is not None",
        ),
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
    evidence=(
        r"^name='Google Gemini: Gemini 3\.1 Flash Image'$",
        r"\bmodel=gemini-3\.1-flash-image upstream=\['streamGenerateContent'\]",
    ),
    # streamed with the browser's tools, which the image model rejects
    log_patterns=(
        (
            "function_gemini:_handle_streaming_response",
            "is not enabled for models/gemini-3.1-flash-image'",
        ),
    ),
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
    # the answer itself arrived, only the usage is missing
    evidence=(r"^HTTP 200 .*Hello from mock \(non-stream\)\. usage=None ",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_181_IMAGES = KnownIssue(
    "#181-images",
    "Gemini 3 image models: the interim thought images (thought=true inline "
    "images) are uploaded and attached like final images",
    GEMINI_FIX,
    ref=ISSUE_181,
    evidence=(r"\binterim_thought_images=[1-9]",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_181_MODELS = KnownIssue(
    "#181-models",
    "Gemini image models without the old name patterns (gemini-nano-banana-2.1, "
    "gemini-3.1-flash-lite-image, IDs listed in IMAGE_GENERATION_MODELS) are not "
    "recognised: no image marker, streamed without IMAGE modality, no ImageConfig",
    GEMINI_FIX,
    ref=ISSUE_181,
    evidence=(
        r"^name='Google Gemini: Nano Banana 2\.1'$",
        r"\bmodel=(gemini-nano-banana-2\.1|gemini-4-flash-image) "
        r"upstream=\['streamGenerateContent'\]",
        # the request worked (HTTP 200, generateContent), only the ImageConfig
        # is missing; with a failing upstream the action is None
        r"^HTTP 200 .*\bmodel=gemini-3\.1-flash-lite-image imageConfig=None "
        r"action=generateContent\b",
    ),
    # streamed with the browser's tools, which the image model rejects
    log_patterns=(
        (
            "function_gemini:_handle_streaming_response",
            "is not enabled for models/gemini-nano-banana-2.1'",
        ),
    ),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_181_TOOLS = KnownIssue(
    "#181-tools",
    "Gemini image models get Open WebUI's native tools (functionDeclarations), "
    "urlContext and, for gemini-2.5-flash-image / gemini-3.1-flash-lite-image, "
    "Search grounding, which these models reject with HTTP 400",
    GEMINI_FIX,
    ref=ISSUE_181,
    # the mock answers like the real API (HTTP 400 INVALID_ARGUMENT)
    evidence=(
        r"Function calling is not enabled for models/gemini-[\w.-]*image",
        r"Url context is not supported for models/gemini-[\w.-]*image",
        r"Search Grounding is not supported for models/gemini-[\w.-]*image",
    ),
    log_patterns=(
        ("function_gemini:", "Function calling is not enabled for models/"),
        ("function_gemini:", "Url context is not supported for models/"),
        ("function_gemini:", "Search Grounding is not supported for models/"),
    ),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_TASK_DETAILS = KnownIssue(
    "gemini-task-details",
    "Gemini background task answers (title, tags, follow-ups) start with the "
    "<details> thinking summary instead of the JSON Open WebUI parses",
    GEMINI_FIX,
    ref=FOUND_BY_E2E,
    evidence=(r"\banswer=<details>",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_TASK_GROUNDING = KnownIssue(
    "gemini-task-grounding",
    "Gemini background tasks of a web_search chat inherit the google_search_tool "
    "flag and are sent with googleSearch / urlContext grounding",
    GEMINI_FIX,
    ref=FOUND_BY_E2E,
    evidence=(r"\btask_tools=\[\[[^\]]*'(googleSearch|urlContext)'",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_VEO_TEXT = KnownIssue(
    "gemini-veo-text",
    "Gemini Veo answers a stream request with a bare chat.completion dict, so "
    "the browser path saves the video but not the answer text",
    GEMINI_FIX,
    ref=FOUND_BY_E2E,
    evidence=(r"\bsaved_text='' videos=1\b",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_STREAM_IMAGES = KnownIssue(
    "gemini-stream-images",
    "Gemini's streaming path drops inline images, so an image from a model the "
    "pipe does not detect as image model is never attached",
    GEMINI_FIX,
    ref=FOUND_BY_E2E,
    evidence=(r"\bupstream=\['streamGenerateContent'\] image_files=0\b",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_IMAGE_ERROR_STATUS = KnownIssue(
    "gemini-image-error-status",
    "Gemini image requests that fail upstream leave the 'Processing image "
    "request...' status running (no final status with done=true)",
    GEMINI_FIX,
    ref=FOUND_BY_E2E,
    evidence=(r"\bimage_status=\{[^}]*'done': False",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_VERTEX_TEXT = KnownIssue(
    "gemini-vertex-text",
    "Gemini Vertex AI Search sources read retrieved_context.chunk_text, which "
    "google-genai does not set, so the source documents are empty",
    GEMINI_FIX,
    ref=FOUND_BY_E2E,
    evidence=(r"\bvertex_document=\[''\]",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_API_IMAGE_DELIVERY = KnownIssue(
    "gemini-api-image-delivery",
    "Gemini image answers to API clients (no chat message) only announce the "
    "image in a 'files' event that reaches nobody; the answer has no link",
    GEMINI_FIX,
    ref=FOUND_BY_REVIEW,
    evidence=(r"\banswer_text=True image_links=0\b",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_STREAM_DATA_PREFIX = KnownIssue(
    "gemini-stream-data-prefix",
    "Gemini's streaming path yields the answer as a str, which Open WebUI "
    "passes through as a raw SSE line when it starts with 'data:', so the "
    "answer is lost",
    GEMINI_FIX,
    ref=FOUND_BY_REVIEW,
    evidence=(r"unparsable SSE line: starts like SSE\.",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_MODEL_CACHE = KnownIssue(
    "gemini-model-cache",
    "Gemini's model list cache ignores valve changes (MODEL_WHITELIST, "
    "MODEL_ADDITIONAL, IMAGE_GENERATION_MODELS) until MODEL_CACHE_TTL expires",
    GEMINI_FIX,
    ref=FOUND_BY_REVIEW,
    # the model list worked (several real models before, still several after the
    # whitelist change); a failing upstream lists only 'gemini.error' both times
    evidence=(
        r"^before=([2-9]|\d{2,}) models after=\['gemini\.gemini-[^']+', "
        r"'gemini\.[^']+'.*\] stale_list=True$",
    ),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_TERMINAL_STATUS = KnownIssue(
    "gemini-terminal-status",
    "Gemini leaves a started status (video generation, image processing, "
    "thinking) running when the request is stopped or fails",
    GEMINI_FIX,
    ref=FOUND_BY_REVIEW,
    # the Stop request worked, the last video status is still running
    evidence=(r"\bstop_ok=True last_video_status_done=False\b",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_STREAM_RETRY = KnownIssue(
    "gemini-stream-retry",
    "Gemini's RETRY_COUNT has no effect for streams: the lazy stream fails on "
    "its first chunk, outside the retried call",
    GEMINI_FIX,
    ref=FOUND_BY_REVIEW,
    evidence=(r"\bupstream_calls=1 .*Error during streaming: 503\b",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
GEMINI_USER_HEADERS = KnownIssue(
    "gemini-user-headers",
    "Gemini keeps the requesting user on the shared Pipe instance (self.user), "
    "so forwarded X-OpenWebUI-User-* headers can belong to another user (e.g. "
    "the model list request after another user's chat)",
    GEMINI_FIX,
    ref=FOUND_BY_REVIEW,
    evidence=(r"\bmodel_list_user=e2e-gemini-user@example\.com\b",),
    file=GEMINI_FILE,
    fixed_in=GEMINI_FIXED_IN,
)
