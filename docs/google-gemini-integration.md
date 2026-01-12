# Google Gemini Integration

This integration enables **Open WebUI** to interact with **Google Gemini** models via the official Google Generative AI API (using API Keys) or through Google Cloud Vertex AI (leveraging Google Cloud's infrastructure and authentication). It provides a robust and customizable pipeline to access text and multimodal generation capabilities from Google’s latest AI models.

🔗 [Learn More About Google AI](https://ai.google.dev/)

## Pipeline

- 🧩 [Google Gemini Pipeline](../pipelines/google/google_gemini.py)

## Features

- **Asynchronous API Calls**  
  Improves performance and scalability with non-blocking requests.

- **Model Caching**  
  Caches available model lists for faster subsequent access.

- **Dynamic Model Handling**  
  Automatically strips provider prefixes for seamless integration.

- **Streaming Response Support**  
  Handles token-by-token responses with built-in safety enforcement.

> [!Note]
> Streaming is automatically disabled for image generation models to prevent chunk size issues.

- **Thinking Support**  
  Support reasoning and thinking steps, allowing models to break down complex tasks. Includes configurable thinking levels for Gemini 3 Pro ("low"/"high") and thinking budgets (0-32768 tokens) for other thinking-capable models.

  > [!Note]
  > **Thinking Levels vs Thinking Budgets**: Gemini 3 Pro models use `thinking_level` ("low" or "high"), while other models like Gemini 2.5 use `thinking_budget` (token count). See [Gemini Thinking Documentation](https://ai.google.dev/gemini-api/docs/thinking) for details.

- **Multimodal Input Support**  
  Accepts both text and image data for more expressive interactions with configurable image optimization.

- **Advanced Image Generation**  
  Support for text-to-image and image-to-image generation with Gemini 2.5 Flash Image Preview models.

- **Flexible Error Handling**  
  Retries failed requests and logs errors for transparency.

- **Integration with Google Generative AI or Vertex AI API**  
  Connect using either the Google Generative AI API or Google Cloud Vertex AI for content generation.

- **Secure API Key Storage**  
  API keys (for the Google Generative AI API method) are encrypted and never exposed in plaintext.

- **Safety Configuration**  
  Control safety behavior using a permissive or strict mode.

- **Customizable Generation Settings**  
  Use environment variables to configure token limits, temperature, etc.

- **Grounding with Google search**  
  Improve the accuracy and recency of Gemini responses with Google search grounding.

- **Ability to forward User Headers and change gemini base url**  
  Forward user information headers (like Name, Id, Email and Role) to Google API or LiteLLM for better context and analytics. Also, change the base URL for the Google Generative AI API if needed.

- **Native tool calling support**  
  Leverage Google genai native function calling to orchestrate the use of tools

## Environment Variables

Set the following environment variables to configure the Google Gemini integration.

### General Settings (Applicable to both connection methods)

```bash
# Use permissive safety settings for content generation (true/false)
# Default: false
USE_PERMISSIVE_SAFETY=false

# Model list cache duration (in seconds)
# Default: 600
GOOGLE_MODEL_CACHE_TTL=600

# Number of retry attempts for failed API calls
# Default: 2
GOOGLE_RETRY_COUNT=2

# Default system prompt applied to all chats
# If a user-defined system prompt exists, this is prepended to it
# Leave empty to disable
# Default: "" (empty, disabled)
GOOGLE_DEFAULT_SYSTEM_PROMPT=""

# Model configuration
# Add models that the SDK doesn't return but you want to make available
# Comma-separated list of model IDs (e.g., "gemini-exp-1206,gemini-2.0-flash-exp")
# Default: "" (empty, no additional models)
GOOGLE_MODEL_ADDITIONAL=""

# Filter models to only show specific ones in the model list
# Comma-separated list of model IDs to show (e.g., "gemini-2.0-flash-exp,gemini-1.5-pro")
# If empty, all models are shown. If set, only these models will be available.
# This filter is applied AFTER MODEL_ADDITIONAL, so you can add and then filter models.
# Default: "" (empty, show all models)
GOOGLE_MODEL_WHITELIST=""

# Image processing optimization settings
# Maximum image size in MB before compression is applied
# Default: 15.0
GOOGLE_IMAGE_MAX_SIZE_MB=15.0

# Maximum width or height in pixels before resizing
# Default: 2048
GOOGLE_IMAGE_MAX_DIMENSION=2048

# JPEG compression quality (1-100, higher = better quality but larger size)
# Default: 85
GOOGLE_IMAGE_COMPRESSION_QUALITY=85

# Enable intelligent image optimization for API compatibility
# Default: true
GOOGLE_IMAGE_ENABLE_OPTIMIZATION=true

# PNG files above this size (MB) will be converted to JPEG for better compression
# Default: 0.5
GOOGLE_IMAGE_PNG_THRESHOLD_MB=0.5

# Maximum number of images (history + current message) sent per request
# Default: 5
GOOGLE_IMAGE_HISTORY_MAX_REFERENCES=5

# Add inline labels like [Image 1] before each image to allow references in follow-up prompts
# Default: true
GOOGLE_IMAGE_ADD_LABELS=true

# Deduplicate identical images from history (hash-based) to reduce payload size
# Default: true
GOOGLE_IMAGE_DEDUP_HISTORY=true

# Boolean: When true (default) history images come before current message images.
# When false, current message images are placed first.
# Default: true
GOOGLE_IMAGE_HISTORY_FIRST=true

# Enable fallback to data URL when image upload fails
# Default: true
GOOGLE_IMAGE_UPLOAD_FALLBACK=true

# Image generation configuration (only for Gemini 3 image models like gemini-3-pro-image-preview)
# Note: These settings do not apply to Gemini 2.5 image models
# Default aspect ratio for generated images
# Valid values: "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
# Default: "1:1"
GOOGLE_IMAGE_GENERATION_ASPECT_RATIO="1:1"

# Default resolution for generated images
# Valid values: "1K", "2K", "4K"
# Default: "2K"
GOOGLE_IMAGE_GENERATION_RESOLUTION="2K"

# Enable Gemini thoughts outputs globally
# Default: true
GOOGLE_INCLUDE_THOUGHTS=true

# Thinking budget for Gemini 2.5 models (not used for Gemini 3 models)
# -1 = dynamic (model decides), 0 = disabled, 1-32768 = fixed token limit
# Default: -1 (dynamic)
# Note: Gemini 3 models use GOOGLE_THINKING_LEVEL instead
GOOGLE_THINKING_BUDGET=-1

# Thinking level for Gemini 3 models only
# Valid values: "low", "high", or empty string for model default
# - "low": Minimizes latency and cost, suitable for simple tasks
# - "high": Maximizes reasoning depth, ideal for complex problem-solving
# Default: "" (empty, uses model default)
# Note: This setting is ignored for non-Gemini 3 models
GOOGLE_THINKING_LEVEL=""

# Enable streaming responses globally
# Default: true
GOOGLE_STREAMING_ENABLED=true
```

### Connection Method: Google Generative AI API (Default)

Use these settings if you are connecting directly via the Google Generative AI API. This is the default method if `GOOGLE_GENAI_USE_VERTEXAI` is not set to `true`.

```bash
# API key for authenticating with Google Generative AI.
# Required if GOOGLE_GENAI_USE_VERTEXAI is "false" or not set.
GOOGLE_API_KEY="your-google-api-key"
```

> [!TIP]
> You can obtain your API key from the [Google AI Studio](https://aistudio.google.com/) dashboard after signing up.

### Connection Method: Google Cloud Vertex AI

Use these settings if you are connecting via Google Cloud Vertex AI. This method typically uses Application Default Credentials (ADC) or a service account for authentication, and `GOOGLE_API_KEY` is not used.

```bash
# Set to "true" to use Google Cloud Vertex AI.
# Default: false
GOOGLE_GENAI_USE_VERTEXAI="true"

# Your Google Cloud Project ID.
# Required if GOOGLE_GENAI_USE_VERTEXAI is "true".
GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# The Google Cloud region for Vertex AI (e.g., "us-central1").
# Defaults to "global" if not set.
GOOGLE_CLOUD_LOCATION="your-gcp-location"

# Vertex AI RAG Store path for grounding (e.g., projects/PROJECT/locations/global/collections/default_collection/dataStores/DATA_STORE_ID)
# Optional: Can also be set via metadata params or filter
# Auto-enabled when USE_VERTEX_AI is true and this is set
VERTEX_AI_RAG_STORE="projects/your-project/locations/global/collections/default_collection/dataStores/your-data-store-id"
```

> [!IMPORTANT]
> **Image Generation Limitations**
>
> **Streaming Support**: Image generation models automatically disable streaming mode to prevent "chunk too big" errors. All image generation requests use non-streaming mode regardless of the streaming setting.
>
> **Image Optimization Direction**: The current image processing configuration **only applies to input images** (Open WebUI → Google API), such as images uploaded to chat or used for image-to-image editing. Generated images from Google API are not yet subject to these optimization settings. This means:
>
> - ✅ **Input images**: Optimized using configuration settings
> - ❌ **Generated images**: Use original API output without additional optimization
>
> Future versions may extend these settings to also optimize generated images before upload/display.

## Image Generation Configuration

The Google Gemini pipeline supports configurable aspect ratios and resolutions for image generation with **Gemini 3 image models** (e.g., `gemini-3-pro-image-preview`, `gemini-3-flash-image-preview`).

> [!IMPORTANT]
> **Model Compatibility**: The `aspect_ratio` and `image_size` parameters (ImageConfig) are **only supported by Gemini 3 image models**. Gemini 2.5 image models (e.g., `gemini-2.5-flash-image-preview`) support image generation but do not support these configuration parameters. When using Gemini 2.5 image models, default aspect ratio and resolution will be used automatically.

### Aspect Ratio

Control the shape and proportions of generated images using the aspect ratio setting:

**Valid Values:**
- `1:1` - Square (default)
- `2:3`, `3:2` - Classic photo ratios
- `3:4`, `4:3` - Standard display ratios
- `4:5`, `5:4` - Portrait/landscape variants
- `9:16`, `16:9` - Mobile and widescreen ratios
- `21:9` - Ultra-wide format

**Configuration:**
```bash
# Set via environment variable (global default)
GOOGLE_IMAGE_GENERATION_ASPECT_RATIO="16:9"
```

Or configure through the pipeline valves in Open WebUI's Admin panel.

### Resolution

Control the quality and size of generated images:

**Valid Values:**
- `1K` - Lower resolution, faster generation
- `2K` - Balanced quality and speed (default)
- `4K` - Highest quality, slower generation

**Configuration:**
```bash
# Set via environment variable (global default)
GOOGLE_IMAGE_GENERATION_RESOLUTION="4K"
```

Or configure through the pipeline valves in Open WebUI's Admin panel.

### Per-Request Override

You can override the default settings on a per-request basis by including these parameters in the request body:

**Example API Usage:**

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="your-api-key")

# Generate a 4K widescreen image with Gemini 3
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents="A serene mountain landscape at sunset",
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size="4K"
        ),
    )
)

for part in response.parts:
    if part.text:
        print(part.text)
    elif image := part.as_image():
        image.save("landscape.png")
```

### Use Cases

**Portrait Photography (`3:4` or `4:5`)**
- Social media profile images
- Portrait-oriented artwork

**Widescreen Content (`16:9` or `21:9`)**
- Desktop wallpapers
- YouTube thumbnails
- Presentation slides

**Square Images (`1:1`)**
- Instagram posts
- Icons and logos
- Product photos

**Mobile-First (`9:16`)**
- Instagram Stories
- TikTok content
- Mobile app screens

### Model Compatibility

| Model                     | ImageConfig Support (aspect_ratio, image_size) |
| ------------------------- | ----------------------------------------------- |
| gemini-3-pro-image-\*     | ✅ Supported                                    |
| gemini-3-flash-image-\*   | ✅ Supported                                    |
| gemini-2.5-flash-image-\* | ❌ Not supported (uses defaults)                |
| Other gemini-3-\* models  | ❌ Not image generation models                  |
| Other models              | ❌ Not image generation models                  |

## Model Configuration

The Google Gemini pipeline provides two complementary mechanisms for controlling which models appear in the model list: `MODEL_ADDITIONAL` and `MODEL_WHITELIST`.

### MODEL_ADDITIONAL

Use `MODEL_ADDITIONAL` to add models that are not returned by the Google AI SDK but that you want to make available. This is useful for:

- **Experimental models** that haven't been added to the SDK yet (e.g., `gemini-exp-1206`)
- **Preview models** that require explicit configuration
- **Custom model endpoints** that follow the Gemini API format

**Configuration:**
```bash
# Add experimental models not in the SDK
GOOGLE_MODEL_ADDITIONAL="gemini-exp-1206,gemini-2.0-flash-exp"
```

**Example Use Case:**
```bash
# Google releases a new experimental model that's not in the SDK yet
GOOGLE_MODEL_ADDITIONAL="gemini-exp-1206"
# This model will now appear in your model list alongside SDK-provided models
```

### MODEL_WHITELIST

Use `MODEL_WHITELIST` to restrict which models appear in the model list. This is useful for:

- **Limiting choices** to prevent users from selecting deprecated models
- **Cost control** by only showing economical models
- **Feature-specific deployments** (e.g., only showing image-capable models)
- **Organizational policies** requiring specific model versions

**Configuration:**
```bash
# Only show specific models in the list
GOOGLE_MODEL_WHITELIST="gemini-2.0-flash-exp,gemini-1.5-pro,gemini-1.5-flash"
```

**Example Use Case:**
```bash
# Only allow the latest production models
GOOGLE_MODEL_WHITELIST="gemini-2.0-flash-exp,gemini-1.5-pro"
# Users will only see these two models, even if more are available
```

### Using Both Together

`MODEL_ADDITIONAL` and `MODEL_WHITELIST` work together in the following order:

1. **Fetch** models from the Google AI SDK
2. **Add** models specified in `MODEL_ADDITIONAL`
3. **Filter** to only show models in `MODEL_WHITELIST` (if configured)

**Example Combined Use:**
```bash
# Add an experimental model and limit the list to specific models
GOOGLE_MODEL_ADDITIONAL="gemini-exp-1206"
GOOGLE_MODEL_WHITELIST="gemini-exp-1206,gemini-2.0-flash-exp,gemini-1.5-pro"
# Result: Users see only the experimental model plus two production models
```

**Practical Scenarios:**

1. **Beta Testing Environment**
   ```bash
   # Add experimental models and limit to testing models only
   GOOGLE_MODEL_ADDITIONAL="gemini-exp-1206,gemini-3-flash-preview"
   GOOGLE_MODEL_WHITELIST="gemini-exp-1206,gemini-3-flash-preview"
   ```

2. **Cost-Controlled Production**
   ```bash
   # Only show flash models (most economical)
   GOOGLE_MODEL_WHITELIST="gemini-2.0-flash-exp,gemini-1.5-flash"
   ```

3. **Image Generation Only**
   ```bash
   # Only show image-capable models
   GOOGLE_MODEL_WHITELIST="gemini-2.5-flash-image-preview,gemini-3-pro-image-preview"
   ```

4. **Adding Preview Model Not in SDK**
   ```bash
   # Add a preview model and include it in the whitelist
   GOOGLE_MODEL_ADDITIONAL="gemini-2.5-flash-preview-0514"
   GOOGLE_MODEL_WHITELIST="gemini-2.5-flash-preview-0514,gemini-1.5-pro"
   ```

> [!TIP]
> Leave both variables empty (default) to show all available models from the SDK.

## Web search and access 

[Grounding with Google search](https://ai.google.dev/gemini-api/docs/google-search) together with the [URL context tool](https://ai.google.dev/gemini-api/docs/url-context) are enabled/disabled together via the `google_search_tool` feature, which can be switched on/off in a Filter.

For instance, the following [Filter (google_search_tool.py)](../filters/google_search_tool.py) will replace Open Web UI default web search function with Google search grounding + the URL context tool.

When enabled, sources and google queries from the search used by Gemini will be displayed with the response.

## Grounding with Vertex AI Search

Improve the accuracy and recency of Gemini responses by grounding them with your own data in Vertex AI Search.

### Configuration

To enable Vertex AI Search grounding, you need to:

1. **Set up a Vertex AI Search Data Store**: Follow the [Google Cloud documentation](https://cloud.google.com/vertex-ai/docs/search/overview) to create a Data Store in Discovery Engine and ingest your documents.
2. **Provide the RAG Store Path**: The path should be in the format `projects/PROJECT/locations/LOCATION/ragCorpora/DATA_STORE_ID` or `projects/PROJECT/locations/global/collections/default_collection/dataStores/DATA_STORE_ID`.
   - Set the `VERTEX_AI_RAG_STORE` environment variable, or
   - Use the [Filter (vertex_ai_search_tool.py)](../filters/vertex_ai_search_tool.py) to enable the feature and optionally pass the store ID via chat metadata.
3. **Enable Vertex AI**: Set `GOOGLE_GENAI_USE_VERTEXAI=true` to use Vertex AI (required for Vertex AI Search grounding).

When `USE_VERTEX_AI` is `true` and `VERTEX_AI_RAG_STORE` is configured, Vertex AI Search grounding will be automatically enabled. You can also explicitly enable it via the `vertex_ai_search` feature flag.

When enabled, Gemini will use the specified Vertex AI Search Data Store to retrieve relevant information and ground its responses, providing citations to the source documents.

### Example Filter Usage

The [vertex_ai_search_tool.py](../filters/vertex_ai_search_tool.py) filter enables Vertex AI Search grounding when the `vertex_ai_search` feature is requested:

```python
# filters/vertex_ai_search_tool.py
# ... (filter code) ...
```

To use this filter, ensure it's enabled in your Open WebUI configuration. Then, in your chat settings or via metadata, you can enable the `vertex_ai_search` feature:

```json
{
  "features": {
    "vertex_ai_search": true
  },
  "params": {
    "vertex_rag_store": "projects/your-project/locations/global/collections/default_collection/dataStores/your-data-store-id"
  }
}
```

## Native tool calling support

Native tool calling is enabled/disabled via the standard 'Function calling' Open Web UI toggle.

## Default System Prompt

The Google Gemini pipeline supports a configurable default system prompt that is applied to all chats. This is useful when you want to consistently apply certain behaviors or instructions to all Gemini models without having to configure each model individually.

### How It Works

- **Default Only**: If only `GOOGLE_DEFAULT_SYSTEM_PROMPT` is set and no user-defined system prompt exists, the default prompt is used as the system instruction.
- **User Only**: If only a user-defined system prompt exists (from model settings), it is used as-is.
- **Both**: If both are set, the default system prompt is **prepended** to the user-defined prompt, separated by a blank line. This allows you to have base instructions that apply to all chats while still allowing model-specific customization.

### Configuration

Set via environment variable:

```bash
# Default system prompt applied to all chats
# If a user-defined system prompt exists, this is prepended to it
GOOGLE_DEFAULT_SYSTEM_PROMPT="You are a helpful AI assistant. Always be concise and accurate."
```

Or configure through the pipeline valves in Open WebUI's Admin panel.

### Example

If your default system prompt is:

```
You are a helpful AI assistant.
```

And your model-specific system prompt is:

```
Always respond in formal English.
```

The combined system prompt sent to Gemini will be:

```
You are a helpful AI assistant.

Always respond in formal English.
```

## Thinking Configuration

The Google Gemini pipeline supports advanced thinking configuration to control how much reasoning and computation is applied by the model.

> [!Note]
> For detailed information about thinking capabilities, see the [Google Gemini Thinking Documentation](https://ai.google.dev/gemini-api/docs/thinking).

### Thinking Levels (Gemini 3 models)

Gemini 3 models support the `thinking_level` parameter, which controls the depth of reasoning:

- **`"low"`**: Minimizes latency and cost, suitable for simple tasks, chat, or high-throughput APIs.
- **`"high"`**: Maximizes reasoning depth, ideal for complex problem-solving, code analysis, and agentic workflows.

> [!Note]
> Gemini 3 models use `thinking_level` and do **not** use `thinking_budget`. The thinking budget setting is ignored for Gemini 3 models.

Set via environment variable:

```bash
# Use low thinking level for faster responses
GOOGLE_THINKING_LEVEL="low"

# Use high thinking level for complex reasoning
GOOGLE_THINKING_LEVEL="high"
```

#### Per-Chat Override (Reasoning Effort)

The per-chat `reasoning_effort` value can override the environment-level `GOOGLE_THINKING_LEVEL` setting. When a chat specifies a `reasoning_effort` value (e.g., "low" or "high"), it takes precedence over the global environment setting. This allows users to customize reasoning depth on a per-conversation basis.

**Example API Usage:**

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="Provide a list of 3 famous physicists and their key contributions",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="low")
    ),
)

print(response.text)
```

### Thinking Budget (Gemini 2.5 models)

For Gemini 2.5 models, you can control the maximum number of tokens used during internal reasoning using `thinking_budget`:

- **`0`**: Disables thinking entirely for fastest responses
- **`-1`**: Dynamic thinking (model decides based on query complexity) - default
- **`1-32768`**: Fixed token limit for reasoning

> [!Note]
> Gemini 3 models do **not** use `thinking_budget`. Use `GOOGLE_THINKING_LEVEL` for Gemini 3 models instead.

Set via environment variable:

```bash
# Disable thinking for fastest responses
GOOGLE_THINKING_BUDGET=0

# Use dynamic thinking (model decides)
GOOGLE_THINKING_BUDGET=-1

# Set a specific token budget for reasoning
GOOGLE_THINKING_BUDGET=1024
```

#### Per-Chat Override (thinking_budget)

Similar to `reasoning_effort` for Gemini 3 models, the per-chat `thinking_budget` value can override the environment-level `GOOGLE_THINKING_BUDGET` setting. When a chat request includes a `thinking_budget` value, it takes precedence over the global environment setting. This allows users to customize the thinking budget on a per-conversation basis.

**Example API Usage:**

```python
from google import genai
from google.genai import types

client = genai.Client()

# Example with a specific thinking budget
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Provide a list of 3 famous physicists and their key contributions",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=1024)
    ),
)
print(response.text)

# Turn off thinking entirely
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="What is 2+2?",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    ),
)
print(response.text)

# Use dynamic thinking (model decides based on query complexity)
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Explain quantum computing",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1)
    ),
)
print(response.text)
```

### Model Compatibility

| Model                     | thinking_level               | thinking_budget        |
| ------------------------- | ---------------------------- | ---------------------- |
| gemini-3-\*               | ✅ Supported ("low", "high") | ❌ Not used            |
| gemini-2.5-\*             | ❌ Not used                  | ✅ Supported (0-32768) |
| gemini-2.5-flash-image-\* | ❌ Not supported             | ❌ Not supported       |
| Other models              | ❌ Not used                  | ✅ May be supported    |
