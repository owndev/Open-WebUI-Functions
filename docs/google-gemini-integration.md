# Google Gemini Integration

This integration enables **Open WebUI** to interact with **Google Gemini** via the official Google Generative AI API. It provides a robust and customizable pipeline to access text and multimodal generation capabilities from Google’s latest AI models.

🔗 [Learn More About Google AI](https://own.dev/ai-google-dev)

## Pipeline

- 🧩 [Google Gemini Pipeline](https://own.dev/github-owndev-open-webui-functions-google-gemini)

## Features

- **Asynchronous API Calls**  
  Improves performance and scalability with non-blocking requests.

- **Model Caching**  
  Caches available model lists for faster subsequent access.

- **Dynamic Model Handling**  
  Automatically strips provider prefixes for seamless integration.

- **Streaming Response Support**  
  Handles token-by-token responses with built-in safety enforcement.

- **Multimodal Input Support**  
  Accepts both text and image data for more expressive interactions.

- **Flexible Error Handling**  
  Retries failed requests and logs errors for transparency.

- **Secure API Key Storage**  
  API keys are encrypted and never exposed in plaintext.

- **Safety Configuration**  
  Control safety behavior using a permissive or strict mode.

- **Customizable Generation Settings**  
  Use environment variables to configure token limits, temperature, etc.

## Environment Variables

Set the following environment variables to configure the Google Gemini integration:

```bash
# API key for authenticating with Google Generative AI
GOOGLE_API_KEY="your-google-api-key"

# Use permissive safety settings for content generation (true/false)
USE_PERMISSIVE_SAFETY=false

# Model list cache duration (in seconds)
GOOGLE_MODEL_CACHE_TTL=600

# Number of retry attempts for failed API calls
GOOGLE_RETRY_COUNT=2
```

> [!TIP]  
> You can obtain your API key from the [Google AI Studio](https://own.dev/aistudio-google-com) dashboard after signing up.