# Open-WebUI-Functions

![GitHub stars](https://img.shields.io/github/stars/owndev/Open-WebUI-Functions?style=social)
![GitHub forks](https://img.shields.io/github/forks/owndev/Open-WebUI-Functions?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/owndev/Open-WebUI-Functions?style=social)
![GitHub top language](https://img.shields.io/github/languages/top/owndev/Open-WebUI-Functions)
![GitHub contributors](https://img.shields.io/github/contributors/owndev/Open-WebUI-Functions)
![GitHub License](https://img.shields.io/github/license/owndev/Open-WebUI-Functions)

![Main Version](https://owndev-public.github.io/badges/owndev/Open-WebUI-Functions/version/main.svg)

[![GitVersion Badge](https://github.com/owndev/Open-WebUI-Functions/actions/workflows/gitversion-badge.yml/badge.svg)](https://github.com/owndev/Open-WebUI-Functions/actions/workflows/gitversion-badge.yml)
[![GitHub Release](https://github.com/owndev/Open-WebUI-Functions/actions/workflows/github-release.yml/badge.svg)](https://github.com/owndev/Open-WebUI-Functions/actions/workflows/github-release.yml)

**Open-WebUI-Functions** is a collection of Python-based functions that extend [Open WebUI](https://github.com/open-webui/open-webui) with additional **pipelines**, **filters**, and **integrations**. These functions make it easier to connect external AI providers, process data, and tailor the Open WebUI experience to real-world workflows.

## 📚 Contents

- [🔎 Overview](#-overview)
- [🏷️ Version](#️-version)
- [✨ Features](#-features)
- [🏗️ Project structure](#️-project-structure)
- [🔗 Prerequisites](#-prerequisites)
- [🚀 Installation](#-installation)
- [🛡️ Security features](#️-security-features)
- [🧩 Pipelines](#-pipelines)
- [🔍 Filters](#-filters)
- [🤝 Integrations](#-integrations)
- [💪 Contributing](#-contributing)
- [📜 License](#-license)
- [💬 Support](#-support)
- [💫 Star history](#-star-history)

## 🔎 Overview

This repository focuses on reusable Python functions for Open WebUI. It includes provider-specific pipelines, request and response filters, optional analytics helpers, secure secret handling, and both streaming and non-streaming integrations.

## 🏷️ Version

| Branch | Version |
| --- | --- |
| **`main`** | ![Main Version](https://owndev-public.github.io/badges/owndev/Open-WebUI-Functions/version/main.svg) |
| **`hotfix/*`** | ![Hotfix Version](https://owndev-public.github.io/badges/owndev/Open-WebUI-Functions/version/hotfix.svg) |
| **`release/*`** | ![Release Version](https://owndev-public.github.io/badges/owndev/Open-WebUI-Functions/version/release.svg) |
| **`dev`** | ![Dev Version](https://owndev-public.github.io/badges/owndev/Open-WebUI-Functions/version/dev.svg) |
| **`feature/*`** | ![Feature Version](https://owndev-public.github.io/badges/owndev/Open-WebUI-Functions/version/feature.svg) |

> These badges are generated and updated automatically by the [GitVersion Badge workflow](https://github.com/owndev/Open-WebUI-Functions/actions/workflows/gitversion-badge.yml) for all GitFlow branch types.

## ✨ Features

- 🧩 **Custom pipelines**: Extend Open WebUI with AI processing pipelines, including model inference and data transformations.
- 🔍 **Filters for data processing**: Apply custom filtering logic to refine, manipulate, or preprocess input and output data.
- 🤝 **Azure AI support**: Seamlessly connect Open WebUI with **Azure OpenAI** and other **Azure AI** models.
- 🤝 **N8N workflow integration**: Enable interactions with [N8N](https://n8n.io/) for automation.
- 📱 **Flexible configuration**: Use environment variables to adjust function settings dynamically.
- 🚀 **Streaming and non-streaming support**: Handle both real-time and batch processing efficiently.
- 🛡️ **Secure API key management**: Automatically encrypt sensitive information such as API keys.

## 🏗️ Project structure

```text
.
├── docs/
│   ├── azure-ai-citations.md
│   ├── azure-ai-integration.md
│   ├── google-gemini-integration.md
│   ├── infomaniak-integration.md
│   ├── n8n-integration.md
│   ├── n8n-tool-usage-display.md
│   └── setup-azure-log-analytics.md
├── filters/
│   ├── google_search_tool.py
│   ├── time_token_tracker.py
│   └── vertex_ai_search_tool.py
└── pipelines/
    ├── azure/
    ├── google/
    ├── infomaniak/
    └── n8n/
```

## 🔗 Prerequisites

> [!IMPORTANT]
> To use these functions, make sure the following requirements are met:
>
> 1. **An active Open WebUI instance**: You must have [Open WebUI](https://github.com/open-webui/open-webui) installed and running.
> 2. **Required AI services (if applicable)**: Some pipelines depend on external AI services, such as [Azure AI](https://ai.azure.com/).
> 3. **Admin access**: You must have administrator privileges in Open WebUI to install functions.

## 🚀 Installation

> [!TIP]
> Follow these steps to install and configure functions in Open WebUI.

1. **Ensure admin access**

   > [!NOTE]
   > You must be an admin in Open WebUI to install functions.

2. **Open Admin Settings**
   - Navigate to the **Admin Settings** section in Open WebUI.

3. **Open the Functions tab**
   - Go to the **Functions** tab in the admin panel.

4. **Create a new function**
   - Click **Add New Function**.
   - Copy the function code from this repository and paste it into the function editor.

5. **Set environment variables if required**
   - Some functions require API keys or provider-specific configuration through environment variables.

   > [!IMPORTANT]
   > Set [WEBUI_SECRET_KEY](https://docs.openwebui.com/getting-started/env-configuration/#webui_secret_key) for secure encryption of sensitive API keys. This is **required** for the encryption features to work properly.

6. **Save and activate**
   - Save the function, and it will be available inside Open WebUI.

## 🛡️ Security Features

> [!WARNING]
> **API key security**: Always use encryption for sensitive information such as API keys.

### API key encryption

The functions include a built-in encryption mechanism for sensitive information:

- **Automatic encryption**: API keys and other sensitive data are automatically encrypted when stored.
- **Encrypted storage**: Values are stored with an `encrypted:` prefix followed by the encrypted data.
- **Transparent usage**: Encryption and decryption happen automatically when values are accessed.
- **No extra configuration required**: Everything works out of the box when [WEBUI_SECRET_KEY](https://docs.openwebui.com/getting-started/env-configuration/#webui_secret_key) is set.

> [!IMPORTANT]
> To enable encryption, set the `WEBUI_SECRET_KEY` environment variable:
>
> ```bash
> # Set this in your Open WebUI environment or .env file
> WEBUI_SECRET_KEY="your-secure-random-string"
> ```

## 🧩 Pipelines

> [!NOTE]
> Pipelines are processing functions that extend Open WebUI with **custom AI models**, **external integrations**, and **data manipulation logic**.

### 1. [Azure AI Foundry Pipeline](./pipelines/azure/azure_ai_foundry.py)

> [!TIP]
> **Azure OpenAI integration made easy**
>
> This pipeline provides seamless integration with Azure OpenAI and other Azure AI models, with advanced features such as Azure Search integration and multiple model support.

- Enables interaction with **Azure OpenAI** and other **Azure AI** models.
- Supports Azure Search / RAG integration for enhanced document retrieval (Azure OpenAI only).
- **Native OpenWebUI citations support** 🎯: Rich citation cards, source previews, relevance scores, and automatic `[docX]` → clickable markdown link conversion (Azure OpenAI only).
- **Relevance scores**: BM25 keyword and semantic rerank scores from Azure AI Search are displayed as a relevance percentage on citation cards, with independently configurable normalization via `BM25_SCORE_MAX` and `RERANK_SCORE_MAX`.
- Supports multiple models via `AZURE_AI_MODEL` (semicolon- or comma-separated, for example `gpt-4o;gpt-4o-mini`) or automatic model extraction from the Azure OpenAI URL.
- **Large predefined model catalogue** (GPT-4o, GPT-5, o3, o4-mini, Phi-4, DeepSeek-R1/V3, Mistral, Llama 3.x, Cohere, Grok, and more) via `USE_PREDEFINED_AZURE_AI_MODELS`.
- Customizable pipeline display prefix via `AZURE_AI_PIPELINE_PREFIX`.
- **Flexible authentication**: `api-key` header (default) or `Authorization: Bearer` token via `AZURE_AI_USE_AUTHORIZATION_HEADER`.
- **Token usage tracking**: Requests `stream_options.include_usage` in streaming mode so token counts are saved to the Open WebUI database.
- Filters valid parameters to ensure clean requests.
- Handles both streaming and non-streaming responses.
- Provides configurable error handling and timeouts.
- Supports encryption of sensitive information such as API keys.

🔗 [Azure AI Pipeline in Open WebUI](https://openwebui.com/f/owndev/azure_ai)

🔗 [Learn more about Azure AI](https://azure.microsoft.com/en-us/solutions/ai)

📖 [Azure AI citations documentation](./docs/azure-ai-citations.md)

### 2. [N8N Pipeline](./pipelines/n8n/n8n.py)

> [!TIP]
> **N8N workflow automation integration**
>
> Connect Open WebUI with N8N to leverage powerful workflow automation. It includes configurable AI agent tool usage display for better transparency into agent actions.

- Integrates **Open WebUI** with **N8N**, an automation and workflow platform.
- **AI agent tool usage display (v2.2.0)** 🛠️: Shows tool calls from N8N AI Agent workflows with three verbosity levels (minimal, compact, detailed) and customizable length limits (non-streaming mode only).
- Streaming and non-streaming support for real-time and batch data processing.
- Sends messages from Open WebUI to an **N8N webhook**.
- Supports real-time message processing with dynamic field handling.
- Enables automation of AI-generated responses inside an **N8N workflow**.
- Supports encryption of sensitive information such as API keys.
- Includes an example [N8N workflow](./pipelines/n8n/Open_WebUI_Test_Agent.json) for the [N8N Pipeline](./pipelines/n8n/n8n.py).

> [!IMPORTANT]
> **Tool usage display limitation**: The AI agent tool call display currently works only in **non-streaming mode** due to N8N's current streaming implementation. The code is future-proof and will work automatically when N8N adds `intermediateSteps` to streaming responses.

🔗 [N8N Pipeline in Open WebUI](https://openwebui.com/f/owndev/n8n_pipeline)

🔗 [Learn more about N8N](https://n8n.io/)

📖 [N8N tool usage display documentation](./docs/n8n-tool-usage-display.md)

### 3. [Infomaniak](./pipelines/infomaniak/infomaniak.py)

- Integrates **Open WebUI** with **Infomaniak**, a Swiss web hosting and cloud services provider.
- Sends messages from Open WebUI to an **Infomaniak AI Tool**.
- Supports encryption of sensitive information such as API keys.

🔗 [Infomaniak Pipeline in Open WebUI](https://openwebui.com/f/owndev/infomaniak_ai_tools)

🔗 [Learn more about Infomaniak](https://www.infomaniak.com/en/hosting/ai-tools)

### 4. [Google Gemini](./pipelines/google/google_gemini.py)

- Integrates **Open WebUI** with **Google Gemini**, a generative AI model by Google.
- Supports integration with the Google Generative AI API or Vertex AI API for content generation.
- Sends messages from Open WebUI to **Google Gemini**.
- Supports encryption of sensitive information such as API keys.
- Supports both streaming and non-streaming responses (streaming is automatically disabled for image generation models).
- **Thinking & reasoning**: Configurable thinking levels (`low` / `high`) for Gemini 3 models and thinking budgets (0–32 768 tokens) for Gemini 2.5 models, with per-chat override support.
- Provides configurable error handling and timeouts.
- **Advanced image processing**: Optimized image handling with configurable compression, resizing, and quality settings.
- **Configurable parameters**: Environment variables for image optimization (quality, max dimensions, format conversion).
- **Multi-image history**: Configurable history image limit, hash-based deduplication, and automatic `[Image N]` labels so the model can reference earlier images.
- **Image generation (Gemini 3)**: Configurable aspect ratio (for example `16:9` or `1:1`) and resolution (`1K`, `2K`, or `4K`) for Gemini 3 image models, with per-user valve overrides.
- **Video generation (Veo)**: Generate videos with Google Veo models (3.1, 3, 2). Configurable aspect ratio, resolution, duration, negative prompt, and person generation controls. Supports text-to-video and image-to-video for all supported Veo models. Videos are automatically uploaded and embedded with playback controls.
- **Token usage tracking**: Returns prompt, completion, and total token counts to Open WebUI for automatic persistence in the database.
- **Model whitelist & additional models**: Restrict the visible model list via `GOOGLE_MODEL_WHITELIST` and add SDK-unsupported models via `GOOGLE_MODEL_ADDITIONAL`.
- Grounding with Google Search via the [google_search_tool.py filter](./filters/google_search_tool.py)
- Grounding with Vertex AI Search via the [vertex_ai_search_tool.py filter](./filters/vertex_ai_search_tool.py)
- Native tool calling support
- Configurable API version support

🔗 [Google Gemini Pipeline in Open WebUI](https://openwebui.com/f/owndev/google_gemini)

🔗 [Learn more about Google Gemini](https://ai.google.dev/gemini-api/docs?hl=en)

> [!NOTE]
> **For LiteLLM users**: To use Google Gemini models through LiteLLM, configure LiteLLM directly in Open WebUI under **Admin Panel → Settings → Connections → OpenAI** instead of using this pipeline. For more information about LiteLLM, visit the [official LiteLLM GitHub repository](https://github.com/BerriAI/litellm).

## 🔍 Filters

> [!NOTE]
> Filters allow **preprocessing and post-processing** of data within Open WebUI.

### 1. [Time Token Tracker](./filters/time_token_tracker.py)

> [!NOTE]
> **Performance monitoring for AI interactions**
>
> Track response times, token usage, and optionally send analytics to Azure Log Analytics for more complete observability.

- Measures **response time** and **token usage** for AI interactions.
- Supports tracking of **total token usage** and **per-message token counts**.
- Can calculate token usage for all messages or only a subset.
- Uses OpenAI's `tiktoken` library for token counting (accurate only for OpenAI models).
- Optionally sends logs to an [Azure Log Analytics Workspace](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/log-analytics-workspace-overview).

🔗 [Time Token Tracker in Open WebUI](https://openwebui.com/f/owndev/time_token_tracker)

🔗 [How to set up Azure Log Analytics](./docs/setup-azure-log-analytics.md)

## 🤝 Integrations

### Azure AI

See the [Azure AI integration guide](./docs/azure-ai-integration.md).

### N8N

See the [N8N integration guide](./docs/n8n-integration.md).

### Infomaniak

See the [Infomaniak integration guide](./docs/infomaniak-integration.md).

### Google

See the [Google Gemini integration guide](./docs/google-gemini-integration.md).

## 💪 Contributing

> [!TIP]
> We welcome contributions of all kinds. You do not need to write code to contribute.
>
> For detailed onboarding and contribution guidance, see [CONTRIBUTING.md](./.github/CONTRIBUTING.md).

## 📜 License

This project is licensed under the [Apache License 2.0](./LICENSE). See the [LICENSE](./LICENSE) file for details.

## 💬 Support

> [!NOTE]
> If you have any questions, suggestions, or need assistance, please open an [issue](https://github.com/owndev/Open-WebUI-Functions/issues/new/choose) to connect with us. 🤝

## 💫 Star History

<a href="https://star-history.com/#owndev/Open-WebUI-Functions&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=owndev/Open-WebUI-Functions&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=owndev/Open-WebUI-Functions&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=owndev/Open-WebUI-Functions&type=Date" />
  </picture>
</a>

---

Created by [owndev](https://github.com/owndev) — let's make Open WebUI even more amazing together. 💪
