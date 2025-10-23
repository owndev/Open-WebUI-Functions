# Open-WebUI-Functions

![GitHub stars](https://img.shields.io/github/stars/owndev/Open-WebUI-Functions?style=social)
![GitHub forks](https://img.shields.io/github/forks/owndev/Open-WebUI-Functions?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/owndev/Open-WebUI-Functions?style=social)
![GitHub top language](https://img.shields.io/github/languages/top/owndev/Open-WebUI-Functions)
![GitHub contributors](https://img.shields.io/github/contributors/owndev/Open-WebUI-Functions)
![GitHub License](https://img.shields.io/github/license/owndev/Open-WebUI-Functions)

**Open-WebUI-Functions** is a collection of Python-based functions designed to extend the capabilities of [Open WebUI](https://github.com/open-webui/open-webui) with additional **pipelines**, **filters**, and **integrations**. These functions allow users to interact with various AI models, process data efficiently, and customize the Open WebUI experience.

## Features ⭐

- 🧩 **Custom Pipelines**: Extend Open WebUI with AI processing pipelines, including model inference and data transformations.

- 🔍 **Filters for Data Processing**: Apply custom filtering logic to refine, manipulate, or preprocess input and output data.

- 🤝 **Azure AI Support**: Seamlessly connect Open WebUI with **Azure OpenAI** and other **Azure AI** models.

- 🤝 **N8N Workflow Integration**: Enable interactions with [N8N](https://n8n.io/) for automation.

- 📱 **Flexible Configuration**: Use environment variables to adjust function settings dynamically.

- 🚀 **Streaming and Non-Streaming Support**: Handle both real-time and batch processing efficiently.

- 🛡️ **Secure API Key Management**: Automatic encryption of sensitive information like API keys.

## Prerequisites 🔗

> [!IMPORTANT]
> To use these functions, ensure the following requirements are met:
>
> 1. **An Active Open WebUI Instance**: You must have [Open WebUI](https://github.com/open-webui/open-webui) installed and running.
> 2. **Required AI Services (if applicable)**: Some pipelines require external AI services, such as [Azure AI](https://ai.azure.com/).
> 3. **Admin Access**: To install functions in Open WebUI, you must have administrator privileges.

## Installation 🚀

> [!TIP]
> Follow these steps to install and configure functions in Open WebUI:

1. **Ensure Admin Access**:

> [!NOTE]
> You must be an admin in Open WebUI to install functions.

2. **Access Admin Settings**:
   - Navigate to the **Admin Settings** section in Open WebUI.

3. **Go to the Function Tab**:
   - Open the **Functions** tab in the admin panel.

4. **Create a New Function**:
   - Click **Add New Function**.
   - Copy the function code from this repository and paste it into the function editor.

5. **Set Environment Variables (if required)**:
   - Some functions require API keys or specific configurations via environment variables.

> [!IMPORTANT]
> Set [WEBUI_SECRET_KEY](https://docs.openwebui.com/getting-started/env-configuration/#webui_secret_key) for secure encryption of sensitive API keys. This is **required** for the encryption features to work properly.

6. **Save and Activate**:
   - Save the function, and it will be available for use within Open WebUI.

## Security Features 🛡️

> [!WARNING]
> **API Key Security**: Always use encryption for sensitive information like API keys!

### API Key Encryption

The functions include a built-in encryption mechanism for sensitive information:

- **Automatic Encryption**: API keys and other sensitive data are automatically encrypted when stored.
- **Encrypted Storage**: Values are stored with an "encrypted:" prefix followed by the encrypted data.
- **Transparent Usage**: The encryption/decryption happens automatically when values are accessed.
- **No Configuration Required**: Works out-of-the-box when [WEBUI_SECRET_KEY](https://docs.openwebui.com/getting-started/env-configuration/#webui_secret_key) is set.

> [!IMPORTANT]
> **To enable encryption**, set the `WEBUI_SECRET_KEY` environment variable:
>
> ```bash
> # Set this in your Open WebUI environment or .env file
> WEBUI_SECRET_KEY="your-secure-random-string"
> ```

## Pipelines 🧩

> [!NOTE]
> Pipelines are processing functions that extend Open WebUI with **custom AI models**, **external integrations**, and **data manipulation logic**.

### **1. [Azure AI Foundry Pipeline](./pipelines/azure/azure_ai_foundry.py)**

> [!TIP]
> **Azure OpenAI Integration Made Easy**
>
> This pipeline provides seamless integration with Azure OpenAI and other Azure AI models with advanced features like Azure Search integration and multiple model support.

- Enables interaction with **Azure OpenAI** and other **Azure AI** models.
- Supports Azure Search integration for enhanced document retrieval.
- Supports multiple Azure AI models selection via the `AZURE_AI_MODEL` environment variable (e.g. `gpt-4o;gpt-4o-mini`).
- Customizable pipeline display with configurable prefix via `AZURE_AI_PIPELINE_PREFIX`.
- Azure AI Search / RAG integration with enhanced collapsible citation display (Azure OpenAI only).
- Filters valid parameters to ensure clean requests.
- Handles both streaming and non-streaming responses.
- Provides configurable error handling and timeouts.
- Predefined models for easy access.
- Supports encryption of sensitive information like API keys.

🔗 [Azure AI Pipeline in Open WebUI](https://openwebui.com/f/owndev/azure_ai)

🔗 [Learn More About Azure AI](https://azure.microsoft.com/en-us/solutions/ai)

### **2. [N8N Pipeline](./pipelines/n8n/n8n.py)**

> [!TIP]
> **N8N Workflow Automation Integration**
>
> Connect Open WebUI with N8N to leverage powerful workflow automation. Includes configurable AI Agent tool usage display for complete transparency into your agent's actions.

- Integrates **Open WebUI** with **N8N**, an automation and workflow platform.
- **AI Agent Tool Usage Display (v2.2.0)** 🛠️: Shows tool calls from N8N AI Agent workflows with three verbosity levels (minimal, compact, detailed) and customizable length limits (non-streaming mode only).
- Streaming and non-streaming support for real-time and batch data processing.
- Sends messages from Open WebUI to an **N8N webhook**.
- Supports real-time message processing with dynamic field handling.
- Enables automation of AI-generated responses within an **N8N workflow**.
- Supports encryption of sensitive information like API keys.
- Here is an example [N8N workflow](./pipelines/n8n/Open_WebUI_Test_Agent.json) for [N8N Pipeline](./pipelines/n8n/n8n.py)

> [!IMPORTANT]
> **Tool Usage Display Limitation**: The AI Agent tool call display currently only works in **non-streaming mode** due to N8N's current streaming implementation. The code is future-proof and will automatically work when N8N adds `intermediateSteps` to streaming responses.

🔗 [N8N Pipeline in Open WebUI](https://openwebui.com/f/owndev/n8n_pipeline)

🔗 [Learn More About N8N](https://n8n.io/)

📖 [N8N Tool Usage Display Documentation](./docs/n8n-tool-usage-display.md)

### **3. [Infomaniak](./pipelines/infomaniak/infomaniak.py)**

- Integrates **Open WebUI** with **Infomaniak**, a Swiss web hosting and cloud services provider.
- Sends messages from Open WebUI to an **Infomaniak AI Tool**.
- Supports encryption of sensitive information like API keys.

🔗 [Infomaniak Pipeline in Open WebUI](https://openwebui.com/f/owndev/infomaniak_ai_tools)

🔗 [Learn More About Infomaniak](https://www.infomaniak.com/en/hosting/ai-tools)

### **4. [Google Gemini](./pipelines/google/google_gemini.py)**

- Integrates **Open WebUI** with **Google Gemini**, a generative AI model by Google.
- Integration with Google Generative AI or Vertex AI API for content generation.
- Sends messages from Open WebUI to **Google Gemini**.
- Supports encryption of sensitive information like API keys.
- Supports both streaming and non-streaming responses (streaming automatically disabled for image generation models).
- Supports thinking and reasoning capabilities.
- Provides configurable error handling and timeouts.
- **Advanced Image Processing**: Optimized image handling with configurable compression, resizing, and quality settings.
- **Configurable Parameters**: Environment variables for image optimization (quality, max dimensions, format conversion).
- Grounding with Google search with [google_search_tool.py filter](./filters/google_search_tool.py)
- Native tool calling support
- Configurable API version support

🔗 [Google Gemini Pipeline in Open WebUI](https://openwebui.com/f/owndev/google_gemini)

🔗 [Learn More About Google Gemini](https://ai.google.dev/gemini-api/docs?hl=de)

> [!NOTE]
> **For LiteLLM Users**: To use Google Gemini models through LiteLLM, configure LiteLLM directly in Open WebUI's Admin Panel → Settings → Connections → OpenAI section instead of using this pipeline. For more information about LiteLLM, visit the [official LiteLLM GitHub repository](https://github.com/BerriAI/litellm).

## Filters 🔍

> [!NOTE]
> Filters allow for **preprocessing and postprocessing** of data within Open WebUI.

### **1. [Time Token Tracker](./filters/time_token_tracker.py)**

> [!NOTE]
> **Performance Monitoring for AI Interactions**
>
> Track response times, token usage, and optionally send analytics to Azure Log Analytics for comprehensive monitoring.

- Measures **response time** and **token usage** for AI interactions.
- Supports tracking of **total token usage** and **per-message token counts**.
- Can calculate token usage for all messages or only a subset.
- Uses OpenAI's `tiktoken` library for token counting (only accurate for OpenAI models).
- Optional: Can send logs to [Azure Log Analytics Workspace](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/log-analytics-workspace-overview).

🔗 [Time Token Tracker in Open WebUI](https://openwebui.com/f/owndev/time_token_tracker)

🔗 [How to Setup Azure Log Analytics](./docs/setup-azure-log-analytics.md)

## Integrations 🤝

### Azure AI

Look here for [Azure AI Integration](./docs/azure-ai-integration.md).

### N8N

Look here for [N8N Integration](./docs/n8n-integration.md).

### Infomaniak

Look here for [Infomaniak Integration](./docs/infomaniak-integration.md).

### Google

Look here for [Google Gemini Integration](./docs/google-gemini-integration.md).

## Contribute 💪

> [!TIP]
> We welcome contributions of all kinds! You don't need to write code to contribute.
>
> For detailed instructions on how to get started with our project, see [about contributing to Open-WebUI-Functions](./.github/CONTRIBUTING.md).

## License 📜

This project is licensed under the [Apache License 2.0](./LICENSE) - see the [LICENSE](./LICENSE) file for details. 📄

## Support 💬

> [!NOTE]
> If you have any questions, suggestions, or need assistance, please open an [issue](../../issues/new/choose) to connect with us! 🤝

## Star History 💫

<a href="https://star-history.com/#owndev/Open-WebUI-Functions&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=owndev/Open-WebUI-Functions&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=owndev/Open-WebUI-Functions&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=owndev/Open-WebUI-Functions&type=Date" />
  </picture>
</a>

---

Created by [owndev](https://github.com/owndev) - Let's make Open WebUI even more amazing together! 💪
