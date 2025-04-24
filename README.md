# Open-WebUI-Functions

**Open-WebUI-Functions** is a collection of Python-based functions designed to extend the capabilities of [Open WebUI](https://own.dev/github-com-open-webui-open-webui) with additional **pipelines**, **filters**, and **integrations**. These functions allow users to interact with various AI models, process data efficiently, and customize the Open WebUI experience.

---

## Features

- **Custom Pipelines**: Extend Open WebUI with AI processing pipelines, including model inference and data transformations.
- **Filters for Data Processing**: Apply custom filtering logic to refine, manipulate, or preprocess input and output data.
- **Azure AI Support**: Seamlessly connect Open WebUI with **Azure OpenAI** and other **Azure AI** models.
- **N8N Workflow Integration**: Enable interactions with [N8N](https://own.dev/n8n-io) for automation.
- **Flexible Configuration**: Use environment variables to adjust function settings dynamically.
- **Streaming and Non-Streaming Support**: Handle both real-time and batch processing efficiently.
- **Secure API Key Management**: Automatic encryption of sensitive information like API keys.

---

## Prerequisites

To use these functions, ensure the following:

1. **An Active Open WebUI Instance**: You must have [Open WebUI](https://own.dev/github-com-open-webui-open-webui) installed and running.
2. **Required AI Services (if applicable)**: Some pipelines require external AI services, such as [Azure AI](https://own.dev/ai-azure-com).
3. **Admin Access**: To install functions in Open WebUI, you must have administrator privileges.

---

## Installation

To install and configure functions in Open WebUI, follow these steps:

1. **Ensure Admin Access**:
   - You must be an admin in Open WebUI to install functions.

2. **Access Admin Settings**:
   - Navigate to the **Admin Settings** section in Open WebUI.

3. **Go to the Function Tab**:
   - Open the **Functions** tab in the admin panel.

4. **Create a New Function**:
   - Click **Add New Function**.
   - Copy the function code from this repository and paste it into the function editor.

5. **Set Environment Variables (if required)**:
   - Some functions require API keys or specific configurations via environment variables.
   - Set [WEBUI_SECRET_KEY](https://own.dev/docs-openwebui-com-getting-started-env-configuration-webui-secret-key) for secure encryption of sensitive API keys.

6. **Save and Activate**:
   - Save the function, and it will be available for use within Open WebUI.

---

## Security Features

### API Key Encryption

The functions include a built-in encryption mechanism for sensitive information:

- **Automatic Encryption**: API keys and other sensitive data are automatically encrypted when stored.
- **Encrypted Storage**: Values are stored with an "encrypted:" prefix followed by the encrypted data.
- **Transparent Usage**: The encryption/decryption happens automatically when values are accessed.
- **No Configuration Required**: Works out-of-the-box when [WEBUI_SECRET_KEY](https://own.dev/docs-openwebui-com-getting-started-env-configuration-webui-secret-key) is set.

To enable encryption:
```bash
# Set this in your Open WebUI environment or .env file
WEBUI_SECRET_KEY="your-secure-random-string"
```

---

## Pipelines

Pipelines are processing functions that extend Open WebUI with **custom AI models**, **external integrations**, and **data manipulation logic**.

### **1. [Azure AI Foundry Pipeline](https://own.dev/github-owndev-open-webui-functions-azure-ai-foundry)**

- Enables interaction with **Azure OpenAI** and other **Azure AI** models.
- Supports multiple Azure AI models selection via the `AZURE_AI_MODEL` environment variable (e.g. `gpt-4o, gpt-4o-mini`).
- Filters valid parameters to ensure clean requests.
- Handles both streaming and non-streaming responses.
- Provides configurable error handling and timeouts.
- Predefined models for easy access.
- Supports encryption of sensitive information like API keys.

🔗 [Azure AI Pipeline in Open WebUI](https://own.dev/openwebui-com-f-owndev-azure-ai)


### **2. [N8N Pipeline](https://own.dev/github-owndev-open-webui-functions-n8n-pipeline)**

- Integrates **Open WebUI** with **N8N**, an automation and workflow platform.
- Sends messages from Open WebUI to an **N8N webhook**.
- Supports real-time message processing with dynamic field handling.
- Enables automation of AI-generated responses within an **N8N workflow**.
- Supports encryption of sensitive information like API keys.
- Here is an example [N8N workflow](https://own.dev/github-owndev-open-webui-functions-open-webui-test-agent) for [N8N Pipeline](https://own.dev/github-owndev-open-webui-functions-n8n-pipeline)

🔗 [N8N Pipeline in Open WebUI](https://own.dev/openwebui-com-f-owndev-n8n-pipeline)

🔗 [Learn More About N8N](https://own.dev/n8n-io)


### **3. [Infomaniak](https://own.dev/github-owndev-open-webui-functions-infomaniak)**

- Integrates **Open WebUI** with **Infomaniak**, a Swiss web hosting and cloud services provider.
- Sends messages from Open WebUI to an **Infomaniak AI Tool**.
- Supports encryption of sensitive information like API keys.

🔗 [Infomaniak Pipeline in Open WebUI](https://own.dev/openwebui-com-f-owndev-infomaniak-ai-tools)

🔗 [Learn More About Infomaniak](https://own.dev/infomaniak-com-en-hosting-ai-tools)

---

## Filters

Filters allow for **preprocessing and postprocessing** of data within Open WebUI.

### **1. [Time Token Tracker](https://own.dev/github-owndev-open-webui-functions-time-token-tracker)**

- Measures **response time** and **token usage** for AI interactions.
- Supports tracking of **total token usage** and **per-message token counts**.
- Can calculate token usage for all messages or only a subset.
- Uses OpenAI's `tiktoken` library for token counting (only accurate for OpenAI models).
- Optional: Can send logs to [Azure Log Analytics Workspace](https://own.dev/learn-microsoft-com-en-us-azure-azure-monitor-logs-log-analytics-workspace-overview).

🔗 [How to Setup](https://own.dev/github-owndev-open-webui-functions-setup-azure-log-analytics)

🔗 [Time Token Tracker in Open WebUI](https://own.dev/openwebui-com-f-owndev-time-token-tracker)

---

## Azure AI Integration

The repository includes functions specifically designed for **Azure AI**, supporting both **Azure OpenAI** models and general **Azure AI** services.

### Features:
- **Azure OpenAI API Support**: Access models like **GPT-4o, o3**, and **other fine-tuned AI models** via Azure.
- **Azure AI Model Deployment**: Connect to **custom models** hosted on Azure AI.
- **Secure API Requests**: Supports API key authentication and environment variable configurations.

### Environment Variables:
For Azure AI-based functions, set the following:
```bash
AZURE_AI_API_KEY="your-api-key"
AZURE_AI_ENDPOINT="https://your-service.openai.azure.com/chat/completions?api-version=2024-05-01-preview"
AZURE_AI_MODEL="gpt-4o, gpt-4o-mini"  # Optional model name, only necessary if not Azure OpenAI or if model name not in URL (e.g. "https://<your-endpoint>/openai/deployments/<model-name>/chat/completions").
```

---

## Contribute

We accept different types of contributions, including some that don't require you to write a single line of code. 
For detailed instructions on how to get started with our project, see [About contributing to Open-WebUI-Functions](https://own.dev/github-owndev-open-webui-functions-contributing).
