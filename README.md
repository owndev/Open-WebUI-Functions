# Open-WebUI-Functions

**Open-WebUI-Functions** is a collection of Python-based functions designed to extend the capabilities of [Open WebUI](https://github.com/open-webui) with additional **pipelines**, **filters**, and **integrations**. These functions allow users to interact with various AI models, process data efficiently, and customize the Open WebUI experience.

---

## Features

- **Custom Pipelines**: Extend Open WebUI with AI processing pipelines, including model inference and data transformations.
- **Filters for Data Processing**: Apply custom filtering logic to refine, manipulate, or preprocess input and output data.
- **Azure AI Support**: Seamlessly connect Open WebUI with **Azure OpenAI** and other **Azure AI** models.
- **N8N Workflow Integration**: Enable interactions with [N8N](https://n8n.io/) for automation.
- **Flexible Configuration**: Use environment variables to adjust function settings dynamically.
- **Streaming and Non-Streaming Support**: Handle both real-time and batch processing efficiently.

---

## Prerequisites

To use these functions, ensure the following:

1. **An Active Open WebUI Instance**: You must have [Open WebUI](https://github.com/open-webui/open-webui) installed and running.
2. **Required AI Services (if applicable)**: Some pipelines require external AI services, such as [Azure AI](https://ai.azure.com/).
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

6. **Save and Activate**:
   - Save the function, and it will be available for use within Open WebUI.

---

## Pipelines

Pipelines are processing functions that extend Open WebUI with **custom AI models**, **external integrations**, and **data manipulation logic**.

### **1. [Azure AI Foundry Pipeline](./pipelines/azure/azure_ai_foundry.py)**

- Enables interaction with **Azure OpenAI** and other **Azure AI** models.
- Supports dynamic model selection via `x-ms-model-mesh-model-name` headers.
- Filters valid parameters to ensure clean requests.
- Handles both streaming and non-streaming responses.
- Provides configurable error handling and timeouts.
- Predefined models for easy access.

🔗 [Azure AI Pipeline in Open WebUI](https://openwebui.com/f/owndev/azure_ai/)

### **2. [N8N Pipeline](./pipelines/n8n/n8n.py)**

- Integrates **Open WebUI** with **N8N**, an automation and workflow platform.
- Sends messages from Open WebUI to an **N8N webhook**.
- Supports real-time message processing with dynamic field handling.
- Enables automation of AI-generated responses within an **N8N workflow**.
- Here is an example [N8N workflow](./pipelines/n8n/Open_WebUI_Test_Agent.json) for [N8N Pipeline](./pipelines/n8n/n8n.py)

🔗 [N8N Pipeline in Open WebUI](https://openwebui.com/f/owndev/n8n_pipeline/)

🔗 [Learn More About N8N](https://n8n.io/)

### **3. [Infomaniak](./pipelines/infomaniak/infomaniak.py)**

- Integrates **Open WebUI** with **Infomaniak**, a Swiss web hosting and cloud services provider.
- Sends messages from Open WebUI to an **Infomaniak AI Tool**.

> **Important**: The function ID in Open WebUI must not contain the name `infomaniak`. Because of a [bug](https://github.com/open-webui/open-webui/discussions/10914) in Open WebUI, the function will not work if the id contains `infomaniak`.


🔗 [Infomaniak Pipeline in Open WebUI](https://openwebui.com/f/owndev/im_ai_tools/)

🔗 [Learn More About Infomaniak](https://www.infomaniak.com/en/hosting/ai-tools)

---

## Filters

Filters allow for **preprocessing and postprocessing** of data within Open WebUI.

### **1. [Time Token Tracker](./filters/time_token_tracker.py)**

- Measures **response time** and **token usage** for AI interactions.
- Supports tracking of **total token usage** and **per-message token counts**.
- Can calculate token usage for all messages or only a subset.
- Uses OpenAI's `tiktoken` library for accurate token counting.

🔗 [Time Token Tracker in Open WebUI](https://openwebui.com/f/owndev/time_token_tracker)

---

## Azure AI Integration

The repository includes functions specifically designed for **Azure AI**, supporting both **Azure OpenAI** models and general **Azure AI** services.

### Features:
- **Azure OpenAI API Support**: Access models like **GPT-4, GPT-3.5**, and **other fine-tuned AI models** via Azure.
- **Azure AI Model Deployment**: Connect to **custom models** hosted on Azure AI.
- **Dynamic Model Selection**: Choose models via the `x-ms-model-mesh-model-name` header or environment variables.
- **Secure API Requests**: Supports API key authentication and environment variable configurations.

### Environment Variables:
For Azure AI-based functions, set the following:
```bash
AZURE_AI_API_KEY="your-api-key"
AZURE_AI_ENDPOINT="https://your-service.openai.azure.com/chat/completions?api-version=2024-05-01-preview"
AZURE_AI_MODEL="gpt-4o"  # Optional model name, only necessary if not Azure OpenAI or if model name not in URL (e.g. "https://<your-endpoint>/openai/deployments/<model-name>/chat/completions").
```

For further details, check the [Azure AI Function in Open WebUI.](https://openwebui.com/f/owndev/).

## Contribute

We welcome contributions from the community! To contribute, follow these steps:

1. **Fork the Repository**: Click the "Fork" button at the top right corner of this page to create a copy of this repository in your GitHub account.
2. **Clone Your Fork**: Clone your fork to your local machine using the following command:
   ```bash
   git clone https://github.com/<your-username>/Open-WebUI-Functions.git
   ```
3. **Create a New Branch**: Create a new branch for your changes:
   ```bash
   git checkout -b my-feature-branch
   ```
4. **Make Your Changes**: Make your changes to the code or documentation.
5. **Commit Your Changes**: Commit your changes with a meaningful commit message:
   ```bash
   git commit -m "Add new feature or fix bug"
   ```
6. **Push to Your Fork**: Push your changes to your forked repository:
   ```bash
   git push origin my-feature-branch
   ```
7. **Create a Pull Request**: Open a pull request on the original repository, describing your changes and why they should be merged.

Thank you for contributing!
