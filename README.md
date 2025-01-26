# Open-WebUI-Azure

**Open-WebUI-Azure** is a Python-based pipeline designed to integrate Azure AI services into the [Open WebUI](https://github.com/open-webui) ecosystem. This flexible pipeline supports both Azure OpenAI and other Azure AI models, providing a robust and configurable solution for AI-driven applications.

## Features

- **Azure OpenAI Support**: Seamlessly connect to Azure OpenAI endpoints for model inferencing.
- **Support for Other Azure AI Models**: Works with various Azure AI models by dynamically configuring headers and endpoints.
- **Dynamic Model Specification**: Specify models via the `x-ms-model-mesh-model-name` header or environment variables.
- **Parameter Filtering**: Only sends valid parameters to the Azure API to ensure clean and efficient requests.
- **Streaming and Non-Streaming Support**: Handles both streaming and traditional responses for a wide range of use cases.
- **Flexible Configuration**: Set API keys, endpoints, and models through environment variables.

---

## Prerequisites

To use this pipeline, ensure the following:

1. **Azure AI Services**: A valid [Azure AI](https://ai.azure.com/) subscription and access to your desired models.
2. **Open WebUI**: A Open WebUI instance. Refer to the [Open WebUI]([https://github.com/open-webui](https://github.com/open-webui/open-webui)) for setup instructions.

---

## Installation

To install and configure the Azure AI Foundry Pipeline as a function in Open WebUI, follow these steps:

1. **Ensure Admin Access**:
   - You must have admin privileges to add new functions in Open WebUI.

2. **Access Admin Settings**:
   - Navigate to the **Admin Settings** section in Open WebUI.

3. **Go to the Function Tab**:
   - In the admin panel, open the **Functions** tab to manage available functions.

4. **Create a New Function**:
   - Click **Add New Function**.
   - Copy the pipeline code from this repository and paste it into the code section of the new function.

5. **Set Environment Variables**:
   - After creating the function, configure the required environment variables in the function's settings:
     - `AZURE_AI_API_KEY`: Your Azure API key.
     - `AZURE_AI_ENDPOINT`: The endpoint for your Azure AI service.
     - `AZURE_AI_MODEL`: (Optional) The model to use.

6. **Save and Activate**:
   - Save the function, and it will be ready for use within Open WebUI.

---

## Funding
Support the development of this project by donating:
- Bitcoin (BTC): `bc1qjkcp5acdnvnhyqtezwrqaeyq8542rgdcxx728z`
- Ethereum (ETH): `0x5bD03A83dD470568e56E465f1D4B0f0Ff930E49C`
