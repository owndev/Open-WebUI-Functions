# Azure AI Integration

The repository includes functions specifically designed for **Azure AI**, supporting both **Azure OpenAI** models and general **Azure AI** services.

🔗 [Learn More About Azure AI](https://own.dev/azure-microsoft-com-en-us-solutions-ai)


## Pipeline
- 🧩 [Azure AI Foundry Pipeline](https://own.dev/github-owndev-open-webui-functions-azure-ai-foundry)


### Features:

- **Azure OpenAI API Support**  
  Access models like **GPT-4o, o3**, and **other fine-tuned AI models** via Azure.

- **Azure AI Model Deployment**  
  Connect to **custom models** hosted on Azure AI.

- **Secure API Requests**  
  Supports API key authentication and environment variable configurations.


### Environment Variables:

Configure the following environment variables to enable Azure AI support:

```bash
# API key or token for Azure AI
AZURE_AI_API_KEY="your-api-key"

# Azure AI endpoint
# Examples:
# - For general Azure AI: "https://<your-endpoint>/chat/completions?api-version=2024-05-01-preview"
# - For Azure OpenAI: "https://<your-endpoint>/openai/deployments/<model-name>/chat/completions?api-version=2024-08-01-preview"
AZURE_AI_ENDPOINT="https://<your project>.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"

# Optional: model names (if not embedded in the URL)
# Supports semicolon or comma separated values: "gpt-4o;gpt-4o-mini" or "gpt-4o,gpt-4o-mini"
AZURE_AI_MODEL="gpt-4o;gpt-4o-mini"

# If true, the model name will be included in the request body
AZURE_AI_MODEL_IN_BODY=true

# Whether to use a predefined list of Azure AI models
USE_PREDEFINED_AZURE_AI_MODELS=false

# If true, use "Authorization: Bearer" instead of "api-key" header
AZURE_AI_USE_AUTHORIZATION_HEADER=false
```

> [!TIP]  
> To use **Azure OpenAI** and other **Azure AI** models **simultaneously**, you can use the following URL: `https://<your project>.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview`