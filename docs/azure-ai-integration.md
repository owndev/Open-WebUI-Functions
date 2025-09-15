# Azure AI Integration

The repository includes functions specifically designed for **Azure AI**, supporting both **Azure OpenAI** models and general **Azure AI** services.

🔗 [Learn More About Azure AI](https://azure.microsoft.com/en-us/solutions/ai)

## Pipeline

- 🧩 [Azure AI Foundry Pipeline](../pipelines/azure/azure_ai_foundry.py)

### Features

- **Azure OpenAI API Support**  
  Access models like **GPT-4o, o3**, and **other fine-tuned AI models** via Azure.

- **Azure AI Model Deployment**  
  Connect to **custom models** hosted on Azure AI.

- **Secure API Requests**  
  Supports API key authentication and environment variable configurations.

### Environment Variables

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

# Azure Search / RAG Configuration (optional)
# Azure Search endpoint for document retrieval
AZURE_SEARCH_ENDPOINT="https://your-search-service.search.windows.net"

# Azure Search index name containing the documents
AZURE_SEARCH_INDEX_NAME="your-index-name"

# Azure Search project resource ID (optional)
AZURE_SEARCH_PROJECT_RESOURCE_ID="your-project-resource-id"

# Azure Search API key (if using api_key authentication)
AZURE_SEARCH_KEY="your-search-api-key"

# Authentication type for Azure Search
AZURE_SEARCH_AUTHENTICATION_TYPE="system_assigned_managed_identity"

# Semantic configuration name for Azure Search
AZURE_SEARCH_SEMANTIC_CONFIGURATION="azureml-default"

# Azure Search embedding endpoint (optional)
AZURE_SEARCH_EMBEDDING_ENDPOINT="your-embedding-endpoint"

# Azure Search embedding API key (optional)
AZURE_SEARCH_EMBEDDING_KEY="your-embedding-key"

# Query type for Azure Search
AZURE_SEARCH_QUERY_TYPE="vectorSimpleHybrid"

# Whether to limit search to indexed documents only
AZURE_SEARCH_IN_SCOPE=false

# Role information for Azure Search responses
AZURE_SEARCH_ROLE_INFORMATION="You are an AI assistant."

# Azure Search strictness level (1-5)
AZURE_SEARCH_STRICTNESS=5

# Number of top documents to retrieve
AZURE_SEARCH_TOP_N_DOCUMENTS=20
```

### Azure Search / RAG Integration

The pipeline now supports **Azure Search** integration for **Retrieval-Augmented Generation (RAG)**. When configured, the pipeline will automatically include a `data_sources` field in requests to Azure AI, enabling document-based AI responses.

#### Configuration

Configure Azure Search by setting the following environment variables:

- **AZURE_SEARCH_ENDPOINT**: Your Azure Search service endpoint
- **AZURE_SEARCH_INDEX_NAME**: Name of the search index containing your documents
- **AZURE_SEARCH_AUTHENTICATION_TYPE**: Authentication method (`system_assigned_managed_identity` or `api_key`)
- **AZURE_SEARCH_KEY**: API key (if using `api_key` authentication)

#### Optional Settings

- **AZURE_SEARCH_PROJECT_RESOURCE_ID**: Project resource ID
- **AZURE_SEARCH_SEMANTIC_CONFIGURATION**: Semantic configuration name
- **AZURE_SEARCH_EMBEDDING_ENDPOINT**: Embedding service endpoint
- **AZURE_SEARCH_EMBEDDING_KEY**: Embedding service API key
- **AZURE_SEARCH_QUERY_TYPE**: Query type (`vectorSimpleHybrid`, `vector`, `semantic`)
- **AZURE_SEARCH_IN_SCOPE**: Limit to indexed documents only
- **AZURE_SEARCH_ROLE_INFORMATION**: Role information for responses
- **AZURE_SEARCH_STRICTNESS**: Strictness level (1-5)
- **AZURE_SEARCH_TOP_N_DOCUMENTS**: Number of documents to retrieve

> [!TIP]  
> To use **Azure OpenAI** and other **Azure AI** models **simultaneously**, you can use the following URL: `https://<your project>.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview`
