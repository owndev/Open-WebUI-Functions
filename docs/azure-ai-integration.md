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
# Custom prefix for pipeline display name (default: "Azure AI")
# The colon ":" will be added automatically between prefix and model name
# Examples: "Azure AI" → "Azure AI: gpt-4o", "My Azure" → "My Azure: gpt-4o"
AZURE_AI_PIPELINE_PREFIX="Azure AI"

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

# Azure AI Search endpoint for document retrieval (Only for Azure OpenAI)
# IMPORTANT: Azure AI Search only works with Azure OpenAI endpoints in this format:
# https://<deployment>.openai.azure.com/openai/deployments/<model>/chat/completions?api-version=2025-01-01-preview
AZURE_AI_ENDPOINT="https://<deployment>.openai.azure.com/openai/deployments/<model>/chat/completions?api-version=2025-01-01-preview"

# Azure AI Data Sources / RAG Configuration
# Complete JSON configuration for Azure Search - copy exactly and replace placeholder values
AZURE_AI_DATA_SOURCES='[{"type":"azure_search","parameters":{"endpoint":"https://<your-search-service>.search.windows.net","index_name":"<your-index-name>","authentication":{"type":"api_key","key":"<your-search-api-key>"}}}]'

# Enable enhanced citation display for better readability (default: true)
AZURE_AI_ENHANCE_CITATIONS=true
```

### Azure AI Search / RAG Integration

The pipeline supports **Azure AI Search** integration for **Retrieval-Augmented Generation (RAG)**. When configured, the pipeline automatically includes a `data_sources` field in requests to Azure AI, enabling document-based AI responses that can cite and reference your indexed content.

> [!IMPORTANT]
> **Azure AI Search integration only works with Azure OpenAI endpoints** in this specific format:
> `https://<deployment>.openai.azure.com/openai/deployments/<model>/chat/completions?api-version=2025-01-01-preview`

#### 📖 Official Documentation

For detailed information about Azure AI Search configuration, please refer to:

- 📚 [Azure AI Search with Azure OpenAI - Official Guide](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/use-your-data-quickstart?tabs=api-key%2Ctypescript-keyless%2Cpython-new&pivots=rest-api)
- 🔧 [Data Sources API Reference](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/references/on-your-data?tabs=rest#data-source)
- 🔍 [Azure Search Parameters Reference](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/references/azure-search?tabs=rest)

#### ⚙️ Configuration

Configure Azure AI Search by setting the `AZURE_AI_DATA_SOURCES` environment variable with your Azure Search configuration.

**Simple Example:**

```bash
AZURE_AI_DATA_SOURCES='[{"type":"azure_search","parameters":{"endpoint":"https://my-search.search.windows.net","index_name":"my-index","authentication":{"type":"api_key","key":"your-search-api-key"}}}]'
```

> [!TIP]
> **Copy the JSON exactly as shown above** - this is the complete configuration that goes into the `AZURE_AI_DATA_SOURCES` field. Just replace:
>
> - `my-search` with your Azure Search service name
> - `my-index` with your search index name
> - `your-search-api-key` with your actual Azure Search API key

#### 📋 Complete Configuration Template

```json
[
  {
    "type": "azure_search",
    "parameters": {
      "endpoint": "https://YOUR-SEARCH-SERVICE.search.windows.net",
      "index_name": "YOUR-INDEX-NAME",
      "authentication": {
        "type": "api_key",
        "key": "YOUR-SEARCH-API-KEY"
      }
    }
  }
]
```

#### 🔧 Advanced Configuration Options

For advanced use cases, you can include additional parameters:

```json
[
  {
    "type": "azure_search",
    "parameters": {
      "endpoint": "https://YOUR-SEARCH-SERVICE.search.windows.net",
      "index_name": "YOUR-INDEX-NAME",
      "authentication": {
        "type": "api_key",
        "key": "YOUR-SEARCH-API-KEY"
      },
      "query_type": "vectorSimpleHybrid",
      "semantic_configuration": "default",
      "top_n_documents": 20,
      "strictness": 3,
      "role_information": "You are an AI assistant that helps with questions based on the provided documents."
    }
  }
]
```

#### 🚀 Quick Setup Steps

1. **Create Azure Search Service** - Set up an Azure Search service in the Azure portal
2. **Create and populate index** - Upload your documents to a search index
3. **Get API key** - Copy the API key from your Azure Search service
4. **Configure pipeline** - Add the `AZURE_AI_DATA_SOURCES` environment variable
5. **Use Azure OpenAI endpoint** - Ensure you're using the correct Azure OpenAI URL format

#### ⚠️ Common Issues

- **Wrong endpoint format**: Make sure you're using Azure OpenAI URLs, not regular Azure AI endpoints
- **Invalid JSON**: Copy the JSON template exactly and only change the placeholder values
- **Missing API key**: Ensure your Azure Search API key has proper permissions
- **Index not found**: Verify your index name matches exactly (case-sensitive)

#### Enhanced Citation Display

The pipeline automatically enhances Azure AI Search responses to make citations and source documents more accessible and readable. When Azure AI Search is configured, the pipeline transforms the raw citation data into a user-friendly format.

**Original Azure AI Response:**

```json
{
  "choices": [
    {
      "message": {
        "content": "**Docker container actions** are a type of GitHub Actions [doc1]...",
        "context": {
          "citations": [
            {
              "content": "environment variable. The token can be used to authenticate...",
              "title": "README.md",
              "chunk_id": "0"
            }
          ]
        }
      }
    }
  ]
}
```

**Enhanced Response with Collapsible Citations:**

```html
**Docker container actions** are a type of GitHub Actions [doc1]...

<details>
<summary>📚 Sources and References</summary>

<details>
<summary>[doc1] - README.md</summary>

📁 **File:** `README.md`
📄 **Chunk ID:** 0
**Content:**
> environment variable. The token can be used to authenticate the workflow when accessing GitHub resources...

</details>

<details>
<summary>[doc2] - Documentation.md</summary>

📁 **File:** `Documentation.md`
📄 **Chunk ID:** 1
**Content:**
> Docker container actions contain all their dependencies in the container and are therefore very consistent...

</details>

</details>
```

**Enhanced Citation Features:**

- **Collapsible interface** with expandable sections for clean presentation
- **Two-level organization** - main sources section and individual document details
- **Complete content display** - full document content, not just previews
- **Document references** with clear [doc1], [doc2] labels for easy cross-referencing
- **Source metadata** including file paths, URLs, and chunk IDs for precise tracking
- **Streaming support** with citations properly formatted for both streaming and non-streaming responses
- **Space efficient** - collapsed by default to avoid overwhelming the main response

> [!TIP]  
> To use **Azure OpenAI** and other **Azure AI** models **simultaneously**, you can use the following URL: `https://<your project>.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview`
