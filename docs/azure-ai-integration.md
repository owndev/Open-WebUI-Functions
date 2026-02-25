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

# Enable relevance score extraction from Azure Search (default: true)
# When enabled, automatically adds include_contexts to get original_search_score and rerank_score
AZURE_AI_INCLUDE_SEARCH_SCORES=true
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

#### Index Schema and Field Mapping for Citations

A common source of confusion is understanding **which fields your Azure AI Search index needs** and **how to map them** so that citations (with titles, URLs, and content) work correctly in OpenWebUI.

##### How Azure OpenAI On Your Data Uses Index Fields

When Azure OpenAI queries your search index, it looks for specific fields to build the citation objects it returns in the response. These citation objects contain `title`, `content`, `url`, `filepath`, and `chunk_id` fields. The pipeline then uses these fields to render citation cards in OpenWebUI.

If the field names in your index don't match the default names Azure expects, you must provide an explicit `fields_mapping` in your `AZURE_AI_DATA_SOURCES` configuration.

##### Default Index Fields (Auto-Ingested Data)

When you use the Azure AI Foundry portal to upload files (PDF, DOCX, TXT, etc.), Azure automatically creates an index with a schema similar to this:

| Index Field Name | Type | Attributes | Purpose |
|---|---|---|---|
| `id` | `Edm.String` | Key | Unique document identifier |
| `content` | `Edm.String` | Searchable, Retrievable | The chunked text content |
| `title` | `Edm.String` | Searchable, Retrievable, Filterable | Document title (shown on citation cards) |
| `filepath` | `Edm.String` | Retrievable, Filterable | Original file path or name (used as citation name) |
| `url` | `Edm.String` | Retrievable | URL to access the source document |
| `chunk_id` | `Edm.String` | Retrievable, Filterable | Identifies specific chunks within a document |
| `metadata_storage_path` | `Edm.String` | Retrievable | Blob storage path of the source file |
| `contentVector` | `Collection(Edm.Single)` | Searchable (vector) | Vector embedding for semantic/vector search |

If your index was auto-generated by Azure, the default field names typically work **without** any `fields_mapping` configuration.

##### Custom Index Fields

If you created your own index with different field names, you **must** configure `fields_mapping` so that Azure OpenAI knows which fields to use for citations.

**Example: Custom index schema**

Suppose your index has these fields:

| Your Index Field | Type | Purpose |
|---|---|---|
| `document_id` | `Edm.String` | Key |
| `body` | `Edm.String` | The main text content |
| `doc_title` | `Edm.String` | Document title |
| `source_file` | `Edm.String` | Original filename |
| `source_url` | `Edm.String` | URL to the source |
| `embedding` | `Collection(Edm.Single)` | Vector data |

You would configure `fields_mapping` as follows:

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
      "fields_mapping": {
        "content_fields": ["body"],
        "title_field": "doc_title",
        "filepath_field": "source_file",
        "url_field": "source_url",
        "vector_fields": ["embedding"]
      }
    }
  }
]
```

##### Fields Mapping Reference

The `fields_mapping` object supports these properties:

| Property | Type | Description | Citation Impact |
|---|---|---|---|
| `content_fields` | `string[]` | Index fields treated as searchable content. Multiple fields can be specified. | Provides the text content shown in citation previews |
| `title_field` | `string` | Index field used as the document title. | Displayed as the citation card title |
| `filepath_field` | `string` | Index field used as the file path/name. | Shown as the citation name; used as fallback for title |
| `url_field` | `string` | Index field used as the document URL. | Makes `[docX]` references clickable links |
| `vector_fields` | `string[]` | Fields containing vector embeddings. | Required for vector/hybrid search |
| `content_fields_separator` | `string` | Separator between multiple content fields. Default: `\n` | -- |

> [!IMPORTANT]
> The `title_field`, `filepath_field`, and `url_field` are critical for citations:
>
> - **`title_field`** → shown as the citation card title (falls back to `filepath_field` or `url_field`)
> - **`filepath_field`** → used to generate the citation name in the response text
> - **`url_field`** → enables clickable `[docX]` links in the response
>
> If none of these fields are populated in your index, citations will show as "Unknown Document" with no clickable links.

##### Indexer Configuration for Blob Storage

If you are indexing documents from **Azure Blob Storage** using an Azure AI Search **indexer**, you typically need to configure **field mappings in the indexer** to extract metadata from the blobs and map them to your index fields.

**Example: Indexer field mappings (REST API)**

```json
{
  "name": "my-blob-indexer",
  "dataSourceName": "my-blob-datasource",
  "targetIndexName": "my-index",
  "fieldMappings": [
    {
      "sourceFieldName": "metadata_storage_name",
      "targetFieldName": "title"
    },
    {
      "sourceFieldName": "metadata_storage_path",
      "targetFieldName": "filepath"
    },
    {
      "sourceFieldName": "metadata_storage_path",
      "targetFieldName": "url"
    }
  ],
  "parameters": {
    "configuration": {
      "dataToExtract": "contentAndMetadata",
      "parsingMode": "default"
    }
  }
}
```

Key indexer field mapping concepts:

- **`metadata_storage_name`** → the blob filename (e.g., `report.pdf`). Map this to your `title` field so citation cards show a readable name.
- **`metadata_storage_path`** → the full blob URL. Map this to both `filepath` and `url` to enable clickable `[docX]` links.
- **`metadata_storage_content_type`** → the MIME type of the blob (optional, useful for filtering).
- **`metadata_storage_last_modified`** → last modified timestamp (optional, useful for sorting/filtering).
- **`content`** → the extracted text content from the document. This is automatically mapped if your index has a field called `content`.
- **`id` (key field)** → the blob indexer **automatically** maps `metadata_storage_path` (base64-encoded) to the key field. You don't need an explicit mapping for `id`.

> [!TIP]
> The indexer also supports **output field mappings** (`outputFieldMappings`) for mapping enriched fields from AI skillsets (e.g., after chunking and embedding via integrated vectorization).

##### Proven Working Example: Blob Storage with Keyword Search

This is a complete, tested configuration for indexing documents from Azure Blob Storage and getting citations with clickable links in OpenWebUI.

**1. Data Source** (connects to your Blob Storage container):

```json
{
  "name": "my-blob-datasource",
  "type": "azureblob",
  "credentials": {
    "connectionString": "ResourceId=/subscriptions/YOUR-SUBSCRIPTION-ID/resourceGroups/YOUR-RESOURCE-GROUP/providers/Microsoft.Storage/storageAccounts/YOUR-STORAGE-ACCOUNT;"
  },
  "container": {
    "name": "YOUR-CONTAINER-NAME",
    "query": "YOUR-FOLDER-PREFIX"
  }
}
```

> [!TIP]
> The `query` field is optional and acts as a folder prefix filter. Omit it to index all blobs in the container.

**2. Index** (with all fields needed for citations):

```json
{
  "name": "my-docs-index",
  "fields": [
    { "name": "id", "type": "Edm.String", "key": true, "retrievable": true },
    { "name": "title", "type": "Edm.String", "searchable": true, "retrievable": true },
    { "name": "filepath", "type": "Edm.String", "filterable": true, "retrievable": true },
    { "name": "url", "type": "Edm.String", "retrievable": true },
    { "name": "content", "type": "Edm.String", "searchable": true, "retrievable": true, "analyzer": "standard.lucene" },
    { "name": "last_modified", "type": "Edm.DateTimeOffset", "filterable": true, "sortable": true, "retrievable": true },
    { "name": "metadata_content_type", "type": "Edm.String", "filterable": true, "retrievable": true },
    { "name": "metadata_content_length", "type": "Edm.Int64", "filterable": true, "sortable": true, "retrievable": true }
  ],
  "similarity": {
    "@odata.type": "#Microsoft.Azure.Search.BM25Similarity"
  }
}
```

**3. Indexer** (maps blob metadata to index fields):

```json
{
  "name": "my-docs-indexer",
  "dataSourceName": "my-blob-datasource",
  "targetIndexName": "my-docs-index",
  "parameters": {
    "configuration": {
      "dataToExtract": "contentAndMetadata",
      "parsingMode": "default",
      "imageAction": "none"
    }
  },
  "fieldMappings": [
    { "sourceFieldName": "metadata_storage_name", "targetFieldName": "title" },
    { "sourceFieldName": "metadata_storage_path", "targetFieldName": "filepath" },
    { "sourceFieldName": "metadata_storage_path", "targetFieldName": "url" },
    { "sourceFieldName": "metadata_storage_last_modified", "targetFieldName": "last_modified" },
    { "sourceFieldName": "metadata_content_type", "targetFieldName": "metadata_content_type" },
    { "sourceFieldName": "metadata_content_length", "targetFieldName": "metadata_content_length" }
  ]
}
```

> [!NOTE]
> No explicit mapping for `id` is required — the blob indexer automatically maps `metadata_storage_path` (base64-encoded) to the key field.

**4. Pipeline Configuration** (`AZURE_AI_DATA_SOURCES`):

Since the index uses the default field names (`title`, `filepath`, `url`, `content`), no `fields_mapping` is needed:

```json
[{"type":"azure_search","parameters":{"endpoint":"https://YOUR-SEARCH-SERVICE.search.windows.net","index_name":"my-docs-index","authentication":{"type":"api_key","key":"YOUR-SEARCH-API-KEY"}}}]
```

**Result:** Citation cards in OpenWebUI will show the blob filename as the title and the blob URL as a clickable link.

##### Complete Example: Custom Index Creation (REST API)

This example creates an index suitable for use with this pipeline's citation features:

```json
PUT https://YOUR-SEARCH-SERVICE.search.windows.net/indexes/my-docs-index?api-version=2024-07-01
Content-Type: application/json
api-key: YOUR-ADMIN-API-KEY

{
  "name": "my-docs-index",
  "fields": [
    { "name": "id", "type": "Edm.String", "key": true, "filterable": true },
    { "name": "content", "type": "Edm.String", "searchable": true, "retrievable": true },
    { "name": "title", "type": "Edm.String", "searchable": true, "retrievable": true, "filterable": true },
    { "name": "filepath", "type": "Edm.String", "retrievable": true, "filterable": true },
    { "name": "url", "type": "Edm.String", "retrievable": true },
    { "name": "chunk_id", "type": "Edm.String", "retrievable": true, "filterable": true },
    { "name": "contentVector", "type": "Collection(Edm.Single)", "searchable": true, "dimensions": 1536, "vectorSearchProfile": "my-vector-profile" }
  ],
  "semantic": {
    "configurations": [
      {
        "name": "default",
        "prioritizedFields": {
          "titleField": { "fieldName": "title" },
          "contentFields": [
            { "fieldName": "content" }
          ]
        }
      }
    ]
  },
  "vectorSearch": {
    "algorithms": [
      { "name": "my-hnsw", "kind": "hnsw" }
    ],
    "profiles": [
      { "name": "my-vector-profile", "algorithmConfigurationName": "my-hnsw" }
    ]
  }
}
```

> [!NOTE]
>
> - The `dimensions` value (1536) matches the `text-embedding-ada-002` model. Use 3072 for `text-embedding-3-large`.
> - The `semantic` configuration enables semantic reranking, which improves citation relevance scores.
> - Make all citation-related fields (`title`, `filepath`, `url`, `content`) **retrievable** so Azure can include them in the citation response.

##### How the Pipeline Uses Citation Fields

The pipeline reads these fields from Azure's citation response to build OpenWebUI citation cards:

```
Azure Citation Response          →  OpenWebUI Citation Card
─────────────────────────────────────────────────────────────
citation.title                   →  Card title (with [docX] prefix)
citation.content                 →  Document preview / snippet
citation.url                     →  Clickable link on [docX] references
citation.filepath                →  Fallback for title and URL
citation.chunk_id                →  Used for score matching
citation.original_search_score   →  Relevance % (BM25/keyword)
citation.rerank_score            →  Relevance % (semantic reranker)
citation.filter_reason           →  Selects which score to display
```

#### 🚀 Quick Setup Steps

1. **Create Azure Search Service** - Set up an Azure Search service in the Azure portal
2. **Create and populate index** - Upload your documents to a search index (ensure fields like `title`, `filepath`, `url`, and `content` are present and **retrievable**)
3. **Get API key** - Copy the API key from your Azure Search service
4. **Configure field mappings** - If your index uses custom field names, add `fields_mapping` to your `AZURE_AI_DATA_SOURCES` JSON
5. **Configure pipeline** - Add the `AZURE_AI_DATA_SOURCES` environment variable
6. **Use Azure OpenAI endpoint** - Ensure you're using the correct Azure OpenAI URL format

#### ⚠️ Common Issues

- **Wrong endpoint format**: Make sure you're using Azure OpenAI URLs, not regular Azure AI endpoints
- **Invalid JSON**: Copy the JSON template exactly and only change the placeholder values
- **Missing API key**: Ensure your Azure Search API key has proper permissions
- **Index not found**: Verify your index name matches exactly (case-sensitive)
- **Citations showing "Unknown Document"**: Your index is missing `title`, `filepath`, or `url` fields, or those fields are not set as **retrievable**
- **No clickable links on `[docX]`**: Your index has no `url` field, or `url_field` is not mapped in `fields_mapping`
- **Custom field names not working**: Add a `fields_mapping` object to your `AZURE_AI_DATA_SOURCES` configuration (see [Index Schema and Field Mapping](#index-schema-and-field-mapping-for-citations) above)

#### Native OpenWebUI Citation Support

The pipeline automatically provides native OpenWebUI citation support for Azure AI Search responses. When Azure AI Search is configured, the pipeline:

1. **Emits citation events** via `__event_emitter__` for the OpenWebUI frontend to display interactive citation cards
2. **Converts `[docX]` references** to clickable markdown links that link directly to document URLs
3. **Extracts relevance scores** when `AZURE_AI_INCLUDE_SEARCH_SCORES=true`
4. **Filters citations** to only show documents actually referenced in the response

**Example: Clickable Document Links**

```markdown
# Original Azure AI response
**Docker container actions** are a type of GitHub Actions [doc1]...

# Enhanced response (with clickable links)
**Docker container actions** are a type of GitHub Actions [[doc1]](https://example.com/README.md)...
```

**Citation Card Features:**

- **Source information** with `[docX]` prefix for easy identification
- **Relevance percentage** displayed on citation cards (requires `AZURE_AI_INCLUDE_SEARCH_SCORES=true`)
- **Document preview** with content snippets
- **Clickable links** to source documents when URLs are available
- **Streaming support** with links converted inline as content streams

**Relevance Score Selection:**

The pipeline uses the `filter_reason` field from Azure Search to select the appropriate score:

- `filter_reason="rerank"` → uses `rerank_score`
- `filter_reason="score"` or not present → uses `original_search_score`

For more details, see the [Azure AI Citations Documentation](azure-ai-citations.md).

> [!TIP]  
> To use **Azure OpenAI** and other **Azure AI** models **simultaneously**, you can use the following URL: `https://<your project>.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview`
