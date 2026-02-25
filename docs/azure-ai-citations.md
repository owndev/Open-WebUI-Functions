# Azure AI Foundry Pipeline - Native OpenWebUI Citations

This document describes the native OpenWebUI citation support in the Azure AI Foundry Pipeline, which enables rich citation cards and source previews in the OpenWebUI frontend.

## Overview

The Azure AI Foundry Pipeline supports **native OpenWebUI citations** for Azure AI Search (RAG) responses. This feature is **automatically enabled** when you configure Azure AI Search data sources (`AZURE_AI_DATA_SOURCES`). The OpenWebUI frontend will display:

- **Citation cards** with source information and relevance scores
- **Source previews** with content snippets
- **Relevance percentage** displayed on citation cards (requires `AZURE_AI_INCLUDE_SEARCH_SCORES=true`)
- **Clickable `[docX]` references** that link directly to document URLs
- **Interactive citation UI** with expandable source details

## Features

### Automatic Citation Support

When Azure AI Search is configured, the pipeline automatically:

1. Emits citation events via `__event_emitter__` for the OpenWebUI frontend
2. Converts `[docX]` references in the response to clickable markdown links
3. Filters citations to only show documents actually referenced in the response
4. Extracts relevance scores from Azure Search when available

### Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `AZURE_AI_DATA_SOURCES` | `""` | JSON configuration for Azure AI Search (required for citations) |
| `AZURE_AI_INCLUDE_SEARCH_SCORES` | `true` | Enable relevance score extraction from Azure Search |

### How It Works

#### Streaming Responses

When Azure AI Search returns citations in a streaming response:

1. The pipeline detects citations in the SSE (Server-Sent Events) stream
2. `[docX]` references in each chunk are converted to markdown links with document URLs
3. After the stream ends, citation events are emitted via `__event_emitter__`
4. Citations are filtered to only include documents referenced in the response

#### Non-Streaming Responses

When Azure AI Search returns citations in a non-streaming response:

1. The pipeline extracts citations from the response context
2. `[docX]` references in the content are converted to markdown links
3. Individual citation events are emitted via `__event_emitter__` for each referenced source

## Citation Format

### OpenWebUI Citation Event Structure

Each citation is emitted as a separate event to ensure all sources appear in the UI. Citation events follow the official OpenWebUI specification (see [OpenWebUI Events Documentation](https://docs.openwebui.com/features/plugin/development/events#source-or-citation-and-code-execution)):

```python
{
    "type": "citation",
    "data": {
        "document": ["Document content..."],  # Content from this citation
        "metadata": [{"source": "https://..."}],  # Metadata with source URL
        "source": {
            "name": "[doc1] Document Title",  # Unique name with index
            "url": "https://..."  # Source URL if available
        },
        "distances": [0.95]  # Relevance score (displayed as percentage)
    }
}
```

Key points:

- Each source document gets its own citation event
- The `source.name` includes the doc index (`[doc1]`, `[doc2]`, etc.) to prevent grouping
- The `distances` array contains relevance scores from Azure AI Search, which OpenWebUI displays as a percentage on the citation cards

### Azure Citation Format (Input)

Azure AI Search returns citations in this format:

```python
{
    "title": "Document Title",
    "content": "Full or partial content",
    "url": "https://...",
    "filepath": "/path/to/file",
    "chunk_id": "chunk-123",
    "score": 0.95,
    "metadata": {}
}
```

The pipeline automatically converts Azure citations to OpenWebUI format.

## Usage

### Basic Setup

Configure Azure AI Search to enable citation support:

```bash
# Azure AI Search configuration (required for citations)
AZURE_AI_DATA_SOURCES='[{"type":"azure_search","parameters":{"endpoint":"https://YOUR-SEARCH-SERVICE.search.windows.net","index_name":"YOUR-INDEX-NAME","authentication":{"type":"api_key","key":"YOUR-SEARCH-API-KEY"}}}]'

# Enable relevance scores (default: true)
AZURE_AI_INCLUDE_SEARCH_SCORES=true
```

### Clickable Document Links

The pipeline automatically converts `[docX]` references to clickable markdown links:

```markdown
# Input from Azure AI
The answer can be found in [doc1] and [doc2].

# Output (converted by pipeline)
The answer can be found in [[doc1]](https://example.com/doc1.pdf) and [[doc2]](https://example.com/doc2.pdf).
```

This works for both streaming and non-streaming responses.

### Relevance Scores

When `AZURE_AI_INCLUDE_SEARCH_SCORES=true` (default), the pipeline:

1. Automatically adds `include_contexts: ["citations", "all_retrieved_documents"]` to Azure Search requests
2. Extracts scores based on the `filter_reason` field:
   - `filter_reason="rerank"` → uses `rerank_score`
   - `filter_reason="score"` or not present → uses `original_search_score`
3. Displays the score as a percentage on citation cards

## Implementation Details

### Helper Functions

The pipeline includes these helper functions for citation processing:

1. **`_extract_citations_from_response()`**: Extracts citations from Azure responses
2. **`_normalize_citation_for_openwebui()`**: Converts Azure citations to OpenWebUI format
3. **`_emit_openwebui_citation_events()`**: Emits citation events via `__event_emitter__`
4. **`_merge_score_data()`**: Matches citations with score data from `all_retrieved_documents`
5. **`_build_citation_urls_map()`**: Builds mapping of citation indices to URLs
6. **`_format_citation_link()`**: Creates markdown links for `[docX]` references
7. **`_convert_doc_refs_to_links()`**: Converts all `[docX]` references in content to markdown links

### Title Fallback Logic

The pipeline uses intelligent title fallback:

1. Use `title` field if available
2. Fallback to filename extracted from `filepath` or `url`
3. Fallback to `"Unknown Document"` if all are empty

This ensures every citation has a meaningful display name.

### Citation Filtering

Citations are filtered to only show documents that are actually referenced in the response content. For example, if Azure returns 5 citations but the response only references `[doc1]` and `[doc3]`, only those 2 citations will appear in the UI.

## Index Schema Requirements for Citations

For citations to work correctly, your Azure AI Search index must contain the right fields with the right attributes. This section explains exactly which fields the pipeline reads and how they map to citation cards in OpenWebUI.

### Required and Recommended Index Fields

| Index Field | Type | Required? | Must Be Retrievable? | Citation Purpose |
|---|---|---|---|---|
| `content` | `Edm.String` | Yes | Yes | Provides the text snippet shown in the citation preview |
| `title` | `Edm.String` | Recommended | Yes | Displayed as the citation card title |
| `filepath` | `Edm.String` | Recommended | Yes | Used as the citation name in the response; fallback for title |
| `url` | `Edm.String` | Recommended | Yes | Makes `[docX]` references into clickable links |
| `chunk_id` | `Edm.String` | Optional | Yes | Helps match citations with relevance scores |
| `contentVector` | `Collection(Edm.Single)` | For vector search | N/A | Enables vector/hybrid search |

> **Key point**: The `title`, `filepath`, and `url` fields must be marked as **retrievable** in your index schema. If they are not retrievable, Azure will not include them in the citation response, and the pipeline cannot display them.

### Title Fallback Chain

The pipeline determines each citation's display title using this fallback chain:

1. `title` field → if present and non-empty
2. `filepath` field → if title is empty
3. `url` field → if both title and filepath are empty
4. `"Unknown Document"` → if all are empty

To avoid seeing "Unknown Document", ensure at least one of `title`, `filepath`, or `url` is populated in your index documents.

### Custom Field Names and `fields_mapping`

If your index uses different field names (e.g., `body` instead of `content`, or `doc_title` instead of `title`), you must tell Azure OpenAI how to map them using the `fields_mapping` parameter in your `AZURE_AI_DATA_SOURCES` configuration.

**`fields_mapping` properties:**

| Property | Type | Maps To |
|---|---|---|
| `content_fields` | `string[]` | The index fields to use as document content |
| `title_field` | `string` | The index field to use as the document title |
| `filepath_field` | `string` | The index field to use as the file path/name |
| `url_field` | `string` | The index field to use as the document URL |
| `vector_fields` | `string[]` | The index fields containing vector embeddings |
| `content_fields_separator` | `string` | Separator pattern between content fields (default: `\n`) |

**Example with custom field names:**

```json
[
  {
    "type": "azure_search",
    "parameters": {
      "endpoint": "https://my-search.search.windows.net",
      "index_name": "my-custom-index",
      "authentication": {
        "type": "api_key",
        "key": "YOUR-SEARCH-API-KEY"
      },
      "fields_mapping": {
        "content_fields": ["body", "summary"],
        "title_field": "doc_title",
        "filepath_field": "source_file",
        "url_field": "source_url",
        "vector_fields": ["embedding"]
      }
    }
  }
]
```

### Creating an Index with the Right Fields

If you are creating a new index manually, here is a minimal schema that supports all citation features:

```json
{
  "name": "my-docs-index",
  "fields": [
    { "name": "id", "type": "Edm.String", "key": true, "filterable": true },
    { "name": "content", "type": "Edm.String", "searchable": true, "retrievable": true },
    { "name": "title", "type": "Edm.String", "searchable": true, "retrievable": true, "filterable": true },
    { "name": "filepath", "type": "Edm.String", "retrievable": true, "filterable": true },
    { "name": "url", "type": "Edm.String", "retrievable": true },
    { "name": "chunk_id", "type": "Edm.String", "retrievable": true, "filterable": true }
  ]
}
```

For vector/hybrid search, add a vector field:

```json
{ "name": "contentVector", "type": "Collection(Edm.Single)", "searchable": true, "dimensions": 1536, "vectorSearchProfile": "my-vector-profile" }
```

### Indexer Field Mappings (Blob Storage)

If you index documents from Azure Blob Storage using an indexer, you need to map blob metadata to your index fields. Common blob metadata fields:

| Blob Metadata Field | Description | Typical Index Mapping |
|---|---|---|
| `metadata_storage_name` | Blob filename (e.g., `report.pdf`) | `title` |
| `metadata_storage_path` | Full blob URL (e.g., `https://account.blob.core.windows.net/container/file.pdf`) | `filepath` and `url` |
| `metadata_storage_last_modified` | Last modified timestamp | `last_modified` (optional, useful for sorting) |
| `metadata_storage_content_type` | MIME type | (optional, useful for filtering) |
| `content` | Extracted text from the document | `content` (auto-mapped if names match) |

**Example indexer with field mappings:**

```json
{
  "name": "my-blob-indexer",
  "dataSourceName": "my-blob-datasource",
  "targetIndexName": "my-docs-index",
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
    },
    {
      "sourceFieldName": "metadata_storage_last_modified",
      "targetFieldName": "last_modified"
    }
  ],
  "parameters": {
    "configuration": {
      "dataToExtract": "contentAndMetadata"
    }
  }
}
```

> **Note**: The `content` field is automatically mapped when the source and target field names match. The blob indexer also **automatically** maps `metadata_storage_path` (base64-encoded) to the `id` key field — no explicit mapping is needed for `id`. Mapping `metadata_storage_name` → `title` gives citation cards a readable name from the blob filename.

### How the Pipeline Reads Citation Fields

When Azure OpenAI returns a response with citations, each citation object looks like this:

```json
{
  "title": "Architecture Overview",
  "content": "The system uses a microservices architecture...",
  "url": "https://storageaccount.blob.core.windows.net/docs/architecture.pdf",
  "filepath": "architecture.pdf",
  "chunk_id": "0",
  "original_search_score": 12.5,
  "rerank_score": 3.2,
  "filter_reason": "rerank"
}
```

The pipeline maps these fields to the OpenWebUI citation event:

| Azure Citation Field | OpenWebUI Citation Property | Display |
|---|---|---|
| `title` | `source.name` | `[doc1] - Architecture Overview` |
| `content` | `document[0]` | Preview text in citation card |
| `url` / `filepath` | `source.url` | Clickable link |
| `rerank_score` / `original_search_score` | `distances[0]` | Relevance percentage |

## Troubleshooting

### Citations Not Appearing

**Problem**: Citations don't appear in the OpenWebUI frontend

**Solutions**:

1. Check that Azure AI Search is properly configured (`AZURE_AI_DATA_SOURCES`)
2. Ensure you're using an Azure OpenAI endpoint (not a generic Azure AI endpoint)
3. Verify the response contains `[docX]` references
4. Check browser console and server logs for errors

### Citations Showing "Unknown Document"

**Problem**: Citation cards display "Unknown Document" instead of a meaningful title

**Solutions**:

1. Verify your index has `title`, `filepath`, or `url` fields and that they are marked as **retrievable**
2. If using custom field names, add `fields_mapping` with `title_field`, `filepath_field`, and `url_field` to your `AZURE_AI_DATA_SOURCES` JSON
3. Verify the fields are actually populated in your indexed documents (empty fields cause fallback to "Unknown Document")

### No Clickable Links on [docX] References

**Problem**: `[docX]` references appear as plain text, not clickable links

**Solutions**:

1. Your index needs a `url` field (or mapped `url_field`) that contains valid URLs
2. If your index stores URLs in a field with a different name, map it using `"url_field": "your_field_name"` in `fields_mapping`
3. Verify that the `url` field is marked as **retrievable** in your index schema

### Relevance Scores Showing 0%

**Problem**: All citation cards show 0% relevance

**Solutions**:

1. Verify `AZURE_AI_INCLUDE_SEARCH_SCORES=true` is set
2. Check that your Azure Search index supports scoring
3. Enable DEBUG logging to see the raw score values from Azure

### Links Not Working

**Problem**: `[docX]` references are not clickable

**Solutions**:

1. Ensure citations have valid `url` or `filepath` fields
2. Check that the document URL is accessible
3. Verify the markdown link format is being generated correctly

## References

- [OpenWebUI Pipelines Citation Feature Discussion](https://github.com/open-webui/pipelines/issues/229)
- [OpenWebUI Event Emitter Documentation](https://docs.openwebui.com/features/plugin/development/events)
- [Azure AI Search Documentation](https://learn.microsoft.com/en-us/azure/search/)
- [Azure On Your Data API Reference](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/references/on-your-data)
- [Azure Search Fields Mapping Options](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/references/azure-search#fields-mapping-options)
- [Azure AI Search Indexer Field Mappings](https://learn.microsoft.com/en-us/azure/search/search-indexer-field-mappings)
- [Azure OpenAI On Your Data - Index Field Mapping](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/use-your-data#index-field-mapping)

## Version History

- **v2.6.0**: Major refactor - removed `AZURE_AI_ENHANCE_CITATIONS` and `AZURE_AI_OPENWEBUI_CITATIONS` valves; citation support is now always enabled when `AZURE_AI_DATA_SOURCES` is configured; added clickable `[docX]` markdown links; improved score extraction using `filter_reason` field
- **v2.5.x**: Dual citation modes (OpenWebUI events + markdown/HTML)
