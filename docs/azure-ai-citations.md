# Azure AI Foundry Pipeline - Native OpenWebUI Citations

This document describes the native OpenWebUI citation support in the Azure AI Foundry Pipeline, which enables rich citation cards and source previews in the OpenWebUI frontend.

> [!WARNING]
> **Azure OpenAI On Your Data is retired on October 14, 2026.** This affects only the legacy mode `AZURE_AI_SEARCH_MODE=on_your_data`, in which the pipeline sends `data_sources` to Azure OpenAI and builds the citations from the `context` of the answer. Microsoft retires On Your Data on **October 14, 2026** ([On Your Data API reference](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/references/on-your-data)) and, until then, supports it only for GPT-4o and GPT-4o-mini. Since v2.9.0 the default `AZURE_AI_SEARCH_MODE=pipeline` queries Azure AI Search itself, works with every chat endpoint and model and does not use On Your Data ([#187](https://github.com/owndev/Open-WebUI-Functions/issues/187)). In `on_your_data` mode the pipeline logs the retirement notice as a warning once per process (see [Logging](#logging)).

## Overview

The Azure AI Foundry Pipeline supports **native OpenWebUI citations** for Azure AI Search (RAG) responses. This feature is **automatically enabled** when you configure Azure AI Search (`AZURE_AI_DATA_SOURCES`). In the default pipeline mode the citations come from the pipeline's own search; in `on_your_data` mode from Azure OpenAI's answer. Both run through the same citation code. The OpenWebUI frontend will display:

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

Azure AI Search is only used for chat answers. Open WebUI background tasks (title, tags and follow-up generation) run **without** a search (and without `data_sources`) and never emit citation or status events, so they cannot add sources to a chat message (see [Background Tasks](#background-tasks-titles-tags-follow-ups)).

### Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `AZURE_AI_DATA_SOURCES` | `""` | JSON configuration for Azure AI Search (required for citations) |
| `AZURE_AI_SEARCH_MODE` | `pipeline` | `pipeline`: the pipeline queries Azure AI Search itself (any chat endpoint and model); `on_your_data`: legacy Azure OpenAI On Your Data |
| `AZURE_AI_SEARCH_KEY` | `""` | Azure AI Search key, stored encrypted; wins over `authentication.key` in the JSON |
| `AZURE_AI_SEARCH_API_VERSION` | `2026-04-01` | Search REST API version (pipeline mode) |
| `AZURE_AI_SEARCH_QUERY_GENERATION` | `auto` | Search queries written by the model: `auto` (follow-up turns), `always`, `off` (pipeline mode) |
| `AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS` | `-1` | Tokens of document text in the prompt (characters / 4); `-1` auto = `min(32000, 1600 × top_n_documents)`, `0` no limit (pipeline mode) |
| `AZURE_AI_INCLUDE_SEARCH_SCORES` | `true` | Relevance percentages on citation cards (pipeline mode: scores of the search hits; `on_your_data`: adds `include_contexts`) |
| `AZURE_AI_SHOW_ALL_CITATIONS_WITHOUT_REFERENCES` | `true` | Sources for an answer that contains no `[docX]` reference: `true` shows all documents of the answer, `false` shows none (see [Citation Filtering](#citation-filtering)) |

See the [Azure AI integration guide](azure-ai-integration.md#azure-ai-search--rag-integration) for the search itself (query types, query generation, strictness, token budget, authentication, errors).

### How It Works

In **pipeline mode** the pipeline searches the index, adds the documents to the prompt as `[doc1]` … `[docN]` and builds one citation per document in On Your Data's format (`context.citations` plus `context.intent`, the JSON list of the search queries). It hands this `context` to the same code that handles On Your Data answers:

- **streamed answers**: a first SSE event `{"id": "chatcmpl-azure-search", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "context": {...}}, "finish_reason": null}]}` before Azure's chunks, so `[docX]` links work from the first delta and API clients get the citations as with On Your Data. Answers with no document found have no such event.
- **non-streamed answers**: `choices[0].message.context` is added to Azure's response.

In **`on_your_data` mode** Azure OpenAI returns the `context` itself.

#### Streaming Responses

When a streamed answer has citations:

1. The pipeline detects citations in the SSE (Server-Sent Events) stream
2. `[docX]` references in each chunk are converted to markdown links with document URLs. Models usually stream a reference in pieces (`[`, `doc`, `1`, `]`), so the end of a chunk that may still become a reference or a link around one (`[`, `[[`, `[d` … `[doc1`, `[doc1]`, `[[doc1]]`, `[[doc1]](https://…` up to the closing `)`) is held back and sent together with the next chunk. Text that is still held back when the stream ends is sent as a separate event before `data: [DONE]`, so nothing is lost, also in the saved chat message
3. After the stream ends, citation events are emitted via `__event_emitter__`
4. Citations are filtered to only include documents referenced in the response

In `on_your_data` mode Azure sends the citations (and, with `AZURE_AI_INCLUDE_SEARCH_SCORES=true`, `all_retrieved_documents`) in one SSE event, which grows with the number and length of the retrieved documents. The pipeline reads events of up to 4 MiB. If the stream fails anyway (a larger event, a dropped connection, a timeout), the text received so far is kept, an `Error: …` message is added to the answer and the stream is ended with `data: [DONE]`, so API clients do not get an empty or silently cut off answer; the chat UI also shows the error as the final status. See [Streamed Answer Ends With an Error](#streamed-answer-ends-with-an-error).

#### Non-Streaming Responses

When a non-streamed answer has citations:

1. The pipeline extracts citations from the response context
2. `[docX]` references in the content are converted to markdown links
3. Individual citation events are emitted via `__event_emitter__` for each referenced source

#### Tool Rounds (Pipeline Mode)

With native function calling Open WebUI calls the pipeline again for every tool round of an answer, with the same message. The pipeline reuses the documents of the first round (same numbering, no new search; only for a tool round of the same user and message with unchanged search valves), a round that ends in tool calls emits only the documents it references (never the "show all" fallback), and every document is emitted at most once per answer. The fallback for an answer without references applies only when no round referenced a document. The documents are kept per Open WebUI worker process, so with several workers a tool round on another process searches again and can add a source twice.

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

In `on_your_data` mode Azure returns citations in this format:

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

In pipeline mode the pipeline builds the same shape from each search hit it puts into the prompt (list order = `[docN]`):

| Citation key | From the search hit |
|---|---|
| `title`, `url`, `filepath` | the fields named by `fields_mapping` (`title_field`, `url_field`, `filepath_field`; default `title`, `url`, `filepath`); lists joined with `, `, `null` when empty |
| `content` | the `content_fields` (default `content`) joined with `content_fields_separator`, after removing `<documents>` tags (a `<` that would still start such a tag afterwards, e.g. from nested tags like `</docu<documents>ments>`, becomes `‹`) and turning `[doc` into `[ doc`, cut to the token budget: exactly the text the model received |
| `chunk_id` | the hit's `chunk_id` field, else the document number (informative only) |
| `original_search_score` | `@search.score` (only with `AZURE_AI_INCLUDE_SEARCH_SCORES=true`) |
| `rerank_score` | `@search.rerankerScore` of semantic queries (only with scores on, left out when absent) |
| `relevance` | 0-1 value shown on the card, computed for the score type (see [Relevance Scores](#relevance-scores); only with scores on) |

Pipeline citations have no `metadata` (a `source` key there would replace the `[docX] - title` card name) and no `filter_reason` (On Your Data used it to mark dropped documents). The `context` is "On Your Data shaped", not identical: there is no `all_retrieved_documents`.

The pipeline automatically converts these citations to OpenWebUI format.

## Usage

### Basic Setup

Configure Azure AI Search to enable citation support:

```bash
# Azure AI Search configuration (required for citations)
AZURE_AI_DATA_SOURCES='[{"type":"azure_search","parameters":{"endpoint":"https://YOUR-SEARCH-SERVICE.search.windows.net","index_name":"YOUR-INDEX-NAME","authentication":{"type":"api_key"}}}]'

# Azure AI Search key, stored encrypted
AZURE_AI_SEARCH_KEY="YOUR-SEARCH-QUERY-KEY"

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

References that are already links (`[[doc1]](url)` or `[doc1](url)`) are left as they are, so they are never wrapped twice, also when such a link is streamed in pieces. A reference the model writes as `[[doc1]]` (without a link) is converted like `[doc1]`. Parentheses in document URLs are percent-encoded (`(` → `%28`, `)` → `%29`) so that a link always ends at its own `)`. The links that the pipeline added to earlier answers are sent back to Azure as plain `[docX]` in the chat history, so the model does not copy the link syntax into its next answer; this also works for links that versions before v2.8.0 saved with unencoded parentheses in the URL (for example `[[doc1]](https://example.com/manual_(v2).pdf)`).

### Background Tasks (Titles, Tags, Follow-ups)

Open WebUI uses the chat model for background tasks such as title, tag and follow-up generation. For these requests (`__task__` is set) the pipeline:

- does **not** search (pipeline mode: no Azure AI Search query and no `<documents>` in the prompt; `on_your_data` mode: no `data_sources`), so there are no citations in the result
- does **not** emit citation or status events, so nothing is added to the chat message the task belongs to

This fixes the "too many sources" problem ([#123](https://github.com/owndev/Open-WebUI-Functions/issues/123)): a task answer such as a title contains no `[docX]` reference, so the "no references → show all citations" fallback emitted every retrieved document once per task. Where Open WebUI passes the chat message's event emitter to background tasks, these citations were added to the message shortly after the answer was finished. Open WebUI 0.6.41 (current when #123 was reported) does this for saved chats; Open WebUI 0.11.4 does it for saved-chat requests without a websocket session, but not for chats in the browser, where the change saves one Azure AI Search query per task.

### Tools and `stream_options` with Azure AI Search

In **pipeline mode** (default since v2.9.0) `tools`, `tool_choice` and `stream_options` are forwarded as in chats without Azure AI Search: function calling works (see [Tool Rounds](#tool-rounds-pipeline-mode)) and streamed answers report token usage.

In **`on_your_data` mode**, when `data_sources` is used for a request, the pipeline does not forward:

- **`tools` / `tool_choice`** (behavior change in v2.8.0, up to v2.7.0 they were forwarded): with tools in the request, Azure OpenAI On Your Data [ignores the data sources](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/concepts/use-your-data#function-calling) unless `tool_choice` is `none`. Open WebUI 0.10+ adds its built-in tools to chats in the web UI (native function calling), which would silently turn off the search. The tools are always dropped, also tools you selected yourself.
- **`stream_options`**: On Your Data rejects it (`Validation error at #/stream_options: Extra inputs are not permitted`). Open WebUI 0.11.1+ adds `stream_options.include_usage` to streaming requests of models with the usage capability. Streaming answers with `data_sources` therefore contain no token usage.

### Relevance Scores

When `AZURE_AI_INCLUDE_SEARCH_SCORES=true` (default), citation cards show a relevance percentage.

**Pipeline mode**: each citation carries `relevance` (0-1), which the card shows:

| Query type | Score | `relevance` |
|---|---|---|
| `semantic`, `vector_semantic_hybrid` | `@search.rerankerScore` (0-4) | score / `RERANK_SCORE_MAX` (default 4.0) |
| `simple`, `semantic` without reranker score (partial result) | `@search.score` (BM25, unbounded) | score / `BM25_SCORE_MAX` (default 100), at most 1 |
| `vector` over one field | `@search.score` (0.333-1 for cosine) | as is |
| `vector_simple_hybrid`, `vector` over several fields, `vector_semantic_hybrid` without reranker scores | `@search.score` (reciprocal rank fusion, about 1/60 per fused list) | score × 60 / number of fused lists (text query + vector fields), at most 1 |

Unlike the On Your Data path, a BM25 or reranker score below 1 is divided as well, so the percentages grow with the score. With `AZURE_AI_INCLUDE_SEARCH_SCORES=false` the citations have no score keys and the cards show 0 %.

**`on_your_data` mode**: the pipeline

1. Automatically adds `include_contexts: ["citations", "all_retrieved_documents"]` to Azure Search requests
2. Extracts scores based on the `filter_reason` field:
   - `filter_reason="rerank"` → uses `rerank_score`
   - `filter_reason="score"` or not present → uses `original_search_score`
3. Displays the score as a percentage on citation cards (a score of at most 1.0 is shown as is)

## Implementation Details

### Helper Functions

The pipeline includes these helper functions for citation processing:

1. **`_extract_citations_from_response()`**: Extracts citations from Azure responses
2. **`_normalize_citation_for_openwebui()`**: Converts Azure citations to OpenWebUI format (uses `relevance` first when a citation has it)
3. **`_emit_openwebui_citation_events()`**: Emits citation events via `__event_emitter__` and returns the emitted indices
4. **`_merge_score_data()`**: Matches citations with score data from `all_retrieved_documents` (`on_your_data` mode)
5. **`_build_citation_urls_map()`**: Builds mapping of citation indices to URLs
6. **`_format_citation_link()`**: Creates markdown links for `[docX]` references
7. **`_convert_doc_refs_to_links()`**: Converts all `[docX]` references in content to markdown links

Pipeline mode adds:

1. **`_get_search_config()`**: Reads `AZURE_AI_DATA_SOURCES` as the search configuration (configuration errors raise `AzureSearchError`)
2. **`_retrieve()`**: Query text, query generation (`_generate_queries()`), embeddings (`_embed()`), the searches (`_search_once()`), `strictness` (`_apply_strictness()`), merge (`_merge_search_results()`), token budget (`_fit_documents_to_budget()`) and the citations (`_search_citation()`); reuses the result in tool rounds
3. **`_inject_sources()`**: Adds the `<documents>` block to the current user message and the rules to the system message (new message objects)
4. **`_prepend_context_event()`**: The first SSE event with `delta.context` of a streamed answer
5. **`_emit_search_citation_events()`**: Citation events per tool round, every document at most once per answer

### Title Fallback Logic

The pipeline uses intelligent title fallback:

1. Use `title` field if available
2. Fallback to filename extracted from `filepath` or `url`
3. Fallback to `"Unknown Document"` if all are empty

This ensures every citation has a meaningful display name.

### Citation Filtering

Citations are filtered to only show documents that are actually referenced in the response content. For example, if Azure returns 5 citations but the response only references `[doc1]` and `[doc3]`, only those 2 citations will appear in the UI. If a chat answer contains no `[docX]` reference at all (for example "The requested information is not available in the retrieved data."), all citations are shown by default. Set `AZURE_AI_SHOW_ALL_CITATIONS_WITHOUT_REFERENCES=false` to show no sources for such answers. References to documents that Azure did not return (for example `[doc9]` with 3 citations) do not count, so an answer that references only such documents is treated like an answer without references. A non-streamed answer whose `content` is `null` (for example a filtered answer) is returned as Azure sent it and also counts as an answer without references.

### Logging

Citation helpers log only counts at `INFO` level. Document content, titles and URLs are logged at `DEBUG` level only.

In pipeline mode each search logs one `INFO` line with the number of queries, hits per query, documents kept, characters added to the prompt and the time of query generation, embeddings and search (no query text, no content). Search errors are logged as `ERROR` (`Error in Azure AI request: Azure AI Search: …`, without traceback) with Search's `request-id` at `INFO`. The query text, the generated queries and the document titles are logged at `DEBUG` only. Keys, tokens, request headers and the `AZURE_AI_DATA_SOURCES` JSON are never logged. Warnings (once per process): unknown `AZURE_AI_SEARCH_MODE`, unknown keys in the JSON (names only), several data sources, the key in both `AZURE_AI_SEARCH_KEY` and the JSON, missing `vector_fields` / `content_fields`, query generation paused for a model.

In `on_your_data` mode (since v2.8.1) the first request that uses `data_sources` (from `AZURE_AI_DATA_SOURCES` or sent by the client; an empty list does not count) logs a `WARNING` that Azure OpenAI On Your Data is retired on October 14, 2026, with a link to [#187](https://github.com/owndev/Open-WebUI-Functions/issues/187) and the hint to set `AZURE_AI_SEARCH_MODE=pipeline`. It is logged once per process (Open WebUI loads the function again when it is saved, for example after its code is changed, which logs it once more) and contains no part of the `data_sources`, so no search key. Background tasks are sent without `data_sources` and do not log it; pipeline mode never logs it.

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

> **Key point**: The `title`, `filepath`, and `url` fields must be marked as **retrievable** in your index schema. If they are not retrievable, the search does not return them, and the pipeline cannot display them.

### Title Fallback Chain

The pipeline determines each citation's display title using this fallback chain:

1. `title` field → if present and non-empty
2. `filepath` field → if title is empty
3. `url` field → if both title and filepath are empty
4. `"Unknown Document"` → if all are empty

To avoid seeing "Unknown Document", ensure at least one of `title`, `filepath`, or `url` is populated in your index documents.

### Custom Field Names and `fields_mapping`

If your index uses different field names (e.g., `body` instead of `content`, or `doc_title` instead of `title`), you must tell the pipeline (or Azure OpenAI in `on_your_data` mode) how to map them using the `fields_mapping` parameter in your `AZURE_AI_DATA_SOURCES` configuration. In pipeline mode a mapping also limits the fields the search returns (`select`); a role it leaves out stays empty (see [How the Index Fields Are Used](azure-ai-integration.md#how-the-index-fields-are-used)).

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
        "type": "api_key"
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

In pipeline mode a citation built from a semantic search hit looks like this:

```json
{
  "title": "Architecture Overview",
  "content": "The system uses a microservices architecture...",
  "url": "https://storageaccount.blob.core.windows.net/docs/architecture.pdf",
  "filepath": "architecture.pdf",
  "chunk_id": "0",
  "original_search_score": 12.5,
  "rerank_score": 3.2,
  "relevance": 0.8
}
```

When Azure OpenAI returns a response with citations (`on_your_data` mode), each citation object looks like this:

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
| `relevance` (pipeline mode), else `rerank_score` / `original_search_score` | `distances[0]` | Relevance percentage |

## Troubleshooting

### Citations Not Appearing

**Problem**: Citations don't appear in the OpenWebUI frontend

**Solutions**:

1. Check that Azure AI Search is properly configured (`AZURE_AI_DATA_SOURCES`)
2. In pipeline mode, check the status lines: `No documents found in Azure AI Search` means the search returned nothing that passed `strictness` (try a lower `strictness`); an `Error: Azure AI Search: …` answer names the problem (see [Troubleshooting Pipeline Mode](azure-ai-integration.md#troubleshooting-pipeline-mode))
3. Only in `on_your_data` mode: ensure you're using an Azure OpenAI endpoint (not a generic Azure AI endpoint)
4. Verify the response contains `[docX]` references
5. Check browser console and server logs for errors

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
4. In pipeline mode, low percentages with `simple` queries mean that BM25 scores are small compared with `BM25_SCORE_MAX` (default 100): lower it to match your index

### Links Not Working

**Problem**: `[docX]` references are not clickable

**Solutions**:

1. Ensure citations have valid `url` or `filepath` fields
2. Check that the document URL is accessible
3. Verify the markdown link format is being generated correctly

### More Sources Appear After the Answer

**Problem**: The referenced sources appear, and shortly after the answer is finished more (unreferenced) sources are added

**Solution**: Update to v2.8.0 or later. Background tasks (title, tags, follow-ups) no longer use Azure AI Search and no longer emit citation events. An answer without any `[docX]` reference still shows all citations returned by Azure; set `AZURE_AI_SHOW_ALL_CITATIONS_WITHOUT_REFERENCES=false` to show no sources for such answers.

### Answers Ignore the Search Index / No Citations in the Chat UI

**Problem**: With `AZURE_AI_DATA_SOURCES` configured, answers in the chat UI are not grounded and show no citations, or streaming fails with `Extra inputs are not permitted`

**Solution**: Update to v2.8.0 or later. Earlier versions forwarded Open WebUI's built-in `tools` (Azure then ignores `data_sources`) and `stream_options` (rejected by On Your Data) together with the data sources. Since v2.9.0 the default pipeline mode does not depend on On Your Data at all.

### Streamed Answer Ends With an Error

**Problem**: A streamed answer with Azure AI Search (`on_your_data` mode) ends with `Error: Azure AI sent a stream event larger than 4 MiB, which cannot be read. …`, or (before v2.8.0) stays empty while the server log shows `Got more than 131072 bytes when reading`

**Solution**: In pipeline mode the document text is limited by `AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS`, so this does not happen; switch to `AZURE_AI_SEARCH_MODE=pipeline`. In `on_your_data` mode the citations of an answer arrive in one SSE event. Up to v2.7.0 the pipeline could only read events of up to 128 KiB, which many or long retrieved documents exceed; since v2.8.0 it reads events of up to 4 MiB. If an event is even larger, retrieve fewer or shorter documents (for example a lower `top_n_documents` in `AZURE_AI_DATA_SOURCES`, or shorter chunks in the index), or set `AZURE_AI_INCLUDE_SEARCH_SCORES=false`, so that `all_retrieved_documents` (which repeats the documents together with their scores) is no longer requested; citation cards then show no relevance percentage.

### Requests With Azure AI Search Fail or Show No Citations After October 14, 2026

**Problem**: Since October 14, 2026, chats that use `AZURE_AI_DATA_SOURCES` in `on_your_data` mode (or API requests with `data_sources`) end with an `Error: …` message (for HTTP 400 / 404 with `(Azure OpenAI On Your Data was retired on 2026-10-14; set AZURE_AI_SEARCH_MODE=pipeline)`), or are answered without the search index and without citations

**Solution**: Microsoft retires Azure OpenAI On Your Data, which the legacy mode is built on, on October 14, 2026 (see the notice at the top of this document). Set `AZURE_AI_SEARCH_MODE=pipeline` (the default since v2.9.0): the pipeline then queries Azure AI Search itself with the same `AZURE_AI_DATA_SOURCES`. Check the [authentication](azure-ai-integration.md#authentication-of-the-search) (a managed identity is now the Open WebUI host's) and, for vector query types, `embedding_dependency`. API clients that send `data_sources` themselves must stop doing so in pipeline mode ([#187](https://github.com/owndev/Open-WebUI-Functions/issues/187)).

## References

- [OpenWebUI Pipelines Citation Feature Discussion](https://github.com/open-webui/pipelines/issues/229)
- [OpenWebUI Event Emitter Documentation](https://docs.openwebui.com/features/plugin/development/events)
- [Azure AI Search Documentation](https://learn.microsoft.com/en-us/azure/search/)
- [Azure On Your Data API Reference](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/references/on-your-data)
- [Azure Search Fields Mapping Options](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/references/azure-search#fields-mapping-options)
- [Azure AI Search Indexer Field Mappings](https://learn.microsoft.com/en-us/azure/search/search-indexer-field-mappings)
- [Azure OpenAI On Your Data - Index Field Mapping](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/concepts/use-your-data#index-field-mapping)
- [Connect a Foundry IQ knowledge base](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/foundry-iq-connect) (Microsoft's recommended migration from On Your Data)

## Version History

- **v2.9.0**: New default `AZURE_AI_SEARCH_MODE=pipeline`: the pipeline queries Azure AI Search itself (Search REST API `2026-04-01`; simple, semantic, vector and hybrid queries; `filter`, `fields_mapping`, `top_n_documents`, approximated `strictness`, `in_scope`, `role_information`, `embedding_dependency` with `deployment_name`, `endpoint` or the index vectorizer), works with every chat endpoint and model, writes search queries for follow-up turns (`AZURE_AI_SEARCH_QUERY_GENERATION`), adds the documents to the prompt within a token budget (`AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS`) and synthesizes On Your Data's `context` for the existing citation code (first SSE event / `message.context`); tools and token usage are kept; citations carry `relevance` (reranker, BM25, vector and rank-based RRF scores); tool rounds reuse the documents and emit every source once; search and configuration errors end with `Error: Azure AI Search: …` (fail closed, also for invalid JSON); client `data_sources` are an error in pipeline mode; new encrypted `AZURE_AI_SEARCH_KEY` (both modes) and `AZURE_AI_SEARCH_API_VERSION`; managed identity now means the Open WebUI host's identity. `AZURE_AI_SEARCH_MODE=on_your_data` keeps the v2.8.1 behavior, plus the key valve and a retirement hint on HTTP 400 / 404 ([#187](https://github.com/owndev/Open-WebUI-Functions/issues/187))
- **v2.8.1**: Retirement notice for Azure OpenAI On Your Data (October 14, 2026) in this document, in the `AZURE_AI_DATA_SOURCES` valve description and in the pipeline docstring; the first request with a non-empty `data_sources` (from the valve or sent by the client) logs a warning once per process, without any part of the `data_sources`; background tasks do not log it ([#187](https://github.com/owndev/Open-WebUI-Functions/issues/187))
- **v2.8.0**: Background tasks (title, tags, follow-ups) are sent without `data_sources` and emit no citation/status events ([#123](https://github.com/owndev/Open-WebUI-Functions/issues/123)); new valve `AZURE_AI_SHOW_ALL_CITATIONS_WITHOUT_REFERENCES` (default `true`) to show no sources for answers without `[docX]` references; `tools` and `tool_choice` are dropped (behavior change) and `stream_options` is not forwarded together with `data_sources`; `[docX]` references and links split across streamed chunks are linked once, held back text at the end of a stream reaches the saved message; already linked references are not wrapped again, `[[docX]]` without a link counts as one reference, parentheses in document URLs are percent-encoded and links in the chat history (also older links with parentheses in the URL) are sent back as plain `[docX]`; references to documents that do not exist do not count as references; streamed events of up to 4 MiB (was 128 KiB) are read and a failed stream ends with an `Error: …` message and `data: [DONE]` instead of an empty answer; a non-streamed answer with `content: null` no longer fails with `Error: expected string or bytes-like object`; document content is only logged at `DEBUG` level
- **v2.6.0**: Major refactor - removed `AZURE_AI_ENHANCE_CITATIONS` and `AZURE_AI_OPENWEBUI_CITATIONS` valves; citation support is now always enabled when `AZURE_AI_DATA_SOURCES` is configured; added clickable `[docX]` markdown links; improved score extraction using `filter_reason` field
- **v2.5.x**: Dual citation modes (OpenWebUI events + markdown/HTML)
