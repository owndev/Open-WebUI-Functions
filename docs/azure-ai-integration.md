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

# Azure AI Search / RAG (see "Azure AI Search / RAG Integration" below)
# The search configuration in the Azure OpenAI On Your Data data_sources format.
# Works with every chat endpoint and model in the default pipeline mode.
AZURE_AI_DATA_SOURCES='[{"type":"azure_search","parameters":{"endpoint":"https://<your-search-service>.search.windows.net","index_name":"<your-index-name>","authentication":{"type":"api_key"}}}]'

# Azure AI Search API key, stored encrypted (a read-only query key is enough).
# Wins over authentication.key in AZURE_AI_DATA_SOURCES (in both modes).
AZURE_AI_SEARCH_KEY="<your-search-query-key>"

# pipeline (default): the pipeline queries Azure AI Search itself and adds the documents to the prompt
# on_your_data: legacy Azure OpenAI On Your Data (data_sources), retired by Microsoft on October 14, 2026
AZURE_AI_SEARCH_MODE=pipeline

# Pipeline mode: Azure AI Search REST API version (default: 2026-04-01)
AZURE_AI_SEARCH_API_VERSION=2026-04-01

# Pipeline mode: search queries written by the chat model (default: auto)
# auto: on follow-up turns; always: on every turn; off: always search with the user's message
AZURE_AI_SEARCH_QUERY_GENERATION=auto

# Pipeline mode: tokens of document text in the prompt, estimated as characters / 4
# -1 (default): auto, min(32000, 1600 x top_n_documents); 0: no limit; > 0: that many tokens
AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS=-1

# Relevance percentages on citation cards (default: true)
# Pipeline mode: scores of the search hits; on_your_data: adds include_contexts to get them
AZURE_AI_INCLUDE_SEARCH_SCORES=true

# Sources for Azure AI Search answers that contain no [docX] reference (default: true)
# true: show all documents the answer was given; false: show no sources for such answers.
# Answers with [docX] references always show only the referenced documents.
AZURE_AI_SHOW_ALL_CITATIONS_WITHOUT_REFERENCES=true
```

> [!NOTE]
> Open WebUI's own web search has an "Azure AI Search" engine with the environment variables `AZURE_AI_SEARCH_API_KEY`, `AZURE_AI_SEARCH_ENDPOINT` and `AZURE_AI_SEARCH_INDEX_NAME`. They are unrelated to this pipeline, which is why its key valve is called `AZURE_AI_SEARCH_KEY` (an unrelated web-search key must not end up in the pipeline).

### Azure AI Search / RAG Integration

The pipeline supports **Azure AI Search** for **Retrieval-Augmented Generation (RAG)**: answers are grounded in your indexed documents, cite them as `[doc1]`, `[doc2]`, … and show them as citation cards in Open WebUI. The search is configured with `AZURE_AI_DATA_SOURCES` (the Azure OpenAI On Your Data `data_sources` JSON format, so existing configurations keep working). `AZURE_AI_SEARCH_MODE` selects the engine:

| Mode | What happens | Chat endpoints and models |
| --- | --- | --- |
| `pipeline` (default since v2.9.0) | The pipeline queries Azure AI Search itself (Search REST API), adds the documents to the prompt and sends the chat request **without** `data_sources` | Every endpoint and model the pipeline supports: Azure OpenAI deployments, Foundry `/models/chat/completions`, serverless endpoints, GPT-4.1, GPT-5, o-series, non-OpenAI models |
| `on_your_data` (legacy) | The pipeline sends `AZURE_AI_DATA_SOURCES` as `data_sources` to Azure OpenAI On Your Data, which searches and answers (behavior of v2.8.1) | Azure OpenAI endpoints only (`https://<deployment>.openai.azure.com/openai/deployments/<model>/chat/completions?api-version=...`), GPT-4o and GPT-4o-mini only |

The mode is read case-insensitively and `on-your-data` works as well; any other value counts as `pipeline` (with a warning in the log). Without `AZURE_AI_DATA_SOURCES` the mode does not matter: chats are sent as they are. Installs that update from v2.8.x without setting the valve switch to `pipeline` mode.

> [!WARNING]
> **Azure OpenAI On Your Data is retired on October 14, 2026.** The legacy mode `AZURE_AI_SEARCH_MODE=on_your_data` is built on On Your Data (the `data_sources` API). Microsoft has deprecated it and retires the service on **October 14, 2026**; see the [On Your Data API reference](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/references/on-your-data). Microsoft does not document what a request with `data_sources` returns after that date: expect it to fail with an error, or to be answered without your search index and without citations. When Azure answers such a request with HTTP 400 or 404, the pipeline adds `(Azure OpenAI On Your Data was retired on 2026-10-14; set AZURE_AI_SEARCH_MODE=pipeline)` to the error. The default `pipeline` mode does not use On Your Data and is not affected ([#187](https://github.com/owndev/Open-WebUI-Functions/issues/187)).
>
> In `on_your_data` mode the pipeline logs this retirement notice as a warning once per process (since v2.8.1), for the first request that uses `data_sources`; `pipeline` mode never logs it. Microsoft's own recommendation for new solutions is Foundry Agent Service with Foundry IQ ([Connect a Foundry IQ knowledge base](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/foundry-iq-connect)).

In `on_your_data` mode, the On Your Data API reference lists `2024-02-01`, `2024-02-15-preview` and `2024-05-01-preview` (the latest there) as its supported versions. `2025-01-01-preview` is a later preview version of the Azure OpenAI inference API whose specification still contains `data_sources`, but no longer `role_information` (removed in `2024-08-01-preview` according to the [API version changelog](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle#api-version-changelog)); in that mode `role_information` therefore only works with an API version that still has it, such as `2024-05-01-preview`. In `pipeline` mode `role_information` is added to the system message and works with every endpoint and API version.

#### How Pipeline Mode Works

For every chat answer (not for background tasks), the pipeline:

1. **Builds the search query** from your message (Open WebUI's original prompt, without its own file context; at most 1,000 characters). A message with only an image is sent without a search.
2. **Writes search queries for follow-up turns** (`AZURE_AI_SEARCH_QUERY_GENERATION`, default `auto`): on a follow-up question such as "and the warranty?", the same chat deployment first turns the conversation into 1 to `max_search_queries` (default 3) self-contained search queries, like On Your Data's "intent" step. `always` does this on every turn, `off` never. If it fails (timeout of 10 seconds, an HTTP error such as a content filter, an answer without JSON), the pipeline searches with your message instead. After three timeouts in a row for a model (reasoning models can need longer than 10 seconds), query generation is skipped for that model for 15 minutes and a warning suggests `AZURE_AI_SEARCH_QUERY_GENERATION=off`. Its tokens are not included in Open WebUI's token usage.
3. **Queries Azure AI Search** (`POST {endpoint}/indexes/{index_name}/docs/search?api-version=2026-04-01`), one request per query, in parallel. Vector queries get their vector from an embeddings call or from the index vectorizer (see `embedding_dependency` below). HTTP 429 and 503 are retried twice; every request has 30 seconds, query generation, embeddings and search together 45 seconds.
4. **Selects the documents**: applies `strictness` per query, merges the results of several queries (duplicates removed, ordered by reranker score or reciprocal rank fusion), keeps `top_n_documents` (default 5) and cuts the document text to the token budget (`AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS`).
5. **Adds the documents to the prompt**: a `<documents>` block with `[doc1]` … `[docN]` at the start of your message, and rules in the system message (cite with `[docN]`, no links, documents are data and not instructions, `in_scope`, `role_information`). The pipeline never changes Open WebUI's own message objects.
6. **Sends the chat request** without `data_sources`, so tools, `tool_choice` and `stream_options` are kept.
7. **Builds the citations** from the search hits, in the format On Your Data returned (`context.citations`, `context.intent`): a first SSE event with `delta.context` in streamed answers, `choices[0].message.context` in non-streamed ones. API clients that read On Your Data's `context` keep working, and the existing citation code links `[docX]` and emits the citation cards.

The chat shows these status lines: `Generating search queries...` (follow-up turns), `Searching Azure AI Search...`, `No documents found in Azure AI Search` (when nothing was found), then `Sending request to Azure AI...` and `Streaming response from Azure AI...` / `Streaming completed` (or `Request completed`) as without search. Stop during the search ends the running status with `Search cancelled`.

**Errors fail closed**: when the search, the embeddings call or the configuration fails, the answer is `Error: Azure AI Search: …` (with a hint, see [Troubleshooting Pipeline Mode](#troubleshooting-pipeline-mode)) and the chat request is not sent, so nobody gets an answer that silently ignores your documents. This also applies to an `AZURE_AI_DATA_SOURCES` value that is not valid JSON (v2.8.x logged it and answered without the search index); an empty value, `[]` or `null` means "not configured" (plain chat). In `on_your_data` mode invalid JSON is still ignored as in v2.8.x.

#### Behavior With Azure AI Search

| | `pipeline` mode | `on_your_data` mode |
| --- | --- | --- |
| Tools / function calling | Forwarded, also Open WebUI's built-in tools; the rules allow the model to use tool results | `tools` and `tool_choice` are always dropped: with tools Azure [ignores the data sources](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/concepts/use-your-data#function-calling) unless `tool_choice` is `none` (since v2.8.0) |
| Token usage in streamed answers | Yes (`stream_options.include_usage`) | No: On Your Data rejects `stream_options` (`Extra inputs are not permitted`) |
| Background tasks (title, tags, follow-ups) | No search, no citations, no status messages | Sent without `data_sources`, no citations, no status messages |
| History | `[[docX]](url)` links of earlier answers are sent back as plain `[docX]`; the documents are added only to the current message | Same |
| `data_sources` sent by the client (API or inlet filter) | **Error**: `data_sources in the request is not supported in pipeline mode; remove it (the search is configured by AZURE_AI_DATA_SOURCES) or set AZURE_AI_SEARCH_MODE=on_your_data`, with or without `AZURE_AI_DATA_SOURCES`. The pipeline never calls an endpoint or forwards a key or `filter` chosen by a client, and never silently replaces a client's per-user `filter` with the valve's search. An empty list is ignored | Forwarded to Azure as before |
| Large contexts | Limited by the token budget | Azure sends the documents of a streamed answer in one SSE event; events of up to 4 MiB are read (see [Streamed Answer Ends With an Error](azure-ai-citations.md#streamed-answer-ends-with-an-error)) |
| Context window | The documents come on top of Open WebUI's full chat history (On Your Data cut the history to 2,000 tokens). A chat that grows beyond the model's context window fails with `context_length_exceeded`; the error then suggests lowering `AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS` or `top_n_documents`, or starting a new chat | Azure manages the context |

**Tool rounds**: with native function calling, Open WebUI calls the pipeline again for every tool round of an answer. The pipeline reuses the documents of the first round (same `[docN]` numbering, no second search) and emits every source once. The documents are kept per Open WebUI worker process for 10 minutes (at most 128 answers, none larger than 1 MB); with several workers or replicas a tool round can land on another process, which searches again and may add a source twice.

#### Alternatives

- **Open WebUI Knowledge** (Workspace → Knowledge) with its own document store, if the documents do not have to live in Azure AI Search.
- **Open WebUI web search with the "Azure AI Search" engine** (`AZURE_AI_SEARCH_*` environment variables of Open WebUI): searches an index as a web-search source, independent of this pipeline.
- **Foundry Agent Service with Foundry IQ** (knowledge bases), Microsoft's recommended successor of On Your Data; a separate agent setup, not used by this pipeline.

#### 📖 Official Documentation

For detailed information about Azure AI Search configuration, please refer to:

- 🔍 [Azure Search Parameters Reference](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/references/azure-search?tabs=rest) (the format of `AZURE_AI_DATA_SOURCES`)
- 🔎 [Azure AI Search: Documents - Search Post](https://learn.microsoft.com/en-us/rest/api/searchservice/documents/search-post) (the request pipeline mode sends), [hybrid search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-how-to-query), [vector queries](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-query), [semantic ranking](https://learn.microsoft.com/en-us/azure/search/semantic-how-to-query-request)
- 📚 [Azure OpenAI On Your Data - Concepts and Setup](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/concepts/use-your-data) and [Data Sources API Reference](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/references/on-your-data?tabs=rest#data-source) (`on_your_data` mode)

#### ⚙️ Configuration

Configure Azure AI Search by setting the `AZURE_AI_DATA_SOURCES` environment variable with your Azure Search configuration, and the search key in the encrypted `AZURE_AI_SEARCH_KEY` valve.

**Simple Example:**

```bash
AZURE_AI_DATA_SOURCES='[{"type":"azure_search","parameters":{"endpoint":"https://my-search.search.windows.net","index_name":"my-index","authentication":{"type":"api_key"}}}]'
AZURE_AI_SEARCH_KEY="your-search-query-key"
```

> [!TIP]
> **Copy the JSON exactly as shown above** - this is the complete configuration that goes into the `AZURE_AI_DATA_SOURCES` field. Just replace:
>
> - `my-search` with your Azure Search service name
> - `my-index` with your search index name
> - `your-search-query-key` with a query key of your Azure Search service (a read-only query key is enough; an admin key works too)
>
> The key can also stay in the JSON as `"authentication":{"type":"api_key","key":"..."}` as in v2.8.x, but `AZURE_AI_DATA_SOURCES` is stored in plain text. When both are set, `AZURE_AI_SEARCH_KEY` wins and the log asks you to remove the key from the JSON. In `on_your_data` mode the pipeline puts `AZURE_AI_SEARCH_KEY` into the `data_sources` it sends to Azure (only into the ones from `AZURE_AI_DATA_SOURCES`, never into `data_sources` sent by a client).

#### 📋 Complete Configuration Template

```json
[
  {
    "type": "azure_search",
    "parameters": {
      "endpoint": "https://YOUR-SEARCH-SERVICE.search.windows.net",
      "index_name": "YOUR-INDEX-NAME",
      "authentication": {
        "type": "api_key"
      }
    }
  }
]
```

With the key in `AZURE_AI_SEARCH_KEY`. Pipeline mode accepts any absolute `http://` or `https://` endpoint (a trailing `/` is removed); the index name is URL-encoded into the path.

#### Authentication of the Search

| `authentication.type` | Pipeline mode | `on_your_data` mode |
| --- | --- | --- |
| missing or `api_key` | `api-key` header with `AZURE_AI_SEARCH_KEY`, else `authentication.key`; neither set is a configuration error | Azure uses the key (`AZURE_AI_SEARCH_KEY` is put into the request) |
| `access_token` (`access_token`) | `Authorization: Bearer <access_token>`. A static token expires (about one hour): for tests only | Azure uses the token |
| `system_assigned_managed_identity` | **Meaning changed**: a Microsoft Entra token of the **Open WebUI host's** identity (App Service, Container Apps, VM, AKS workload identity; `AZURE_CLIENT_ID` selects a user-assigned identity; `az login` during development), minted with `azure-identity` (`DefaultAzureCredential`, scope `https://search.azure.com/.default`) | The identity of the Azure OpenAI resource |
| `user_assigned_managed_identity` (`managed_identity_resource_id`) | **Meaning changed**: the host's user-assigned identity with that resource ID (`ManagedIdentityCredential`); if your hosting platform does not accept a resource ID, use `system_assigned_managed_identity` with `AZURE_CLIENT_ID` set to the identity's client ID | The user-assigned identity of the Azure OpenAI resource |
| `connection_string`, `key_and_key_id`, `encoded_api_key`, `username_and_password` | Configuration error (not valid for `azure_search`) | Sent to Azure as is |

With a managed identity the identity of the Open WebUI host needs the **Search Index Data Reader** role on the search service (or the index), and the search service must allow role-based access ("Role-based access control" or "Both" under Keys); role assignments can take 5-10 minutes to take effect. `azure-identity` ships with Open WebUI 0.11.4 and is imported only when a managed identity is configured; if it is missing, the error asks for `AZURE_AI_SEARCH_KEY` instead. Tokens are cached per process until 5 minutes before they expire. Sovereign clouds (other Search scopes) are not supported yet. The key, tokens, request headers and the `AZURE_AI_DATA_SOURCES` JSON are never logged and never shown in an error.

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
        "type": "api_key"
      },
      "query_type": "vector_semantic_hybrid",
      "semantic_configuration": "default",
      "embedding_dependency": {
        "type": "deployment_name",
        "deployment_name": "YOUR-EMBEDDING-DEPLOYMENT"
      },
      "fields_mapping": {
        "content_fields": ["content"],
        "title_field": "title",
        "url_field": "url",
        "filepath_field": "filepath",
        "vector_fields": ["contentVector"]
      },
      "top_n_documents": 20,
      "strictness": 3
    }
  }
]
```

Property keys and enum values are snake case (`vector_semantic_hybrid`, not `vectorSemanticHybrid`; pipeline mode also accepts the camel-case form). `vector`, `vector_simple_hybrid` and `vector_semantic_hybrid` need a query vector, `semantic` and `vector_semantic_hybrid` need `semantic_configuration` (the name of a semantic configuration of your index, or the index default); without vector fields or a semantic configuration, omit `query_type` (default `simple`). See the [Azure Search Parameters Reference](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/references/azure-search?tabs=rest#parameters) for all parameters.

The query vector in pipeline mode comes from `embedding_dependency`:

- `deployment_name` (as above): the pipeline calls `POST <base>/openai/v1/embeddings` with `{"model": "<deployment_name>", "input": [...]}` and the chat key (`AZURE_AI_API_KEY`, also as Bearer with `AZURE_AI_USE_AUTHORIZATION_HEADER`), where `<base>` is `AZURE_AI_ENDPOINT` cut before `/openai/` or `/models/` (a gateway path such as `https://apim.example.com/aoai` is kept). The embedding deployment must be in the same resource as the chat endpoint. Serverless chat endpoints have no embedding deployments: use `endpoint` there.
- `endpoint` (`endpoint`, `authentication`): `POST` to that URL with `{"input": [...]}`. A `.../openai/deployments/<name>/embeddings` URL without `api-version` gets `api-version=2024-10-21`; an `/openai/v1/` URL is a configuration error (it needs the deployment name: use `deployment_name`). `authentication` is `api_key` (`key`) or `access_token` (`access_token`); without it the chat key is used only when the URL has the same host as `AZURE_AI_ENDPOINT`. The embedding key in the JSON is stored in plain text, so prefer `deployment_name` or `integrated`.
- `integrated`, or **no** `embedding_dependency`: the index vectorizer turns the text into the vector (`vectorQueries` of kind `text`, no embeddings call). This needs a vectorizer on the vector field; without one, Azure AI Search answers HTTP 400 and the error says to set `embedding_dependency`. **Configurations copied from older versions of this guide** (`vectorSimpleHybrid` without `embedding_dependency` and without `vector_fields`) therefore fail in pipeline mode unless the index has a vectorizer on `contentVector`: add `embedding_dependency` (and `fields_mapping.vector_fields` if your vector field has another name).
- `dimensions` (optional, `text-embedding-3` models only) is sent when set. `model_id` (Elasticsearch only) is a configuration error.

Without `fields_mapping.vector_fields` a vector query searches the field `contentVector` (the name Microsoft's tools use) and a warning is logged.

#### Parameters in Pipeline Mode

Pipeline mode reads the first entry with `"type": "azure_search"` (with several entries a warning is logged; other types such as `azure_cosmos_db` or `elasticsearch` are only supported in `on_your_data` mode and give an error). It adds no keys of its own to the JSON (in `on_your_data` mode the JSON is sent to Azure as is, which rejects unknown keys); everything pipeline-specific is a valve. Keys it does not know are ignored with a warning that lists their names.

| Parameter | Pipeline mode | |
| --- | --- | --- |
| `endpoint`, `index_name` | `POST {endpoint}/indexes/{index_name}/docs/search?api-version={AZURE_AI_SEARCH_API_VERSION}` | supported |
| `authentication` | see [Authentication of the Search](#authentication-of-the-search) | supported / managed identity changed |
| `query_type` | `simple` (default): `queryType: simple`; `semantic`: `queryType: semantic` + `semanticConfiguration` + `semanticErrorHandling: partial`; `vector`: `vectorQueries` only; `vector_simple_hybrid`: text + vector (RRF); `vector_semantic_hybrid`: text + vector + semantic ranking (`k` 50) | supported |
| `semantic_configuration` | `semanticConfiguration` (left out when not set: the index default is used) | supported |
| `filter` | `filter`, verbatim (OData). One static filter per pipeline instance, see [Security Trimming](#security-trimming-per-user-filters) | supported |
| `top_n_documents` (default 5) | documents in the prompt (1-50); the search asks for `top` = 2 × `top_n_documents` (at least 10, at most 50) to leave room for `strictness` and duplicates | supported |
| `strictness` (default 3) | [approximation](#strictness-top_n_documents-and-the-token-budget) | approximated |
| `in_scope` (default `true`) | prompt rule: answer only from the documents (and from files in the message and tool results), otherwise say that the information is not in the retrieved documents; `false`: own knowledge allowed, marked as such | approximated |
| `role_information` | added to the system message (at most 16,000 characters) | approximated |
| `fields_mapping` | see [Index Schema and Field Mapping](#index-schema-and-field-mapping-for-citations); `image_vector_fields` is ignored | supported |
| `embedding_dependency` | see above | supported |
| `max_search_queries` (default 3) | at most this many generated queries (1-5) | supported |
| `allow_partial_result` (default `false`) | with several generated queries: `false` fails the answer when one search fails, `true` continues with the others | supported |
| `include_contexts` | ignored; scores come from the search hits (`AZURE_AI_INCLUDE_SEARCH_SCORES`) | ignored |

The search text is your message as plain text for the simple parser (never Lucene syntax); a `-` at the start of a word, which the parser would read as "NOT", is replaced by a space.

#### Strictness, `top_n_documents` and the Token Budget

Azure AI Search has no `strictness`; pipeline mode approximates On Your Data's filter **per query** before the results are merged (the results are not identical to On Your Data):

| `strictness` | Semantic types: drop if the reranker score (0-4) is below | Other types (BM25, vector, hybrid): drop if `@search.score` is below this share of the best hit of the same query |
| --- | --- | --- |
| 1 | 0 (nothing dropped) | 0 |
| 2 | 1.0 | 10 % |
| 3 (default) | 1.5 | 25 % |
| 4 | 2.0 | 50 % |
| 5 | 2.5 | 75 % |

All documents can be dropped; the model then gets "No documents were found for this question." (with `in_scope: true`) and says that the information is not available.

The document text in the prompt is limited by `AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS`, estimated as characters / 4. The default `-1` means `min(32000, 1600 × top_n_documents)` tokens: 8,000 for the default 5 documents, 32,000 for 20 or more. When the documents are longer, each is cut to an equal share (at a word boundary, marked with ` …`); if that share would be below 500 characters, the lowest-ranked document is dropped instead. `0` switches the limit off. The budget costs input tokens on every answer, and characters / 4 fits English text: German needs about 3 characters per token and Chinese or Japanese about 1, so the same budget can be 1.3-4 times more tokens there. The citation card shows exactly the text the model received.

Without `fields_mapping`, Azure AI Search returns every retrievable field of each hit, including retrievable vector fields (about 60 KB of JSON per hit for a 3,072-dimension vector). Set `fields_mapping` (then only the mapped fields are requested with `select`) or make vector fields non-retrievable.

#### Security Trimming (Per-User Filters)

On Your Data's documented way to show each user only their documents is a per-request `filter`, for example `data_sources` added by an inlet filter or an API client. Pipeline mode refuses `data_sources` in the request with an error (see [Behavior With Azure AI Search](#behavior-with-azure-ai-search)) instead of silently searching with the valve's configuration, which could show documents the client's filter excludes. Pipeline mode has **one static `filter` per pipeline instance** (from `AZURE_AI_DATA_SOURCES`); per-user filters are a planned follow-up. Until then, use one pipeline instance (and Open WebUI model permissions) per group of users, or `AZURE_AI_SEARCH_MODE=on_your_data` until October 14, 2026.

#### Index Schema and Field Mapping for Citations

A common source of confusion is understanding **which fields your Azure AI Search index needs** and **how to map them** so that citations (with titles, URLs, and content) work correctly in OpenWebUI.

##### How the Index Fields Are Used

Every document of an answer becomes a citation object with `title`, `content`, `url`, `filepath` and `chunk_id`, which the pipeline renders as a citation card in OpenWebUI. In `pipeline` mode the pipeline builds these objects from the search hits itself; in `on_your_data` mode Azure OpenAI builds them and returns them in the answer.

In pipeline mode the fields come **only from the configuration** (the pipeline does not read the index definition, which a query key or the Search Index Data Reader role cannot do):

- **Without `fields_mapping`** the hit fields `content`, `title`, `url` and `filepath` are read (missing ones stay empty) and no `select` is sent. If a hit has no `content` field, its other text fields (except title, url and filepath) are used as content and a warning asks you to set `fields_mapping.content_fields`. On Your Data had an undocumented automatic mapping, which pipeline mode does not reproduce.
- **With `fields_mapping`** only the mapped fields are requested (`select`): a role the mapping leaves out stays empty (there is no fallback to the default name, which could be an unknown field and fail the search with HTTP 400). The card title then falls back to the file path, the URL or "Unknown Document". A mapping without `content_fields` sends no `select` and guesses the content as described above.
- Field values that are lists (`Collection(Edm.String)`) are joined (content with `content_fields_separator`, title / url / filepath with `, `), numbers and booleans are converted to text, vectors are skipped.

If the field names in your index don't match the default names, provide an explicit `fields_mapping` in your `AZURE_AI_DATA_SOURCES` configuration.

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

If you created your own index with different field names, you **must** configure `fields_mapping` so that the pipeline (or Azure OpenAI in `on_your_data` mode) knows which fields to use for citations.

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
        "type": "api_key"
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
| `vector_fields` | `string[]` | Fields containing vector embeddings (at most 10). Default in pipeline mode: `contentVector` | Required for vector/hybrid search |
| `content_fields_separator` | `string` | Separator between multiple content fields. Default: `\n` | -- |
| `image_vector_fields` | `string[]` | Image vector fields | Ignored in pipeline mode |

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
[{"type":"azure_search","parameters":{"endpoint":"https://YOUR-SEARCH-SERVICE.search.windows.net","index_name":"my-docs-index","authentication":{"type":"api_key"}}}]
```

with the key in `AZURE_AI_SEARCH_KEY`.

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
> - Make all citation-related fields (`title`, `filepath`, `url`, `content`) **retrievable** so the search can return them for the citations.
> - Pipeline mode with `vector_*` query types and no `embedding_dependency` needs a vectorizer on `contentVector` (this example has none, so set `embedding_dependency` or add a vectorizer).

##### How the Pipeline Uses Citation Fields

The pipeline reads these fields from each citation (in pipeline mode built from the search hit, in `on_your_data` mode returned by Azure) to build OpenWebUI citation cards:

```text
Search hit (pipeline mode)        Citation                         →  OpenWebUI Citation Card
──────────────────────────────────────────────────────────────────────────────────────────────
title_field (default title)       citation.title                   →  Card title (with [docX] prefix)
content_fields (default content)  citation.content                 →  Document preview (the text the model received)
url_field (default url)           citation.url                     →  Clickable link on [docX] references
filepath_field (default filepath) citation.filepath                →  Fallback for title and URL
chunk_id                          citation.chunk_id                →  Informative (on_your_data: score matching)
@search.score                     citation.original_search_score   →  Relevance % (BM25, vector, hybrid)
@search.rerankerScore             citation.rerank_score            →  Relevance % (semantic reranker)
(computed, pipeline mode only)    citation.relevance               →  Relevance % shown on the card
(on_your_data only)               citation.filter_reason           →  Selects which score to display
```

#### 🚀 Quick Setup Steps

1. **Create Azure Search Service** - Set up an Azure Search service in the Azure portal
2. **Create and populate index** - Upload your documents to a search index (ensure fields like `title`, `filepath`, `url`, and `content` are present and **retrievable**)
3. **Get a query key** - Copy a query key (or an admin key) from your Azure Search service into `AZURE_AI_SEARCH_KEY`, or give the Open WebUI host's managed identity the Search Index Data Reader role
4. **Configure field mappings** - If your index uses custom field names, add `fields_mapping` to your `AZURE_AI_DATA_SOURCES` JSON
5. **Configure pipeline** - Add the `AZURE_AI_DATA_SOURCES` environment variable; keep `AZURE_AI_SEARCH_MODE=pipeline` (default), which works with every chat endpoint
6. **Only for `on_your_data` mode**: use an Azure OpenAI endpoint (`https://<deployment>.openai.azure.com/openai/deployments/<model>/chat/completions?api-version=...`) with GPT-4o or GPT-4o-mini

#### ⚠️ Common Issues

- **Wrong endpoint format** (`on_your_data` mode only): Make sure you're using Azure OpenAI URLs, not regular Azure AI endpoints
- **Invalid JSON**: Copy the JSON template exactly and only change the placeholder values; in pipeline mode every answer then ends with `Error: Azure AI Search: AZURE_AI_DATA_SOURCES is not valid JSON (line L column C)`
- **Missing API key**: Ensure your Azure Search API key has proper permissions
- **Index not found**: Verify your index name matches exactly (case-sensitive)
- **Citations showing "Unknown Document"**: Your index is missing `title`, `filepath`, or `url` fields, or those fields are not set as **retrievable**
- **No clickable links on `[docX]`**: Your index has no `url` field, or `url_field` is not mapped in `fields_mapping`
- **Custom field names not working**: Add a `fields_mapping` object to your `AZURE_AI_DATA_SOURCES` configuration (see [Index Schema and Field Mapping](#index-schema-and-field-mapping-for-citations) above)

#### Troubleshooting Pipeline Mode

Errors are shown as the answer and as the final status, always starting with `Error: Azure AI Search:`; the server log has the same line at `ERROR` level (`Error in Azure AI request: Azure AI Search: …`), plus Search's `request-id` at `INFO` level for Microsoft support. They never contain the key, a token or the JSON.

| Error | Cause and fix |
| --- | --- |
| `HTTP 401` / `HTTP 403 from index '…': … Check the key, or that the identity has the Search Index Data Reader role` | Wrong or missing key; with a managed identity: the Open WebUI host's identity lacks **Search Index Data Reader** (assignments take up to 10 minutes) or the service does not allow role-based access. The identity is no longer the Azure OpenAI resource's (as with On Your Data) |
| `HTTP 402: the semantic ranker's free monthly quota is used up` | Enable the standard plan of the semantic ranker, or use `query_type` `simple` |
| `HTTP 400: … (check fields_mapping)` | A mapped field does not exist or is not retrievable |
| `HTTP 400: … (check vector_fields, dimensions and embedding_dependency…)` | Wrong vector field name, a vector of the wrong length (embedding model or `dimensions` differ from the index), or a `vector_*` query type without `embedding_dependency` on an index without vectorizer |
| `HTTP 400: the filter in AZURE_AI_DATA_SOURCES is invalid` | The OData `filter` is wrong; Azure's message (which can quote the filter) is logged at `DEBUG` only |
| `index '…' not found at <host>` | Wrong `index_name` or `endpoint` |
| `the search service is throttling requests (HTTP 503)` | The service is overloaded (also after two retries); add replicas or try again later |
| `no answer from <host> within 30 s` / `within the 45 s retrieval limit` / `cannot connect to <host>` | Network, firewall or private endpoint between the Open WebUI host and the search service (in pipeline mode Open WebUI, not Azure OpenAI, calls the search service) |
| `HTTP 302 from <host>; check parameters.endpoint` | Redirects are not followed (the key would be sent to another host) |
| `embeddings request failed (HTTP …): … Check embedding_dependency.` | Wrong embedding deployment name, key or URL |
| `data source type '…' is only supported with AZURE_AI_SEARCH_MODE=on_your_data` | Pipeline mode supports `azure_search` only |
| `no API key; set AZURE_AI_SEARCH_KEY or authentication.key in AZURE_AI_DATA_SOURCES` | Set the key |
| `data_sources in the request is not supported in pipeline mode; …` | A client or an inlet filter sends `data_sources`; see [Security Trimming](#security-trimming-per-user-filters) |
| `… (lower AZURE_AI_SEARCH_MAX_CONTEXT_TOKENS or top_n_documents, or start a new chat)` | The chat request with the documents exceeds the model's context window |

Answers that are not grounded although documents were found usually mean that the model ignores the rules; check the documents in the citation cards and set `role_information` or a system prompt. Query generation failures are logged at `INFO` level (`query generation failed (…); searching with the user message`) and never fail the answer.

#### Native OpenWebUI Citation Support

The pipeline automatically provides native OpenWebUI citation support for Azure AI Search responses. When Azure AI Search is configured, the pipeline:

1. **Emits citation events** via `__event_emitter__` for the OpenWebUI frontend to display interactive citation cards
2. **Converts `[docX]` references** to clickable markdown links that link directly to document URLs
3. **Extracts relevance scores** when `AZURE_AI_INCLUDE_SEARCH_SCORES=true`
4. **Filters citations** to only show documents actually referenced in the response (for an answer without any `[docX]` reference: all documents, or none with `AZURE_AI_SHOW_ALL_CITATIONS_WITHOUT_REFERENCES=false`)

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
- **Streaming support** with links converted inline as content streams, also when a `[docX]` reference or a link around one arrives in several pieces
- **No double links**: references that are already markdown links are not wrapped again; parentheses in document URLs are percent-encoded (`%28`, `%29`) so each link ends at its own `)`

**Relevance Score Selection:**

In pipeline mode each citation carries a `relevance` (0-1) computed from its search hit:

- semantic types (`semantic`, `vector_semantic_hybrid`): reranker score (0-4) / `RERANK_SCORE_MAX` (default 4.0)
- `simple` (BM25): `@search.score` / `BM25_SCORE_MAX` (default 100)
- `vector` over one field: the similarity score as is (0.333-1 for cosine)
- `vector_simple_hybrid`, `vector` over several fields and a semantic hybrid answer without reranker scores (reciprocal rank fusion, about 0.016 per fused list): rescaled by its rank, `score × 60 / number of fused lists`, so a top hit shows close to 100 % instead of 1-3 %

In `on_your_data` mode the pipeline uses the `filter_reason` field from Azure Search to select the appropriate score:

- `filter_reason="rerank"` → uses `rerank_score`
- `filter_reason="score"` or not present → uses `original_search_score`

For more details, see the [Azure AI Citations Documentation](azure-ai-citations.md).

> [!TIP]  
> To use **Azure OpenAI** and other **Azure AI** models **simultaneously**, you can use the following URL: `https://<your project>.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview`

## Authentication Modes

The pipeline supports two authentication headers:

| Mode | Header | When to use |
| --- | --- | --- |
| `api-key` (default) | `api-key: <key>` | Standard Azure AI API key authentication |
| Bearer token | `Authorization: Bearer <token>` | Managed identity, Entra ID tokens, or services that require Bearer auth |

**Configuration:**

```bash
# Use Bearer token instead of api-key header
AZURE_AI_USE_AUTHORIZATION_HEADER=true
```

## Token Usage Tracking

The pipeline automatically requests token usage metadata and returns it to Open WebUI so it is saved to the database and shown in the UI.

### How it works

- **Streaming mode**: `stream_options: {"include_usage": true}` is automatically added to every streaming request without `data_sources`, also to chats with Azure AI Search in pipeline mode. The final SSE chunk contains the usage object which the Open WebUI middleware extracts. The tokens of the query generation call of pipeline mode are not included.
- **Non-streaming mode**: The response body already contains the standard `usage` field returned by the Azure / OpenAI API.
- **`on_your_data` mode (`data_sources`)**: Azure OpenAI On Your Data does not support `stream_options`, so it is removed from these requests (also when Open WebUI adds it). Streaming answers with data sources have no token usage.

No additional configuration is required.

## Function Calling (Tools)

Since Open WebUI 0.10, **Native** function calling is the default: in chats in the web UI, Open WebUI adds its built-in tools (27 in Open WebUI 0.11.4, for example `get_current_timestamp`, memory and notes tools) and any selected tools to the request as `tools`. The pipeline forwards them to Azure, also in chats with Azure AI Search in pipeline mode (since v2.9.0; the documents of the answer are reused for every tool round). Only requests with `data_sources` (`on_your_data` mode) have `tools` and `tool_choice` dropped, since v2.8.0 (see [Behavior With Azure AI Search](#behavior-with-azure-ai-search)).

If a deployment does not support tool calling, Azure returns an error for these requests. Switch the model to **Legacy** function calling: per chat in the chat controls, per model in the model's advanced parameters (Workspace → Models), or for all models in the default model parameters of the admin settings. In Legacy mode Open WebUI selects tools with a separate prompt and does not send `tools` to the model. Alternatively, disable the built-in tools for that model in its settings.

## Model Selection

The pipeline resolves which model(s) to expose in Open WebUI in the following priority order:

1. **`AZURE_AI_MODEL`** – Explicit model name(s). Supports semicolons, commas, or spaces as separators (e.g. `gpt-4o;gpt-4o-mini`). Each model becomes its own entry in the model list.
2. **URL extraction** – If `AZURE_AI_MODEL` is empty and the endpoint URL follows the Azure OpenAI pattern `…/deployments/<model>/chat/completions`, the model name is extracted automatically.
3. **`USE_PREDEFINED_AZURE_AI_MODELS=true`** – Exposes a curated catalogue of the most popular Azure AI Foundry models including GPT-4o, GPT-5, o3, o4-mini, Phi-4, DeepSeek-R1/V3, Mistral, Llama 3.x, Cohere Command, Grok and more.
4. **Fallback** – A single generic `azure_ai` entry using the pipeline prefix.

### Model in Body vs Header

By default the model name is sent via the `x-ms-model-mesh-model-name` HTTP header (used by the Azure AI Models-as-a-Service endpoint). Set `AZURE_AI_MODEL_IN_BODY=true` to place the model name in the JSON request body instead — required for Azure OpenAI deployments.

Model names that contain dots, such as `gpt-4.1` or `Phi-3.5-mini-instruct`, are sent unchanged in both modes (v2.7.0 shortened them to `1` / `5-mini-instruct` for non-streaming requests and requests with Azure AI Search).

```bash
# Include model name in request body (required for Azure OpenAI)
AZURE_AI_MODEL_IN_BODY=true
```
