# Azure AI Foundry Pipeline - Native OpenWebUI Citations

This document describes the native OpenWebUI citation support in the Azure AI Foundry Pipeline, which enables rich citation cards and source previews in the OpenWebUI frontend.

## Overview

The Azure AI Foundry Pipeline now supports **native OpenWebUI citations** for Azure AI Search (RAG) responses. This feature enables the OpenWebUI frontend to display:

- **Citation cards** with source information and relevance scores
- **Source previews** with content snippets
- **Relevance percentage** displayed on citation cards
- **Interactive citation UI** with clickable sources

## Features

### Dual Citation Modes

The pipeline supports two modes for displaying citations:

1. **Native OpenWebUI Citations** (new): Structured citation events emitted via `__event_emitter__` for frontend consumption
2. **Markdown/HTML Citations** (existing): Collapsible HTML details with formatted citation information

Both modes can be enabled simultaneously or independently via configuration.

### Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `AZURE_AI_OPENWEBUI_CITATIONS` | `true` | Enable native OpenWebUI citation events |
| `AZURE_AI_ENHANCE_CITATIONS` | `true` | Enable markdown/HTML citation display (collapsible sections) |

### How It Works

#### Streaming Responses

When Azure AI Search returns citations in a streaming response:

1. The pipeline detects citations in the SSE (Server-Sent Events) stream
2. **If `AZURE_AI_OPENWEBUI_CITATIONS` is enabled**: Citation events are emitted immediately via `__event_emitter__`
3. **If `AZURE_AI_ENHANCE_CITATIONS` is enabled**: A formatted markdown/HTML citation section is appended at the end of the stream

#### Non-Streaming Responses

When Azure AI Search returns citations in a non-streaming response:

1. The pipeline extracts citations from the response
2. **If `AZURE_AI_OPENWEBUI_CITATIONS` is enabled**: Individual citation events are emitted via `__event_emitter__` for each source
3. **If `AZURE_AI_ENHANCE_CITATIONS` is enabled**: The response content is enhanced with a formatted citation section

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

## Usage Examples

### Basic Setup with Native Citations

```python
# Enable native OpenWebUI citations (default)
AZURE_AI_OPENWEBUI_CITATIONS=true

# Optionally disable markdown/HTML citations if you only want native citations
AZURE_AI_ENHANCE_CITATIONS=false
```

### Both Citation Modes Enabled (Default)

```python
# Enable both native and markdown/HTML citations (default)
AZURE_AI_OPENWEBUI_CITATIONS=true
AZURE_AI_ENHANCE_CITATIONS=true
```

This configuration provides:
- Native citation cards in the OpenWebUI frontend
- Markdown/HTML citation section as fallback for non-supported clients

### Only Markdown/HTML Citations (Legacy)

```python
# Disable native citations, use only markdown/HTML
AZURE_AI_OPENWEBUI_CITATIONS=false
AZURE_AI_ENHANCE_CITATIONS=true
```

## Implementation Details

### Helper Functions

The pipeline includes three new helper functions:

1. **`_extract_citations_from_response()`**: Extracts citations from Azure responses
2. **`_normalize_citation_for_openwebui()`**: Converts Azure citations to OpenWebUI format
3. **`_emit_openwebui_citation_events()`**: Emits citation events via `__event_emitter__`

### Title Fallback Logic

The pipeline uses intelligent title fallback:

1. Use `title` field if available
2. Fallback to `filepath` if title is empty
3. Fallback to `url` if both title and filepath are empty
4. Fallback to `"Unknown Document"` if all are empty

This ensures every citation has a meaningful display name.

### Streaming Citation Emission

Citations are emitted **as soon as they are detected** in the stream, ensuring:
- Low latency for citation display
- Frontend can start rendering citations while content is still streaming
- No waiting for the complete response

### Backward Compatibility

The implementation maintains full backward compatibility:

- Existing markdown/HTML citation display continues to work
- No breaking changes to the API
- Both citation modes can be enabled simultaneously
- Default configuration enables both modes

## Troubleshooting

### Citations Not Appearing

**Problem**: Citations don't appear in the OpenWebUI frontend

**Solutions**:
1. Verify `AZURE_AI_OPENWEBUI_CITATIONS=true` is set
2. Check that Azure AI Search is properly configured (`AZURE_AI_DATA_SOURCES`)
3. Ensure you're using an Azure OpenAI endpoint (not a generic Azure AI endpoint)
4. Check browser console for errors

### Citation Cards vs. Markdown Section

**Problem**: Seeing both citation cards and markdown section

**Solution**: This is the default behavior. To show only citation cards:
```bash
AZURE_AI_OPENWEBUI_CITATIONS=true
AZURE_AI_ENHANCE_CITATIONS=false
```

### Missing Citation Metadata

**Problem**: Some citation fields (URL, filepath, score) are missing

**Solution**: These fields are optional. Azure AI Search may not return all fields depending on your index configuration. The pipeline gracefully handles missing fields.

## References

- [OpenWebUI Pipelines Citation Feature Discussion](https://github.com/open-webui/pipelines/issues/229)
- [OpenWebUI Event Emitter Documentation](https://docs.openwebui.com/features/plugin/development/events)
- [Azure AI Search Documentation](https://learn.microsoft.com/en-us/azure/search/)

## Version History

- **v2.6.0**: Added native OpenWebUI citations support
- **v2.5.x**: Markdown/HTML citation display only
