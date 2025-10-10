# N8N AI Agent Tool Usage Display

## Overview

Starting with version 2.2.0, the N8N pipeline automatically displays AI Agent tool calls in a user-friendly format. When your N8N workflow includes an AI Agent node that uses tools (like Wikipedia, Date/Time, Calculator, etc.), the pipeline will extract and display detailed information about each tool invocation.

## Important Limitation

> [!IMPORTANT]
> **⚠️ Non-Streaming Mode Only**: Tool usage display is currently only available in **non-streaming mode**. N8N's AI Agent streaming responses do not include the `intermediateSteps` field, which is required to show tool calls. This is a limitation of N8N's streaming implementation, not the pipeline.
>
> **To see tool calls**: Configure your N8N workflow to use **non-streaming responses** (remove or disable streaming in the "Respond to Webhook" node).

## Features

### Automatic Detection

- Works with **non-streaming** N8N responses
- Automatically extracts `intermediateSteps` from the N8N response payload
- No additional configuration required

### Rich Display Format

Each tool call is displayed with:

- 🔧 **Tool Name**: The name of the tool that was invoked
- 🆔 **Call ID**: Unique identifier for debugging (e.g., `call_FB0sIgrwuIGJkOaROor7raU2`)
- 📥 **Input**: The parameters passed to the tool (formatted as JSON)
- 📤 **Result**: The tool's response/observation
- 📝 **Log**: Optional log messages from the tool execution

### Collapsible UI

Uses HTML `<details>` tags for a clean, expandable interface:

```txt
🛠️ Tool Calls (3 steps) ▶
  ├─ Step 1: Date_Time ▶
  ├─ Step 2: Wikipedia ▶
  └─ Step 3: Wikipedia ▶
```

Click to expand each step and view full details.

## Example

### N8N Response Format

Your N8N AI Agent workflow should return data in this format:

```json
[
  {
    "output": "Current time in Europe/London: 2025-10-10 09:46:45 BST (UTC+1)...",
    "intermediateSteps": [
      {
        "action": {
          "tool": "Date_Time",
          "toolInput": {
            "Include_Current_Time": true,
            "Timezone": "Europe/London"
          },
          "toolCallId": "call_FB0sIgrwuIGJkOaROor7raU2",
          "log": "Calling Date_Time with input: {...}"
        },
        "observation": "[{\"currentDate\":\"2025-10-10T09:46:45.754+01:00\"}]"
      },
      {
        "action": {
          "tool": "Wikipedia",
          "toolInput": {
            "input": "Europe/London time zone Wikipedia"
          },
          "toolCallId": "call_QFUtaSdUI2PtgjhkDTmbRknt",
          "log": "Calling Wikipedia with input: {...}"
        },
        "observation": "Page: Time zone\nSummary: Time zones are regions..."
      }
    ]
  }
]
```

### UI Display

The user will see:

1. **Main Response**: The agent's text response from the `output` field
2. **Tool Calls Section**: A collapsible section with all tool invocations

## Implementation Details

### Streaming Mode ⚠️

> **Not Supported**: N8N AI Agent does not include `intermediateSteps` in streaming responses. The streaming mode only sends content chunks, not metadata. This is a limitation of N8N's implementation.

### Non-Streaming Mode ✅

- Tool calls are extracted from the complete response JSON
- Supports both array `[{...}]` and object `{...}` response formats
- Automatically detects and formats all tool calls from `intermediateSteps`

### Data Structure Support

The pipeline handles both response formats from N8N:

**Array Format (typical for streaming):**

```json
[
  {
    "output": "...",
    "intermediateSteps": [...]
  }
]
```

**Object Format (typical for non-streaming):**

```json
{
  "output": "...",
  "intermediateSteps": [...]
}
```

## N8N Workflow Configuration

To enable this feature, your N8N workflow must:

1. **Use AI Agent Node**: Include an AI Agent node with tools
2. **Disable Streaming**: In the "Respond to Webhook" node, ensure streaming is disabled
3. **Return intermediateSteps**: Ensure your workflow returns the `intermediateSteps` array in the response

### Example N8N Workflow Structure

```txt
Webhook Trigger
  ↓
AI Agent (with tools: Wikipedia, Date/Time, etc.)
  ↓
Function Node (format response)
  ↓
Respond to Webhook
```

**Function Node Code Example:**

```javascript
// Get the AI Agent output
const agentOutput = $('AI Agent').item.json;

return {
  output: agentOutput.output,
  intermediateSteps: agentOutput.intermediateSteps || []
};
```

## Supported Tools

The display works with any N8N tool, including:

- 📅 Date/Time
- 📚 Wikipedia
- 🔍 Search
- 🧮 Calculator
- 🌐 HTTP Request
- 📧 Email
- 💾 Database queries
- And any custom tools you create!

## Troubleshooting

### Tool Calls Not Showing?

Check that:

1. ✅ Your N8N workflow includes an AI Agent node with tools
2. ✅ The response includes the `intermediateSteps` array
3. ✅ The N8N pipeline version is 2.2.0 or higher
4. ✅ The response structure matches the expected format (see examples above)

### Debugging

Enable debug logging in the pipeline to see:

- Number of intermediate steps found
- Tool call extraction process
- Response parsing details

The pipeline logs helpful messages like:

```txt
Found 3 intermediate steps
Added 3 tool calls to response
```

## Related Documentation

- [N8N Integration Overview](./n8n-integration.md)
- [N8N Template Workflows](../pipelines/n8n/)
- [N8N AI Agent Documentation](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.agent/)
