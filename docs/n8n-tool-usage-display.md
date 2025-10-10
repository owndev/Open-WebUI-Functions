# N8N AI Agent Tool Usage Display

## Overview

Starting with version 2.2.0, the N8N pipeline automatically displays AI Agent tool calls in a user-friendly format. When your N8N workflow includes an AI Agent node that uses tools (like Wikipedia, Date/Time, Calculator, etc.), the pipeline will extract and display detailed information about each tool invocation.

Version 2.2.0 includes **configurable verbosity levels** and **length limits** for tool display, allowing you to control how detailed the tool information is shown.

## Important Limitation

> [!IMPORTANT]
> **⚠️ Non-Streaming Mode Only**: Tool usage display is currently only available in **non-streaming mode**. N8N's AI Agent streaming responses do not include the `intermediateSteps` field, which is required to show tool calls. This is a limitation of N8N's streaming implementation, not the pipeline.
>
> **To see tool calls**: Configure your N8N workflow to use **non-streaming responses** (remove or disable streaming in the "Respond to Webhook" node).

## Features

### Automatic Detection

- Works with **non-streaming** N8N responses
- Automatically extracts `intermediateSteps` from the N8N response payload
- Configurable verbosity levels (v2.2.0+)
- Customizable length limits for inputs and outputs (v2.2.0+)

### Verbosity Levels (v2.2.0+)

> [!TIP]
> **Configure via Pipeline Settings**: Set `TOOL_DISPLAY_VERBOSITY` to control the detail level.

#### 1. **Minimal** (`minimal`)

Shows only tool names in a collapsible list:

```txt
🛠️ Tool Calls (4 steps) ▶
  1. Date_Time
  2. Crypto
  3. Wikipedia
  4. Calculator
```

**Best for**: Quick overview, minimal UI clutter, collapsible for space-saving

#### 2. **Compact** (`compact`)

Shows tool names with short result previews:

```txt
🛠️ Tool Calls (4 steps) ▶
  Step 1: Date_Time → [{"currentDate":"2025-10-10T11:22:38.697+01:00"}]
  Step 2: Crypto → [{"Property_Name":"uuid4","uuid4":"c28495ca-1eb8-419f-8941-7a38c753e809"}]
  Step 3: Wikipedia → Page: List of tz database time zones...
  Step 4: Calculator → 128675.5415958103
```

**Best for**: Balance between overview and detail

#### 3. **Detailed** (`detailed`) - Default

Shows full collapsible sections with all information:

```txt
🛠️ Tool Calls (4 steps) ▶
  ├─ Step 1: Date_Time ▶
  │   🔧 Tool: Date_Time
  │   🆔 Call ID: call_i7JgGhwh7xV0ghqR9mD4qTJ9
  │   📥 Input: {"Include_Current_Time": true, "Timezone": "Europe/London"}
  │   📤 Result: [{"currentDate":"2025-10-10T11:22:38.697+01:00"}]
  │   📝 Log: Invoking "Date_Time" with...
  ├─ Step 2: Crypto ▶
  └─ ...
```

**Best for**: Debugging, full transparency

### Length Limits (v2.2.0+)

> [!NOTE]
> **Control Output Size**: Set `TOOL_INPUT_MAX_LENGTH` and `TOOL_OUTPUT_MAX_LENGTH` to limit text length.

- **`TOOL_INPUT_MAX_LENGTH`** (default: 500): Maximum characters for tool input display in `detailed` and `compact` modes
- **`TOOL_OUTPUT_MAX_LENGTH`** (default: 500): Maximum characters for tool output/observation display
- Set to **0** for unlimited length (no truncation)

> [!IMPORTANT]
> **Behavior with `0` (unlimited)**:
>
> - **Detailed mode**: Shows complete input and output without any truncation
> - **Compact mode**: For inputs, shows full data. For outputs, still uses a 100-character preview for UI readability

**Example**: For very long Wikipedia results, set `TOOL_OUTPUT_MAX_LENGTH` to 200 to show only the first 200 characters, or set to 0 to show everything.

### Rich Display Format (Detailed Mode)

Each tool call is displayed with:

- 🔧 **Tool Name**: The name of the tool that was invoked
- 🆔 **Call ID**: Unique identifier for debugging (e.g., `call_FB0sIgrwuIGJkOaROor7raU2`)
- 📥 **Input**: The parameters passed to the tool (formatted as JSON, respects `TOOL_INPUT_MAX_LENGTH`)
- 📤 **Result**: The tool's response/observation (respects `TOOL_OUTPUT_MAX_LENGTH`)
- 📝 **Log**: Optional log messages from the tool execution (max 200 chars)

### Collapsible UI

Uses HTML `<details>` tags for a clean, expandable interface:

Click to expand each step and view full details (in detailed mode).

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

## Configuration

### Pipeline Settings

Configure tool display in the N8N pipeline settings (Admin Panel → Functions → N8N Pipeline):

| Setting | Default | Description |
|---------|---------|-------------|
| `TOOL_DISPLAY_VERBOSITY` | `detailed` | Display mode: `minimal`, `compact`, or `detailed` |
| `TOOL_INPUT_MAX_LENGTH` | `500` | Maximum characters for tool input. Set to `0` for unlimited (no truncation) |
| `TOOL_OUTPUT_MAX_LENGTH` | `500` | Maximum characters for tool output. Set to `0` for unlimited in detailed mode (compact mode uses 100 char preview) |

### Recommendations

> [!TIP]
> **For Production Use**:
>
> - Use `compact` mode for cleaner UI with essential info
> - Set `TOOL_OUTPUT_MAX_LENGTH` to 200-300 for long outputs like Wikipedia
>
> **For Development/Debugging**:
>
> - Use `detailed` mode to see all information
> - Set lengths to 0 (unlimited) to see complete data
>
> **For Minimal UI**:
>
> - Use `minimal` mode to just show tool names
> - Perfect for when you only need to know which tools were called

## Troubleshooting

### Tool Calls Not Showing?

Check that:

1. ✅ Your N8N workflow includes an AI Agent node with tools
2. ✅ The response includes the `intermediateSteps` array
3. ✅ The N8N pipeline version is 2.2.0 or higher
4. ✅ The response structure matches the expected format (see examples above)
5. ✅ You're using non-streaming mode (streaming doesn't support tool display)

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
