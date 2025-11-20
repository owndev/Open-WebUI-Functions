# Code Interpreter / Tool Call Fix for Gemini Pipeline

## Problem

The Google Gemini pipeline did not properly handle tool calls (function calls) in responses from the Gemini API. This caused the code interpreter feature in Open WebUI to not work - when users asked Gemini to execute code, the response would repeat text instead of executing the code.

## Root Cause

The Gemini pipeline was configured to send tools to the API (via `__tools__` parameter) but did not handle tool call responses. When Gemini returned a response containing a `function_call` part (indicating it wants to call a tool like the code interpreter), the pipeline:

1. **Streaming mode**: Ignored the `function_call` part entirely, only processing `text` and `thought` parts
2. **Non-streaming mode**: Also ignored `function_call` parts, only processing `text`, `thought`, and `inline_data` parts

This meant Open WebUI never received the tool call and couldn't execute the code.

## Solution

Added proper detection and emission of tool calls in both streaming and non-streaming response handling:

### Streaming Response Changes

In `_handle_streaming_response()` method (around line 1684):

```python
# Function call parts (tool calls)
if getattr(part, "function_call", None):
    function_call = part.function_call
    # Emit tool call event for Open WebUI
    await __event_emitter__(
        {
            "type": "chat:message:delta",
            "data": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{function_call.name}",
                        "type": "function",
                        "function": {
                            "name": function_call.name,
                            "arguments": json.dumps(
                                dict(function_call.args)
                            ),
                        },
                    }
                ],
            },
        }
    )
```

### Non-Streaming Response Changes

In the `pipe()` method's non-streaming path (around line 2083):

1. **Detection**: Added check for `function_call` attribute on response parts
2. **Collection**: Store tool calls in a list during part processing
3. **Emission**: Emit tool calls via event emitter
4. **Response Format**: Return OpenAI-compatible format with tool calls

```python
# Handle function calls (tool calls)
if getattr(part, "function_call", None):
    function_call = part.function_call
    tool_call = {
        "id": f"call_{function_call.name}",
        "type": "function",
        "function": {
            "name": function_call.name,
            "arguments": json.dumps(dict(function_call.args)),
        },
    }
    tool_calls_list.append(tool_call)
```

Then at the end of processing:

```python
# If there are tool calls, return them in the expected format
if tool_calls_list:
    # Emit tool calls for Open WebUI
    await __event_emitter__(
        {
            "type": "chat:message",
            "data": {
                "role": "assistant",
                "content": full_response or "",
                "tool_calls": tool_calls_list,
            },
        }
    )
    # Return in OpenAI-compatible format
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": full_response or "",
                    "tool_calls": tool_calls_list,
                }
            }
        ]
    }
```

## Tool Call Format

Tool calls are formatted in OpenAI-compatible structure:

```json
{
  "id": "call_<function_name>",
  "type": "function",
  "function": {
    "name": "<function_name>",
    "arguments": "<json_string_of_args>"
  }
}
```

For example, a code execution call:

```json
{
  "id": "call_python",
  "type": "function",
  "function": {
    "name": "python",
    "arguments": "{\"code\": \"print('Hello, World!')\"}"
  }
}
```

## Testing

To test the fix:

1. Enable code interpreter in Open WebUI
2. Select a Gemini model
3. Ask it to execute code, e.g., "Calculate pi to 10 decimal places using Python"
4. The code should now execute and show results instead of repeating text

## Compatibility

This fix maintains backward compatibility:
- When no tool calls are present, behavior is unchanged
- Tool calls follow OpenAI format for compatibility with Open WebUI
- Works with both streaming and non-streaming modes
- Does not interfere with existing features (thoughts, images, grounding)

## References

- Google Generative AI SDK documentation on function calling
- OpenAI API specification for tool calls
- Open WebUI event emitter documentation
