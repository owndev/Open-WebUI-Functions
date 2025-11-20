# Code Interpreter Solution - Complete Explanation

## The Problem

Code interpreter didn't work with Gemini pipeline:
- **Non-streaming**: Code executed but response repeated 4-5 times, OpenWebUI didn't recognize message end
- **Streaming**: Code didn't execute, just showed `<code_interpreter>` tags as text

## The Investigation

### Initial Assumptions (WRONG)
1. ❌ Thought tools weren't being registered due to `function_calling` mode
2. ❌ Thought we needed to handle `function_call` parts from Gemini
3. ❌ Thought OpenWebUI provided tools via `__tools__` parameter

### What Logs Revealed (CORRECT)
```
INFO: Tools parameter: __tools__ is None
INFO: Enabling tools: []
```

**OpenWebUI doesn't provide any tools!** But part attributes showed:
- `executable_code` 
- `code_execution_result`

This revealed Gemini has **native code execution** built-in.

## The Solution

### Gemini Native Code Execution

Gemini has a built-in code execution feature that:
1. Doesn't require external tools
2. Executes Python code internally
3. Returns results via special part types

### Implementation (Commit cfd5270)

#### 1. Enable Code Execution in Generation Config

```python
# In _configure_generation method (~line 1380)
gen_config_params["tools"] = [types.Tool(code_execution={})]
```

This tells Gemini it's allowed to execute code.

#### 2. Handle executable_code Parts

**Streaming** (~line 1707):
```python
if getattr(part, "executable_code", None):
    executable_code = part.executable_code
    code = getattr(executable_code, "code", None)
    language = getattr(executable_code, "language", "python")
    
    # Display code in markdown block
    code_content = f"```{language}\n{code}\n```\n"
    answer_chunks.append(code_content)
    
    # Emit to OpenWebUI
    await __event_emitter__({
        "type": "chat:message:delta",
        "data": {
            "role": "assistant",
            "content": code_content,
        },
    })
```

**Non-Streaming** (~line 2139): Same logic, appends to `answer_segments`

#### 3. Handle code_execution_result Parts

**Streaming** (~line 1724):
```python
elif getattr(part, "code_execution_result", None):
    result = part.code_execution_result
    outcome = getattr(result, "outcome", "UNKNOWN")
    output = getattr(result, "output", "")
    
    # Display result with outcome
    result_content = f"**Execution Result ({outcome}):**\n```\n{output}\n```\n"
    answer_chunks.append(result_content)
    
    # Emit to OpenWebUI
    await __event_emitter__({
        "type": "chat:message:delta",
        "data": {
            "role": "assistant",
            "content": result_content,
        },
    })
```

**Non-Streaming** (~line 2147): Same logic, appends to `answer_segments`

## How It Works

### Request Flow

1. User asks: "Print Hello World in Python"
2. Pipeline enables code execution in generation config
3. Gemini receives the request with code execution enabled

### Response Flow

4. Gemini decides to execute code
5. Gemini returns `executable_code` part with the Python code
6. Pipeline detects and displays: ` ``` python\nprint("Hello World")\n``` `
7. Gemini executes the code internally
8. Gemini returns `code_execution_result` part with output
9. Pipeline detects and displays: `**Execution Result (SUCCESS):**\n```\nHello World\n```\n`

### Key Points

- **No external tools**: Gemini handles execution internally
- **No OpenWebUI tools needed**: Works without `__tools__` parameter
- **Clean display**: Code and results in formatted blocks
- **Both modes work**: Streaming and non-streaming

## Why Previous Attempts Failed

### Attempt 1: Handle function_call parts
- ❌ Gemini wasn't using function_call for code execution
- ❌ It uses executable_code/code_execution_result instead

### Attempt 2: Fix tool registration
- ❌ OpenWebUI doesn't provide tools
- ❌ Looking for `__tools__` that don't exist

### Attempt 3: Enable auto mode
- ❌ Still looking for external tools
- ❌ Missed that Gemini has native feature

## Verification

To verify this works, check logs for:

```
INFO: Enabled Gemini native code execution
INFO: CODE EXECUTION: language=python, code_length=...
INFO: CODE RESULT: outcome=SUCCESS, output_length=...
```

## References

- Gemini SDK: `types.Tool(code_execution={})` enables native code execution
- Part types: `executable_code` and `code_execution_result`
- No dependency on OpenWebUI's `__tools__` mechanism
