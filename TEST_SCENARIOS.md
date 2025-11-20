# Test Scenarios for Code Interpreter Fix

## Prerequisites
1. Open WebUI is running with the updated Gemini pipeline
2. Code interpreter feature is enabled in Open WebUI settings
3. Google API key or Vertex AI credentials are configured
4. A Gemini model is selected (e.g., gemini-2.5-pro, gemini-2.0-flash)

## Test Scenario 1: Simple Code Execution

**Prompt:**
```
Write and execute Python code to calculate pi to 10 decimal places using the Leibniz formula.
```

**Expected Behavior:**
- Gemini should respond with code explanation
- Code should execute automatically
- Results should be displayed showing pi ≈ 3.1415926536

**What to Check:**
- No repeated text (the original bug symptom)
- Code is displayed in a code block
- Execution results are shown
- No errors in console logs

## Test Scenario 2: Data Visualization

**Prompt:**
```
Create a simple bar chart showing the first 5 Fibonacci numbers using matplotlib.
```

**Expected Behavior:**
- Code generates a bar chart
- Chart is displayed in the response
- No repeated text errors

**What to Check:**
- Code executes successfully
- Image/chart is visible
- Proper error handling if matplotlib isn't available

## Test Scenario 3: Error Handling

**Prompt:**
```
Execute this Python code: print(1/0)
```

**Expected Behavior:**
- Code attempts to execute
- Division by zero error is caught and displayed
- Error message is clear and doesn't break the UI

**What to Check:**
- Error is handled gracefully
- No system crash or hung requests
- Error message is visible to user

## Test Scenario 4: Multi-turn Conversation

**Prompt 1:**
```
Create a Python function to calculate factorial of a number.
```

**Prompt 2:**
```
Now use that function to calculate factorial of 10.
```

**Expected Behavior:**
- First response creates and shows the function
- Second response uses the function and shows result (3628800)
- Context is maintained between turns

**What to Check:**
- Multi-turn context works correctly
- Variables/functions from previous turns are available
- No repeated text issues

## Test Scenario 5: Complex Calculation

**Prompt:**
```
Calculate the first 20 prime numbers using Python.
```

**Expected Behavior:**
- Code is generated and executed
- List of primes is displayed: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

**What to Check:**
- Complex logic executes correctly
- Results are accurate
- No timeout errors

## Test Scenario 6: Streaming Mode

Enable streaming responses and test:

**Prompt:**
```
Generate a simple "Hello, World!" program in Python and execute it.
```

**Expected Behavior:**
- Response streams in real-time
- Code execution still works
- No "chunk too big" errors

**What to Check:**
- Streaming works smoothly
- Tool calls are detected in streaming mode
- No response corruption

## Test Scenario 7: Non-Streaming Mode

Disable streaming responses and test:

**Prompt:**
```
Calculate the sum of numbers from 1 to 100 using Python.
```

**Expected Behavior:**
- Response arrives all at once
- Code executes and shows result (5050)
- No repeated text

**What to Check:**
- Non-streaming mode works
- Tool calls are detected and emitted
- Format is correct

## Debugging Tips

If tests fail, check:

1. **Browser Console**: Look for JavaScript errors or failed API calls
2. **Open WebUI Logs**: Check for Python exceptions or warnings
3. **Network Tab**: Inspect the API request/response format
4. **Event Emitter**: Verify events are being emitted correctly

Key indicators of success:
- ✅ No repeated text in responses
- ✅ Code blocks are properly formatted
- ✅ Execution results are displayed
- ✅ Tool call events appear in logs
- ✅ Format matches OpenAI tool call structure

Key indicators of issues:
- ❌ Text repeats multiple times
- ❌ Code doesn't execute
- ❌ "function_call" errors in logs
- ❌ Missing tool_calls in API response
- ❌ Malformed JSON in arguments

## Log Monitoring

Watch for these log messages (set log level to DEBUG):

**Success indicators:**
```
Detected tool call: <function_name> with args: <args>
Emitted tool call: <function_name> with args: <args>
```

**Error indicators:**
```
Error processing content part: ...
Failed to access content parts: ...
```

## Comparison with Azure Pipeline

To verify the fix matches Azure's behavior, test the same prompts with both:
1. Azure AI pipeline (known working)
2. Gemini pipeline (after fix)

Both should execute code successfully and show results.
