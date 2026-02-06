#!/usr/bin/env python3
"""
Test to verify the model name extraction bug with gpt-5.2-chat
"""


def extract_model_name_buggy(model_name: str) -> str:
    """Current buggy implementation"""
    return (
        model_name.split(".", 1)[1]
        if "." in model_name
        else model_name
    )


def extract_model_name_fixed(model_name: str) -> str:
    """Fixed implementation - only strip pipeline prefix if it looks like one"""
    # Only strip prefix if it looks like a pipeline prefix (e.g., "Azure AI: gpt-4")
    # Pipeline prefixes typically contain a space after the colon
    if "." in model_name and ": " in model_name.split(".", 1)[0]:
        return model_name.split(".", 1)[1]
    return model_name


def test_model_extraction():
    """Test various model name formats"""

    test_cases = [
        # (input, expected_output, description)
        ("gpt-5.2-chat", "gpt-5.2-chat", "GPT-5.2 with version number"),
        ("gpt-4o", "gpt-4o", "Simple model name"),
        ("gpt-4.1-mini", "gpt-4.1-mini", "GPT-4.1 with version number"),
        ("Azure AI: .gpt-5", "gpt-5", "Pipeline prefix format"),
        ("o1-mini", "o1-mini", "O-series model"),
        ("gpt-4.5-preview", "gpt-4.5-preview", "GPT-4.5 preview"),
    ]

    print("Testing BUGGY implementation:")
    buggy_failures = []
    for input_name, expected, desc in test_cases:
        result = extract_model_name_buggy(input_name)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {desc}: '{input_name}' → '{result}' (expected: '{expected}')")
        if result != expected:
            buggy_failures.append((input_name, result, expected))

    print("\nTesting FIXED implementation:")
    fixed_failures = []
    for input_name, expected, desc in test_cases:
        result = extract_model_name_fixed(input_name)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {desc}: '{input_name}' → '{result}' (expected: '{expected}')")
        if result != expected:
            fixed_failures.append((input_name, result, expected))

    if buggy_failures:
        print(f"\n✗ Buggy implementation failed {len(buggy_failures)} tests")
    else:
        print("\n✓ Buggy implementation passed all tests")

    if fixed_failures:
        print(f"\n✗ Fixed implementation failed {len(fixed_failures)} tests")
        return False
    else:
        print("\n✓ Fixed implementation passed all tests")
        return True


if __name__ == "__main__":
    import sys
    success = test_model_extraction()
    sys.exit(0 if success else 1)
