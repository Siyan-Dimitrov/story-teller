"""Test the intelligent split JSON extraction logic."""

import sys
sys.path.insert(0, 'backend')

from backend.main import _extract_json_from_llm_response, _find_split_position, INTELLIGENT_SPLIT_PROMPT

# Test 1: JSON extraction with markdown fences
print("Test 1: JSON with markdown fences")
content1 = """```json
{
  "parts": [
    {
      "part_number": 1,
      "title": "The Beginning",
      "summary": "Introduction",
      "split_after_text": "and so it began.",
      "char_count": 1500
    },
    {
      "part_number": 2,
      "title": "The Middle",
      "summary": "The story continues",
      "split_after_text": "things got interesting.",
      "char_count": 1600
    }
  ],
  "reasoning": "Split at scene changes"
}
```"""

result1 = _extract_json_from_llm_response(content1)
if result1 and "parts" in result1:
    print(f"  [PASS] Extracted {len(result1['parts'])} parts")
else:
    print(f"  [FAIL] Failed to extract JSON")
    print(f"  Raw: {result1}")

# Test 2: Plain JSON without fences
print("\nTest 2: Plain JSON without fences")
content2 = """{
  "parts": [
    {
      "part_number": 1,
      "title": "Part One",
      "summary": "First part",
      "split_after_text": "end of first part.",
      "char_count": 1000
    }
  ],
  "reasoning": "Natural break"
}"""

result2 = _extract_json_from_llm_response(content2)
if result2 and "parts" in result2:
    print(f"  [PASS] Extracted {len(result2['parts'])} parts")
else:
    print(f"  [FAIL] Failed to extract JSON")

# Test 3: Check the prompt format
print("\nTest 3: Prompt format check")
prompt = INTELLIGENT_SPLIT_PROMPT.format(num_parts=3)
if "{num_parts}" not in prompt:
    print(f"  [PASS] Prompt formatted correctly")
else:
    print(f"  [FAIL] Prompt has unformatted placeholders")

# Test 4: Split position finding
print("\nTest 4: Split position finding")
text = """Once upon a time in a land far away, there lived a young princess. She spent her days exploring the castle gardens and reading books in the library.

One day, a mysterious stranger arrived at the castle gates. He brought news from the neighboring kingdom.

The princess listened carefully to his tale. It was a story of adventure and danger."""

marker = "castle gates. He brought news"
pos = _find_split_position(text, marker)
if pos > 0:
    print(f"  [PASS] Found split position at {pos}")
else:
    print(f"  [FAIL] Could not find split position")

print("\n--- Tests complete ---")
