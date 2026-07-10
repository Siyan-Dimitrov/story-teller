"""Test Ollama connection and intelligent split directly."""
import asyncio
import httpx
import json
import re
import sys
sys.path.insert(0, 'backend')

from backend import config

OLLAMA_URL = config.OLLAMA_URL
MODEL = "mistral:latest"

INTELLIGENT_SPLIT_PROMPT = """You are a literary editor. Your task: split the given text into {num_parts} logical narrative parts.

EXTREMELY IMPORTANT: Your ENTIRE response must be ONLY a valid JSON object. NO explanations before. NO explanations after. NO markdown code fences with ```. Just raw JSON starting with {{ and ending with }}.

JSON format required:
- parts: array of {num_parts} objects, each with:
  - part_number: integer (1, 2, 3...)
  - title: string describing this section
  - summary: brief description of what happens
  - split_after_text: the EXACT last 50-100 characters from this part (copy verbatim from source text)
  - char_count: approximate character count
- reasoning: brief explanation of why you chose these split points

Rules:
1. Narrative coherence over equal sizing
2. Find breaks at scene endings, time shifts, or story pauses
3. split_after_text MUST be copied exactly from the source text
4. Output ONLY JSON - nothing else
"""

TEST_TEXT = """Once upon a time in a land far away, there lived a young princess. She spent her days exploring the castle gardens and reading books in the library.

One day, a mysterious stranger arrived at the castle gates. He brought news from the neighboring kingdom. The princess listened carefully to his tale.

The princess decided to embark on an adventure to help her people. She packed her things and set off early the next morning."""


async def test_ollama_connection():
    """Test basic Ollama connectivity."""
    print("=" * 60)
    print("TEST 1: Ollama Connection")
    print(f"URL: {OLLAMA_URL}")
    print("=" * 60)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            print(f"Status: {resp.status_code}")

            if resp.status_code == 200:
                data = resp.json()
                models = [m.get('name', m.get('model', '')) for m in data.get('models', [])]
                print(f"Available models: {models}")

                if MODEL in models:
                    print(f"[PASS] Model '{MODEL}' is available")
                    return True
                else:
                    print(f"[FAIL] Model '{MODEL}' NOT found in available models")
                    print(f"         You may need to run: ollama pull {MODEL}")
                    return False
            else:
                print(f"[FAIL] Ollama returned status {resp.status_code}")
                print(f"       Response: {resp.text[:500]}")
                return False

    except httpx.ConnectError as e:
        print(f"[FAIL] Cannot connect to Ollama: {e}")
        print(f"       Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {type(e).__name__}: {e}")
        return False


async def test_chat_completion():
    """Test a simple chat completion."""
    print("\n" + "=" * 60)
    print("TEST 2: Simple Chat Completion")
    print("=" * 60)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "user", "content": "Say 'hello' and nothing else."}
                    ],
                    "stream": False,
                },
            )

            print(f"Status: {resp.status_code}")

            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                print(f"Response: {content}")
                print("[PASS] Chat completion works")
                return True
            else:
                print(f"[FAIL] Status {resp.status_code}: {resp.text[:500]}")
                return False

    except Exception as e:
        print(f"[FAIL] Error: {type(e).__name__}: {e}")
        return False


async def test_intelligent_split():
    """Test the full intelligent split flow."""
    print("\n" + "=" * 60)
    print("TEST 3: Intelligent Split")
    print("=" * 60)

    prompt = INTELLIGENT_SPLIT_PROMPT.format(num_parts=2)
    user_message = f"Please split this text into exactly 2 logical parts.\n\nText ({len(TEST_TEXT)} chars total):\n\n{TEST_TEXT}"

    print(f"Sending request to {OLLAMA_URL}/api/chat")
    print(f"Model: {MODEL}")
    print(f"Text length: {len(TEST_TEXT)} chars")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2000,
                    },
                },
            )

            print(f"Response status: {resp.status_code}")

            if resp.status_code != 200:
                print(f"[FAIL] HTTP {resp.status_code}: {resp.text[:500]}")
                try:
                    error_data = resp.json()
                    print(f"Error details: {error_data}")
                except:
                    pass
                return False

            data = resp.json()
            print(f"Response keys: {list(data.keys())}")

            msg = data.get("message", {})
            content = msg.get("content", "") or ""

            if not content:
                content = msg.get("thinking", "") or ""

            print(f"Content length: {len(content)}")
            print(f"\n--- Raw LLM Response (first 1000 chars) ---")
            print(content[:1000])
            print("\n... [truncated] ...\n")
            print(f"--- Last 500 chars ---")
            print(content[-500:])
            print("--- End Response ---\n")

            # Try to parse JSON with multiple strategies
            result = None

            # Try markdown fences first
            json_match = re.search(r'```json\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1).strip())
                    print("[PASS] Parsed JSON from ```json fence")
                except:
                    pass

            if not result:
                json_match = re.search(r'```\s*\n?(.*?)\n?```', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1).strip())
                        print("[PASS] Parsed JSON from ``` fence")
                    except:
                        pass

            # Try to find JSON with "parts" key (try LAST match first)
            if not result:
                all_matches = list(re.finditer(r'\{[\s\S]*?"parts"[\s\S]*?\}', content))
                for match in reversed(all_matches):
                    try:
                        json_str = match.group(0)
                        # Find balanced braces
                        brace_count = 0
                        start = match.start()
                        for i, char in enumerate(content[start:]):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = content[start:start+i+1]
                                    break
                        result = json.loads(json_str)
                        print(f"[PASS] Parsed JSON with 'parts' key (from {len(all_matches)} matches)")
                        break
                    except:
                        continue

            # Try first { to last }
            if not result:
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1 and end > start:
                    try:
                        json_str = content[start:end+1]
                        result = json.loads(json_str)
                        print("[PASS] Parsed JSON from first '{' to last '}'")
                    except:
                        pass

            if not result:
                print(f"[FAIL] Could not parse JSON from response")
                return False

            parts = result.get("parts", [])
            print(f"[PASS] Found {len(parts)} parts")
            for i, part in enumerate(parts):
                print(f"  Part {i+1}: {part.get('title', 'No title')} ({part.get('char_count', 'N/A')} chars)")
            return True

    except httpx.TimeoutException:
        print("[FAIL] Request timed out after 60s")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("\nOLLAMA DIAGNOSTIC TESTS\n")

    results = []
    results.append(("Connection", await test_ollama_connection()))
    results.append(("Chat", await test_chat_completion()))
    results.append(("Split", await test_intelligent_split()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    if not all(r[1] for r in results):
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama is running: ollama serve")
        print(f"  2. Pull the model: ollama pull {MODEL}")
        print(f"  3. Check Ollama URL (currently: {OLLAMA_URL})")
        print("  4. Check the model name is correct")


if __name__ == "__main__":
    asyncio.run(main())
