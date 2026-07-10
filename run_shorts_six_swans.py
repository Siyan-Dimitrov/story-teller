"""Direct invocation of ShortsDirector on 'The Six Swans' (project 366f3648582d).

Bypasses FastAPI — runs the director coroutine in an asyncio loop against
the already-running VoiceBox + Ollama services.
"""

import asyncio
import json
import sys

from backend.agents import shorts_director

PROJECT_ID = "366f3648582d"


async def main() -> None:
    print(f"Generating short for project {PROJECT_ID} (The Six Swans)…", flush=True)
    result = await shorts_director.generate_short(PROJECT_ID)
    print("DONE", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"FAILED: {e!r}", flush=True)
        sys.exit(1)
