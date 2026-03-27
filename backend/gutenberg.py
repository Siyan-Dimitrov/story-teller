"""Project Gutenberg search via the Gutendex API."""

import logging

import httpx

from . import config

log = logging.getLogger(__name__)

GUTENDEX_BASE = "https://gutendex.com/books/"


async def search_gutenberg(
    query: str, page: int = 1, topic: str = "", languages: str = "",
) -> dict:
    """Search Project Gutenberg via the Gutendex API."""
    params: dict[str, str | int] = {"page": page}
    if query:
        params["search"] = query
    if topic:
        params["topic"] = topic
    if languages:
        params["languages"] = languages

    log.info(f"Gutenberg search: query={query!r}, topic={topic!r}, languages={languages!r}, page={page}")

    try:
        async with httpx.AsyncClient(timeout=config.GUTENBERG_TIMEOUT_SECONDS) as client:
            resp = await client.get(GUTENDEX_BASE, params=params)
            resp.raise_for_status()
    except httpx.TimeoutException:
        raise RuntimeError(f"Gutendex API timed out after {config.GUTENBERG_TIMEOUT_SECONDS}s — try again")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Gutendex API returned {e.response.status_code}")
    except httpx.ConnectError:
        raise RuntimeError("Could not connect to Gutendex API (gutendex.com)")

    data = resp.json()
    results = []
    for book in data.get("results", []):
        text_url = _find_text_url(book.get("formats", {}))
        results.append({
            "gutenberg_id": book["id"],
            "title": book.get("title", ""),
            "authors": [
                {
                    "name": a.get("name", ""),
                    "birth_year": a.get("birth_year"),
                    "death_year": a.get("death_year"),
                }
                for a in book.get("authors", [])
            ],
            "subjects": book.get("subjects", []),
            "bookshelves": book.get("bookshelves", []),
            "languages": book.get("languages", []),
            "download_count": book.get("download_count", 0),
            "text_url": text_url,
        })

    return {
        "count": data.get("count", 0),
        "next": data.get("next"),
        "previous": data.get("previous"),
        "results": results,
    }


async def fetch_gutenberg_text(text_url: str, max_chars: int = 0) -> dict:
    """Fetch plain text of a Gutenberg book.

    If max_chars > 0, returns only that many characters (preview mode).
    If max_chars == 0, returns the full text.
    """
    try:
        async with httpx.AsyncClient(
            timeout=config.GUTENBERG_TEXT_TIMEOUT_SECONDS,
            follow_redirects=True,
        ) as client:
            resp = await client.get(text_url)
            resp.raise_for_status()
    except httpx.TimeoutException:
        raise RuntimeError(f"Text download timed out after {config.GUTENBERG_TEXT_TIMEOUT_SECONDS}s — try again")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Text download failed with HTTP {e.response.status_code}")
    except httpx.ConnectError:
        raise RuntimeError("Could not connect to gutenberg.org")

    full_text = _strip_gutenberg_boilerplate(resp.text)

    if max_chars > 0:
        return {
            "text": full_text[:max_chars],
            "total_chars": len(full_text),
            "truncated": len(full_text) > max_chars,
        }

    return {
        "text": full_text,
        "total_chars": len(full_text),
        "truncated": False,
    }


def _find_text_url(formats: dict[str, str]) -> str | None:
    """Pick the best plain-text URL from Gutenberg's formats dict."""
    skip = ("readme", "_readme")
    # Prefer UTF-8 .txt URLs (e.g. 2591-0.txt, 1597-0.txt)
    for mime, url in formats.items():
        if "text/plain" in mime and "utf-8" in mime and not any(s in url.lower() for s in skip):
            return url
    # Then any .txt URL that isn't a readme
    for mime, url in formats.items():
        if "text/plain" in mime and url.endswith(".txt") and not any(s in url.lower() for s in skip):
            return url
    # Fallback: any text/plain key that isn't a readme
    for mime, url in formats.items():
        if "text/plain" in mime and not any(s in url.lower() for s in skip):
            return url
    return None


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer license text."""
    markers_start = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
    ]
    markers_end = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    for marker in markers_start:
        idx = text.find(marker)
        if idx != -1:
            newline = text.find("\n", idx)
            if newline != -1:
                text = text[newline + 1:]
            break

    for marker in markers_end:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()
