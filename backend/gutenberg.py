"""Project Gutenberg search via the Gutendex API, with OPDS fallback."""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET

import httpx

from . import config

log = logging.getLogger(__name__)

GUTENDEX_BASE = "https://gutendex.com/books/"
OPDS_SEARCH = "https://www.gutenberg.org/ebooks/search.opds/"
OPDS_BOOK = "https://www.gutenberg.org/ebooks/{id}.opds"

MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 2.0  # seconds — doubles each attempt

# Gutendex gets a short timeout; OPDS is fast so it gets its own.
_GUTENDEX_TIMEOUT = 12.0
_OPDS_TIMEOUT = 15.0

# XML namespace used by Atom / OPDS feeds
_ATOM = "{http://www.w3.org/2005/Atom}"
_DCTERMS = "{http://purl.org/dc/terms/}"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


# ── HTTP helpers ────────────────────────────────────────────────


async def _get_with_retries(
    url: str,
    params: dict,
    timeout: float,
    *,
    follow_redirects: bool = False,
) -> httpx.Response:
    """GET with exponential-backoff retries on timeout / 5xx."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=follow_redirects,
            ) as client:
                resp = await client.get(url, params=params)
                if resp.status_code >= 500:
                    log.warning(
                        "Request to %s returned %s on attempt %d/%d",
                        url, resp.status_code, attempt + 1, MAX_RETRIES,
                    )
                    last_exc = httpx.HTTPStatusError(
                        f"Server error {resp.status_code}",
                        request=resp.request, response=resp,
                    )
                else:
                    resp.raise_for_status()
                    return resp
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            log.warning(
                "Request to %s failed on attempt %d/%d: %s",
                url, attempt + 1, MAX_RETRIES, exc,
            )
            last_exc = exc

        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF_BASE * (2 ** attempt)
            log.info("Retrying in %.1fs …", wait)
            await asyncio.sleep(wait)

    # All retries exhausted
    if isinstance(last_exc, httpx.TimeoutException):
        raise httpx.TimeoutException(
            f"Timed out after {MAX_RETRIES} attempts ({timeout}s each)"
        )
    if isinstance(last_exc, httpx.ConnectError):
        raise httpx.ConnectError(
            f"Could not connect after {MAX_RETRIES} attempts"
        )
    if isinstance(last_exc, httpx.HTTPStatusError):
        raise last_exc
    raise RuntimeError("Request failed after retries")


# ── Gutendex (primary) ──────────────────────────────────────────


async def _search_gutendex(
    query: str, page: int, topic: str, languages: str,
) -> dict:
    """Search via the third-party Gutendex JSON API."""
    params: dict[str, str | int] = {"page": page}
    if query:
        params["search"] = query
    if topic:
        params["topic"] = topic
    if languages:
        params["languages"] = languages

    resp = await _get_with_retries(GUTENDEX_BASE, params, _GUTENDEX_TIMEOUT)
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


# ── OPDS fallback (gutenberg.org native) ─────────────────────────


def _opds_extract_id(id_text: str) -> int | None:
    """Extract Gutenberg book ID from an OPDS <id> URL."""
    m = re.search(r"/ebooks/(\d+)", id_text)
    return int(m.group(1)) if m else None


def _opds_text_url(gid: int) -> str:
    """Construct the standard plain-text URL for a Gutenberg book."""
    return f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"


def _opds_parse_search(xml_text: str) -> dict:
    """Parse an OPDS Atom search feed into our standard result format."""
    root = ET.fromstring(xml_text)

    results = []
    for entry in root.findall(f"{_ATOM}entry"):
        id_el = entry.find(f"{_ATOM}id")
        if id_el is None or id_el.text is None:
            continue
        gid = _opds_extract_id(id_el.text)
        if gid is None:
            continue

        title_el = entry.find(f"{_ATOM}title")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""

        # Author from <content> (plain text: "Author Name")
        content_el = entry.find(f"{_ATOM}content")
        author_name = ""
        if content_el is not None and content_el.text:
            author_name = content_el.text.strip()

        authors = [{"name": author_name, "birth_year": None, "death_year": None}] if author_name else []

        results.append({
            "gutenberg_id": gid,
            "title": title,
            "authors": authors,
            "subjects": [],
            "bookshelves": [],
            "languages": [],
            "download_count": 0,
            "text_url": _opds_text_url(gid),
        })

    # totalResults from opensearch namespace
    total_el = root.find(f".//{_OPENSEARCH}totalResults")
    # Fall back to a reasonable count
    count = int(total_el.text) if total_el is not None and total_el.text else len(results)

    # next/previous page links
    next_url = None
    prev_url = None
    for link in root.findall(f"{_ATOM}link"):
        rel = link.get("rel", "")
        href = link.get("href", "")
        if rel == "next" and href:
            next_url = f"https://www.gutenberg.org{href}" if href.startswith("/") else href
        elif rel == "previous" and href:
            prev_url = f"https://www.gutenberg.org{href}" if href.startswith("/") else href

    return {
        "count": count,
        "next": next_url,
        "previous": prev_url,
        "results": results,
    }


async def _search_opds(
    query: str, page: int, topic: str, languages: str,
) -> dict:
    """Fallback search using gutenberg.org's native OPDS feed."""
    params: dict[str, str | int] = {}
    # Build a combined query string (OPDS only supports a single search term)
    terms = []
    if query:
        terms.append(query)
    if topic:
        terms.append(topic)
    if terms:
        params["query"] = " ".join(terms)

    # OPDS uses start_index (1-based, 25 per page)
    if page > 1:
        params["start_index"] = (page - 1) * 25 + 1

    log.info("OPDS fallback search: params=%s", params)
    resp = await _get_with_retries(OPDS_SEARCH, params, _OPDS_TIMEOUT)
    return _opds_parse_search(resp.text)


# ── Public API ──────────────────────────────────────────────────


async def search_gutenberg(
    query: str, page: int = 1, topic: str = "", languages: str = "",
) -> dict:
    """Search Project Gutenberg — races Gutendex and OPDS, returns first success."""
    log.info("Gutenberg search: query=%r, topic=%r, languages=%r, page=%d", query, topic, languages, page)

    async def _try_gutendex() -> dict:
        return await _search_gutendex(query, page, topic, languages)

    async def _try_opds() -> dict:
        return await _search_opds(query, page, topic, languages)

    # Race both sources concurrently — use whichever responds first.
    tasks = [
        asyncio.create_task(_try_gutendex(), name="gutendex"),
        asyncio.create_task(_try_opds(), name="opds"),
    ]
    errors: list[str] = []

    # As each task completes, return the first success.
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            # Cancel the slower task
            for t in tasks:
                t.cancel()
            return result
        except Exception as exc:
            errors.append(str(exc))

    raise RuntimeError(f"Both Gutendex and OPDS search failed: {'; '.join(errors)}")


async def fetch_gutenberg_text(text_url: str, max_chars: int = 0) -> dict:
    """Fetch plain text of a Gutenberg book.

    If max_chars > 0, returns only that many characters (preview mode).
    If max_chars == 0, returns the full text.
    """
    resp = await _get_with_retries(
        text_url, {}, config.GUTENBERG_TEXT_TIMEOUT_SECONDS,
        follow_redirects=True,
    )

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


# ── Helpers ─────────────────────────────────────────────────────


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
