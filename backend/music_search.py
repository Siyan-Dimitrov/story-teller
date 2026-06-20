"""Royalty-free music search (Jamendo) + local music library helpers.

Ported from yt_facts_video_gen so both projects share one integration pattern.
Jamendo client ID is provided via JAMENDO_CLIENT_ID in the environment.
"""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Optional

import httpx

from . import config

log = logging.getLogger(__name__)


# ── Jamendo ──────────────────────────────────────────────────

async def search_jamendo(query: str, limit: int = 8) -> list[dict]:
    """Search Jamendo for royalty-free instrumental music.

    Returns a list of dicts with: id, title, artist, duration, url, license, source.
    Empty list when no client ID is configured or the API returns nothing usable.
    """
    if not config.JAMENDO_CLIENT_ID:
        log.debug("Jamendo client ID not configured")
        return []

    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            resp = await client.get(
                f"{config.JAMENDO_URL}/tracks/",
                params={
                    "client_id": config.JAMENDO_CLIENT_ID,
                    "format": "json",
                    "search": query,
                    "vocalinstrumental": "instrumental",
                    "order": "popularity_total_desc",
                    "limit": limit,
                    "audioformat": "mp32",
                    "include": "musicinfo",
                },
                timeout=30.0,
            )
        except httpx.HTTPError as e:
            log.warning(f"Jamendo request failed: {e}")
            return []

        if resp.status_code != 200:
            log.warning(f"Jamendo API error {resp.status_code}: {resp.text[:200]}")
            return []

        data = resp.json()
        if data.get("headers", {}).get("status") != "success":
            log.warning(f"Jamendo API non-success: {data.get('headers')}")
            return []

        results: list[dict] = []
        for track in data.get("results", []):
            download_url = track.get("audiodownload") or track.get("audio")
            if not download_url:
                continue
            if not track.get("audiodownload_allowed", True):
                continue

            results.append({
                "id": f"jamendo_{track['id']}",
                "title": track.get("name", "Unknown"),
                "artist": track.get("artist_name", "Unknown"),
                "duration": track.get("duration", 0),
                "url": download_url,
                "license": track.get("license_ccurl", ""),
                "source": "jamendo",
            })

        log.info(f"Jamendo: found {len(results)} tracks for '{query}'")
        return results


# ── Local library ────────────────────────────────────────────

def find_local_music() -> list[dict]:
    """Return descriptors for every music file currently sitting in data/music/."""
    music_dir = config.MUSIC_DIR
    if not music_dir.exists():
        return []

    tracks: list[dict] = []
    for path in sorted(music_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in config.MUSIC_EXTENSIONS:
            continue
        tracks.append({
            "id": f"local_{path.stem}",
            "title": path.stem.replace("_", " ").replace("-", " ").title(),
            "name": path.name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "source": "local",
        })
    return tracks


# ── Download + cache ─────────────────────────────────────────

def _cache_filename_for_url(url: str, fallback_ext: str = ".mp3") -> str:
    """Deterministic filename for a downloaded remote track."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix not in config.MUSIC_EXTENSIONS:
        suffix = fallback_ext
    return f"jamendo_{digest}{suffix}"


def download_music_to_cache(url: str) -> Optional[Path]:
    """Download a remote music URL into data/music/ and return the cached path.

    Reuses existing files when the same URL has already been downloaded.
    """
    if not url.startswith(("http://", "https://")):
        return None

    cache_path = config.MUSIC_DIR / _cache_filename_for_url(url)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    try:
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
    except httpx.HTTPError as e:
        log.error(f"Failed to download music {url}: {e}")
        return None

    log.info(f"Cached remote music to {cache_path.name} ({cache_path.stat().st_size} bytes)")
    return cache_path


# ── High-level picker ────────────────────────────────────────

async def suggest_background_music(mood: str = "cinematic") -> Optional[dict]:
    """Pick a reasonable background track: Jamendo search first, local fallback."""
    tracks = await search_jamendo(f"{mood} background", limit=5)
    if not tracks:
        tracks = await search_jamendo("cinematic background", limit=5)
    if tracks:
        return tracks[0]

    local = find_local_music()
    if local:
        return random.choice(local)
    return None
