"""Background music helpers."""

import logging
import random
from pathlib import Path
from typing import Optional

import httpx

from . import config

log = logging.getLogger(__name__)


async def search_jamendo(query: str, limit: int = 5) -> list[dict]:
    """Search Jamendo for royalty-free instrumental music."""
    if not config.JAMENDO_CLIENT_ID:
        log.debug("Jamendo client ID not configured")
        return []

    async with httpx.AsyncClient(follow_redirects=True) as client:
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

    if resp.status_code != 200:
        log.warning(f"Jamendo API error {resp.status_code}: {resp.text[:200]}")
        return []

    data = resp.json()
    if data.get("headers", {}).get("status") != "success":
        log.warning(f"Jamendo API returned non-success: {data.get('headers')}")
        return []

    results: list[dict] = []
    for track in data.get("results", []):
        download_url = track.get("audiodownload") or track.get("audio")
        if not download_url:
            continue
        if not track.get("audiodownload_allowed", True):
            continue

        results.append(
            {
                "id": f"jamendo_{track['id']}",
                "title": track.get("name", "Unknown"),
                "artist": track.get("artist_name", "Unknown"),
                "duration": track.get("duration", 0),
                "url": download_url,
                "path": None,
                "license": track.get("license_ccurl", ""),
                "source": "jamendo",
            }
        )

    log.info(f"Jamendo: found {len(results)} tracks for {query}")
    return results


def find_local_music() -> list[dict]:
    """Find music files in the local data/music directory."""
    if not config.MUSIC_DIR.exists():
        return []

    tracks: list[dict] = []
    for ext in ("*.mp3", "*.wav", "*.ogg", "*.m4a"):
        for path in config.MUSIC_DIR.glob(ext):
            tracks.append(
                {
                    "id": f"local_{path.stem}",
                    "title": path.stem.replace("_", " ").replace("-", " ").title(),
                    "path": str(path),
                    "url": None,
                    "source": "local",
                }
            )
    return tracks


async def suggest_background_music(tone: str) -> Optional[dict]:
    """Find background music via Jamendo, falling back to local files."""
    primary_tag = tone.split(",")[0].strip() if tone else ""

    if config.JAMENDO_CLIENT_ID:
        try:
            query = f"{primary_tag} ambient" if primary_tag else "ambient"
            tracks = await search_jamendo(query, limit=5)
            if not tracks:
                tracks = await search_jamendo("cinematic ambient", limit=5)
            if tracks:
                track = random.choice(tracks)
                log.info(
                    f"Selected Jamendo track: {track['title']} by {track.get('artist', '?')}"
                )
                return track
        except Exception as e:
            log.warning(f"Jamendo search failed: {e}")

    local_tracks = find_local_music()
    if local_tracks:
        track = random.choice(local_tracks)
        log.info(f"Selected local music: {track['title']}")
        return track

    log.info(
        f"No music found. Set JAMENDO_CLIENT_ID env var or add .mp3/.wav files to "
        f"{config.MUSIC_DIR}"
    )
    return None


async def download_music(url: str, target_path: Path) -> bool:
    """Download a music file to disk."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                target_path.write_bytes(resp.content)
                log.info(f"Downloaded music to {target_path}")
                return True

            log.warning(f"Music download failed: HTTP {resp.status_code}")
            return False
    except Exception as e:
        log.warning(f"Music download failed: {e}")
        return False
