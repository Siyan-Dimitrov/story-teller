"""SSRF guards for endpoints that fetch user-supplied URLs.

Two server-side fetch paths take a URL from the client: the Gutenberg text
fetch and the music-download cache. Both are validated here so a caller can't
point them at internal/cloud-metadata addresses or non-allowlisted hosts.

Usage:
    assert_allowlisted_url(url, ALLOWED_GUTENBERG_HOSTS)  # exact-host allowlist
    assert_public_url(url)                                # block private ranges

Both raise ``UnsafeURLError`` (a ValueError) on rejection; callers map that to
an HTTP 400.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

# Gutenberg only ever serves from these hosts.
ALLOWED_GUTENBERG_HOSTS = frozenset({
    "gutenberg.org",
    "www.gutenberg.org",
    "gutenberg.net",
    "www.gutenberg.net",
})


# Cap on manual redirect hops when callers follow redirects themselves so each
# target can be re-validated.
MAX_REDIRECTS = 5


class UnsafeURLError(ValueError):
    """Raised when a user-supplied URL fails an SSRF safety check."""


def _hostname(url: str) -> tuple[str, str]:
    """Return (scheme, normalized_hostname) or raise UnsafeURLError."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise UnsafeURLError(f"URL scheme must be http(s), got {scheme!r}")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise UnsafeURLError("URL has no hostname")
    return scheme, host


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    """True if the address is in a range we must never fetch from."""
    return (
        ip.is_private          # RFC1918 (10/8, 172.16/12, 192.168/16) + IPv6 ULA
        or ip.is_loopback      # 127.0.0.0/8, ::1
        or ip.is_link_local    # 169.254.0.0/16 (incl. cloud metadata), fe80::/10
        or ip.is_reserved
        or ip.is_unspecified   # 0.0.0.0, ::
        or ip.is_multicast
    )


def _assert_resolved_host_is_public(host: str) -> None:
    """Resolve host to every A/AAAA record and reject if any is non-public.

    Resolving (rather than parsing) defeats DNS names that point at internal
    IPs. We reject if *any* resolved address is blocked, so a dual-record host
    can't smuggle one public + one internal answer.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise UnsafeURLError(f"could not resolve host {host!r}: {e}") from e

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            raise UnsafeURLError(f"unparseable address for {host!r}: {addr!r}")
        if _is_blocked_ip(ip):
            raise UnsafeURLError(
                f"host {host!r} resolves to non-public address {addr} — refusing to fetch"
            )


def assert_allowlisted_url(url: str, allowed_hosts: frozenset[str]) -> None:
    """Require the URL's host to be exactly one of ``allowed_hosts`` AND resolve
    to a public address. Use for known single-provider fetches (e.g. Gutenberg)."""
    _, host = _hostname(url)
    if host not in allowed_hosts:
        raise UnsafeURLError(
            f"host {host!r} is not allowed (permitted: {', '.join(sorted(allowed_hosts))})"
        )
    _assert_resolved_host_is_public(host)


def assert_public_url(url: str) -> None:
    """Require the URL to resolve to a public address (no allowlist). Use for
    open-ended fetches where the host set isn't fixed (e.g. music downloads)."""
    _, host = _hostname(url)
    _assert_resolved_host_is_public(host)
