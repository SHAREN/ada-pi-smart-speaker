"""Small internet-search tools for ADA with strict SSRF protection."""
from __future__ import annotations

import asyncio
import html
import ipaddress
import logging
import re
import socket
import time
from typing import Any
from urllib.parse import quote_plus, urljoin, urlsplit

import httpx
from ddgs import DDGS
from lxml import html as lxml_html

logger = logging.getLogger("voice.web")

MAX_SEARCH_RESULTS = 6
MAX_FETCH_BYTES = 600_000
MAX_TEXT_CHARS = 10_000
MAX_REDIRECTS = 4
SEARCH_CACHE_TTL_SECONDS = 300.0
USER_AGENT = "ADA-Pi-SmartSpeaker/1.0"
_SEARCH_CACHE: dict[tuple[str, str, int], tuple[float, dict[str, Any]]] = {}


def _clean_text(value: str, limit: int = MAX_TEXT_CHARS) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _bing_html_search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    url = "https://www.bing.com/search?q=" + quote_plus(query) + "&setlang=ru&cc=ru"
    response = httpx.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 Chrome/131 Safari/537.36"},
        timeout=8.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    doc = lxml_html.fromstring(response.text)
    results: list[dict[str, Any]] = []
    for node in doc.xpath('//li[contains(concat(" ", normalize-space(@class), " "), " b_algo ")]'):
        anchors = node.xpath(".//h2/a")
        if not anchors:
            continue
        anchor = anchors[0]
        href = str(anchor.get("href") or "").strip()
        title = _clean_text(" ".join(anchor.itertext()), 400)
        paragraphs = node.xpath(".//p")
        snippet = _clean_text(" ".join(paragraphs[0].itertext()), 1200) if paragraphs else ""
        if href and title:
            results.append({"title": title, "href": href, "body": snippet, "source": "Bing"})
        if len(results) >= max_results:
            break
    if not results:
        raise RuntimeError("Bing HTML returned no parseable results")
    return results


def _search_sync(query: str, max_results: int, kind: str) -> tuple[list[dict[str, Any]], str]:
    ddgs = DDGS()
    backends = ("bing", "yahoo", "duckduckgo") if kind == "news" else ("yahoo", "google", "duckduckgo")
    last_error: Exception | None = None
    raw: list[dict[str, Any]] = []
    used_backend = ""
    for backend in backends:
        try:
            if kind == "news":
                raw = ddgs.news(query, max_results=max_results, region="ru-ru", backend=backend)
            else:
                raw = ddgs.text(query, max_results=max_results, region="ru-ru", backend=backend)
            if raw:
                used_backend = backend
                break
        except Exception as exc:
            last_error = exc
            logger.info("web search backend=%s failed query=%r error=%r", backend, query[:120], exc)
    if not raw:
        try:
            raw = _bing_html_search_sync(query, max_results)
            used_backend = "bing-html"
        except Exception as exc:
            logger.info("web search backend=bing-html failed query=%r error=%r", query[:120], exc)
            if last_error is not None:
                raise last_error
            raise

    results: list[dict[str, Any]] = []
    for row in raw or []:
        if not isinstance(row, dict):
            continue
        url = str(row.get("href") or row.get("url") or "").strip()
        title = _clean_text(str(row.get("title") or ""), 400)
        snippet = _clean_text(str(row.get("body") or row.get("description") or ""), 1200)
        source = _clean_text(str(row.get("source") or ""), 200)
        date = _clean_text(str(row.get("date") or ""), 100)
        if url:
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                **({"source": source} if source else {}),
                **({"date": date} if date else {}),
            })
    return results[:max_results], used_backend


async def web_search(query: str, max_results: int = 5, kind: str = "text") -> dict[str, Any]:
    query = " ".join(str(query or "").split())[:500]
    if not query:
        return {"success": False, "error": "Search query is empty"}
    count = min(MAX_SEARCH_RESULTS, max(1, int(max_results or 5)))
    kind = "news" if str(kind).lower() == "news" else "text"
    cache_key = (kind, query.casefold(), count)
    cached = _SEARCH_CACHE.get(cache_key)
    if cached is not None:
        cached_at, cached_payload = cached
        age = time.monotonic() - cached_at
        if age <= SEARCH_CACHE_TTL_SECONDS:
            payload = dict(cached_payload)
            payload["cached"] = True
            payload["cache_age_seconds"] = round(age, 3)
            logger.info("web search cache hit kind=%s query=%r age=%.2fs", kind, query[:120], age)
            return payload
        _SEARCH_CACHE.pop(cache_key, None)

    try:
        results, backend = await asyncio.wait_for(
            asyncio.to_thread(_search_sync, query, count, kind), timeout=12.0
        )
        payload = {
            "success": True,
            "query": query,
            "kind": kind,
            "backend": backend,
            "cached": False,
            "results": results,
        }
        _SEARCH_CACHE[cache_key] = (time.monotonic(), payload)
        # Bound this tiny in-memory cache; newest entries are enough for spoken follow-ups.
        if len(_SEARCH_CACHE) > 64:
            oldest_key = min(_SEARCH_CACHE, key=lambda key: _SEARCH_CACHE[key][0])
            _SEARCH_CACHE.pop(oldest_key, None)
        logger.info("web search kind=%s backend=%s query=%r results=%d", kind, backend, query[:120], len(results))
        return payload
    except Exception as exc:
        logger.warning("web search failed query=%r: %s", query[:120], exc)
        return {"success": False, "error": f"Web search failed: {type(exc).__name__}: {exc}"}


def _public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


async def _validate_public_url(url: str) -> str:
    parsed = urlsplit(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public http/https URLs are allowed")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".local"):
        raise ValueError("Local-network URLs are blocked")
    try:
        direct_ip = ipaddress.ip_address(host)
    except ValueError:
        direct_ip = None
    if direct_ip is not None:
        if not _public_ip(host):
            raise ValueError("Private/reserved IP addresses are blocked")
    else:
        loop = asyncio.get_running_loop()
        infos = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)),
            timeout=4.0,
        )
        addresses = {info[4][0] for info in infos}
        if not addresses or any(not _public_ip(address) for address in addresses):
            raise ValueError("URL resolves to a private/reserved address")
    return parsed.geturl()


def _extract_html_text(raw: bytes, content_type: str) -> str:
    decoded = raw.decode("utf-8", errors="replace")
    if "html" not in content_type.lower():
        return _clean_text(decoded)
    try:
        doc = lxml_html.fromstring(decoded)
        for bad in doc.xpath("//script|//style|//noscript|//svg|//template"):
            bad.drop_tree()
        return _clean_text(doc.text_content())
    except Exception:
        return _clean_text(re.sub(r"<[^>]+>", " ", decoded))


async def web_read(url: str) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,application/xhtml+xml;q=0.9,*/*;q=0.2"}
    try:
        current = await _validate_public_url(url)
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0), follow_redirects=False, headers=headers) as client:
            for _ in range(MAX_REDIRECTS + 1):
                current = await _validate_public_url(current)
                async with client.stream("GET", current) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            return {"success": False, "error": "Redirect without Location header"}
                        current = urljoin(current, location)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if not any(token in content_type.lower() for token in ("text/", "html", "xhtml", "json", "xml")):
                        return {"success": False, "error": f"Unsupported content type: {content_type[:120]}"}
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > MAX_FETCH_BYTES:
                            break
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    text = _extract_html_text(raw, content_type)
                    logger.info("web read host=%s status=%d chars=%d", urlsplit(current).hostname, response.status_code, len(text))
                    return {
                        "success": True,
                        "url": current,
                        "status": response.status_code,
                        "content_type": content_type[:120],
                        "text": text,
                        "truncated": total > MAX_FETCH_BYTES or len(text) >= MAX_TEXT_CHARS,
                    }
            return {"success": False, "error": "Too many redirects"}
    except Exception as exc:
        logger.warning("web read failed url=%r: %s", str(url)[:180], exc)
        return {"success": False, "error": f"Web read failed: {type(exc).__name__}: {exc}"}


WEB_TOOL_DECLARATIONS = [
    {
        "name": "web_search",
        "description": "Search the public internet for current or changing information. Use kind='news' for recent news. Returns titles, snippets and public URLs; call web_read on a useful result when the snippet is not enough.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Concise web search query."},
                "max_results": {"type": "integer", "description": "Number of results, 1 to 6."},
                "kind": {"type": "string", "enum": ["text", "news"], "description": "General web or recent news search."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "web_read",
        "description": "Read a public HTTP(S) page returned by web_search when more detail is needed. Private/LAN/localhost addresses are blocked.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Public http/https URL from a search result."}},
            "required": ["url"],
            "additionalProperties": False,
        },
    },
]
