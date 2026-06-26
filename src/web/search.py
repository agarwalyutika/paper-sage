"""
Web search -- the fallback source when the paper corpus doesn't cover a question.

Swappable backend (like the LLM provider):
  - "duckduckgo": free, no API key (default)
  - "tavily":     free key, returns cleaner page content (set TAVILY_API_KEY + WEB_SEARCH_BACKEND=tavily)

Each result is a dict: {title, url, content}. The answer generator cites these by [n].
"""
from urllib.parse import urlparse

from src.config import settings


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "web"


def _duckduckgo(query: str, k: int) -> list[dict]:
    from ddgs import DDGS
    out = []
    with DDGS() as d:
        for r in d.text(query, max_results=k):
            url = r.get("href", "") or r.get("url", "")
            out.append({
                "title": r.get("title", "")[:120] or _domain(url),
                "url": url,
                "content": r.get("body", "") or "",
                "domain": _domain(url),
            })
    return out


def _tavily(query: str, k: int) -> list[dict]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    resp = client.search(query, max_results=k)
    out = []
    for r in resp.get("results", []):
        out.append({
            "title": r.get("title", "")[:120] or _domain(r.get("url", "")),
            "url": r.get("url", ""),
            "content": r.get("content", "") or "",
            "domain": _domain(r.get("url", "")),
        })
    return out


def web_search(query: str, k: int | None = None) -> list[dict]:
    """Search the web and return result dicts (empty list on failure)."""
    k = k or settings.WEB_SEARCH_K
    try:
        if settings.WEB_SEARCH_BACKEND == "tavily" and settings.TAVILY_API_KEY:
            return _tavily(query, k)
        return _duckduckgo(query, k)
    except Exception as e:
        print(f"  ! web search failed: {e}")
        return []
