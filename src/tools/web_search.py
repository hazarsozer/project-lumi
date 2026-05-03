"""Web search tool for Project Lumi — DuckDuckGo HTML scrape, no API key."""

from __future__ import annotations

import logging
import urllib.parse

from src.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

_DDG_URL = "https://html.duckduckgo.com/html/"
_MAX_RESULTS = 5
_TIMEOUT_S = 8
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Lumi/0.1)"}


class WebSearchTool:
    """Search the web via DuckDuckGo HTML endpoint (no API key required).

    Schema::

        {"tool": "web_search", "args": {"query": "<search terms>"}}

    Returns up to 5 result snippets as a numbered list.
    """

    name: str = "web_search"
    description: str = (
        "Search the web for information. "
        "Args: query (str). "
        "Returns top search result snippets."
    )

    def execute(self, **kwargs: object) -> ToolResult:  # noqa: D102
        query = kwargs.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                success=False,
                output="web_search requires a non-empty 'query' string.",
                data={},
            )

        try:
            import requests
            from bs4 import BeautifulSoup

            resp = requests.get(
                _DDG_URL,
                params={"q": query},
                headers=_HEADERS,
                timeout=_TIMEOUT_S,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("web_search HTTP error for %r: %s", query, exc)
            return ToolResult(success=False, output=f"Search request failed: {exc}", data={})

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(resp.text, "html.parser")
            snippets = [el.get_text(" ", strip=True) for el in soup.select("a.result__snippet")]
            titles = [el.get_text(" ", strip=True) for el in soup.select("a.result__a")]
        except Exception as exc:
            logger.warning("web_search parse error: %s", exc)
            return ToolResult(success=False, output=f"Failed to parse results: {exc}", data={})

        if not snippets:
            return ToolResult(
                success=False,
                output=f"No results found for: {query}",
                data={"query": query, "results": []},
            )

        items = snippets[:_MAX_RESULTS]
        item_titles = titles[:_MAX_RESULTS]
        lines = [
            f"{i + 1}. {item_titles[i] + ': ' if i < len(item_titles) else ''}{items[i]}"
            for i in range(len(items))
        ]
        output = f"Search results for '{query}':\n" + "\n".join(lines)

        return ToolResult(
            success=True,
            output=output,
            data={"query": query, "results": [{"title": t, "snippet": s} for t, s in zip(item_titles, items)]},
        )
