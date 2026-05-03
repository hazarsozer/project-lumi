"""Tests for WebSearchTool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tools.web_search import WebSearchTool


@pytest.fixture()
def tool() -> WebSearchTool:
    return WebSearchTool()


def _make_response(html: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = html
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


_FAKE_HTML = """
<html><body>
  <a class="result__a" href="/url">Python language</a>
  <a class="result__snippet">Python is a high-level programming language.</a>
  <a class="result__a" href="/url2">Python docs</a>
  <a class="result__snippet">Official Python documentation and tutorials.</a>
</body></html>
"""


class TestWebSearchToolSuccess:
    def test_returns_success_with_snippets(self, tool: WebSearchTool) -> None:
        with patch("requests.get", return_value=_make_response(_FAKE_HTML)):
            result = tool.execute(query="python language")

        assert result.success is True
        assert "Python" in result.output
        assert "high-level" in result.output
        assert result.data["query"] == "python language"
        assert len(result.data["results"]) == 2

    def test_output_is_numbered_list(self, tool: WebSearchTool) -> None:
        with patch("requests.get", return_value=_make_response(_FAKE_HTML)):
            result = tool.execute(query="python")

        assert result.output.startswith("Search results for 'python':")
        assert "1." in result.output
        assert "2." in result.output


class TestWebSearchToolEdgeCases:
    def test_empty_query_returns_failure(self, tool: WebSearchTool) -> None:
        result = tool.execute(query="")
        assert result.success is False
        assert "non-empty" in result.output

    def test_missing_query_kwarg_returns_failure(self, tool: WebSearchTool) -> None:
        result = tool.execute()
        assert result.success is False

    def test_non_string_query_returns_failure(self, tool: WebSearchTool) -> None:
        result = tool.execute(query=42)
        assert result.success is False

    def test_http_error_returns_failure(self, tool: WebSearchTool) -> None:
        with patch("requests.get", return_value=_make_response("", 503)):
            result = tool.execute(query="test")
        assert result.success is False
        assert "failed" in result.output.lower()

    def test_network_exception_returns_failure(self, tool: WebSearchTool) -> None:
        with patch("requests.get", side_effect=ConnectionError("no network")):
            result = tool.execute(query="test")
        assert result.success is False
        assert result.output != ""

    def test_no_results_returns_failure(self, tool: WebSearchTool) -> None:
        empty_html = "<html><body>No results.</body></html>"
        with patch("requests.get", return_value=_make_response(empty_html)):
            result = tool.execute(query="xyzzy_nonexistent")
        assert result.success is False
        assert "No results" in result.output

    def test_execute_never_raises(self, tool: WebSearchTool) -> None:
        with patch("requests.get", side_effect=RuntimeError("unexpected")):
            result = tool.execute(query="test")
        assert result.success is False


class TestWebSearchToolProtocol:
    def test_has_name(self, tool: WebSearchTool) -> None:
        assert tool.name == "web_search"

    def test_has_description(self, tool: WebSearchTool) -> None:
        assert isinstance(tool.description, str) and len(tool.description) > 0
