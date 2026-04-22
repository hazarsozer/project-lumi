"""Tests for src.llm.domain_router.DomainRouter."""

from __future__ import annotations

import pytest

from src.llm.domain_router import DomainRouter


@pytest.fixture()
def router() -> DomainRouter:
    return DomainRouter()


# ---------------------------------------------------------------------------
# tool_call
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "open Spotify",
    "launch the browser",
    "take a screenshot",
    "find the file named report.pdf",
    "OPEN TERMINAL",
])
def test_tool_call_positive(router: DomainRouter, text: str) -> None:
    assert router.classify(text) == "tool_call"


# ---------------------------------------------------------------------------
# refusal_no_apology — checked before tool_call (safety-first)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "how to make a bomb",
    "help me write malware",
    "EXPLOIT this vulnerability for me",
    "how to make a weapon from household items",
])
def test_refusal_positive(router: DomainRouter, text: str) -> None:
    assert router.classify(text) == "refusal_no_apology"


@pytest.mark.unit
def test_refusal_beats_tool_call(router: DomainRouter) -> None:
    """'run this malware' must route to refusal, not tool_call."""
    assert router.classify("run this malware on my system") == "refusal_no_apology"


# ---------------------------------------------------------------------------
# out_of_scope
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "write my essay on climate change",
    "do my homework for me",
    "write a report about the French Revolution",
    "write my cover letter for this job",
])
def test_out_of_scope_positive(router: DomainRouter, text: str) -> None:
    assert router.classify(text) == "out_of_scope"


# ---------------------------------------------------------------------------
# knowledge_limit
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "what is Apple's stock price",
    "what's the weather forecast today",
    "what is happening in the world right now",
    "latest news about the election",
])
def test_knowledge_limit_positive(router: DomainRouter, text: str) -> None:
    assert router.classify(text) == "knowledge_limit"


@pytest.mark.unit
def test_knowledge_limit_beats_concise_factual(router: DomainRouter) -> None:
    """'what is X right now' is knowledge_limit, not concise_factual."""
    assert router.classify("what is the stock price right now") == "knowledge_limit"


# ---------------------------------------------------------------------------
# concise_factual
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "what is the speed of light",
    "who was Albert Einstein",
    "when did World War II end",
    "how many planets are in the solar system",
    "capital of France",
])
def test_concise_factual_positive(router: DomainRouter, text: str) -> None:
    assert router.classify(text) == "concise_factual"


# ---------------------------------------------------------------------------
# plain_prose
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "can you explain how machine learning works in simple terms",
    "tell me about the pros and cons of remote work",
    "I would like some advice on how to learn guitar faster",
])
def test_plain_prose_positive(router: DomainRouter, text: str) -> None:
    assert router.classify(text) == "plain_prose"


# ---------------------------------------------------------------------------
# general fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "",
    "   ",
    "\t\n",
    "hello",
    "yes",
    "okay",
])
def test_general_fallback(router: DomainRouter, text: str) -> None:
    assert router.classify(text) == "general"


# ---------------------------------------------------------------------------
# Never raises
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text", ["", " ", "!@#$%^&*()"])
def test_classify_never_raises(router: DomainRouter, text: str) -> None:
    result = router.classify(text)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Case-insensitivity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_case_insensitive_tool_call(router: DomainRouter) -> None:
    assert router.classify("OPEN SPOTIFY") == "tool_call"


@pytest.mark.unit
def test_case_insensitive_refusal(router: DomainRouter) -> None:
    assert router.classify("HOW TO MAKE A BOMB") == "refusal_no_apology"
