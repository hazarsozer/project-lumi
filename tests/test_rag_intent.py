"""Tests for RAG intent detection in ReflexRouter.route_rag_intent()."""

import pytest

from src.llm.reflex_router import ReflexRouter


@pytest.fixture()
def router():
    return ReflexRouter()


class TestRouteRagIntent:
    def test_search_my_docs(self, router):
        assert router.route_rag_intent("search my docs for the meeting notes") is True

    def test_look_up_my_notes(self, router):
        assert router.route_rag_intent("look up my notes on Python") is True

    def test_find_in_my_files(self, router):
        assert router.route_rag_intent("find the contract in my files") is True

    def test_check_my_wiki(self, router):
        assert router.route_rag_intent("check my wiki for the deployment steps") is True

    def test_search_in_my_journal(self, router):
        assert router.route_rag_intent("search in my journal for last week") is True

    def test_find_my_documents(self, router):
        assert router.route_rag_intent("find my documents about the project") is True

    def test_ordinary_question_returns_false(self, router):
        assert router.route_rag_intent("what time is it") is False

    def test_greeting_returns_false(self, router):
        assert router.route_rag_intent("hello there") is False

    def test_empty_string_returns_false(self, router):
        assert router.route_rag_intent("") is False

    def test_whitespace_only_returns_false(self, router):
        assert router.route_rag_intent("   ") is False

    def test_case_insensitive(self, router):
        assert router.route_rag_intent("SEARCH MY DOCS") is True

    def test_route_still_works_independently(self, router):
        # route() should be unaffected by route_rag_intent() additions
        assert router.route("hello") == "Hello! How can I help you?"

    def test_search_without_personal_qualifier_returns_false(self, router):
        # "search the web" has no "my docs/notes/files" qualifier
        assert router.route_rag_intent("search the web for news") is False
