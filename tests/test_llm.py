"""
tests/test_llm.py — Unit tests for core/llm.py (GroqClientWrapper)

Tests:
  - Sentence splitting regex handles abbreviations correctly
  - Sentence-end detection logic
  - Live API call (skipped if GROQ_API_KEY not set)
  - Invalid/missing key returns friendly error message, not exception
  - Empty user input returns graceful response
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Sentence splitting tests (no API needed)
# ---------------------------------------------------------------------------

class TestSplitIntoSentences:
    """Test the sentence splitter with real-world edge cases."""

    def test_basic_split(self):
        from core.llm import split_into_sentences
        result = split_into_sentences("Hello there. How are you?")
        assert len(result) == 2
        assert result[0] == "Hello there."
        assert result[1] == "How are you?"

    def test_abbreviation_dr_not_split(self):
        from core.llm import split_into_sentences
        # "Dr. Smith" should NOT be split at the period
        result = split_into_sentences("Dr. Smith is here. How can I help?")
        assert len(result) == 2
        assert "Dr. Smith" in result[0]

    def test_abbreviation_mr_not_split(self):
        from core.llm import split_into_sentences
        result = split_into_sentences("Mr. Johnson called. Please call back.")
        assert len(result) == 2

    def test_exclamation_and_question(self):
        from core.llm import split_into_sentences
        result = split_into_sentences("Great! Really? Absolutely.")
        assert len(result) == 3

    def test_single_sentence_no_split(self):
        from core.llm import split_into_sentences
        result = split_into_sentences("This is a single sentence")
        assert len(result) == 1

    def test_empty_string(self):
        from core.llm import split_into_sentences
        result = split_into_sentences("")
        assert result == []

    def test_whitespace_only(self):
        from core.llm import split_into_sentences
        result = split_into_sentences("   ")
        assert result == []

    def test_multiple_periods_in_abbreviation(self):
        from core.llm import split_into_sentences
        # "i.e." should not be split on inner periods
        result = split_into_sentences("Use it correctly, i.e. with care. That is all.")
        assert len(result) == 2


class TestEndsWithTerminator:
    def test_period(self):
        from core.llm import _ends_with_sentence_terminator
        assert _ends_with_sentence_terminator("Hello.") is True

    def test_question(self):
        from core.llm import _ends_with_sentence_terminator
        assert _ends_with_sentence_terminator("How are you?") is True

    def test_exclamation(self):
        from core.llm import _ends_with_sentence_terminator
        assert _ends_with_sentence_terminator("Wow!") is True

    def test_no_terminator(self):
        from core.llm import _ends_with_sentence_terminator
        assert _ends_with_sentence_terminator("Hello there") is False

    def test_trailing_space(self):
        from core.llm import _ends_with_sentence_terminator
        assert _ends_with_sentence_terminator("Hello.  ") is True


# ---------------------------------------------------------------------------
# GroqClientWrapper tests
# ---------------------------------------------------------------------------

class TestGroqClientWrapperNoKey:
    """Test graceful degradation when no API key is present."""

    def _make_wrapper_no_key(self):
        from config import LLMConfig
        from core.llm import GroqClientWrapper
        cfg = LLMConfig(api_key="")
        return GroqClientWrapper(cfg)

    def test_no_key_returns_error_message(self):
        """Without API key, stream_sentences should yield an error string."""
        wrapper = self._make_wrapper_no_key()
        sentences = list(wrapper.stream_sentences("Hello"))
        assert len(sentences) >= 1
        assert isinstance(sentences[0], str)
        # Should contain a helpful message, not raise
        assert len(sentences[0]) > 5

    def test_empty_text_returns_message(self):
        """Empty input string should yield a graceful prompt."""
        wrapper = self._make_wrapper_no_key()
        sentences = list(wrapper.stream_sentences(""))
        assert len(sentences) >= 1
        assert isinstance(sentences[0], str)

    def test_no_exception_on_bad_key(self):
        """Invalid key should not propagate an exception to caller."""
        from config import LLMConfig
        from core.llm import GroqClientWrapper
        cfg = LLMConfig(api_key="invalid_key_xyz")
        wrapper = GroqClientWrapper(cfg)
        try:
            sentences = list(wrapper.stream_sentences("What time is it?"))
            assert len(sentences) >= 1
        except Exception as exc:
            pytest.fail(f"Exception should not propagate to caller: {exc}")


class TestGroqClientWrapperLive:
    """Live API tests — skipped if GROQ_API_KEY is not set."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        key = os.getenv("GROQ_API_KEY", "")
        if not key or key == "your_groq_api_key_here":
            pytest.skip("GROQ_API_KEY not set — skipping live API test.")

    def test_live_response_yields_sentences(self):
        """Live call should yield at least one sentence string."""
        from config import CONFIG
        from core.llm import GroqClientWrapper
        wrapper = GroqClientWrapper(CONFIG.llm)
        sentences = list(wrapper.stream_sentences("Say hello in one sentence."))
        assert len(sentences) >= 1
        for s in sentences:
            assert isinstance(s, str)
            assert len(s.strip()) > 0

    def test_live_response_is_str_type(self):
        """All yielded values must be strings."""
        from config import CONFIG
        from core.llm import GroqClientWrapper
        wrapper = GroqClientWrapper(CONFIG.llm)
        for sentence in wrapper.stream_sentences("What is 2 plus 2?"):
            assert isinstance(sentence, str)
