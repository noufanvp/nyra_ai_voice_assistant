"""
core/llm.py — Groq API LLM integration with sentence-level streaming.

Features:
  - GroqClientWrapper: streams token chunks and yields complete sentences
  - Sentence splitter that handles abbreviations (Dr., Mr., etc.)
  - Exponential backoff retry on HTTP 429 (rate limit)
  - Graceful error fallback: yields friendly error message instead of crashing
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Generator
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence-splitting utilities
# ---------------------------------------------------------------------------

# Common abbreviations that should NOT be treated as sentence terminators
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
    "i.e", "e.g", "fig", "no", "vol", "approx", "dept", "est",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}

# Simple split: sentence terminator + whitespace + uppercase
_NAIVE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences, respecting common abbreviations.

    Strategy:
      1. Naively split on `. `, `! `, `? ` followed by uppercase.
      2. Re-join any split that happened on an abbreviation (e.g. "Dr. Smith").

    Parameters
    ----------
    text : str
        Raw text to split.

    Returns
    -------
    list[str]
        List of sentence strings. Never returns empty strings.
    """
    if not text or not text.strip():
        return []

    parts = _NAIVE_SPLIT_RE.split(text)

    # Merge back parts that were incorrectly split at abbreviations
    merged: list[str] = []
    for part in parts:
        if merged:
            # Check if the last token of the previous part is an abbreviation
            prev_last_word = merged[-1].rstrip().rstrip(".").lower().split(".")[-1]
            if prev_last_word in _ABBREVIATIONS:
                # Re-join: this was a false split on an abbreviation period
                merged[-1] = merged[-1] + " " + part
                continue
        merged.append(part)

    return [s.strip() for s in merged if s.strip()]


def _ends_with_sentence_terminator(text: str) -> bool:
    """Return True if text ends with ., !, or ?"""
    return bool(re.search(r'[.!?]["\'"]?\s*$', text))


# ---------------------------------------------------------------------------
# Groq Client Wrapper
# ---------------------------------------------------------------------------

class GroqClientWrapper:
    """Wraps the Groq SDK to provide sentence-level streaming generation."""

    def __init__(self, llm_config):
        self.cfg = llm_config
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.cfg.api_key:
            logger.warning("GROQ_API_KEY is not set. LLM will return error messages only.")
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=self.cfg.api_key)
            logger.info("Groq client initialized (model=%s).", self.cfg.model)
        except ImportError:
            logger.error("groq package not installed. Run: pip install groq")
        except Exception as exc:
            logger.error("Failed to initialize Groq client: %s", exc)

    def stream_sentences(
        self,
        user_text: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> Generator[str, None, None]:
        """Stream the LLM response and yield complete sentences one at a time."""
        if not self._client:
            yield "I'm sorry, the AI service is not configured. Please set your API key."
            return

        if not user_text.strip():
            yield "I didn't catch that. Could you please repeat?"
            return

        messages = [{"role": "system", "content": self.cfg.system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_text})

        buffer = ""
        first_token_logged = False
        t0 = time.monotonic()
        attempt = 0

        while attempt <= self.cfg.max_retries:
            try:
                stream = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    stream=True,
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    token = getattr(delta, "content", None) or ""

                    if not first_token_logged and token:
                        elapsed_ms = (time.monotonic() - t0) * 1000
                        logger.info("Groq first-token latency: %.0f ms", elapsed_ms)
                        first_token_logged = True

                    buffer += token

                    while True:
                        sentences = split_into_sentences(buffer)
                        if len(sentences) >= 2:
                            complete = sentences[0]
                            buffer = " ".join(sentences[1:])
                            logger.debug("LLM sentence: '%s'", complete)
                            yield complete
                        elif len(sentences) == 1 and _ends_with_sentence_terminator(buffer):
                            complete = sentences[0]
                            buffer = ""
                            logger.debug("LLM sentence (final): '%s'", complete)
                            yield complete
                            break
                        else:
                            break

                if buffer.strip():
                    logger.debug("LLM sentence (flush): '%s'", buffer.strip())
                    yield buffer.strip()

                total_ms = (time.monotonic() - t0) * 1000
                logger.info("Groq streaming complete in %.0f ms.", total_ms)
                return

            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str or "rate_limit" in exc_str.lower():
                    wait_s = self.cfg.retry_backoff_base_s * (2 ** attempt)
                    logger.warning("Groq rate limit hit. Retrying in %.1fs…", wait_s)
                    time.sleep(wait_s)
                    attempt += 1
                    continue
                else:
                    logger.error("Groq API error: %s", exc)
                    yield "I'm sorry, the AI service is currently unavailable."
                    return

        logger.error("Groq API: max retries exceeded.")
        yield "I'm sorry, I'm having trouble reaching the AI service right now."
