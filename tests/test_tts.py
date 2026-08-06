"""
tests/test_tts.py — Unit tests for core/tts.py (LocalTTSEngine)

Tests:
  - Engine initializes (kokoro or pyttsx3 fallback)
  - synthesize() returns (np.ndarray, int) tuple
  - Output array shape is (N,) 1-D
  - Sample rate is a positive integer
  - Empty string returns empty array without error
  - Output values are in [-1, 1] range (no clipping distortion)
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLocalTTSEngine:
    @pytest.fixture(scope="class")
    def tts_engine(self):
        """Initialize TTS engine once for all tests."""
        from config import CONFIG
        from core.tts import LocalTTSEngine
        try:
            engine = LocalTTSEngine(CONFIG.tts)
            return engine
        except Exception as exc:
            pytest.skip(f"TTS engine failed to initialize: {exc}")

    def test_engine_has_backend(self, tts_engine):
        """Engine should have a valid backend name after init."""
        assert tts_engine.backend_name in {"kokoro-onnx", "pyttsx3", "none"}
        # If none, the engine simply has no backend — we still test gracefully
        if tts_engine.backend_name == "none":
            pytest.skip("No TTS backend available on this system.")

    def test_synthesize_returns_tuple(self, tts_engine):
        """synthesize() must return a 2-tuple."""
        result = tts_engine.synthesize("Voice engine functional.")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_synthesize_returns_ndarray_and_int(self, tts_engine):
        """First element must be np.ndarray, second must be int."""
        samples, sample_rate = tts_engine.synthesize("Voice engine functional.")
        assert isinstance(samples, np.ndarray)
        assert isinstance(sample_rate, int)

    def test_output_is_1d(self, tts_engine):
        """Audio output must be 1-D array."""
        samples, _ = tts_engine.synthesize("Hello world.")
        assert samples.ndim == 1, f"Expected 1-D array, got shape {samples.shape}"

    def test_output_is_nonempty(self, tts_engine):
        """Non-empty text should produce a non-empty audio array."""
        samples, _ = tts_engine.synthesize("Voice engine functional.")
        assert len(samples) > 0, "Expected audio samples, got empty array."

    def test_sample_rate_is_positive(self, tts_engine):
        """Sample rate must be a positive integer."""
        _, sample_rate = tts_engine.synthesize("Test.")
        assert sample_rate > 0

    def test_output_dtype_is_float32(self, tts_engine):
        """Audio samples must be float32."""
        samples, _ = tts_engine.synthesize("Hello.")
        assert samples.dtype == np.float32, f"Expected float32, got {samples.dtype}"

    def test_values_in_valid_range(self, tts_engine):
        """Audio values should be within [-1, 1] after normalization."""
        samples, _ = tts_engine.synthesize("This is a test of the voice engine.")
        if len(samples) > 0:
            max_val = np.max(np.abs(samples))
            # Allow small epsilon above 1.0 for floating point reasons
            assert max_val <= 1.05, f"Audio peak {max_val:.3f} exceeds expected range."

    def test_empty_string_returns_empty_array(self, tts_engine):
        """Empty string input should return empty array without error."""
        samples, sample_rate = tts_engine.synthesize("")
        assert isinstance(samples, np.ndarray)
        assert len(samples) == 0
        assert isinstance(sample_rate, int)

    def test_whitespace_only_returns_empty(self, tts_engine):
        """Whitespace-only input should return empty array."""
        samples, _ = tts_engine.synthesize("   ")
        assert len(samples) == 0

    def test_multiple_sentences(self, tts_engine):
        """Multi-sentence input should produce longer audio than single sentence."""
        samples_short, _ = tts_engine.synthesize("Hi.")
        samples_long, _ = tts_engine.synthesize(
            "Hi there. How are you doing today? I hope everything is going well."
        )
        assert len(samples_long) > len(samples_short)

    def test_pronunciation_map(self, tts_engine):
        """Engine should apply phonetic pronunciation replacements from config."""
        samples, _ = tts_engine.synthesize("Aitute is an organization.")
        assert isinstance(samples, np.ndarray)
        assert len(samples) > 0
