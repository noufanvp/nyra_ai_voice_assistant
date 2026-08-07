"""
tests/test_stt.py — Unit tests for core/stt.py (WhisperTranscriber)

Tests:
  - Empty audio returns empty string
  - Numpy array → in-memory WAV conversion
  - Transcriber returns a string type
  - Sine wave (non-speech) is handled gracefully
"""

from __future__ import annotations

import io
import sys
import os
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helper: generate synthetic audio arrays
# ---------------------------------------------------------------------------

def make_sine_wave(freq: float = 440.0, duration_s: float = 1.0,
                   sample_rate: int = 16_000) -> np.ndarray:
    """Generate a pure sine wave as float32 numpy array."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence(duration_s: float = 1.0, sample_rate: int = 16_000) -> np.ndarray:
    """Generate silence as float32 numpy array."""
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


# ---------------------------------------------------------------------------
# WAV conversion utility tests (no model needed)
# ---------------------------------------------------------------------------

class TestNumpyToWav:
    def test_wav_has_correct_params(self):
        """_numpy_to_wav_bytes should produce a valid mono 16kHz WAV."""
        from core.stt import _numpy_to_wav_bytes
        audio = make_sine_wave(duration_s=0.5)
        buf = _numpy_to_wav_bytes(audio, 16_000)
        assert isinstance(buf, io.BytesIO)

        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16_000
            assert wf.getsampwidth() == 2  # int16 = 2 bytes

    def test_empty_array_produces_valid_wav(self):
        """Empty array should produce a valid WAV with 0 frames."""
        from core.stt import _numpy_to_wav_bytes
        audio = np.array([], dtype=np.float32)
        buf = _numpy_to_wav_bytes(audio, 16_000)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 0

    def test_clipping_is_applied(self):
        """Values outside [-1, 1] should be clamped."""
        from core.stt import _numpy_to_wav_bytes
        # Audio with values beyond [-1, 1]
        audio = np.array([2.0, -3.0, 0.5], dtype=np.float32)
        buf = _numpy_to_wav_bytes(audio, 16_000)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            frames = np.frombuffer(wf.readframes(3), dtype=np.int16)
        # Max int16 = 32767, min mapped from -1.0 * 32767 = -32767
        assert frames[0] == 32767
        assert frames[1] == -32767


class TestNormalizeTranscript:
    def test_normalizes_neera(self):
        from core.stt import normalize_transcript
        assert normalize_transcript("Hi Neera, can you solve my doubts?") == "Hi Nyra, can you solve my doubts?"

    def test_normalizes_nira_naira(self):
        from core.stt import normalize_transcript
        assert normalize_transcript("Hey nira what is the weather") == "Hey Nyra what is the weather"
        assert normalize_transcript("Hello naira") == "Hello Nyra"
        assert normalize_transcript("Hello near a") == "Hello Nyra"

    def test_preserves_other_text(self):
        from core.stt import normalize_transcript
        assert normalize_transcript("Hello Nyra, how are you?") == "Hello Nyra, how are you?"
        assert normalize_transcript("") == ""


# ---------------------------------------------------------------------------
# WhisperTranscriber tests (requires model download ~150MB on first run)
# ---------------------------------------------------------------------------

class TestWhisperTranscriber:
    @pytest.fixture(scope="class")
    def transcriber(self):
        """Load the Whisper model once for all tests in this class."""
        pytest.importorskip("faster_whisper")
        from config import CONFIG
        from core.stt import WhisperTranscriber
        try:
            return WhisperTranscriber(CONFIG.stt)
        except Exception as exc:
            pytest.skip(f"Could not load Whisper model: {exc}")

    def test_empty_audio_returns_empty_string(self, transcriber):
        """Empty numpy array must return empty string."""
        result = transcriber.transcribe(np.array([], dtype=np.float32))
        assert isinstance(result, str)
        assert result == ""

    def test_none_audio_returns_empty_string(self, transcriber):
        """None input must return empty string."""
        result = transcriber.transcribe(None)
        assert isinstance(result, str)
        assert result == ""

    def test_sine_wave_returns_string(self, transcriber):
        """Sine wave (non-speech) should return a string (possibly empty)."""
        audio = make_sine_wave(freq=440.0, duration_s=1.5)
        result = transcriber.transcribe(audio)
        assert isinstance(result, str)

    def test_silence_returns_string(self, transcriber):
        """Silent audio should return a string (likely empty)."""
        audio = make_silence(duration_s=1.0)
        result = transcriber.transcribe(audio)
        assert isinstance(result, str)

    def test_return_type_is_always_str(self, transcriber):
        """Transcriber must always return str, never None."""
        for audio in [
            make_silence(0.5),
            make_sine_wave(440, 1.0),
            np.random.randn(16_000).astype(np.float32) * 0.01,
        ]:
            result = transcriber.transcribe(audio)
            assert isinstance(result, str), f"Expected str, got {type(result)}"
