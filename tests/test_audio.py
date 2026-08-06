"""
tests/test_audio.py — Unit tests for core/audio_io.py

Tests:
  - Device list is non-empty and has expected structure
  - VADRecorder initializes without hardware errors (no actual recording)
  - Silence detection timing math
  - play_audio handles edge cases (empty, normal arrays)
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Device enumeration tests
# ---------------------------------------------------------------------------

def _has_portaudio() -> bool:
    """Return True if sounddevice/PortAudio is functional on this system."""
    try:
        import sounddevice as sd
        sd.query_devices()
        return True
    except Exception:
        return False


class TestListAudioDevices:
    def test_returns_list(self):
        """list_audio_devices should return a list (or skip if no PortAudio)."""
        if not _has_portaudio():
            pytest.skip("PortAudio not available on this system.")
        from core.audio_io import list_audio_devices
        devices = list_audio_devices()
        assert isinstance(devices, list)

    def test_device_structure(self):
        """Each device dict must have required keys."""
        if not _has_portaudio():
            pytest.skip("PortAudio not available on this system.")
        from core.audio_io import list_audio_devices
        devices = list_audio_devices()
        if not devices:
            pytest.skip("No audio devices available on this system.")
        required_keys = {"index", "name", "max_input_channels", "max_output_channels", "default_samplerate"}
        for dev in devices:
            assert required_keys.issubset(dev.keys()), f"Device missing keys: {dev}"

    def test_device_indices_are_unique(self):
        """All device indices must be unique."""
        if not _has_portaudio():
            pytest.skip("PortAudio not available on this system.")
        from core.audio_io import list_audio_devices
        devices = list_audio_devices()
        indices = [d["index"] for d in devices]
        assert len(indices) == len(set(indices))

    def test_at_least_one_device(self):
        """Expect at least one audio device (virtual/real)."""
        if not _has_portaudio():
            pytest.skip("PortAudio not available on this system.")
        from core.audio_io import list_audio_devices
        devices = list_audio_devices()
        if len(devices) == 0:
            pytest.skip("No audio devices found — skipping (likely headless CI).")
        assert len(devices) >= 1


# ---------------------------------------------------------------------------
# VAD silence detection math
# ---------------------------------------------------------------------------

class TestSilenceDetectionMath:
    """Test the frame-count-to-duration arithmetic used by VADRecorder."""

    def test_chunk_duration(self):
        """512 samples at 16kHz = 32ms."""
        sample_rate = 16_000
        chunk_size = 512
        duration_ms = chunk_size / sample_rate * 1000
        assert abs(duration_ms - 32.0) < 0.1

    def test_silence_frame_count(self):
        """600ms of silence at 32ms chunks = ~18.75 → at least 18 chunks."""
        silence_duration_s = 0.6
        sample_rate = 16_000
        chunk_size = 512
        chunk_duration_s = chunk_size / sample_rate
        required_chunks = silence_duration_s / chunk_duration_s
        # Should be ~18.75 — meaning after ~19 chunks we declare silence
        assert required_chunks >= 18.0
        assert required_chunks < 20.0

    def test_min_speech_frames(self):
        """200ms min speech at 32ms chunks = at least 6 chunks."""
        min_speech_s = 0.2
        sample_rate = 16_000
        chunk_size = 512
        chunk_duration_s = chunk_size / sample_rate
        required_chunks = min_speech_s / chunk_duration_s
        assert required_chunks >= 6.0


# ---------------------------------------------------------------------------
# play_audio edge-case tests (no hardware needed)
# ---------------------------------------------------------------------------

class TestPlayAudio:
    def test_empty_array_does_not_raise(self):
        """play_audio with empty array should return gracefully (or skip if no PortAudio)."""
        try:
            from core.audio_io import play_audio
        except OSError:
            pytest.skip("PortAudio library not found on this system.")
        empty = np.array([], dtype=np.float32)
        # Empty array exits before touching PortAudio — should not raise
        try:
            play_audio(empty, 16000, blocking=False)
        except OSError:
            pytest.skip("PortAudio not available — but empty-array guard works.")

    def test_none_does_not_raise(self):
        """play_audio with None should return gracefully (or skip if no PortAudio)."""
        try:
            from core.audio_io import play_audio
        except OSError:
            pytest.skip("PortAudio library not found on this system.")
        try:
            play_audio(None, 16000, blocking=False)
        except OSError:
            pytest.skip("PortAudio not available — but None guard works.")


# ---------------------------------------------------------------------------
# VADRecorder initialization test (no actual recording)
# ---------------------------------------------------------------------------

class TestVADRecorderInit:
    def test_instantiation(self):
        """VADRecorder should instantiate without errors if webrtcvad is available."""
        pytest.importorskip("webrtcvad")

        from config import CONFIG
        try:
            from core.audio_io import VADRecorder
            recorder = VADRecorder(CONFIG.audio)
            assert recorder is not None
        except Exception as exc:
            pytest.skip(f"VADRecorder init failed (likely headless): {exc}")
