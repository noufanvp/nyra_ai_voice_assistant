"""
core/tts.py — Local TTS engine with kokoro-onnx (primary) and pyttsx3 (fallback).

Features:
  - Auto-downloads kokoro model files if missing
  - Returns numpy float32 PCM array (no disk I/O required)
  - Falls back to pyttsx3 if kokoro fails to initialize
  - Logs synthesis latency per call
"""

from __future__ import annotations

import io
import logging
import re
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model auto-downloader
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path) -> None:
    """Download a file from `url` to `dest` with progress logging."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                logger.debug("  %.1f%%", pct)

    logger.info("Download complete: %s", dest.name)


# ---------------------------------------------------------------------------
# Kokoro-ONNX backend
# ---------------------------------------------------------------------------

class _KokoroBackend:
    """Kokoro-ONNX TTS backend."""

    def __init__(self, tts_config):
        self.cfg = tts_config
        self._engine = None
        self._load()

    def _ensure_models(self) -> bool:
        """Ensure model files exist, downloading if cfg.auto_download is True."""
        model_path = Path(self.cfg.model_path)
        voices_path = Path(self.cfg.voices_path)

        for path, url in [
            (model_path, self.cfg.model_url),
            (voices_path, self.cfg.voices_url),
        ]:
            if not path.exists():
                if self.cfg.auto_download:
                    try:
                        _download_file(url, path)
                    except Exception as exc:
                        logger.error("Failed to download %s: %s", path.name, exc)
                        return False
                else:
                    logger.error("Model file missing: %s", path)
                    return False
        return True

    def _load(self) -> None:
        if not self._ensure_models():
            raise RuntimeError("Kokoro model files not available.")
        try:
            from kokoro_onnx import Kokoro
            self._engine = Kokoro(self.cfg.model_path, self.cfg.voices_path)
            logger.info("Kokoro-ONNX TTS loaded (voice=%s).", self.cfg.voice)
        except Exception as exc:
            logger.error("Failed to load Kokoro-ONNX: %s", exc)
            raise

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        samples, sample_rate = self._engine.create(
            text,
            voice=self.cfg.voice,
            speed=self.cfg.speed,
            lang=self.cfg.lang,
        )
        return np.array(samples, dtype=np.float32), int(sample_rate)


# ---------------------------------------------------------------------------
# pyttsx3 fallback backend
# ---------------------------------------------------------------------------

class _Pyttsx3Backend:
    """pyttsx3 offline TTS backend (fallback). Returns audio via temp WAV."""

    def __init__(self):
        import pyttsx3
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 165)
        logger.info("pyttsx3 TTS fallback initialized.")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize via pyttsx3 → temp WAV → numpy array."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._engine.save_to_file(text, tmp_path)
            self._engine.runAndWait()

            # Read WAV back as numpy
            with wave.open(tmp_path, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return audio, sr
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# gTTS cloud fallback backend (0 MB RAM model footprint for Render free tier)
# ---------------------------------------------------------------------------

class _GTTSBackend:
    """gTTS online backend (0 MB RAM model footprint, ideal for Render 512MB limit)."""

    def __init__(self):
        from gtts import gTTS
        logger.info("gTTS online backend initialized (0 MB RAM model footprint).")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        import os
        import tempfile
        import av
        from gtts import gTTS

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tts = gTTS(text=text, lang="en")
            tts.save(tmp_path)

            container = av.open(tmp_path)
            resampler = av.AudioResampler(format="flt", layout="mono", rate=22050)
            samples_list = []
            for frame in container.decode(audio=0):
                for rframe in resampler.resample(frame):
                    samples_list.append(rframe.to_ndarray())
            container.close()

            if samples_list:
                audio = np.concatenate(samples_list, axis=1).squeeze()
            else:
                audio = np.array([], dtype=np.float32)
            return audio, 22050
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Public TTS Engine
# ---------------------------------------------------------------------------

class LocalTTSEngine:
    """
    Unified local TTS engine. Uses kokoro-onnx by default for local desktop,
    gTTS for Render cloud free tier (0 MB RAM), and falls back to pyttsx3.

    Parameters
    ----------
    tts_config : TTSConfig
        Configuration dataclass from config.py.
    """

    def __init__(self, tts_config):
        self.cfg = tts_config
        self._backend = None
        self._backend_name = "none"
        self._init_backend()

    def _init_backend(self) -> None:
        import os

        # On Render or when USE_GTTS=true, use gTTS to fit within 512MB RAM cap
        use_gtts = os.getenv("USE_GTTS", "").lower() in ("true", "1") or os.getenv("RENDER") is not None
        if use_gtts:
            try:
                self._backend = _GTTSBackend()
                self._backend_name = "gtts"
                return
            except Exception as exc:
                logger.warning("gTTS backend init failed (%s), trying Kokoro-ONNX...", exc)

        # Try kokoro-onnx first for local desktop
        try:
            self._backend = _KokoroBackend(self.cfg)
            self._backend_name = "kokoro-onnx"
            return
        except Exception as exc:
            logger.warning("Kokoro-ONNX unavailable (%s). Trying gTTS fallback…", exc)

        try:
            self._backend = _GTTSBackend()
            self._backend_name = "gtts"
            return
        except Exception as exc:
            logger.warning("gTTS fallback failed (%s). Trying pyttsx3 fallback...", exc)

        # Fallback to pyttsx3
        try:
            self._backend = _Pyttsx3Backend()
            self._backend_name = "pyttsx3"
        except Exception as exc:
            logger.error("pyttsx3 also failed: %s. TTS is non-functional.", exc)
            self._backend = None

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Convert text to speech audio.

        Parameters
        ----------
        text : str
            Text to synthesize.

        Returns
        -------
        tuple[np.ndarray, int]
            (float32 PCM samples, sample_rate). Returns (empty_array, 22050) on error.
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32), 22050

        if self._backend is None:
            logger.error("No TTS backend available.")
            return np.array([], dtype=np.float32), 22050

        # Apply phonetic pronunciation overrides for proper nouns
        if hasattr(self.cfg, "pronunciation_map") and self.cfg.pronunciation_map:
            for word, replacement in self.cfg.pronunciation_map.items():
                text = re.sub(rf'\b{re.escape(word)}\b', replacement, text, flags=re.IGNORECASE)

        t0 = time.monotonic()
        try:
            samples, sample_rate = self._backend.synthesize(text)
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "TTS [%s] ▸ %.0f ms | %d samples @ %dHz",
                self._backend_name, elapsed_ms, len(samples), sample_rate,
            )
            return samples, sample_rate
        except Exception as exc:
            logger.error("TTS synthesis error: %s", exc)
            return np.array([], dtype=np.float32), 22050

    @property
    def backend_name(self) -> str:
        return self._backend_name
