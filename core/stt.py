"""
core/stt.py — Local Speech-to-Text via faster-whisper.

Wraps faster_whisper.WhisperModel with:
  - Model caching on first load
  - Configurable compute_type and device
  - Latency logging
  - numpy array input (no disk I/O)
"""

from __future__ import annotations

import io
import logging
import time
import wave
from typing import Optional

import numpy as np

import re

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


WHISPER_HALLUCINATIONS = {
    "you", "you.", "you!", "you?",
    "thank you", "thank you.", "thank you!", "thank you very much.", "thank you very much!",
    "subtitles", "subtitles by", "subtitles by amara.org", "amara.org",
    "bye", "bye.", "goodbye", "goodbye.",
    "subscribe", "like and subscribe",
    "the", "a", "so", "oh", "um", "uh", "peace", "peace.",
    "english speech.", "english speech", "english speech!",
}


def _is_allowed_script(text: str) -> bool:
    """Return True if text contains only English/ASCII characters."""
    for char in text:
        code = ord(char)
        if not (code < 128 or char in " \t\n\r"):
            return False
    return True


def _is_repetitive(text: str, repeat_threshold: int = 4) -> bool:
    """
    Detect Whisper hallucination loops like 'I'm a little bit of a little bit of...'.

    Checks for any phrase of 2-6 words that appears more than `repeat_threshold`
    times in the text. Real speech very rarely repeats a phrase 4+ times.

    Also rejects transcripts that are clearly too long (>200 words) since our
    VAD limits recordings to a few seconds of actual speech.
    """
    words = text.split()
    # Hard cap: real utterances in this context are ≤ ~50 words
    if len(words) > 80:
        return True
    if len(words) < 10:
        return False  # too short to have a meaningful repeat pattern
    for phrase_len in range(2, 7):
        if len(words) < phrase_len * repeat_threshold:
            continue
        phrase = " ".join(words[:phrase_len]).lower()
        count = text.lower().count(phrase)
        if count >= repeat_threshold:
            return True
    return False


class WhisperTranscriber:
    """
    Local STT engine wrapping faster-whisper's WhisperModel.

    Parameters
    ----------
    stt_config : STTConfig
        Configuration dataclass from config.py.
    """

    def __init__(self, stt_config, progress_callback=None):
        self.cfg = stt_config
        self._model = None
        self._progress_callback = progress_callback
        self._load_model()

    def _progress(self, message: str, percent: int = -1) -> None:
        """Forward progress to callback if provided."""
        if self._progress_callback:
            self._progress_callback(message, percent)

    def _load_model(self) -> None:
        """Download (if needed) and load the Whisper model."""
        logger.info(
            "Loading Whisper model '%s' [device=%s, compute=%s]…",
            self.cfg.model_size,
            self.cfg.device,
            self.cfg.compute_type,
        )

        # Check if model is already in the HuggingFace cache
        model_id = f"Systran/faster-whisper-{self.cfg.model_size}"
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(model_id, "config.json")
            is_cached = cached is not None and cached != "_NOT_CACHED"
        except Exception:
            is_cached = False

        if is_cached:
            self._progress(f"Loading Whisper model ({self.cfg.model_size})…", -1)
        else:
            self._progress(
                f"Downloading Whisper model ({self.cfg.model_size}) — first run only…",
                -1,
            )

        t0 = time.monotonic()
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.cfg.model_size,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
                cpu_threads=self.cfg.cpu_threads,
                download_root=None,  # uses HuggingFace cache default
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.info("Whisper model ready in %.0f ms.", elapsed_ms)
            self._progress(f"Whisper ready ✔ ({elapsed_ms/1000:.1f}s)", 100)
        except Exception as exc:
            logger.exception("Failed to load Whisper model: %s", exc)
            raise

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """
        Transcribe a numpy float32 audio array to text.

        Parameters
        ----------
        audio : np.ndarray
            Mono float32 audio at `sample_rate` Hz.
        sample_rate : int
            Audio sample rate (must be 16000 for best results).

        Returns
        -------
        str
            Transcribed text. Returns empty string for silent/empty audio.
        """
        if audio is None or len(audio) == 0:
            logger.debug("STT: empty audio input, returning empty string.")
            return ""

        # Convert numpy array → in-memory WAV (faster-whisper accepts file-like)
        wav_buffer = _numpy_to_wav_bytes(audio, sample_rate)

        t0 = time.monotonic()
        try:
            # Single-pass transcription (VAD clipping is already done by VADRecorder)
            segments, info = self._model.transcribe(
                wav_buffer,
                beam_size=self.cfg.beam_size,
                language=self.cfg.language or "en",
                temperature=0.0,
                vad_filter=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,
                initial_prompt=(
                    "Hello, I have a question about science, math, biology, photosynthesis, "
                    "multiplication, mathematics, technology, or general topics."
                ),
            )

            # Filter out segments Whisper itself is uncertain about
            filtered_segments = [
                seg.text.strip()
                for seg in segments
                if seg.no_speech_prob < 0.6 and seg.text.strip()
            ]
            text = " ".join(filtered_segments).strip()

            # Reject common Whisper background noise hallucinations (e.g. 'You', 'Thank you.')
            clean_text = text.lower().strip()
            if clean_text in WHISPER_HALLUCINATIONS:
                logger.info("STT: Discarding common Whisper noise hallucination: '%s'", text)
                return ""

            # Reject repetition-loop hallucinations ('I'm a little bit of a little bit of...')
            if _is_repetitive(text):
                logger.warning("STT: Discarding repetitive hallucination (%d words): '%s...'",
                               len(text.split()), text[:80])
                return ""

            # Reject transcripts containing non-English/ASCII characters
            if text and not _is_allowed_script(text):
                logger.warning("STT: Discarding transcript with non-English script: '%s'", text)
                return ""

        except Exception as exc:
            logger.error("Transcription error: %s", exc)
            return ""

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info("STT ▸ '%s'  [%.0f ms]", text, elapsed_ms)
        return text

    def transcribe_wake_word(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """
        Specialised transcription optimised for wake-word detection.

        Differences from transcribe():
        - initial_prompt lists the wake word and common Whisper mishearings,
          biasing the token probabilities toward the expected phrase.
        - beam_size=1 for minimum latency (~50% faster).
        - More permissive no_speech_threshold so quiet utterances aren’t
          silently dropped.
        - Returns lowercase text (caller does substring matching).

        Parameters
        ----------
        audio : np.ndarray
            Mono float32 audio at `sample_rate` Hz.
        sample_rate : int
            Audio sample rate.

        Returns
        -------
        str
            Lowercase transcribed text, or empty string.
        """
        if audio is None or len(audio) == 0:
            return ""

        wav_buffer = _numpy_to_wav_bytes(audio, sample_rate)

        t0 = time.monotonic()
        try:
            segments, _info = self._model.transcribe(
                wav_buffer,
                # Speed over accuracy — wake word scanning runs in a tight loop.
                beam_size=1,
                language="en",
                temperature=0.0,
                # Prime Whisper with the expected wake word and its common mishearings
                initial_prompt="Hey Nyra. Hey Nira. Nyra. Hey Aura.",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,
            )
            # Accept segments with confidence and non-empty text
            parts = [
                seg.text.strip()
                for seg in segments
                if seg.no_speech_prob < 0.5 and seg.text.strip()
            ]
            text = " ".join(parts).strip().lower()
        except Exception as exc:
            logger.error("Wake-word transcription error: %s", exc)
            return ""

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("Wake-word STT ▸ '%s'  [%.0f ms]", text, elapsed_ms)
        return text


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    """
    Convert a float32 numpy array to an in-memory WAV BytesIO object.

    faster-whisper can accept any file-like object that `soundfile` can read.
    """
    # Clamp and convert to int16 PCM
    audio_clamped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clamped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)          # mono
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)
    return buf
