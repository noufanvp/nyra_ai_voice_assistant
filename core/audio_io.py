"""
core/audio_io.py — Microphone capture, WebRTC VAD, and audio playback.

Key components:
  - list_audio_devices()  : enumerate sounddevice I/O devices
  - VADRecorder           : records until VAD detects sustained silence
  - play_audio()          : non-blocking playback via sounddevice

VAD backend: webrtcvad (lightweight, no PyTorch dependency).
For higher accuracy, set USE_SILERO=True in config and install torch+silero-vad.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------

def list_audio_devices() -> list[dict]:
    """
    Return a list of available audio devices with their properties.

    Returns
    -------
    list[dict]
        Each dict has keys: index, name, max_input_channels, max_output_channels,
        default_samplerate.
    """
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        result.append({
            "index": i,
            "name": dev["name"],
            "max_input_channels": dev["max_input_channels"],
            "max_output_channels": dev["max_output_channels"],
            "default_samplerate": dev["default_samplerate"],
        })
    return result


def get_default_input_device() -> Optional[dict]:
    """Return the default input device info, or None if unavailable."""
    try:
        idx = sd.default.device[0]
        if idx is None or idx < 0:
            return None
        devices = list_audio_devices()
        return next((d for d in devices if d["index"] == idx), None)
    except Exception as exc:
        logger.warning("Could not query default input device: %s", exc)
        return None


# ---------------------------------------------------------------------------
# WebRTC VAD wrapper
# ---------------------------------------------------------------------------

class WebRTCVAD:
    """
    Thin wrapper around the webrtcvad library.
    webrtcvad operates on 10ms, 20ms, or 30ms frames of 16kHz int16 PCM.
    """

    # webrtcvad requires frames of exactly 10, 20, or 30ms
    FRAME_DURATION_MS = 30  # ms per frame

    def __init__(self, sample_rate: int = 16_000, aggressiveness: int = 3):
        """
        Parameters
        ----------
        sample_rate : int
            Must be 8000, 16000, 32000, or 48000.
        aggressiveness : int
            0 (least aggressive) – 3 (most aggressive in filtering non-speech).
            Use 3 to eliminate background noise / mic hum false positives.
        """
        import webrtcvad
        self._vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * self.FRAME_DURATION_MS / 1000)
        # RMS energy gate: frames below this level are always treated as silence
        # regardless of WebRTC decision (prevents mic hum / fan noise triggering VAD).
        # Calibrated for typical laptop/desktop mic at 0.3-0.5m speaking distance.
        self.energy_threshold: float = 0.010  # roughly -40 dBFS

    def is_speech(self, audio_float32: np.ndarray) -> bool:
        """
        Determine if an audio frame contains speech.

        Uses a two-stage check:
          1. RMS energy gate — if the frame is too quiet it can't be speech.
          2. WebRTC VAD — linguistic pattern detection on frames that pass the gate.

        Parameters
        ----------
        audio_float32 : np.ndarray
            Float32 audio frame. Must be exactly frame_length samples.

        Returns
        -------
        bool
        """
        # Stage 1: energy gate — fast path for silence
        rms = float(np.sqrt(np.mean(audio_float32 ** 2)))
        if rms < self.energy_threshold:
            return False

        # Stage 2: WebRTC linguistic VAD
        audio_int16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()

        # webrtcvad is strict: frame must be exactly the right number of bytes
        expected_bytes = self.frame_length * 2  # 2 bytes per int16 sample
        if len(pcm_bytes) != expected_bytes:
            # Pad or truncate silently
            pcm_bytes = pcm_bytes[:expected_bytes].ljust(expected_bytes, b"\x00")

        try:
            return self._vad.is_speech(pcm_bytes, self.sample_rate)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# VAD Recorder
# ---------------------------------------------------------------------------

class VADRecorder:
    """
    Records microphone audio until VAD detects sustained silence.

    Usage
    -----
    recorder = VADRecorder(config.audio)
    audio_array = recorder.record()   # blocks until speech+silence detected
    """

    def __init__(
        self,
        audio_config,
        device: Optional[int] = None,
        amplitude_callback: Optional[Callable[[float], None]] = None,
        abort_event: Optional["threading.Event"] = None,
    ):
        """
        Parameters
        ----------
        audio_config : AudioConfig
            Configuration dataclass from config.py.
        device : int, optional
            sounddevice device index. None = system default.
        amplitude_callback : Callable[[float], None], optional
            Called with the RMS amplitude (0.0 – 1.0) of each incoming audio
            frame. Invoked from the sounddevice callback thread — consumers
            MUST be thread-safe (e.g. Qt queued signals).
        abort_event : threading.Event, optional
            When set, causes an in-progress record() call to abort immediately
            and return an empty array. Use this to stop listening the instant
            TTS playback begins so the speaker audio is never captured.
        """
        self.cfg = audio_config
        self.device = device
        self.amplitude_callback = amplitude_callback
        self.abort_event = abort_event
        self._vad = WebRTCVAD(
            sample_rate=self.cfg.sample_rate,
            aggressiveness=3,  # maximum noise filtering for noisy environments
        )

    def record(
        self,
        max_duration_s: Optional[float] = None,
        abort_check: Optional[Callable[[], bool]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        """
        Block until a complete speech segment (followed by silence) is captured.

        Uses a sliding-window majority vote over the last N VAD frames to
        decide speech/silence, preventing a single noisy non-speech frame
        from prematurely cutting off mid-sentence.

        Parameters
        ----------
        max_duration_s : float, optional
            Maximum recording duration in seconds before forcing recording to end.
            If None, uses self.cfg.max_recording_duration_s.
        abort_check : Callable[[], bool], optional
            Optional callback returning True to abort recording immediately (discards audio).
        stop_check : Callable[[], bool], optional
            Optional callback returning True to stop recording immediately and return recorded audio.

        Returns
        -------
        np.ndarray
            float32 mono audio array at cfg.sample_rate.
            Returns empty array if silence timeout or no speech detected.
        """
        logger.debug("🎙  Listening… (speak now)")

        frame_length = self._vad.frame_length
        frames: list[np.ndarray] = []
        speech_detected = False
        silence_start: Optional[float] = None
        speech_start: Optional[float] = None
        recording_start = time.monotonic()
        max_cap = max_duration_s if max_duration_s is not None else self.cfg.max_recording_duration_s

        # Sliding window: track speech/non-speech decisions for last N frames.
        # Majority vote prevents a single non-speech frame from cutting audio.
        VAD_WINDOW = 5        # frames to smooth over (~150ms at 30ms/frame)
        VAD_SPEECH_THRESH = 3  # min speech frames in window to count as speech (raised from 2→3)
        vad_window: list[bool] = []

        def callback(indata: np.ndarray, frame_count: int, time_info, status):
            if status:
                logger.debug("sounddevice status: %s", status)
            mono = indata[:, 0].copy()
            frames.append(mono)
            # Emit RMS amplitude to any registered UI listener (thread-safe)
            if self.amplitude_callback is not None:
                rms = float(np.sqrt(np.mean(mono ** 2)))
                # Clamp to [0, 1] — RMS of float32 audio is already ≤1 but
                # brief clipping can push it slightly above.
                self.amplitude_callback(min(rms, 1.0))

        try:
            with sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                dtype=self.cfg.dtype,
                blocksize=frame_length,
                device=self.device,
                callback=callback,
            ):
                while True:
                    # Abort immediately if TTS is now playing or abort_check requested
                    if self.abort_event is not None and self.abort_event.is_set():
                        logger.debug("record() aborted — abort_event is set.")
                        return np.array([], dtype=np.float32)
                    if abort_check is not None and abort_check():
                        logger.debug("record() aborted — abort_check condition met.")
                        return np.array([], dtype=np.float32)
                    if stop_check is not None and stop_check():
                        logger.debug("record() stopped — stop_check condition met.")
                        break

                    # Safety timeouts (only enforce silence timeouts when NOT in Push-to-Talk mode)
                    elapsed = time.monotonic() - recording_start
                    if stop_check is None and not speech_detected and elapsed >= self.cfg.initial_silence_timeout_s:
                        logger.debug(
                            "No speech detected within initial timeout (%.1fs). Returning.",
                            elapsed,
                        )
                        break

                    if elapsed >= max_cap:
                        logger.debug("Max recording duration (%.1fs) reached.", max_cap)
                        break

                    # Sleep one frame duration before polling
                    time.sleep(frame_length / self.cfg.sample_rate)

                    if not frames:
                        continue

                    # Analyse latest frame for speech
                    chunk = frames[-1]
                    # Ensure chunk is exactly frame_length samples
                    if len(chunk) < frame_length:
                        chunk = np.pad(chunk, (0, frame_length - len(chunk)))
                    elif len(chunk) > frame_length:
                        chunk = chunk[:frame_length]

                    # Update sliding window
                    frame_is_speech = self._vad.is_speech(chunk)
                    vad_window.append(frame_is_speech)
                    if len(vad_window) > VAD_WINDOW:
                        vad_window.pop(0)

                    # Majority vote: speech if enough recent frames are speech
                    speech_votes = sum(vad_window)
                    is_speech_windowed = speech_votes >= VAD_SPEECH_THRESH

                    now = time.monotonic()

                    if is_speech_windowed:
                        if not speech_detected:
                            logger.debug("Speech start detected.")
                            speech_detected = True
                            speech_start = now
                        silence_start = None  # reset silence counter
                    else:
                        if speech_detected:
                            if silence_start is None:
                                silence_start = now
                            silence_duration = now - silence_start
                            # Only stop on silence in wake-word mode, NOT in Push-to-Talk mode
                            if stop_check is None and silence_duration >= self.cfg.silence_duration_s:
                                logger.debug(
                                    "Silence for %.2fs — stopping.", silence_duration
                                )
                                break

        except sd.PortAudioError as exc:
            logger.error("PortAudio error during recording: %s", exc)
            return np.array([], dtype=np.float32)

        if not frames:
            return np.array([], dtype=np.float32)

        full_audio = np.concatenate(frames)

        # Guard: require minimum speech duration (only in wake-word mode)
        if speech_start is not None:
            speech_duration = time.monotonic() - speech_start
        else:
            speech_duration = 0.0

        if stop_check is None and (not speech_detected or speech_duration < self.cfg.min_speech_duration_s):
            logger.debug(
                "No meaningful speech detected (%.2fs). Ignoring.", speech_duration
            )
            return np.array([], dtype=np.float32)

        duration_s = len(full_audio) / self.cfg.sample_rate
        logger.info("Recording complete: %.2fs of audio captured.", duration_s)
        return full_audio


# ---------------------------------------------------------------------------
# Audio Playback
# ---------------------------------------------------------------------------

def play_audio(samples: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
    """
    Play a numpy audio array through the default output device.

    Parameters
    ----------
    samples : np.ndarray
        float32 mono or stereo audio samples.
    sample_rate : int
        Playback sample rate in Hz.
    blocking : bool
        If True, block until playback is complete.
    """
    if samples is None or len(samples) == 0:
        logger.debug("play_audio: empty samples, skipping.")
        return

    samples = samples.astype(np.float32)

    # Normalize to prevent clipping distortion
    max_val = np.max(np.abs(samples))
    if max_val > 1.0:
        samples = samples / max_val

    try:
        sd.play(samples, samplerate=sample_rate)
        if blocking:
            sd.wait()
    except sd.PortAudioError as exc:
        logger.error("Playback error: %s", exc)


def stop_playback() -> None:
    """Stop any currently playing audio immediately."""
    try:
        sd.stop()
    except Exception as exc:
        logger.warning("Could not stop playback: %s", exc)
