"""
main.py — Hybrid AI Voice Assistant: PySide6 UI + 3-thread backend pipeline.

Pipeline (unchanged from headless version):
  Mic + VAD → STT → Groq LLM (streaming) → TTS → Speaker

Threading model:
  Qt Main Thread  : QApplication.exec(), UI rendering, signal dispatch
  vad_thread      : VADRecorder.record() → emits amplitude + queues audio
  stt_thread      : Whisper transcription → emits transcript
  llm_tts_thread  : Groq streaming + TTS playback → emits sentences

Thread-safety:
  All UI updates are performed via UIBridge signals.
  PySide6 queues cross-thread emissions automatically — no locks needed
  for UI callbacks.

UI-aware workers are thin wrappers: they call the same core objects but
additionally emit signals via UIBridge. The original worker logic is preserved
in the module-level functions for reference and backward compatibility.
"""

from __future__ import annotations

import logging
import queue
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

# Qt imports — must happen before any other app code
from PySide6.QtWidgets import QApplication

from config import CONFIG
from core.audio_io import VADRecorder, play_audio, stop_playback
from core.stt import WhisperTranscriber
from core.llm import GroqClientWrapper
from core.tts import LocalTTSEngine
from ui.bridge import UIBridge
from ui.main_window import MainWindow


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, CONFIG.logging.level, logging.INFO),
        format=CONFIG.logging.fmt,
        datefmt=CONFIG.logging.datefmt,
    )


# ---------------------------------------------------------------------------
# Global stop event
# ---------------------------------------------------------------------------

stop_event = threading.Event()

# speaking_lock is SET while TTS audio is playing.
# The VAD worker waits on it before listening to prevent the assistant's
# own voice from being picked up by the microphone (audio feedback loop).
speaking_lock = threading.Event()

# wake_active is SET when the wake word "Hey Nyra" has been detected.
# While it is CLEAR the assistant is sleeping — only the wake_word_worker
# listens. Once SET, the full VAD → STT → LLM → TTS pipeline activates.
# It auto-clears after CONFIG.wake_word.active_timeout_s of silence.
wake_active = threading.Event()

# mic_muted is SET when the user manually mutes the microphone from the UI.
# While SET, no audio is recorded or processed by any thread.
mic_muted = threading.Event()



def _flush_queue(q: queue.Queue) -> None:
    """Drain all pending items from a queue without blocking.

    Called when TTS starts speaking to discard audio/transcripts that arrived
    just before the speaking_lock was set (i.e. fragments captured while the
    assistant was still generating its previous response).
    """
    flushed = 0
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
            flushed += 1
        except queue.Empty:
            break
    if flushed:
        logging.getLogger("main").debug("Flushed %d stale item(s) from queue.", flushed)


def _sigint_handler(sig, frame):
    print("\n\n[SIGINT] Shutting down gracefully…")
    stop_event.set()
    stop_playback()


signal.signal(signal.SIGINT, _sigint_handler)


# ---------------------------------------------------------------------------
# Wake Word Worker (Thread 0)
# ---------------------------------------------------------------------------

# Cooldown timestamp to suppress wake word detection right after manual sleep
_manual_sleep_until: float = 0.0

# Inactivity timer tracking
_last_activity_time: float = time.monotonic()
_activity_lock = threading.Lock()


def touch_activity() -> None:
    """Record activity timestamp to prevent premature auto-sleep timeouts."""
    global _last_activity_time
    with _activity_lock:
        _last_activity_time = time.monotonic()


def _get_idle_seconds() -> float:
    global _last_activity_time
    with _activity_lock:
        return time.monotonic() - _last_activity_time


def wake_word_worker(
    recorder: VADRecorder,
    transcriber: WhisperTranscriber,
    tts: LocalTTSEngine,
    audio_queue: queue.Queue,
    transcript_queue: queue.Queue,
    stop_event: threading.Event,
    bridge: UIBridge,
) -> None:
    """
    Thread 0 (UI-aware): Listens continuously for the wake word when the assistant is sleeping.
    """
    logger = logging.getLogger("wake_word")
    cfg = CONFIG.wake_word
    was_active = False

    while not stop_event.is_set():
        if mic_muted.is_set():
            time.sleep(0.1)
            continue

        # ── Phase 1: sleeping — wait for wake word ──────────────────────
        if not wake_active.is_set():
            was_active = False
            bridge.emit_state("SLEEPING")
            audio = recorder.record(
                max_duration_s=2.5,
                abort_check=lambda: stop_event.is_set() or wake_active.is_set() or mic_muted.is_set(),
            )
            if audio is None or len(audio) == 0 or wake_active.is_set() or stop_event.is_set() or mic_muted.is_set():
                continue

            # Ignore wake scan if within manual sleep cooldown window
            if time.monotonic() < _manual_sleep_until:
                logger.debug("Wake scan ignored (manual sleep cooldown).")
                continue

            text = transcriber.transcribe_wake_word(audio, CONFIG.audio.sample_rate)
            logger.info("Wake scan heard: '%s'", text)  # INFO so you can tune variants

            # Check for any accepted phrase
            detected = any(phrase in text for phrase in cfg.phrases)
            if not detected or wake_active.is_set() or stop_event.is_set():
                continue

            logger.info("🔔 Wake word detected! Activating Nyra…")
            touch_activity()
            wake_active.set()
            bridge.emit_state("IDLE")

            # Speak greeting via TTS
            speaking_lock.set()
            bridge.emit_state("SPEAKING")
            bridge.emit_assistant_token(cfg.greeting)
            samples, sr = tts.synthesize(cfg.greeting)
            if len(samples) > 0 and wake_active.is_set() and not stop_event.is_set():
                play_audio(samples, sr, blocking=True)
            bridge.emit_assistant_done()
            speaking_lock.clear()
            if wake_active.is_set():
                bridge.emit_state("IDLE")
                touch_activity()

        # ── Phase 2: active — check for inactivity timeout ─────────────
        else:
            if not was_active:
                touch_activity()
                was_active = True

            # Refresh last_activity whenever the pipeline is busy
            queues_busy = not audio_queue.empty() or not transcript_queue.empty()
            if queues_busy or speaking_lock.is_set():
                touch_activity()

            idle_s = _get_idle_seconds()
            if idle_s >= cfg.active_timeout_s:
                logger.info(
                    "No activity for %.0fs — returning to sleep mode.", cfg.active_timeout_s
                )
                wake_active.clear()
                was_active = False
                bridge.emit_state("SLEEPING")
            else:
                time.sleep(0.5)

    logger.info("Wake-word worker stopped.")


# ---------------------------------------------------------------------------
# UI-aware worker: VAD
# ---------------------------------------------------------------------------

def vad_worker_ui(
    recorder: VADRecorder,
    audio_queue: queue.Queue,
    stop_event: threading.Event,
    speaking_lock: threading.Event,
    bridge: UIBridge,
) -> None:
    """
    Thread 1 (UI-aware): Records audio and emits amplitude + state signals.
    """
    logger = logging.getLogger("vad_worker")
    logger.info("VAD listener started.")
    initial_state = "IDLE" if (not CONFIG.wake_word.enabled or wake_active.is_set()) else "SLEEPING"
    bridge.emit_state(initial_state)

    while not stop_event.is_set():
        if mic_muted.is_set():
            bridge.emit_amplitude(0.0)
            time.sleep(0.1)
            continue

        # ── Wake-word gate ───────────────────────────────────────────────
        if CONFIG.wake_word.enabled and not wake_active.is_set():
            # The wake_word_worker owns the mic right now — just wait.
            time.sleep(0.1)
            continue

        # Block here while TTS is playing back — poll every 100ms so we
        # don't spin-lock the CPU.
        if speaking_lock.is_set():
            logger.debug("VAD paused — assistant is speaking.")
            bridge.emit_state("IDLE")
            while speaking_lock.is_set() and not stop_event.is_set():
                time.sleep(0.1)
            # Brief extra delay after playback ends so any reverb/echo in the
            # room can decay before the microphone opens again.
            time.sleep(0.6)
            logger.debug("VAD resumed — microphone open.")

        try:
            if (CONFIG.wake_word.enabled and not wake_active.is_set()) or mic_muted.is_set():
                continue
            touch_activity()
            bridge.emit_state("LISTENING")
            # abort_check will cancel recording immediately if put to sleep, stop requested, TTS starts, or mic muted
            audio = recorder.record(
                max_duration_s=7.0,
                abort_check=lambda: stop_event.is_set() or (CONFIG.wake_word.enabled and not wake_active.is_set()) or speaking_lock.is_set() or mic_muted.is_set(),
            )

            # Belt-and-suspenders: if put to sleep, muted, or TTS started while record() was winding down, discard audio
            if (CONFIG.wake_word.enabled and not wake_active.is_set()) or speaking_lock.is_set() or stop_event.is_set() or mic_muted.is_set():
                logger.debug("Discarding audio captured while sleeping, muted, or TTS playback.")
                time.sleep(0.1)
                continue

            if audio is not None and len(audio) > 0:
                touch_activity()
                audio_queue.put(audio)
                bridge.emit_state("PROCESSING")
                logger.debug("Audio segment enqueued (%d samples).", len(audio))
            else:
                time.sleep(0.05)
        except Exception as exc:
            logger.error("VAD error: %s", exc)
            bridge.emit_state("IDLE" if wake_active.is_set() else "SLEEPING")
            time.sleep(0.5)

    logger.info("VAD listener stopped.")


# ---------------------------------------------------------------------------
# UI-aware worker: STT
# ---------------------------------------------------------------------------

def stt_worker_ui(
    transcriber: WhisperTranscriber,
    audio_queue: queue.Queue,
    transcript_queue: queue.Queue,
    stop_event: threading.Event,
    bridge: UIBridge,
) -> None:
    """
    Thread 2 (UI-aware): Transcribes audio; emits transcript + PROCESSING state.
    """
    logger = logging.getLogger("stt_worker")
    logger.info("STT worker started.")

    while not stop_event.is_set():
        try:
            audio = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if CONFIG.wake_word.enabled and not wake_active.is_set():
            audio_queue.task_done()
            continue

        touch_activity()
        bridge.emit_state("PROCESSING")
        t0 = time.monotonic()
        text = transcriber.transcribe(audio, CONFIG.audio.sample_rate)
        stt_ms = (time.monotonic() - t0) * 1000
        logger.info("[STT latency: %.0f ms] Transcript: '%s'", stt_ms, text)

        # Check wake status again after transcription completes
        if CONFIG.wake_word.enabled and not wake_active.is_set():
            logger.info("STT finished but assistant is sleeping — discarding transcript.")
            bridge.emit_state("SLEEPING")
            audio_queue.task_done()
            continue

        if text.strip():
            touch_activity()
            bridge.emit_user_text(text.strip())
            transcript_queue.put((text.strip(), stt_ms))
        else:
            logger.debug("Empty transcript — discarding.")
            bridge.emit_state("IDLE" if wake_active.is_set() else "SLEEPING")

        audio_queue.task_done()

    logger.info("STT worker stopped.")


# ---------------------------------------------------------------------------
# UI-aware worker: LLM + TTS
# ---------------------------------------------------------------------------

def llm_tts_worker_ui(
    llm: GroqClientWrapper,
    tts: LocalTTSEngine,
    audio_queue: queue.Queue,
    transcript_queue: queue.Queue,
    stop_event: threading.Event,
    bridge: UIBridge,
) -> None:
    """
    Thread 3 (UI-aware): Streams LLM sentences, plays TTS, emits tokens + timings.
    """
    logger = logging.getLogger("llm_tts_worker")
    logger.info("LLM+TTS worker started.")
    conversation_history: list[dict] = []

    while not stop_event.is_set():
        try:
            item = transcript_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if CONFIG.wake_word.enabled and not wake_active.is_set():
            transcript_queue.task_done()
            continue

        # Unpack: stt_ms was added by stt_worker_ui
        if isinstance(item, tuple):
            text, stt_ms = item
        else:
            text, stt_ms = item, 0.0

        logger.info("Processing: '%s'", text)
        touch_activity()
        bridge.emit_state("SPEAKING")

        full_response_parts: list[str] = []
        tts_queue: queue.Queue = queue.Queue()
        playback_done = threading.Event()
        first_token_ms: float = 0.0
        last_tts_ms: float = 0.0
        t_request = time.monotonic()

        def tts_playback_thread():
            """Sub-thread: synthesizes and plays sentences."""
            nonlocal last_tts_ms
            while True:
                tts_item = tts_queue.get()
                if tts_item is None:
                    break
                sentence, t_received = tts_item
                if CONFIG.wake_word.enabled and not wake_active.is_set():
                    tts_queue.task_done()
                    continue

                # Lock BEFORE synthesis so the mic closes the instant we start
                # generating audio — prevents the 4-6s synthesis window from
                # letting speaker audio leak back into the microphone.
                speaking_lock.set()
                # Flush stale audio/transcripts captured before the lock.
                _flush_queue(audio_queue)
                _flush_queue(transcript_queue)

                t0 = time.monotonic()
                samples, sr = tts.synthesize(sentence)
                last_tts_ms = (time.monotonic() - t0) * 1000
                logger.info("[TTS: %.0f ms] Speaking: '%s'", last_tts_ms, sentence)
                if len(samples) > 0 and not stop_event.is_set() and (not CONFIG.wake_word.enabled or wake_active.is_set()):
                    play_audio(samples, sr, blocking=True)
                tts_queue.task_done()
            speaking_lock.clear()            # ← UNLOCK: microphone resumes
            playback_done.set()

        pb_thread = threading.Thread(target=tts_playback_thread, daemon=True)
        pb_thread.start()

        try:
            first_sentence = True
            for sentence in llm.stream_sentences(text, conversation_history):
                if stop_event.is_set() or (CONFIG.wake_word.enabled and not wake_active.is_set()):
                    break

                touch_activity()
                if first_sentence:
                    first_token_ms = (time.monotonic() - t_request) * 1000
                    logger.info("Groq first-sentence latency: %.0f ms", first_token_ms)
                    first_sentence = False

                full_response_parts.append(sentence)
                bridge.emit_assistant_token(sentence)
                tts_queue.put((sentence, time.monotonic()))

        except Exception as exc:
            logger.error("LLM streaming error: %s", exc)
            fallback = "I'm sorry, something went wrong."
            bridge.emit_assistant_token(fallback)
            tts_queue.put((fallback, time.monotonic()))

        tts_queue.put(None)   # sentinel: stop playback thread
        pb_thread.join(timeout=30)

        bridge.emit_assistant_done()
        bridge.emit_state("IDLE" if wake_active.is_set() else "SLEEPING")
        bridge.emit_amplitude(0.0)

        # Emit telemetry
        bridge.emit_telemetry({
            "stt_ms":         stt_ms,
            "first_token_ms": first_token_ms,
            "tts_ms":         last_tts_ms,
        })

        # Update conversation history (bounded to last 10 turns)
        if full_response_parts:
            conversation_history.append({"role": "user", "content": text})
            conversation_history.append({
                "role": "assistant",
                "content": " ".join(full_response_parts),
            })
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

        transcript_queue.task_done()

    logger.info("LLM+TTS worker stopped.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()
    logger = logging.getLogger("main")

    # ── Qt Application (must be first) ────────────────────────────────
    app = QApplication(sys.argv)
    app.setApplicationName("Nyra")
    app.setOrganizationName("ROTECH")

    # ── Signal bridge ─────────────────────────────────────────────────
    bridge = UIBridge()

    # ── Show window immediately so the UI is visible while loading ────
    window = MainWindow(bridge=bridge, config=CONFIG.ui)
    window.show()
    bridge.emit_state("LOADING")
    app.processEvents()   # force the window to paint before blocking work starts

    # ── Manual wake toggle from the UI button ─────────────────────────
    def _on_manual_wake_toggle(active: bool) -> None:
        """Called on the Qt main thread when the user presses Wake Up / Sleep."""
        logger = logging.getLogger("main")
        if active:
            touch_activity()
            wake_active.set()
            _flush_queue(audio_queue)
            _flush_queue(transcript_queue)
            bridge.emit_state("IDLE")
            logger.info("Manual wake: Nyra activated.")
            window._dock.set_wake_active(True)
        else:
            global _manual_sleep_until
            wake_active.clear()
            stop_playback()
            speaking_lock.clear()
            _flush_queue(audio_queue)
            _flush_queue(transcript_queue)
            _manual_sleep_until = time.monotonic() + 2.0
            bridge.emit_state("SLEEPING")
            logger.info("Manual wake: Nyra put to sleep.")
            window._dock.set_wake_active(False)

    bridge.wake_toggled.connect(_on_manual_wake_toggle)

    # ── Manual mute toggle from the UI button ─────────────────────────
    def _on_manual_mute_toggle(muted: bool) -> None:
        """Called on the Qt main thread when the user presses Mute / Unmute."""
        logger = logging.getLogger("main")
        if muted:
            mic_muted.set()
            stop_playback()
            speaking_lock.clear()
            _flush_queue(audio_queue)
            _flush_queue(transcript_queue)
            bridge.emit_amplitude(0.0)
            logger.info("Manual mute: Microphone MUTED.")
        else:
            mic_muted.clear()
            logger.info("Manual mute: Microphone UNMUTED.")

    bridge.mute_toggled.connect(_on_manual_mute_toggle)

    # ── Inter-thread queues ───────────────────────────────────────────
    audio_queue: queue.Queue      = queue.Queue(maxsize=4)
    transcript_queue: queue.Queue = queue.Queue(maxsize=4)

    # ── Initialize backend + start workers in a background thread ────
    # This prevents heavy model loading (Whisper, TTS) from blocking
    # the Qt event loop so the window stays responsive during startup.
    def _init_and_start_workers():
        nonlocal threads_holder

        logger.info("Background init: loading backend components…")

        # ── Stage 1: Microphone / VAD ─────────────────────────────────
        bridge.emit_loading_progress("Initialising microphone…", 5)
        try:
            recorder = VADRecorder(
                CONFIG.audio,
                amplitude_callback=bridge.emit_amplitude,
                abort_event=speaking_lock,   # stops recording the instant TTS plays
            )
        except Exception as exc:
            logger.error("Failed to initialize VAD recorder: %s", exc)
            bridge.emit_loading_progress(f"Microphone error: {exc}", 0)
            stop_event.set()
            return

        if stop_event.is_set():
            return

        # ── Stage 2: Whisper STT (may download on first run) ──────────
        # The progress callback forwards STT-internal messages to the UI
        bridge.emit_loading_progress("Preparing speech recognition…", 15)
        try:
            transcriber = WhisperTranscriber(
                CONFIG.stt,
                progress_callback=bridge.emit_loading_progress,
            )
        except Exception as exc:
            logger.error("Failed to initialize STT: %s", exc)
            bridge.emit_loading_progress(f"STT error: {exc}", 0)
            stop_event.set()
            return

        if stop_event.is_set():
            return

        # ── Stage 3: Groq LLM client ──────────────────────────────────
        bridge.emit_loading_progress("Connecting to Groq LLM…", 85)
        llm = GroqClientWrapper(CONFIG.llm)

        # ── Stage 4: TTS engine ───────────────────────────────────────
        bridge.emit_loading_progress("Starting text-to-speech engine…", 92)
        tts = LocalTTSEngine(CONFIG.tts)

        if stop_event.is_set():
            return

        bridge.emit_loading_progress("All systems ready!", 100)
        logger.info("All components ready. TTS backend: %s", tts.backend_name)
        time.sleep(0.6)  # let user see "All systems ready!" briefly

        if stop_event.is_set():
            return

        # ── Worker threads ────────────────────────────────────────────
        threads: list[threading.Thread] = []

        if CONFIG.wake_word.enabled:
            threads.append(threading.Thread(
                target=wake_word_worker,
                args=(recorder, transcriber, tts, audio_queue, transcript_queue, stop_event, bridge),
                name="wake_word_worker",
                daemon=True,
            ))
            logger.info(
                "Wake-word mode ENABLED — say '%s' to activate.",
                CONFIG.wake_word.phrases[0],
            )
        else:
            # No wake word: activate immediately
            wake_active.set()
            logger.info("Wake-word mode DISABLED — always-on.")

        threads += [
            threading.Thread(
                target=vad_worker_ui,
                args=(recorder, audio_queue, stop_event, speaking_lock, bridge),
                name="vad_worker",
                daemon=True,
            ),
            threading.Thread(
                target=stt_worker_ui,
                args=(transcriber, audio_queue, transcript_queue, stop_event, bridge),
                name="stt_worker",
                daemon=True,
            ),
            threading.Thread(
                target=llm_tts_worker_ui,
                args=(llm, tts, audio_queue, transcript_queue, stop_event, bridge),
                name="llm_tts_worker",
                daemon=True,
            ),
        ]

        for t in threads:
            t.start()
            logger.debug("Thread '%s' started.", t.name)

        threads_holder.extend(threads)
        logger.info("Nyra is ready.")


    threads_holder: list[threading.Thread] = []
    init_thread = threading.Thread(
        target=_init_and_start_workers,
        name="init_worker",
        daemon=True,
    )
    threads_holder.append(init_thread)
    init_thread.start()

    # ── Qt event loop — blocks until window closes ────────────────────
    def _on_quit():
        logger.info("Application closing — setting stop_event.")
        stop_event.set()
        stop_playback()

    app.aboutToQuit.connect(_on_quit)

    exit_code = app.exec()

    # ── Graceful thread shutdown ──────────────────────────────────────
    logger.info("Waiting for worker threads to finish…")
    for t in threads_holder:
        t.join(timeout=5.0)
        if t.is_alive():
            logger.warning("Thread '%s' did not stop cleanly.", t.name)

    logger.info("Goodbye.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
