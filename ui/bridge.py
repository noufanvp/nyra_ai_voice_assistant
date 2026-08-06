"""
ui/bridge.py — Thread-safe Qt Signal hub connecting backend threads to the UI.

UIBridge is a QObject that lives in the main (Qt) thread.
Background threads call its emit_*() helpers; Qt automatically routes the
signal delivery to the correct thread via its queued-connection mechanism.

Usage
-----
    from ui.bridge import UIBridge
    bridge = UIBridge()

    # In background thread — safe to call from any thread:
    bridge.emit_amplitude(0.42)
    bridge.emit_state("LISTENING")
    bridge.emit_user_text("Hello, assistant.")
    bridge.emit_assistant_token("Sure, I can help!")
    bridge.emit_assistant_done()
    bridge.emit_telemetry({"stt_ms": 312, "first_token_ms": 205, "tts_ms": 88})

    # In main thread — connect to UI slots:
    bridge.amplitude_updated.connect(visualizer.set_level)
    bridge.state_changed.connect(status_badge.set_state)
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class UIBridge(QObject):
    """
    Central Qt signal hub for the voice assistant UI.

    All signals are emitted thread-safely: PySide6 queues cross-thread
    signal emissions automatically, so background threads may call
    emit_*() methods without any locking.
    """

    # ── Signal definitions ─────────────────────────────────────────────
    # Mic RMS amplitude float in [0.0, 1.0] — drives visualizer
    amplitude_updated: Signal = Signal(float)

    # Assistant state string: "IDLE" | "LISTENING" | "PROCESSING" | "SPEAKING" | "LOADING" | "SLEEPING"
    state_changed: Signal = Signal(str)

    # Loading progress during startup: (message, percent)
    # percent = -1  → indeterminate (animated pulse bar)
    # percent = 0–100 → determinate fill
    loading_progress: Signal = Signal(str, int)

    # Full transcribed user utterance
    user_transcribed: Signal = Signal(str)

    # One complete LLM sentence streamed from Groq
    assistant_token: Signal = Signal(str)

    # Emitted when LLM streaming is complete for the current turn
    assistant_done: Signal = Signal()

    # Timing telemetry: {"stt_ms": float, "first_token_ms": float, "tts_ms": float}
    telemetry_updated: Signal = Signal(dict)

    # Emitted by the UI when the user manually toggles the wake state.
    # True = activate (user pressed Wake Up), False = go to sleep.
    wake_toggled: Signal = Signal(bool)

    # Emitted by the UI when the user manually toggles microphone mute.
    # True = muted, False = unmuted.
    mute_toggled: Signal = Signal(bool)

    # ── Thread-safe emit helpers ───────────────────────────────────────

    def emit_amplitude(self, rms: float) -> None:
        """Emit mic RMS level. Safe to call from any thread."""
        self.amplitude_updated.emit(float(rms))

    def emit_state(self, state: str) -> None:
        """Emit assistant state change. Safe to call from any thread."""
        self.state_changed.emit(state)

    def emit_user_text(self, text: str) -> None:
        """Emit a completed user transcript. Safe to call from any thread."""
        self.user_transcribed.emit(text)

    def emit_assistant_token(self, sentence: str) -> None:
        """Emit one streamed LLM sentence. Safe to call from any thread."""
        self.assistant_token.emit(sentence)

    def emit_assistant_done(self) -> None:
        """Signal that the current LLM response is complete."""
        self.assistant_done.emit()

    def emit_telemetry(self, data: dict) -> None:
        """Emit latency telemetry dict. Safe to call from any thread."""
        self.telemetry_updated.emit(data)

    def emit_loading_progress(self, message: str, percent: int = -1) -> None:
        """Emit startup loading progress. percent=-1 = indeterminate bar."""
        self.loading_progress.emit(message, percent)

    def emit_wake_toggle(self, active: bool) -> None:
        """Emit manual wake toggle from the UI button. Safe to call from main thread."""
        self.wake_toggled.emit(active)

    def emit_mute_toggle(self, muted: bool) -> None:
        """Emit manual mute toggle from the UI button. Safe to call from main thread."""
        self.mute_toggled.emit(muted)
