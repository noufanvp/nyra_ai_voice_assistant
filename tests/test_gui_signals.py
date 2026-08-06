"""
tests/test_gui_signals.py — Validate UIBridge Qt signal emissions.

Tests run headless (no display required) via pytest-qt's qtbot fixture,
which creates a QApplication automatically and manages event-loop pumping.

Coverage
--------
1. All UIBridge signals fire without errors.
2. Signal payloads match what was emitted.
3. Correct state sequence: IDLE → LISTENING → PROCESSING → SPEAKING → IDLE.
4. assistant_token and assistant_done fire in the right order.
5. Telemetry dict is forwarded intact.
"""

from __future__ import annotations

import sys
import os

import pytest

# Ensure project root is on path when running from tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

@pytest.fixture()
def bridge(qapp):
    """Return a fresh UIBridge connected to the test QApplication."""
    from ui.bridge import UIBridge
    return UIBridge()


# ── Basic signal emission tests ────────────────────────────────────────────────

class TestAmplitudeSignal:
    def test_emits_float(self, bridge, qtbot):
        received = []
        bridge.amplitude_updated.connect(received.append)
        bridge.emit_amplitude(0.75)
        qtbot.wait(10)
        assert received == [pytest.approx(0.75)]

    def test_clamped_value_passthrough(self, bridge, qtbot):
        """UIBridge does not clamp — the recorder does. Verify passthrough."""
        received = []
        bridge.amplitude_updated.connect(received.append)
        bridge.emit_amplitude(0.0)
        bridge.emit_amplitude(1.0)
        qtbot.wait(10)
        assert received == [pytest.approx(0.0), pytest.approx(1.0)]


class TestStateSignal:
    def test_idle(self, bridge, qtbot):
        received = []
        bridge.state_changed.connect(received.append)
        bridge.emit_state("IDLE")
        qtbot.wait(10)
        assert received == ["IDLE"]

    def test_all_states(self, bridge, qtbot):
        states = ["IDLE", "LISTENING", "PROCESSING", "SPEAKING", "IDLE"]
        received = []
        bridge.state_changed.connect(received.append)
        for s in states:
            bridge.emit_state(s)
        qtbot.wait(20)
        assert received == states


class TestTranscriptSignals:
    def test_user_text(self, bridge, qtbot):
        received = []
        bridge.user_transcribed.connect(received.append)
        bridge.emit_user_text("Hello, assistant.")
        qtbot.wait(10)
        assert received == ["Hello, assistant."]

    def test_assistant_tokens(self, bridge, qtbot):
        sentences = ["Sure,", "I can help with that."]
        received = []
        bridge.assistant_token.connect(received.append)
        for s in sentences:
            bridge.emit_assistant_token(s)
        qtbot.wait(10)
        assert received == sentences

    def test_assistant_done_fires(self, bridge, qtbot):
        called = []
        bridge.assistant_done.connect(lambda: called.append(True))
        bridge.emit_assistant_done()
        qtbot.wait(10)
        assert called == [True]


class TestTelemetrySignal:
    def test_dict_passthrough(self, bridge, qtbot):
        payload = {"stt_ms": 312.5, "first_token_ms": 205.0, "tts_ms": 88.1}
        received = []
        bridge.telemetry_updated.connect(received.append)
        bridge.emit_telemetry(payload)
        qtbot.wait(10)
        assert len(received) == 1
        assert received[0]["stt_ms"] == pytest.approx(312.5)
        assert received[0]["first_token_ms"] == pytest.approx(205.0)
        assert received[0]["tts_ms"] == pytest.approx(88.1)


class TestMuteSignal:
    def test_mute_toggled_signal(self, bridge, qtbot):
        received = []
        bridge.mute_toggled.connect(received.append)
        bridge.emit_mute_toggle(True)
        bridge.emit_mute_toggle(False)
        qtbot.wait(10)
        assert received == [True, False]


# ── Full turn sequence test ────────────────────────────────────────────────────

class TestFullConversationTurn:
    """
    Simulate a complete user→assistant turn and verify signals fire in order.
    """

    def test_signal_sequence(self, bridge, qtbot):
        log = []
        bridge.state_changed.connect(lambda s: log.append(("state", s)))
        bridge.user_transcribed.connect(lambda t: log.append(("user", t)))
        bridge.assistant_token.connect(lambda t: log.append(("asst_tok", t)))
        bridge.assistant_done.connect(lambda: log.append(("asst_done", None)))
        bridge.telemetry_updated.connect(lambda d: log.append(("tele", d)))

        # Simulate the sequence emitted by worker threads
        bridge.emit_state("LISTENING")
        bridge.emit_amplitude(0.4)
        bridge.emit_state("PROCESSING")
        bridge.emit_user_text("What is the weather like?")
        bridge.emit_state("SPEAKING")
        bridge.emit_assistant_token("It looks sunny today.")
        bridge.emit_assistant_token("Enjoy the day!")
        bridge.emit_assistant_done()
        bridge.emit_state("IDLE")
        bridge.emit_telemetry({"stt_ms": 200, "first_token_ms": 150, "tts_ms": 90})

        qtbot.wait(30)

        states = [v for k, v in log if k == "state"]
        assert states == ["LISTENING", "PROCESSING", "SPEAKING", "IDLE"]

        users = [v for k, v in log if k == "user"]
        assert users == ["What is the weather like?"]

        tokens = [v for k, v in log if k == "asst_tok"]
        assert tokens == ["It looks sunny today.", "Enjoy the day!"]

        dones = [k for k, _ in log if k == "asst_done"]
        assert len(dones) == 1

        # done must appear AFTER last token
        done_idx  = next(i for i, (k, _) in enumerate(log) if k == "asst_done")
        last_tok_idx = max(i for i, (k, _) in enumerate(log) if k == "asst_tok")
        assert done_idx > last_tok_idx
