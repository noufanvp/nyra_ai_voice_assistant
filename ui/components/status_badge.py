"""
ui/components/status_badge.py — Animated state indicator pill widget.

StatusBadge displays the current assistant state with a colour-coded pill
that smoothly crossfades between states using QPropertyAnimation.

States
------
  "IDLE"       → #6B7280  (cool gray)
  "LISTENING"  → #10B981  (emerald green) + blinking dot
  "PROCESSING" → #F59E0B  (amber)
  "SPEAKING"   → #3B82F6  (electric blue)
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    QTimer,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import QWidget


# State → (hex_color, label)
_STATE_MAP: dict[str, tuple[str, str]] = {
    "IDLE":       ("#64748B", "Idle"),
    "SLEEPING":   ("#475569", "Asleep"),
    "LISTENING":  ("#10B981", "Listening"),
    "PROCESSING": ("#F59E0B", "Thinking…"),
    "SPEAKING":   ("#3B82F6", "Speaking"),
    "LOADING":    ("#A855F7", "Loading…"),
}
_DEFAULT_STATE = "IDLE"


class StatusBadge(QWidget):
    """
    Pill-shaped state badge with animated colour transitions.

    Slot
    ----
    set_state(state: str)
        Transition to a new state. Accepts "IDLE", "LISTENING",
        "PROCESSING", or "SPEAKING".
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setMinimumWidth(130)

        self._state = _DEFAULT_STATE
        hex_color, _ = _STATE_MAP[_DEFAULT_STATE]
        self._color = QColor(hex_color)
        self._target_color = QColor(hex_color)

        # Animated interpolation colour (used by Property animation)
        self._anim_color = QColor(hex_color)

        # Blinking dot for LISTENING
        self._blink_phase: float = 0.0
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(50)  # 20 Hz for blink
        self._blink_timer.timeout.connect(self._blink_tick)
        self._blink_timer.start()

        # QPropertyAnimation driving _anim_color
        self._color_anim = QPropertyAnimation(self, b"badge_color", self)
        self._color_anim.setDuration(300)
        self._color_anim.setEasingCurve(QEasingCurve.OutCubic)

    # ── Qt Property for animation ─────────────────────────────────────

    def _get_badge_color(self) -> QColor:
        return self._anim_color

    def _set_badge_color(self, color: QColor) -> None:
        self._anim_color = color
        self.update()

    badge_color = Property(QColor, _get_badge_color, _set_badge_color)

    # ── Public API ────────────────────────────────────────────────────

    def set_state(self, state: str) -> None:
        """Slot: smoothly transition to the new state."""
        state = state.upper()
        if state not in _STATE_MAP:
            return
        if state == self._state:
            return

        self._state = state
        hex_color, _ = _STATE_MAP[state]
        new_color = QColor(hex_color)

        self._color_anim.stop()
        self._color_anim.setStartValue(self._anim_color)
        self._color_anim.setEndValue(new_color)
        self._color_anim.start()

    # ── Internal ──────────────────────────────────────────────────────

    def _blink_tick(self) -> None:
        if self._state == "LISTENING":
            self._blink_phase += 0.12
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        _, label = _STATE_MAP[self._state]

        # ── Pill shape ────────────────────────────────────────────────
        color = self._anim_color
        bg = QColor(color.red(), color.green(), color.blue(), 28)   # ~11% opacity fill
        border = QColor(color.red(), color.green(), color.blue(), 90)  # ~35% border

        path = QPainterPath()
        r = h / 2
        path.addRoundedRect(0, 0, w, h, r, r)

        painter.fillPath(path, bg)
        painter.setPen(border)
        painter.drawPath(path)

        # ── Blinking dot ─────────────────────────────────────────────
        dot_opacity = 1.0
        if self._state == "LISTENING":
            dot_opacity = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(self._blink_phase))

        dot_r = 5
        dot_x = 16
        dot_y = h / 2
        dot_color = QColor(color.red(), color.green(), color.blue(), int(dot_opacity * 255))
        painter.setPen(Qt.NoPen)
        painter.setBrush(dot_color)
        painter.drawEllipse(int(dot_x - dot_r), int(dot_y - dot_r), dot_r * 2, dot_r * 2)

        # ── Label text ────────────────────────────────────────────────
        text_color = QColor(color.red(), color.green(), color.blue(), 220)
        painter.setPen(text_color)
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        painter.setFont(font)

        text_rect = self.rect().adjusted(28, 0, -10, 0)
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, label.upper())

        painter.end()
