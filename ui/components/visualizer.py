"""
ui/components/visualizer.py — AI Character Avatar Stage & Audio Visualizer.

Renders an interactive animated AI assistant character avatar with:
  • Circular glass frame with dynamic state glow (Idle, Listening, Processing, Speaking)
  • RMS Audio-reactive orbital spectrum rings & lip-sync talking wave halo
  • Rotating cyber neural ring in PROCESSING state
  • Multi-layered 60 FPS sine-wave audio visualizer curves
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from config import CONFIG  # type: ignore # type import check only
from ui.theme_manager import ThemeManager


class WaveformVisualizer(QWidget):
    """
    Interactive AI Character Stage and Audio Visualizer.

    Slots
    -----
    set_level(rms: float)
        Update target audio RMS amplitude (0.0 - 1.0).
    set_state(state: str)
        Update assistant state ("IDLE", "LISTENING", "PROCESSING", "SPEAKING", "LOADING").
    set_accent_color(hex_color: str)
        Update primary accent color.
    """

    _LAYERS = [
        (0.6, 0.0, 0.90, 2.5),
        (0.9, math.pi / 3, 0.55, 1.5),
        (1.4, math.pi, 0.30, 1.0),
    ]

    _IDLE_AMPLITUDE = 0.04
    _SMOOTH_FACTOR = 0.14

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(210)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        self._target_level: float = 0.0
        self._smooth_level: float = 0.0
        self._state: str = "IDLE"

        # Phase accumulators
        self._phases: list[float] = [0.0] * len(self._LAYERS)
        self._idle_phase: float = 0.0
        self._cyber_angle: float = 0.0  # rotating cyber ring angle (degrees)
        self._pulse_phase: float = 0.0  # speaking/listening pulse

        # Accent colors
        self._accent_color = QColor("#6366F1")
        self._cyan_color = QColor("#06B6D4")
        self._emerald_color = QColor("#10B981")
        self._amber_color = QColor("#F59E0B")
        self._blue_color = QColor("#3B82F6")

        # Load & prepare circular avatar pixmap
        self._avatar_pixmap: QPixmap | None = None
        self._load_avatar_asset()

        # Background colour — updated by theme
        self._bg_color = QColor("#090A0F")
        self._tm = ThemeManager.instance()
        self._tm.theme_changed.connect(self._on_theme_changed)
        self._on_theme_changed(self._tm.theme)

        # 60 FPS animation timer
        self._timer = QTimer(self)
        self._timer.setInterval(1000 // 60)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self._last_tick = time.monotonic()

    # ── Asset loading ─────────────────────────────────────────────────

    def _load_avatar_asset(self) -> None:
        """Load avatar PNG from assets/avatar.png."""
        avatar_file = Path(__file__).parent.parent.parent / "assets" / "avatar.png"
        if avatar_file.exists():
            pix = QPixmap(str(avatar_file))
            if not pix.isNull():
                self._avatar_pixmap = pix

    # ── Public API / Slots ────────────────────────────────────────────

    def set_level(self, rms: float) -> None:
        """Slot: receive audio level (mic or TTS amplitude)."""
        self._target_level = max(0.0, min(1.0, rms))

    def set_state(self, state: str) -> None:
        """Slot: receive assistant state transition."""
        self._state = state.upper()

    def set_accent_color(self, hex_color: str) -> None:
        """Change visual accent color."""
        self._accent_color = QColor(hex_color)

    def _on_theme_changed(self, theme: str) -> None:
        """Update bg and cyan colours when theme switches."""
        if theme == "dark":
            self._bg_color = QColor("#090A0F")
            self._cyan_color = QColor("#06B6D4")
        else:
            self._bg_color = QColor("#EFF6FF")
            self._cyan_color = QColor("#0891B2")

    # ── Animation Loop ────────────────────────────────────────────────

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now

        # Smooth RMS level lerp
        self._smooth_level += (self._target_level - self._smooth_level) * self._SMOOTH_FACTOR

        # Advance wave phases
        for i, (freq, _, _, _) in enumerate(self._LAYERS):
            self._phases[i] += 2 * math.pi * freq * dt

        # Idle & cyber angles
        self._idle_phase += 2 * math.pi * 0.35 * dt
        self._cyber_angle = (self._cyber_angle + 120.0 * dt) % 360.0
        self._pulse_phase += 2 * math.pi * 1.5 * dt

        self.update()

    # ── Rendering ─────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0 - 10.0  # slightly above center for visual balance

        # 1. Background fill — theme-aware
        painter.fillRect(0, 0, w, h, self._bg_color)

        # Subtle radial glow behind avatar
        glow_color = self._get_state_color()
        radial_bg = QRadialGradient(cx, cy, max(w, h) * 0.5)
        radial_bg.setColorAt(0.0, QColor(glow_color.red(), glow_color.green(), glow_color.blue(), 35))
        radial_bg.setColorAt(0.6, QColor(glow_color.red(), glow_color.green(), glow_color.blue(), 8))
        radial_bg.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(0, 0, w, h, radial_bg)

        # 2. Draw Multi-Layer Audio Waves (bottom layer)
        self._draw_waveforms(painter, w, h, cy + 35.0)

        # 3. Draw AI Character Avatar Stage
        avatar_r = 52.0  # radius of avatar circle (104px diameter)
        self._draw_avatar_stage(painter, cx, cy, avatar_r, glow_color)

        painter.end()

    def _get_state_color(self) -> QColor:
        """Return the primary accent color based on state."""
        if self._state == "LISTENING":
            return self._emerald_color
        elif self._state == "PROCESSING":
            return self._amber_color
        elif self._state == "SPEAKING":
            return self._blue_color
        return self._accent_color

    def _draw_avatar_stage(
        self, painter: QPainter, cx: float, cy: float, r: float, state_color: QColor
    ) -> None:
        """Draw circular AI avatar stage with state reactive rings and particle aura."""

        # ── State Reactive Outer Aura & Energy Rings ───────────────────

        eff_level = self._smooth_level
        if self._state == "LISTENING":
            # Dynamic sound-reactive expansion ring
            ring_r = r + 10.0 + (eff_level * 24.0) + math.sin(self._pulse_phase) * 2.0
            pen = QPen(QColor(16, 185, 129, int(160 + eff_level * 95)), 2.5)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), ring_r, ring_r)

        elif self._state == "SPEAKING":
            # Multi-layer electric speech wave halos emanating from avatar
            for ring_idx in range(3):
                phase_off = ring_idx * 1.8
                pulse = (math.sin(self._pulse_phase + phase_off) + 1.0) * 0.5
                wave_r = r + 8.0 + (ring_idx * 10.0) + (pulse * 8.0) + (eff_level * 18.0)
                alpha = int((1.0 - (ring_idx / 3.0)) * 140 * pulse)
                painter.setPen(QPen(QColor(59, 130, 246, alpha), 2.0))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(cx, cy), wave_r, wave_r)

        elif self._state == "PROCESSING":
            # Rotating Cyber Neural Ring
            painter.save()
            painter.translate(cx, cy)
            painter.rotate(self._cyber_angle)

            cyber_r = r + 12.0
            cyber_pen = QPen(QColor(245, 158, 11, 200), 2.0, Qt.DashLine)
            cyber_pen.setDashPattern([12, 8, 4, 8])
            painter.setPen(cyber_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(0, 0), cyber_r, cyber_r)

            painter.restore()

        else:  # IDLE
            # Breathing ambient pulse ring
            idle_breath = 0.5 + 0.5 * math.sin(self._idle_phase)
            breath_r = r + 6.0 + (idle_breath * 4.0)
            painter.setPen(QPen(QColor(99, 102, 241, int(60 + idle_breath * 60)), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), breath_r, breath_r)

        # ── Avatar Inner Frame & Image ────────────────────────────────

        # Outer Glass Rim
        rim_rect = QRectF(cx - r, cy - r, r * 2.0, r * 2.0)
        glass_grad = QLinearGradient(cx - r, cy - r, cx + r, cy + r)
        glass_grad.setColorAt(0.0, QColor(state_color.red(), state_color.green(), state_color.blue(), 180))
        glass_grad.setColorAt(1.0, QColor(15, 23, 42, 220))

        painter.setPen(QPen(state_color, 2.0))
        painter.setBrush(QBrush(glass_grad))
        painter.drawEllipse(rim_rect)

        # Draw Avatar Pixmap inside clipped circle
        if self._avatar_pixmap and not self._avatar_pixmap.isNull():
            painter.save()

            clip_path = QPainterPath()
            clip_path.addEllipse(rim_rect.adjusted(3, 3, -3, -3))
            painter.setClipPath(clip_path)

            # Draw pixmap centered & scaled
            scaled_pix = self._avatar_pixmap.scaled(
                int(r * 2.0),
                int(r * 2.0),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            painter.drawPixmap(
                int(cx - r), int(cy - r), scaled_pix
            )

            # Talking lip-sync overlay highlight when SPEAKING
            if self._state == "SPEAKING" and eff_level > 0.05:
                talk_alpha = int(min(180, eff_level * 250))
                talk_grad = QRadialGradient(cx, cy + (r * 0.3), r * 0.6)
                talk_grad.setColorAt(0.0, QColor(59, 130, 246, talk_alpha))
                talk_grad.setColorAt(1.0, QColor(59, 130, 246, 0))
                painter.fillRect(rim_rect, talk_grad)

            painter.restore()
        else:
            # Fallback graphic: AI Core Icon
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(state_color.red(), state_color.green(), state_color.blue(), 100))
            painter.drawEllipse(QPointF(cx, cy), r * 0.4, r * 0.4)

    def _draw_waveforms(self, painter: QPainter, w: float, h: float, cx_y: float) -> None:
        """Draw subtle multi-layer sine waves at the bottom of visualizer stage."""

        idle_breath = self._IDLE_AMPLITUDE * (0.5 + 0.5 * math.sin(self._idle_phase))
        effective = max(self._smooth_level * 0.45 + idle_breath, idle_breath)
        amplitude_px = effective * 35.0

        steps = max(int(w), 2)
        wave_color = self._get_state_color()

        for i, (freq, phase_off, opacity, lw) in enumerate(self._LAYERS):
            path = QPainterPath()
            phase = self._phases[i] + phase_off

            for x in range(steps + 1):
                t = x / steps
                y = cx_y + amplitude_px * math.sin(phase + t * 2 * math.pi * (1.2 + i * 0.3))
                if x == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)

            r, g, b = wave_color.red(), wave_color.green(), wave_color.blue()
            alpha = int(opacity * 200)
            pen = QPen(QColor(r, g, b, alpha), lw, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
