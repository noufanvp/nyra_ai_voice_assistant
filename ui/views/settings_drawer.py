"""
ui/views/settings_drawer.py — Collapsible diagnostics and device-selection panel.

SettingsDrawer slides up from below the control dock using QPropertyAnimation.
It exposes:
  • Input / Output device QComboBoxes (populated from sounddevice)
  • Live latency telemetry (STT ms, First-Token ms, TTS ms)
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    Qt,
    Signal,
)
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.theme_manager import ThemeManager

try:
    from core.audio_io import list_audio_devices
except ImportError:
    def list_audio_devices():  # type: ignore[misc]
        return []


_PANEL_HEIGHT = 280  # expanded height in pixels


class SettingsDrawer(QWidget):
    """
    Slide-up settings panel fixed to the bottom of the parent window.

    Call open() / close() or toggle() to animate it.
    """

    closed: Signal = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)

        # Keep panel outside the normal layout — position it manually
        self.setFixedHeight(_PANEL_HEIGHT)
        self.setAttribute(Qt.WA_StyledBackground, True)
        # Initial style applied in _apply_theme() below

        # ── Content layout ────────────────────────────────────────────
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 10, 20, 16)
        outer.setSpacing(12)

        # Top Header Bar with Drag Handle + Title + Close Button
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        title = QLabel("SETTINGS & DIAGNOSTICS")
        title.setStyleSheet("color:#6366F1; font-size:11px; font-weight:800; letter-spacing:1px; background:transparent;")
        header_row.addWidget(title)

        header_row.addStretch(1)

        # Drag handle pill
        self._handle = QFrame()
        self._handle.setFixedSize(36, 4)
        header_row.addWidget(self._handle)

        header_row.addStretch(1)

        # Close button
        self._btn_close = QPushButton("✕")
        self._btn_close.setFixedSize(26, 26)
        self._btn_close.setCursor(Qt.PointingHandCursor)
        self._btn_close.setToolTip("Close Settings [Esc]")
        self._btn_close.clicked.connect(self.close)
        header_row.addWidget(self._btn_close)

        outer.addLayout(header_row)

        # ESC shortcut to close drawer
        self._shortcut_esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._shortcut_esc.activated.connect(self.close)

        # ── Section: Devices ──────────────────────────────────────────
        dev_label = QLabel("AUDIO DEVICES")
        dev_label.setObjectName("lbl_section")
        outer.addWidget(dev_label)

        device_grid = QGridLayout()
        device_grid.setHorizontalSpacing(12)
        device_grid.setVerticalSpacing(8)

        device_grid.addWidget(QLabel("Input"), 0, 0, Qt.AlignVCenter)
        self._combo_input = QComboBox()
        self._combo_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        device_grid.addWidget(self._combo_input, 0, 1)

        device_grid.addWidget(QLabel("Output"), 1, 0, Qt.AlignVCenter)
        self._combo_output = QComboBox()
        self._combo_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        device_grid.addWidget(self._combo_output, 1, 1)

        outer.addLayout(device_grid)
        self._populate_devices()

        # ── Separator ─────────────────────────────────────────────────
        self._sep = QFrame()
        self._sep.setObjectName("separator")
        self._sep.setFixedHeight(1)
        outer.addWidget(self._sep)

        # ── Section: Telemetry ────────────────────────────────────────
        tele_label = QLabel("LATENCY TELEMETRY")
        tele_label.setObjectName("lbl_section")
        outer.addWidget(tele_label)

        tele_grid = QGridLayout()
        tele_grid.setHorizontalSpacing(12)
        tele_grid.setVerticalSpacing(6)

        metrics = [
            ("STT",        "stt_ms",         "ms"),
            ("First Token","first_token_ms",  "ms"),
            ("TTS",        "tts_ms",          "ms"),
        ]
        self._tele_labels: dict[str, QLabel] = {}
        # store refs for theming
        self._tele_name_labels: list[QLabel] = []
        self._tele_unit_labels: list[QLabel] = []

        for row, (display, key, unit) in enumerate(metrics):
            lbl_name = QLabel(display)
            self._tele_name_labels.append(lbl_name)
            tele_grid.addWidget(lbl_name, row, 0, Qt.AlignVCenter)

            lbl_val = QLabel("—")
            lbl_val.setObjectName("lbl_telemetry_value")
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tele_grid.addWidget(lbl_val, row, 1)

            lbl_unit = QLabel(unit)
            self._tele_unit_labels.append(lbl_unit)
            tele_grid.addWidget(lbl_unit, row, 2, Qt.AlignVCenter)

            self._tele_labels[key] = lbl_val

        outer.addLayout(tele_grid)
        outer.addStretch(1)

        # ── Theme support ─────────────────────────────────────────────
        self._tm = ThemeManager.instance()
        self._tm.theme_changed.connect(self._apply_theme)
        self._apply_theme(self._tm.theme)

        # ── Animation ─────────────────────────────────────────────────
        self._anim = QPropertyAnimation(self, b"geometry", self)
        self._anim.setDuration(280)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.finished.connect(self._on_anim_finished)

        self._is_open = False

    # ── Public API ────────────────────────────────────────────────────

    def open(self) -> None:
        if self._is_open:
            return
        self._is_open = True
        p = self.parent()
        pw = p.width() if p else 600
        ph = p.height() if p else 700
        self.show()
        self.raise_()
        self._anim.stop()
        self._anim.setStartValue(QRect(0, ph, pw, _PANEL_HEIGHT))
        self._anim.setEndValue(QRect(0, ph - _PANEL_HEIGHT, pw, _PANEL_HEIGHT))
        self._anim.start()

    def close(self) -> None:  # type: ignore[override]
        if not self._is_open:
            return
        self._is_open = False
        p = self.parent()
        pw = p.width() if p else 600
        ph = p.height() if p else 700
        self._anim.stop()
        self._anim.setStartValue(self.geometry())
        self._anim.setEndValue(QRect(0, ph, pw, _PANEL_HEIGHT))
        self._anim.start()

    def toggle(self) -> None:
        if self._is_open:
            self.close()
        else:
            self.open()

    def update_telemetry(self, data: dict) -> None:
        """Slot: receive telemetry_updated(dict) signal."""
        for key, lbl in self._tele_labels.items():
            val = data.get(key)
            if val is not None:
                lbl.setText(f"{val:.0f}")
            else:
                lbl.setText("—")

    def selected_input_index(self) -> int | None:
        """Return the sounddevice index of the selected input device."""
        idx = self._combo_input.currentData()
        return idx

    def selected_output_index(self) -> int | None:
        """Return the sounddevice index of the selected output device."""
        idx = self._combo_output.currentData()
        return idx

    # ── Internal ──────────────────────────────────────────────────────

    def _apply_theme(self, theme: str) -> None:
        """Re-skin the drawer for dark or light mode."""
        is_dark = (theme == "dark")
        if is_dark:
            panel_bg   = "#1B1E2B"
            panel_bdr  = "#252836"
            sep_color  = "#252836"
            name_color = "#94A3B8"
            unit_color = "#6B7280"
            handle_bg  = "#374151"
            close_bg   = "rgba(255,255,255,0.06)"
            close_fg   = "#94A3B8"
            close_bdr  = "rgba(255,255,255,0.1)"
        else:
            panel_bg   = "#F8FAFC"
            panel_bdr  = "#E2E8F0"
            sep_color  = "#E2E8F0"
            name_color = "#64748B"
            unit_color = "#94A3B8"
            handle_bg  = "#CBD5E1"
            close_bg   = "rgba(0,0,0,0.04)"
            close_fg   = "#64748B"
            close_bdr  = "rgba(148,163,184,0.4)"

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {panel_bg};
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
                border-top: 1px solid {panel_bdr};
            }}
        """)

        if hasattr(self, "_sep"):
            self._sep.setStyleSheet(f"background:{sep_color};")

        if hasattr(self, "_handle"):
            self._handle.setStyleSheet(f"background:{handle_bg}; border-radius:2px;")

        if hasattr(self, "_btn_close"):
            self._btn_close.setStyleSheet(f"""
                QPushButton {{
                    background: {close_bg};
                    color: {close_fg};
                    border: 1px solid {close_bdr};
                    border-radius: 13px;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 0px;
                }}
                QPushButton:hover {{
                    background: #EF4444;
                    color: #FFFFFF;
                    border-color: #EF4444;
                }}
            """)

        for lbl in getattr(self, "_tele_name_labels", []):
            lbl.setStyleSheet(f"color:{name_color}; font-size:12px; background:transparent;")
        for lbl in getattr(self, "_tele_unit_labels", []):
            lbl.setStyleSheet(f"color:{unit_color}; font-size:11px; background:transparent;")

    def _populate_devices(self) -> None:
        """Fill comboboxes with sounddevice input/output devices."""
        devices = list_audio_devices()
        self._combo_input.clear()
        self._combo_output.clear()
        self._combo_input.addItem("System Default", None)
        self._combo_output.addItem("System Default", None)
        for dev in devices:
            if dev["max_input_channels"] > 0:
                self._combo_input.addItem(dev["name"][:40], dev["index"])
            if dev["max_output_channels"] > 0:
                self._combo_output.addItem(dev["name"][:40], dev["index"])

    def _reposition(self) -> None:
        """Pin the drawer to the bottom of the parent widget."""
        if self.parent() is None:
            return
        p = self.parent()
        pw, ph = p.width(), p.height()
        y = ph - _PANEL_HEIGHT if self._is_open else ph
        self.setGeometry(0, y, pw, _PANEL_HEIGHT)

    def _on_anim_finished(self) -> None:
        if not self._is_open:
            self.hide()
            self.closed.emit()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reposition()
