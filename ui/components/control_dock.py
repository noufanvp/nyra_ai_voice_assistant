"""
ui/components/control_dock.py — Floating cyber control dock widget.

Redesigned nav bar:
  • Large, highlighted vector icons on top
  • Small muted text labels below each icon
  • Active/hover states glow with accent colours
  • Pill-shaped glassmorphic container
  • Theme toggle (dark ↔ light) at the far right

Keyboard Shortcuts:
  M → toggle mic mute
  W → toggle wake / sleep
  , → toggle settings drawer
  C → clear chat history
  T → toggle dark / light theme
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QToolButton,
    QWidget,
)

from ui.theme_manager import ThemeManager

# ─── Icon size constants ────────────────────────────────────────────────────────
_ICON_PX = 28        # logical icon size fed to QToolButton
_ICON_DRAW = 32      # canvas draw size for the vector icons


def _draw_mic_icon(color: str = "#F8FAFC", muted: bool = False, size: int = _ICON_DRAW) -> QIcon:
    """Render a crisp, highlighted vector microphone icon."""
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    pen = QPen(QColor(color), 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    # Mic capsule body
    painter.drawRoundedRect(QRectF(11, 4, 10, 14), 5, 5)

    # U-bracket arc
    path = QPainterPath()
    path.arcMoveTo(QRectF(7, 8, 18, 14), 0)
    path.arcTo(QRectF(7, 8, 18, 14), 0, -180)
    painter.drawPath(path)

    # Base & stand
    painter.drawLine(QPointF(16, 22), QPointF(16, 27))
    painter.drawLine(QPointF(10, 27), QPointF(22, 27))

    if muted:
        slash_pen = QPen(QColor("#FF6B6B"), 2.8, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(slash_pen)
        painter.drawLine(QPointF(5, 5), QPointF(27, 27))

    painter.end()
    return QIcon(pixmap)


def _draw_wake_icon(active: bool = True, size: int = _ICON_DRAW) -> QIcon:
    """Render a lightning bolt (awake) or crescent moon (sleeping) icon."""
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    if active:
        # Glowing filled lightning bolt
        poly = QPolygonF([
            QPointF(18, 2), QPointF(6, 17), QPointF(15, 17),
            QPointF(14, 30), QPointF(26, 13), QPointF(17, 13)
        ])
        painter.setPen(QPen(QColor("#10B981"), 1.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(QColor("#10B981")))
        painter.drawPolygon(poly)
    else:
        # Filled crescent moon
        path = QPainterPath()
        path.arcMoveTo(QRectF(4, 3, 24, 24), -55)
        path.arcTo(QRectF(4, 3, 24, 24), -55, 270)
        path.arcTo(QRectF(10, 3, 16, 16), 215, -195)
        path.closeSubpath()

        painter.setPen(QPen(QColor("#6366F1"), 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(QColor("#6366F1")))
        painter.drawPath(path)

    painter.end()
    return QIcon(pixmap)


def _draw_gear_icon(color: str = "#F8FAFC", size: int = _ICON_DRAW) -> QIcon:
    """Render a highlighted 6-tooth gear icon."""
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    painter.setPen(QPen(QColor(color), 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)

    cx, cy = 16.0, 16.0
    r_outer = 11.0
    r_inner = 8.0
    num_teeth = 6

    path = QPainterPath()
    for i in range(num_teeth):
        angle_deg = i * 60
        rad1 = math.radians(angle_deg - 11)
        rad2 = math.radians(angle_deg + 11)
        rad3 = math.radians(angle_deg + 30 - 7)
        rad4 = math.radians(angle_deg + 30 + 7)

        pt1 = QPointF(cx + r_outer * math.cos(rad1), cy + r_outer * math.sin(rad1))
        pt2 = QPointF(cx + r_outer * math.cos(rad2), cy + r_outer * math.sin(rad2))
        pt3 = QPointF(cx + r_inner * math.cos(rad3), cy + r_inner * math.sin(rad3))
        pt4 = QPointF(cx + r_inner * math.cos(rad4), cy + r_inner * math.sin(rad4))

        if i == 0:
            path.moveTo(pt1)
        else:
            path.lineTo(pt1)
        path.lineTo(pt2)
        path.lineTo(pt3)
        path.lineTo(pt4)
    path.closeSubpath()
    painter.drawPath(path)

    painter.drawEllipse(QPointF(cx, cy), 4.0, 4.0)
    painter.end()
    return QIcon(pixmap)


def _draw_theme_icon(dark_mode: bool, size: int = _ICON_DRAW) -> QIcon:
    """Render a sun (light mode) or crescent moon (dark mode) icon."""
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    if dark_mode:
        # Show sun icon (click → switch to light)
        color = QColor("#F59E0B")
        cx, cy = 16.0, 16.0

        # Sun disc
        painter.setPen(QPen(color, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(cx, cy), 6.0, 6.0)

        # Rays
        ray_pen = QPen(color, 2.2, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(ray_pen)
        painter.setBrush(Qt.NoBrush)
        for angle_deg in range(0, 360, 45):
            rad = math.radians(angle_deg)
            x1 = cx + 9.0 * math.cos(rad)
            y1 = cy + 9.0 * math.sin(rad)
            x2 = cx + 12.5 * math.cos(rad)
            y2 = cy + 12.5 * math.sin(rad)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    else:
        # Show moon icon (click → switch to dark)
        color = QColor("#6366F1")
        path = QPainterPath()
        path.arcMoveTo(QRectF(4, 3, 24, 24), -55)
        path.arcTo(QRectF(4, 3, 24, 24), -55, 270)
        path.arcTo(QRectF(10, 3, 16, 16), 215, -195)
        path.closeSubpath()
        painter.setPen(QPen(color, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(color))
        painter.drawPath(path)

    painter.end()
    return QIcon(pixmap)


def _draw_trash_icon(color: str = "#F8FAFC", size: int = _ICON_DRAW) -> QIcon:
    """Render a highlighted vector trash can icon."""
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    painter.setPen(QPen(QColor(color), 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)

    # Lid handle
    painter.drawRoundedRect(QRectF(12, 3, 8, 4), 1.5, 1.5)

    # Lid bar
    painter.drawLine(QPointF(5, 9), QPointF(27, 9))

    # Bucket body
    bucket_path = QPainterPath()
    bucket_path.moveTo(8, 9)
    bucket_path.lineTo(9.5, 28)
    bucket_path.lineTo(22.5, 28)
    bucket_path.lineTo(24, 9)
    painter.drawPath(bucket_path)

    # Vertical ribs
    painter.drawLine(QPointF(13, 13), QPointF(13, 24))
    painter.drawLine(QPointF(19, 13), QPointF(19, 24))

    painter.end()
    return QIcon(pixmap)


# ─── Shared style helpers ───────────────────────────────────────────────────────


def _btn_base(name: str, hover_color: str) -> str:
    """Return a QToolButton base stylesheet using the current theme tokens."""
    tok = ThemeManager.instance().tokens
    btn_bg      = tok["btn_bg"]
    btn_hover   = tok["btn_hover_bg"]
    label_color = tok["btn_label_color"]
    return f"""
    QToolButton#{name} {{
        background-color: {btn_bg};
        color: {label_color};
        border: 1px solid rgba(128, 128, 128, 0.12);
        border-radius: 18px;
        font-size: 8px;
        font-weight: 700;
        letter-spacing: 1.2px;
        padding-top: 4px;
        padding-bottom: 6px;
        text-align: center;
    }}
    QToolButton#{name}:hover {{
        background-color: {btn_hover};
        border-color: {hover_color};
        color: {hover_color};
    }}
"""


_BTN_ACTIVE = """
    QToolButton#{name} {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {grad_start}, stop:1 {grad_end});
        color: #FFFFFF;
        border: 1.5px solid {border_color};
        border-radius: 18px;
        font-size: 8px;
        font-weight: 700;
        letter-spacing: 1.2px;
        padding-top: 4px;
        padding-bottom: 6px;
        text-align: center;
    }}
"""


def _make_btn(name: str, tip: str, width: int = 68, height: int = 62) -> QToolButton:
    """Create a styled nav QToolButton with icon-on-top, label-below layout."""
    btn = QToolButton()
    btn.setObjectName(name)
    btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
    btn.setFixedSize(width, height)
    btn.setIconSize(QSize(_ICON_PX, _ICON_PX))
    btn.setToolTip(tip)
    btn.setCursor(Qt.PointingHandCursor)
    # Small bold label font
    f = btn.font()
    f.setPointSize(7)
    f.setWeight(QFont.Bold)
    f.setLetterSpacing(QFont.AbsoluteSpacing, 0.8)
    btn.setFont(f)
    return btn


# ─── ControlDock ────────────────────────────────────────────────────────────────

class ControlDock(QFrame):
    """
    Floating glassmorphic nav bar — large highlighted icons + tiny labels below.

    Signals
    -------
    mute_toggled(bool)
    wake_toggled(bool)
    settings_toggled(bool)
    clear_chat_requested()
    """

    mute_toggled: Signal = Signal(bool)
    wake_toggled: Signal = Signal(bool)
    settings_toggled: Signal = Signal(bool)
    clear_chat_requested: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(82)
        self.setObjectName("panel_dock")
        self.setStyleSheet("""
            QFrame#panel_dock {
                background-color: rgba(14, 17, 30, 0.96);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-top: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 41px;
            }
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 8, 20, 8)
        layout.setSpacing(6)

        # Track separator widgets for theme re-colouring
        self._separators: list = []

        # ── 1. Mute button ─────────────────────────────────────────────
        self._btn_mute = _make_btn("btn_mute", "Mute / Unmute microphone  [M]")
        self._btn_mute.setCheckable(True)
        self._btn_mute.clicked.connect(self._on_mute_clicked)
        self._update_mute_visual(False)
        layout.addWidget(self._btn_mute)

        # ── Separator ─────────────────────────────────────────────────
        _sep1 = self._make_separator()
        self._separators.append(_sep1)
        layout.addWidget(_sep1)

        # ── 2. Settings button ────────────────────────────────────────
        self._btn_settings = _make_btn("btn_settings", "Settings & Telemetry  [,]")
        self._btn_settings.setText("SETTINGS")
        self._btn_settings.setCheckable(True)
        self._btn_settings.setIcon(_draw_gear_icon("#06B6D4"))
        self._btn_settings.setStyleSheet(
            _btn_base(name="btn_settings", hover_color="#06B6D4") +
            """
            QToolButton#btn_settings:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1E1B4B, stop:1 #312E81);
                color: #A5B4FC;
                border: 1.5px solid #6366F1;
                border-radius: 18px;
                font-size: 8px;
                font-weight: 700;
                letter-spacing: 1.2px;
                padding-top: 4px;
                padding-bottom: 6px;
            }
            """
        )
        self._btn_settings.clicked.connect(self._on_settings_clicked)
        layout.addWidget(self._btn_settings)

        # ── Separator ─────────────────────────────────────────────────
        _sep2 = self._make_separator()
        self._separators.append(_sep2)
        layout.addWidget(_sep2)

        # ── 3. CENTER Hero Mic button (Tap & Hold to Speak) ────────────
        self._btn_wake = _make_btn("btn_wake", "Press and hold to speak  [W]", width=80, height=62)
        self._btn_wake.setCheckable(False)
        self._btn_wake.pressed.connect(self._on_wake_pressed)
        self._btn_wake.released.connect(self._on_wake_released)
        self._apply_wake_visual(active=False)
        layout.addWidget(self._btn_wake)

        # ── Separator ─────────────────────────────────────────────────
        _sep3 = self._make_separator()
        self._separators.append(_sep3)
        layout.addWidget(_sep3)

        # ── 4. Clear button ───────────────────────────────────────────
        self._btn_clear = _make_btn("btn_clear", "Clear Chat History  [C]")
        self._btn_clear.setText("CLEAR")
        self._btn_clear.setIcon(_draw_trash_icon("#94A3B8"))
        self._btn_clear.setStyleSheet(
            _btn_base(name="btn_clear", hover_color="#F87171") +
            """
            QToolButton#btn_clear:pressed {
                background-color: rgba(153, 27, 27, 0.5);
                border-color: #EF4444;
                color: #FCA5A5;
                border-radius: 18px;
            }
            """
        )
        self._btn_clear.clicked.connect(self.clear_chat_requested.emit)
        layout.addWidget(self._btn_clear)

        # ── 5. Theme toggle button ────────────────────────────────────
        sep4 = self._make_separator()
        self._separators.append(sep4)
        layout.addWidget(sep4)

        self._btn_theme = _make_btn("btn_theme", "Toggle Dark / Light Theme  [T]")
        self._btn_theme.clicked.connect(self._on_theme_clicked)
        layout.addWidget(self._btn_theme)

        # Subscribe to theme changes so the icon always reflects reality
        self._tm = ThemeManager.instance()
        self._tm.theme_changed.connect(self._update_theme_visual)
        self._tm.theme_changed.connect(self._refresh_dock_shell)
        self._update_theme_visual(self._tm.theme)

        # ── Keyboard shortcuts ─────────────────────────────────────────
        self._shortcut_mute = QShortcut(QKeySequence("M"), self)
        self._shortcut_mute.activated.connect(self._btn_mute.animateClick)

        self._shortcut_wake = QShortcut(QKeySequence("W"), self)
        self._shortcut_wake.activated.connect(self._btn_wake.animateClick)

        self._shortcut_settings = QShortcut(QKeySequence(","), self)
        self._shortcut_settings.activated.connect(self._btn_settings.animateClick)

        self._shortcut_clear = QShortcut(QKeySequence("C"), self)
        self._shortcut_clear.activated.connect(self.clear_chat_requested.emit)

        self._shortcut_theme = QShortcut(QKeySequence("T"), self)
        self._shortcut_theme.activated.connect(self._on_theme_clicked)

    def _make_separator(self) -> QFrame:
        """Thin vertical divider line between buttons."""
        tok = ThemeManager.instance().tokens
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedSize(1, 28)
        sep.setStyleSheet(f"background-color: {tok['dock_sep']}; border: none;")
        return sep

    def _refresh_dock_shell(self, theme: str) -> None:
        """Re-apply dock pill frame + separators + stateless button styles on theme switch."""
        tok = ThemeManager.instance().tokens
        is_dark = (theme == "dark")

        # Dock pill
        border_top = "rgba(255,255,255,0.15)" if is_dark else "rgba(148,163,184,0.35)"
        self.setStyleSheet(f"""
            QFrame#panel_dock {{
                background-color: {tok['dock_bg']};
                border: 1px solid {tok['dock_border']};
                border-top: 1px solid {border_top};
                border-radius: 41px;
            }}
        """)

        # Separators
        for sep in getattr(self, "_separators", []):
            sep.setStyleSheet(f"background-color: {tok['dock_sep']}; border: none;")

        # Re-apply stateless buttons (don't touch checked/active state buttons)
        if not self._btn_mute.isChecked():
            self._btn_mute.setStyleSheet(_btn_base("btn_mute", "#06B6D4"))

        if not self._btn_wake.isChecked():
            self._btn_wake.setStyleSheet(_btn_base("btn_wake", "#6366F1"))

        self._btn_settings.setStyleSheet(
            _btn_base("btn_settings", "#06B6D4") +
            """
            QToolButton#btn_settings:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1E1B4B, stop:1 #312E81);
                color: #A5B4FC;
                border: 1.5px solid #6366F1;
                border-radius: 18px;
                font-size: 8px;
                font-weight: 700;
                letter-spacing: 1.2px;
                padding-top: 4px;
                padding-bottom: 6px;
            }
            """
        )

        self._btn_clear.setStyleSheet(
            _btn_base("btn_clear", "#F87171") +
            """
            QToolButton#btn_clear:pressed {
                background-color: rgba(153, 27, 27, 0.5);
                border-color: #EF4444;
                color: #FCA5A5;
                border-radius: 18px;
            }
            """
        )

        # Theme button refresh handled by _update_theme_visual

    # ── Visual update helpers ──────────────────────────────────────────────────

    def _update_mute_visual(self, muted: bool) -> None:
        if muted:
            self._btn_mute.setText("MUTED")
            self._btn_mute.setIcon(_draw_mic_icon("#FF6B6B", muted=True))
            self._btn_mute.setStyleSheet(
                _BTN_ACTIVE.format(
                    name="btn_mute",
                    grad_start="#7F1D1D",
                    grad_end="#DC2626",
                    border_color="#FF6B6B",
                ) +
                "QToolButton#btn_mute:hover { border-color: #FCA5A5; }"
            )
        else:
            self._btn_mute.setText("MIC ON")
            self._btn_mute.setIcon(_draw_mic_icon("#06B6D4", muted=False))
            self._btn_mute.setStyleSheet(
                _btn_base(name="btn_mute", hover_color="#06B6D4")
            )

    def _apply_wake_visual(self, active: bool) -> None:
        if active:
            self._btn_wake.setText("LISTENING...")
            self._btn_wake.setIcon(_draw_wake_icon(active=True))
            self._btn_wake.setToolTip("Release to send voice command")
            self._btn_wake.setStyleSheet(
                _BTN_ACTIVE.format(
                    name="btn_wake",
                    grad_start="#7F1D1D",
                    grad_end="#DC2626",
                    border_color="#FF6B6B",
                ) +
                "QToolButton#btn_wake:hover { border-color: #FCA5A5; }"
            )
        else:
            self._btn_wake.setText("HOLD TO SPEAK")
            self._btn_wake.setIcon(_draw_wake_icon(active=False))
            self._btn_wake.setToolTip("Press and hold to speak  [W]")
            self._btn_wake.setStyleSheet(
                _btn_base(name="btn_wake", hover_color="#6366F1")
            )

    def _update_theme_visual(self, theme: str) -> None:
        """Update the theme button icon and tooltip to reflect the current theme."""
        is_dark = (theme == "dark")
        if is_dark:
            # Currently dark → show sun → tooltip says "Switch to Light"
            self._btn_theme.setText("LIGHT")
            self._btn_theme.setToolTip("Switch to Light Theme  [T]")
            self._btn_theme.setIcon(_draw_theme_icon(dark_mode=True))
            self._btn_theme.setStyleSheet(
                _btn_base(name="btn_theme", hover_color="#F59E0B")
            )
        else:
            # Currently light → show moon → tooltip says "Switch to Dark"
            self._btn_theme.setText("DARK")
            self._btn_theme.setToolTip("Switch to Dark Theme  [T]")
            self._btn_theme.setIcon(_draw_theme_icon(dark_mode=False))
            self._btn_theme.setStyleSheet(
                _btn_base(name="btn_theme", hover_color="#6366F1")
            )

    def _on_theme_clicked(self) -> None:
        ThemeManager.instance().toggle()

    # ── Slot handlers ──────────────────────────────────────────────────────────

    def _on_mute_clicked(self, checked: bool) -> None:
        self._update_mute_visual(checked)
        self.mute_toggled.emit(checked)

    def _on_wake_pressed(self) -> None:
        self._apply_wake_visual(active=True)
        self.wake_toggled.emit(True)

    def _on_wake_released(self) -> None:
        self._apply_wake_visual(active=False)
        self.wake_toggled.emit(False)

    def _on_settings_clicked(self, checked: bool) -> None:
        self.settings_toggled.emit(checked)

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_muted(self) -> bool:
        return self._btn_mute.isChecked()

    def set_settings_open(self, open_: bool) -> None:
        self._btn_settings.setChecked(open_)

    def set_wake_active(self, active: bool) -> None:
        self._apply_wake_visual(active)
