"""
ui/main_window.py — Root PySide6 application window for Jangoz AI Voice Assistant.

MainWindow assembles all UI components:
  • Custom Glass Title Bar with branding and window controls
  • AI Character Stage Visualizer (avatar + state orbital rings + audio waves)
  • Live Status Badge & telemetry indicators
  • Streaming Chat Display with circular AI avatar thumbnails
  • Floating Control Dock (Mute, Wake, Settings, Clear Chat)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.bridge import UIBridge
from ui.components.chat_display import ChatDisplay
from ui.components.control_dock import ControlDock
from ui.components.status_badge import StatusBadge
from ui.components.visualizer import WaveformVisualizer
from ui.theme_manager import ThemeManager
from ui.views.settings_drawer import SettingsDrawer
from ui.views.student_presets_drawer import StudentPresetsDrawer

_STYLES_DIR = Path(__file__).parent / "styles"


def _load_stylesheet() -> str:
    """Deprecated: stylesheet now managed by ThemeManager."""
    qss_path = _STYLES_DIR / "dark_theme.qss"
    if qss_path.exists():
        return qss_path.read_text(encoding="utf-8")
    return ""


class _TitleBar(QWidget):
    """Custom draggable title bar replacing OS window decorations."""

    def __init__(self, title: str, subtitle: str, parent: QMainWindow) -> None:
        super().__init__(parent)
        self._drag_pos: QPoint | None = None
        self._window = parent
        self.setFixedHeight(58)
        self.setObjectName("title_bar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 14, 0)
        layout.setSpacing(0)

        # App branding dot
        dot = QLabel("◈")
        dot.setStyleSheet("color: #06B6D4; font-size: 18px; background: transparent;")
        layout.addWidget(dot)

        layout.addSpacing(10)

        # Title & Subtitle
        text_col = QVBoxLayout()
        text_col.setSpacing(1)

        lbl_title = QLabel(title)
        lbl_title.setObjectName("lbl_title")
        text_col.addWidget(lbl_title)

        lbl_sub = QLabel(subtitle)
        lbl_sub.setObjectName("lbl_subtitle")
        text_col.addWidget(lbl_sub)

        layout.addLayout(text_col)
        layout.addStretch(1)

        # Window Action Buttons
        for symbol, slot in [("−", self._minimise), ("✕", self._close)]:
            btn = QLabel(symbol)
            btn.setFixedSize(30, 30)
            btn.setAlignment(Qt.AlignCenter)
            btn.setStyleSheet("""
                QLabel {
                    color: #94A3B8;
                    background: transparent;
                    border-radius: 8px;
                    font-size: 15px;
                    font-weight: bold;
                }
                QLabel:hover { background: #1E2235; color: #F8FAFC; }
            """)
            btn.setCursor(Qt.PointingHandCursor)
            btn.mousePressEvent = lambda e, s=slot: s()
            layout.addWidget(btn)
            layout.addSpacing(4)

    def _minimise(self) -> None:
        self._window.showMinimized()

    def _close(self) -> None:
        self._window.close()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self._window.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            self._window.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_pos = None


class MainWindow(QMainWindow):
    """
    Root desktop UI window.
    """

    def __init__(self, bridge: UIBridge, config) -> None:
        super().__init__()
        self._bridge = bridge
        self._cfg = config
        self._muted = False

        self._setup_window()
        self._setup_palette()
        self._setup_ui()
        self._connect_signals()

    def _setup_window(self) -> None:
        self.setWindowTitle("Nyra — AI Voice Assistant")
        self.setMinimumSize(self._cfg.window_min_width, self._cfg.window_min_height)
        self.resize(self._cfg.window_width, self._cfg.window_height)

        if self._cfg.frameless:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
            self.setAttribute(Qt.WA_TranslucentBackground, False)

        # Initialise ThemeManager and apply the dark stylesheet globally
        self._tm = ThemeManager.instance()
        self._tm.apply("dark")
        self._tm.theme_changed.connect(self._on_theme_changed)

    def _setup_palette(self) -> None:
        palette = QPalette()
        bg = QColor(self._cfg.color_bg)
        palette.setColor(QPalette.Window, bg)
        palette.setColor(QPalette.WindowText, QColor(self._cfg.color_text))
        palette.setColor(QPalette.Base, QColor(self._cfg.color_surface))
        palette.setColor(QPalette.Text, QColor(self._cfg.color_text))
        palette.setColor(QPalette.Button, QColor(self._cfg.color_accent))
        palette.setColor(QPalette.ButtonText, QColor("#FFFFFF"))
        self.setPalette(palette)
        QApplication.instance().setPalette(palette)

    def _setup_ui(self) -> None:
        tok = ThemeManager.instance().tokens
        central = QWidget()
        central.setObjectName("central_widget")
        central.setStyleSheet(f"background-color: {tok['color_bg']};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Custom Title Bar
        self._title_bar = _TitleBar(
            "Nyra AI",
            "Al Irshad Public School · Mentored by Aitute",
            self,
        )
        root.addWidget(self._title_bar)

        # Loading Panel
        self._loading_panel = self._make_loading_panel()
        root.addWidget(self._loading_panel)
        self._loading_panel.hide()

        # AI Character Visualizer Stage Card
        self._vis_card = QFrame()
        self._vis_card.setObjectName("panel_surface")
        self._vis_card.setStyleSheet(f"""
            QFrame#panel_surface {{
                background-color: {self._cfg.color_surface};
                border-radius: 0px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            }}
        """)
        self._vis_card.setFixedHeight(self._cfg.visualizer_height)

        vis_layout = QVBoxLayout(self._vis_card)
        vis_layout.setContentsMargins(0, 0, 0, 0)

        self._visualizer = WaveformVisualizer()
        self._visualizer.set_accent_color(self._cfg.color_accent)
        vis_layout.addWidget(self._visualizer)

        root.addWidget(self._vis_card)

        # Status Badge Row
        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(18, 12, 18, 6)

        self._status_badge = StatusBadge()
        badge_row.addWidget(self._status_badge, 0, Qt.AlignLeft)
        badge_row.addStretch(1)

        # Student Presets Button
        self._btn_presets = QPushButton("🎓 STUDENT PRESETS")
        self._btn_presets.setCursor(Qt.PointingHandCursor)
        self._btn_presets.setToolTip("Browse presets for Math, Science, CS, and Study Skills")
        self._btn_presets.setStyleSheet("""
            QPushButton {
                background: rgba(16, 185, 129, 0.12);
                color: #10B981;
                border: 1px solid #10B981;
                border-radius: 10px;
                padding: 3px 10px;
                font-size: 9px;
                font-weight: 700;
                letter-spacing: 0.6px;
            }
            QPushButton:hover {
                background: #10B981;
                color: #FFFFFF;
            }
        """)
        badge_row.addWidget(self._btn_presets, 0, Qt.AlignRight)
        badge_row.addSpacing(8)

        # Live status pill indicator
        self._live_dot = QLabel("● CYBER ONLINE  [EN]")
        self._live_dot.setStyleSheet(
            "color: #06B6D4; font-size: 10px; font-weight: 700; letter-spacing: 0.8px; background: transparent;"
        )
        badge_row.addWidget(self._live_dot, 0, Qt.AlignRight)

        root.addLayout(badge_row)

        # Chat Container
        self._chat = ChatDisplay()
        root.addWidget(self._chat, 1)

        # Floating Control Dock Area
        dock_container = QWidget()
        dock_container.setStyleSheet("background: transparent;")
        dock_layout = QHBoxLayout(dock_container)
        dock_layout.setContentsMargins(16, 6, 16, 16)

        self._dock = ControlDock()
        dock_layout.addStretch(1)
        dock_layout.addWidget(self._dock)
        dock_layout.addStretch(1)

        root.addWidget(dock_container)

        # Settings Drawer Overlay
        self._drawer = SettingsDrawer(central)
        self._drawer.hide()

        # Student Presets Drawer Overlay
        self._presets_drawer = StudentPresetsDrawer(central)
        self._presets_drawer.hide()

    def _make_loading_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("loading_panel")
        panel.setStyleSheet("""
            QFrame#loading_panel {
                background-color: #141722;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 14, 20, 16)
        layout.setSpacing(8)

        header = QHBoxLayout()
        spinner = QLabel("⧗")
        spinner.setStyleSheet(
            "color: #A855F7; font-size: 18px; background: transparent;"
        )
        header.addWidget(spinner)
        header.addSpacing(8)

        hdr_text = QLabel("Initialising Nyra AI Core…")
        hdr_text.setStyleSheet(
            "color: #F8FAFC; font-size: 13px; font-weight: 700; background: transparent;"
        )
        header.addWidget(hdr_text)
        header.addStretch()
        layout.addLayout(header)

        self._loading_label = QLabel("Loading models…")
        self._loading_label.setStyleSheet(
            "color: #94A3B8; font-size: 11px; background: transparent;"
        )
        layout.addWidget(self._loading_label)

        self._loading_bar = QProgressBar()
        self._loading_bar.setMinimum(0)
        self._loading_bar.setMaximum(0)
        self._loading_bar.setFixedHeight(6)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1E2235;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #06B6D4,
                    stop:0.5 #6366F1,
                    stop:1 #A855F7
                );
                border-radius: 3px;
            }
        """)
        layout.addWidget(self._loading_bar)

        return panel

    def _connect_signals(self) -> None:
        # Backend → UI
        self._bridge.amplitude_updated.connect(self._visualizer.set_level)
        self._bridge.state_changed.connect(self._on_state_changed)
        self._bridge.loading_progress.connect(self._on_loading_progress)
        self._bridge.user_transcribed.connect(self._chat.append_user_message)
        self._bridge.assistant_token.connect(self._chat.stream_assistant_token)
        self._bridge.assistant_done.connect(self._chat.finalize_assistant_message)
        self._bridge.telemetry_updated.connect(self._drawer.update_telemetry)

        # Dock Controls
        self._dock.mute_toggled.connect(self._bridge.emit_mute_toggle)
        self._dock.mute_toggled.connect(self._on_mute_toggled)
        self._dock.wake_toggled.connect(self._bridge.emit_wake_toggle)
        self._dock.settings_toggled.connect(self._on_settings_toggled)
        self._dock.clear_chat_requested.connect(self._chat.clear_chat)
        self._drawer.closed.connect(lambda: self._dock.set_settings_open(False))

        # Student Presets Drawer Controls
        self._btn_presets.clicked.connect(self._presets_drawer.toggle)
        self._presets_drawer.preset_selected.connect(self._bridge.emit_text_query)

    def _on_state_changed(self, state: str) -> None:
        """Propagate state changes to badge and visualizer stage."""
        self._status_badge.set_state(state)
        self._visualizer.set_state(state)

        upper = state.upper()
        is_loading = upper == "LOADING"
        self._loading_panel.setVisible(is_loading)
        self._dock.setEnabled(not is_loading)

    def _on_loading_progress(self, message: str, percent: int) -> None:
        self._loading_label.setText(message)
        if percent < 0:
            self._loading_bar.setMaximum(0)
        else:
            self._loading_bar.setMaximum(100)
            self._loading_bar.setValue(percent)

    def _on_mute_toggled(self, muted: bool) -> None:
        self._muted = muted
        color = "#EF4444" if muted else self._cfg.color_accent
        self._visualizer.set_accent_color(color)

    def _on_settings_toggled(self, open_: bool) -> None:
        if open_:
            self._drawer.open()
        else:
            self._drawer.close()

    def _on_theme_changed(self, theme: str) -> None:
        """Re-apply inline background colours that QSS can't reach."""
        tok = ThemeManager.instance().tokens
        is_dark = (theme == "dark")

        # Central widget background
        cw = self.centralWidget()
        if cw:
            cw.setStyleSheet(f"background-color: {tok['color_bg']};")

        # Title bar
        tb_border = tok["titlebar_border"]
        tb_bg = tok["titlebar_bg"]
        self._title_bar.setStyleSheet(
            f"""QWidget#title_bar {{
                background-color: {tb_bg};
                border-bottom: 1px solid {tb_border};
            }}"""
        )

        # Title text colours
        for child in self._title_bar.findChildren(QLabel):
            obj = child.objectName()
            if obj == "lbl_title":
                child.setStyleSheet(f"color: {tok['title_text']}; font-size: 15px; font-weight: 700; background: transparent;")
            elif obj == "lbl_subtitle":
                child.setStyleSheet(f"color: {tok['title_muted']}; font-size: 11px; background: transparent;")

        # Visualizer card
        surf_border = "rgba(255,255,255,0.06)" if is_dark else "rgba(148,163,184,0.20)"
        self._vis_card.setStyleSheet(
            f"""QFrame#panel_surface {{
                background-color: {tok['color_surface']};
                border-radius: 0px;
                border-bottom: 1px solid {surf_border};
            }}"""
        )

        # Live dot accent
        accent = "#06B6D4" if is_dark else "#0891B2"
        self._live_dot.setStyleSheet(
            f"color: {accent}; font-size: 10px; font-weight: 700; letter-spacing: 0.8px; background: transparent;"
        )

        # Control dock pill
        dock_border_top = "rgba(255,255,255,0.15)" if is_dark else "rgba(148,163,184,0.35)"
        self._dock.setStyleSheet(
            f"""QFrame#panel_dock {{
                background-color: {tok['dock_bg']};
                border: 1px solid {tok['dock_border']};
                border-top: 1px solid {dock_border_top};
                border-radius: 41px;
            }}"""
        )

        # Repaint visualizer accent
        self._visualizer.set_accent_color(tok["color_accent"])

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._drawer.isVisible():
            self._drawer._reposition()
        if hasattr(self, "_presets_drawer") and self._presets_drawer.isVisible():
            self._presets_drawer._reposition()

    def closeEvent(self, event) -> None:  # noqa: N802
        super().closeEvent(event)
