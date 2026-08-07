"""
ui/views/student_presets_drawer.py — Collapsible Student Q&A preset browser panel.

Provides:
  - Slide-up panel with categorized student preset questions (Science, Math, CS, Study Skills).
  - Quick-click chips that emit `preset_selected(text)` to ask the voice assistant.
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
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.student_presets import get_presets_by_category
from ui.theme_manager import ThemeManager

_PANEL_HEIGHT = 360  # expanded height in pixels


class StudentPresetsDrawer(QWidget):
    """
    Slide-up student Q&A browser drawer fixed to the bottom of the parent window.

    Signals
    -------
    preset_selected(str): Emitted when a student clicks a preset question button.
    closed(): Emitted when the drawer closes.
    """

    preset_selected: Signal = Signal(str)
    closed: Signal = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)

        self.setFixedHeight(_PANEL_HEIGHT)
        self.setAttribute(Qt.WA_StyledBackground, True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 10, 18, 14)
        outer.setSpacing(10)

        # Header bar
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        title = QLabel("STUDENT Q&A PRESET BANK")
        title.setStyleSheet("color:#10B981; font-size:11px; font-weight:800; letter-spacing:1px; background:transparent;")
        header_row.addWidget(title)

        header_row.addStretch(1)

        self._handle = QFrame()
        self._handle.setFixedSize(36, 4)
        header_row.addWidget(self._handle)

        header_row.addStretch(1)

        self._btn_close = QPushButton("✕")
        self._btn_close.setFixedSize(26, 26)
        self._btn_close.setCursor(Qt.PointingHandCursor)
        self._btn_close.setToolTip("Close Presets Bank")
        self._btn_close.clicked.connect(self.close)
        header_row.addWidget(self._btn_close)

        outer.addLayout(header_row)

        # Scrollable area for categories & question chips
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        cont_layout = QVBoxLayout(container)
        cont_layout.setContentsMargins(0, 0, 0, 0)
        cont_layout.setSpacing(12)

        presets_by_cat = get_presets_by_category()

        for category, items in presets_by_cat.items():
            cat_label = QLabel(category.upper())
            cat_label.setStyleSheet("color:#06B6D4; font-size:10px; font-weight:800; letter-spacing:0.8px;")
            cont_layout.addWidget(cat_label)

            chip_flow = QHBoxLayout()
            chip_flow.setSpacing(8)

            # Scroll row for chips
            chip_scroll = QScrollArea()
            chip_scroll.setWidgetResizable(True)
            chip_scroll.setFixedHeight(42)
            chip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            chip_scroll.setFrameShape(QFrame.NoFrame)

            chip_widget = QWidget()
            chip_layout = QHBoxLayout(chip_widget)
            chip_layout.setContentsMargins(0, 0, 0, 0)
            chip_layout.setSpacing(8)

            for item in items:
                btn = QPushButton(item["question"])
                btn.setCursor(Qt.PointingHandCursor)
                btn.setStyleSheet("""
                    QPushButton {
                        background: #1E2235;
                        color: #F8FAFC;
                        border: 1px solid rgba(255, 255, 255, 0.12);
                        border-radius: 12px;
                        padding: 6px 14px;
                        font-size: 11px;
                        font-weight: 600;
                    }
                    QPushButton:hover {
                        background: #6366F1;
                        color: #FFFFFF;
                        border-color: #818CF8;
                    }
                """)
                q_text = item["question"]
                btn.clicked.connect(lambda checked=False, q=q_text: self._on_chip_clicked(q))
                chip_layout.addWidget(btn)

            chip_layout.addStretch(1)
            chip_scroll.setWidget(chip_widget)
            cont_layout.addWidget(chip_scroll)

        cont_layout.addStretch(1)
        scroll.setWidget(container)
        outer.addWidget(scroll)

        # ESC shortcut to close drawer
        self._shortcut_esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._shortcut_esc.activated.connect(self.close)

        self._tm = ThemeManager.instance()
        self._tm.theme_changed.connect(self._apply_theme)
        self._apply_theme(self._tm.theme)

        self._anim = QPropertyAnimation(self, b"geometry", self)
        self._anim.setDuration(280)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.finished.connect(self._on_anim_finished)

        self._is_open = False

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

    def _on_chip_clicked(self, question: str) -> None:
        self.close()
        self.preset_selected.emit(question)

    def _apply_theme(self, theme: str) -> None:
        is_dark = (theme == "dark")
        panel_bg = "#1B1E2B" if is_dark else "#F8FAFC"
        panel_bdr = "#252836" if is_dark else "#E2E8F0"
        handle_bg = "#374151" if is_dark else "#CBD5E1"

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {panel_bg};
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
                border-top: 1px solid {panel_bdr};
            }}
        """)
        if hasattr(self, "_handle"):
            self._handle.setStyleSheet(f"background:{handle_bg}; border-radius:2px;")

    def _reposition(self) -> None:
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
