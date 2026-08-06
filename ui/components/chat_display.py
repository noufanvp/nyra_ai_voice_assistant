"""
ui/components/chat_display.py — Cyber-luxe streaming chat display.

Renders user and assistant speech bubbles in a scroll area with:
  • Circular AI Avatar thumbnail icons on assistant responses
  • High contrast readable text with ample spacing & proper bubble expansion
  • Animated typing status indicator dots (● ● ●) while streaming
  • Indigo/Violet gradient styling for user bubbles
  • Obsidian glass cards for assistant responses
  • Smooth auto-scrolling with delayed layout refresh
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from ui.theme_manager import ThemeManager

# User bubble is always an indigo gradient regardless of theme
_USER_BG = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4338CA, stop:1 #6366F1)"
_USER_TEXT = "#FFFFFF"
_STREAMING_DOTS = " ● ● ●"


def _asst_bg() -> str:
    tm = ThemeManager.instance()
    return "#141722" if tm.is_dark else "#FFFFFF"


def _asst_border() -> str:
    tm = ThemeManager.instance()
    return "rgba(255, 255, 255, 0.12)" if tm.is_dark else "rgba(148, 163, 184, 0.30)"


def _asst_text() -> str:
    tm = ThemeManager.instance()
    return "#F1F5F9" if tm.is_dark else "#0F172A"


def _asst_header_color() -> str:
    tm = ThemeManager.instance()
    return "#06B6D4" if tm.is_dark else "#0891B2"


class _AvatarIcon(QWidget):
    """Circular AI avatar thumbnail widget."""

    def __init__(self, size: int = 34, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)

        avatar_file = Path(__file__).parent.parent.parent / "assets" / "avatar.png"
        self._pixmap: QPixmap | None = None
        if avatar_file.exists():
            pix = QPixmap(str(avatar_file))
            if not pix.isNull():
                self._pixmap = pix

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = float(self.width())
        r = w / 2.0
        rect = QRectF(0, 0, w, w)

        # Outer ring glow
        painter.setPen(QPen(QColor(6, 182, 212, 220), 1.5))
        painter.setBrush(QColor(15, 23, 42, 240))
        painter.drawEllipse(rect)

        if self._pixmap and not self._pixmap.isNull():
            path = QPainterPath()
            path.addEllipse(rect.adjusted(1.5, 1.5, -1.5, -1.5))
            painter.setClipPath(path)
            scaled = self._pixmap.scaled(
                int(w), int(w), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            painter.drawPixmap(0, 0, scaled)
        else:
            # Fallback icon
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#06B6D4"))
            painter.drawEllipse(QPointF(r, r), r * 0.4, r * 0.4)

        painter.end()


class _BubbleWidget(QFrame):
    """A single chat message bubble with avatar, speaker tag, and text content."""

    def __init__(
        self,
        text: str,
        is_user: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._is_user = is_user
        self._streaming = not is_user

        self.setObjectName("bubble_frame")
        if is_user:
            self.setStyleSheet(f"""
                QFrame#bubble_frame {{
                    background: {_USER_BG};
                    border: 1px solid rgba(165, 180, 252, 0.4);
                    border-radius: 14px 14px 2px 14px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame#bubble_frame {{
                    background: {_asst_bg()};
                    border: 1px solid {_asst_border()};
                    border-radius: 14px 14px 14px 2px;
                }}
            """)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.setMinimumWidth(180)

        inner = QVBoxLayout(self)
        inner.setContentsMargins(14, 10, 14, 12)
        inner.setSpacing(6)

        # Speaker tag header
        header = QLabel("YOU" if is_user else "NYRA AI")
        header_color = "#E0E7FF" if is_user else _asst_header_color()
        header.setStyleSheet(f"""
            QLabel {{
                color: {header_color};
                font-size: 10px;
                font-weight: 800;
                letter-spacing: 1.2px;
                background: transparent;
                border: none;
            }}
        """)
        inner.addWidget(header)

        # Message text
        fg = _USER_TEXT if is_user else _asst_text()
        initial_text = text + (_STREAMING_DOTS if self._streaming else "")
        self._label = QLabel(initial_text)
        self._label.setWordWrap(True)
        self._label.setTextFormat(Qt.PlainText)
        self._label.setStyleSheet(f"""
            QLabel {{
                color: {fg};
                background: transparent;
                border: none;
                font-size: 13px;
                font-weight: 500;
                line-height: 1.45;
            }}
        """)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        inner.addWidget(self._label)

        self._text_buffer = text

    def append_text(self, token: str) -> None:
        """Append streaming token text and refresh label."""
        self._text_buffer += token
        self._label.setText(self._text_buffer + (_STREAMING_DOTS if self._streaming else ""))

    def finalize(self) -> None:
        """Remove streaming dots and lock bubble."""
        self._streaming = False
        self._label.setText(self._text_buffer)


class ChatDisplay(QScrollArea):
    """
    Scrollable container for user and assistant chat bubbles.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(14)
        self._layout.addStretch(1)

        self.setWidget(self._container)
        self._current_asst_bubble: _BubbleWidget | None = None

        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.setInterval(40)
        self._scroll_timer.timeout.connect(self._do_scroll)

    # ── Public Slots ──────────────────────────────────────────────────

    def append_user_message(self, text: str) -> None:
        """Add a right-aligned user speech bubble."""
        bubble = _BubbleWidget(text, is_user=True)
        bubble.finalize()

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        # Right-align user messages with left spacer
        row.addStretch(1)
        row.addWidget(bubble, 0)

        self._layout.insertLayout(self._layout.count() - 1, row)
        self._schedule_scroll()

    def stream_assistant_token(self, sentence: str) -> None:
        """Stream an assistant response token/sentence with avatar thumbnail."""
        if self._current_asst_bubble is None:
            self._current_asst_bubble = _BubbleWidget("", is_user=False)
            avatar_icon = _AvatarIcon(size=34)

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)
            row.addWidget(avatar_icon, 0, Qt.AlignTop)
            row.addWidget(self._current_asst_bubble, 1)  # Stretch=1 lets bubble expand properly
            row.addStretch(0)

            self._layout.insertLayout(self._layout.count() - 1, row)

        current = self._current_asst_bubble._text_buffer
        separator = " " if current else ""
        self._current_asst_bubble.append_text(separator + sentence)
        self._schedule_scroll()

    def finalize_assistant_message(self) -> None:
        """Lock current assistant bubble."""
        if self._current_asst_bubble is not None:
            self._current_asst_bubble.finalize()
            self._current_asst_bubble = None

    def clear_chat(self) -> None:
        """Remove all message rows."""
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            elif item.widget():
                item.widget().deleteLater()
        self._current_asst_bubble = None

    # ── Internal ──────────────────────────────────────────────────────

    def _schedule_scroll(self) -> None:
        self._scroll_timer.start()

    def _do_scroll(self) -> None:
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())
