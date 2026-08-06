"""
ui/theme_manager.py — Singleton theme manager for dark / light mode switching.

Usage
-----
    from ui.theme_manager import ThemeManager

    tm = ThemeManager.instance()
    tm.toggle()                # flip between dark and light
    tm.theme_changed.connect(some_slot)   # notified on every switch
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

_STYLES_DIR = Path(__file__).parent / "styles"

Theme = Literal["dark", "light"]


class ThemeManager(QObject):
    """Global singleton that owns the dark/light stylesheet swap."""

    # Emitted after a theme switch — carries the new theme name.
    theme_changed: Signal = Signal(str)

    _instance: "ThemeManager | None" = None

    # ── Palette token sets ─────────────────────────────────────────────────
    DARK_TOKENS: dict[str, str] = {
        "color_bg":               "#090A0F",
        "color_surface":          "#141722",
        "color_surface_alt":      "#1E2235",
        "color_accent":           "#6366F1",
        "color_accent_hover":     "#818CF8",
        "color_cyan":             "#06B6D4",
        "color_text":             "#F8FAFC",
        "color_text_muted":       "#94A3B8",
        "color_user_bubble":      "#312E81",
        "color_assistant_bubble": "#131722",
        # control dock
        "dock_bg":          "rgba(14, 17, 30, 0.96)",
        "dock_border":      "rgba(255, 255, 255, 0.10)",
        "dock_sep":         "rgba(255, 255, 255, 0.08)",
        "btn_bg":           "rgba(22, 27, 44, 0.80)",
        "btn_hover_bg":     "rgba(40, 48, 74, 0.95)",
        "btn_label_color":  "rgba(148, 163, 184, 0.80)",
        # title bar
        "titlebar_bg":      "#090A0F",
        "titlebar_border":  "rgba(255, 255, 255, 0.06)",
        "title_text":       "#F8FAFC",
        "title_muted":      "#94A3B8",
        "title_btn_hover":  "#1E2235",
    }

    LIGHT_TOKENS: dict[str, str] = {
        "color_bg":               "#F0F4F8",
        "color_surface":          "#FFFFFF",
        "color_surface_alt":      "#E2E8F0",
        "color_accent":           "#4F46E5",
        "color_accent_hover":     "#6366F1",
        "color_cyan":             "#0891B2",
        "color_text":             "#0F172A",
        "color_text_muted":       "#64748B",
        "color_user_bubble":      "#EEF2FF",
        "color_assistant_bubble": "#F8FAFC",
        # control dock
        "dock_bg":          "rgba(255, 255, 255, 0.96)",
        "dock_border":      "rgba(148, 163, 184, 0.30)",
        "dock_sep":         "rgba(148, 163, 184, 0.25)",
        "btn_bg":           "rgba(241, 245, 249, 0.90)",
        "btn_hover_bg":     "rgba(226, 232, 240, 0.95)",
        "btn_label_color":  "rgba(100, 116, 139, 0.90)",
        # title bar
        "titlebar_bg":      "#F8FAFC",
        "titlebar_border":  "rgba(148, 163, 184, 0.25)",
        "title_text":       "#0F172A",
        "title_muted":      "#64748B",
        "title_btn_hover":  "#E2E8F0",
    }

    # ── Singleton access ───────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        super().__init__()
        self._theme: Theme = "dark"
        self._dark_qss: str = self._read_qss("dark_theme.qss")
        self._light_qss: str = self._read_qss("light_theme.qss")

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def theme(self) -> Theme:
        return self._theme

    @property
    def is_dark(self) -> bool:
        return self._theme == "dark"

    @property
    def tokens(self) -> dict[str, str]:
        return self.DARK_TOKENS if self.is_dark else self.LIGHT_TOKENS

    def apply(self, theme: Theme) -> None:
        """Switch to the specified theme and notify all listeners."""
        if theme == self._theme:
            return
        self._theme = theme
        app = QApplication.instance()
        if app:
            qss = self._dark_qss if self.is_dark else self._light_qss
            app.setStyleSheet(qss)
        self.theme_changed.emit(self._theme)

    def toggle(self) -> None:
        """Flip between dark and light."""
        self.apply("light" if self.is_dark else "dark")

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _read_qss(filename: str) -> str:
        path = _STYLES_DIR / filename
        return path.read_text(encoding="utf-8") if path.exists() else ""
