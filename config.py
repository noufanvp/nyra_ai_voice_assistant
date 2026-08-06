"""
config.py — Central configuration for the Hybrid AI Voice Assistant.

All tunable parameters are defined here using dataclasses.
Environment variables take precedence where applicable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root (one level up from this file)
_PROJECT_ROOT = Path(__file__).parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Audio / VAD Configuration
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    """Parameters for microphone capture and VAD."""

    # Sample rate expected by Silero VAD and faster-whisper
    sample_rate: int = 16_000

    # Mono audio
    channels: int = 1

    # Number of samples per VAD chunk (512 = 32ms at 16kHz)
    vad_chunk_size: int = 512

    # Speech probability threshold for Silero VAD (0.0 – 1.0)
    vad_threshold: float = 0.5

    # How long silence must persist after speaking before recording stops (seconds).
    # 0.5s provides quick, snappy response turnaround after a sentence ends.
    silence_duration_s: float = 0.5

    # Maximum time to wait for speech to begin before timing out (seconds).
    # If no speech starts within this window, recording stops immediately.
    initial_silence_timeout_s: float = 2.0

    # Minimum speech duration before we treat audio as real input (seconds).
    # 0.25s filters out noise bursts and brief accidental sounds.
    min_speech_duration_s: float = 0.25

    # Maximum recording duration safety cap (seconds)
    max_recording_duration_s: float = 6.0

    # sounddevice dtype
    dtype: str = "float32"


# ---------------------------------------------------------------------------
# STT Configuration
# ---------------------------------------------------------------------------

@dataclass
class STTConfig:
    """Parameters for faster-whisper local STT."""

    # model_size: "tiny.en", "base.en", "small.en", "medium"
    # Defaults to "tiny.en" for low-memory environments (Render free tier 512MB limit).
    # Override via environment variable STT_MODEL_SIZE="base.en" for higher local accuracy.
    model_size: str = field(default_factory=lambda: os.getenv("STT_MODEL_SIZE", "tiny.en"))

    # Inference device: "cpu" or "cuda"
    device: str = "cpu"

    # Quantization: "int8", "float16", "float32"
    compute_type: str = "int8"

    # beam size (3 for accurate decoding without speed penalty)
    beam_size: int = 3

    # Language hint ("en" for English-only model)
    language: str | None = "en"

    # Allowed languages for STT filter
    allowed_languages: list = field(default_factory=lambda: ["en"])

    # Number of CPU threads for CTranslate2
    cpu_threads: int = 2


# ---------------------------------------------------------------------------
# LLM Configuration (Groq API)
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Parameters for the Groq API LLM."""

    # Groq model identifier
    model: str = "llama-3.1-8b-instant"

    # Maximum tokens in the LLM response
    max_tokens: int = 200

    # Temperature (0.0 = deterministic, 1.0 = creative)
    temperature: float = 0.7

    # System prompt tuned for spoken output in English
    system_prompt: str = (
        "You are Nyra, a helpful and friendly AI voice assistant. "
        "You were built by students of Al Irshad Public School under the mentoring of Aitute. "
        "When asked who you are, introduce yourself as Nyra, an AI assistant made by the students of Al Irshad Public School with the help of Aitute. "
        "Always keep responses concise (1–2 short sentences) suitable for text-to-speech without markdown or special formatting."
    )

    # Groq API key — loaded from environment
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # Retry settings for rate-limit (HTTP 429) handling
    max_retries: int = 3
    retry_backoff_base_s: float = 1.0


# ---------------------------------------------------------------------------
# TTS Configuration (kokoro-onnx / pyttsx3 fallback)
# ---------------------------------------------------------------------------

@dataclass
class TTSConfig:
    """Parameters for local TTS engine."""

    # Voice identifier for kokoro-onnx
    voice: str = "af_heart"

    # Speaking speed multiplier (1.0 = normal)
    speed: float = 1.0

    # Language code for kokoro-onnx
    lang: str = "en-us"

    # Path to kokoro ONNX model file
    model_path: str = field(
        default_factory=lambda: str(
            _PROJECT_ROOT / "models" / "kokoro-v1.0.onnx"
        )
    )

    # Path to kokoro voices binary
    voices_path: str = field(
        default_factory=lambda: str(
            _PROJECT_ROOT / "models" / "voices-v1.0.bin"
        )
    )

    # Auto-download model files if missing
    auto_download: bool = True

    # Phonetic pronunciation map for proper nouns before TTS synthesis.
    # Replaces words with phonetic spellings (e.g. 'Aitute' -> 'Eye-toot').
    pronunciation_map: dict = field(default_factory=lambda: {
        "Aitute": "AI-toot",
    })

    # Download URLs
    model_url: str = (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/kokoro-v1.0.onnx"
    )
    voices_url: str = (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/voices-v1.0.bin"
    )


# ---------------------------------------------------------------------------
# Wake Word Configuration
# ---------------------------------------------------------------------------

@dataclass
class WakeWordConfig:
    """Settings for wake-word detection mode."""

    # The assistant's public name (used in greeting + UI title)
    assistant_name: str = "Nyra"

    # Enable wake-word gating (False = Push-to-Talk direct mode, True = wait for wake word)
    enabled: bool = False

    # Accepted wake-word phrases (case-insensitive substring match).
    # Whisper often mishears slight variations, so we list common alternatives
    # including phonetically similar English words it substitutes.
    phrases: list = field(default_factory=lambda: [
        # Direct matches & variants for 'Nyra'
        "hey nyra",
        "nyra",
        "hey nira",
        "nira",
        "hey neera",
        "neera",
        "hey naira",
        "naira",
        "hey near a",
        "near a",
        # Legacy matches
        "hey aura",
        "aura",
        "hey ora",
        "ora",
    ])

    # How long (seconds) the assistant stays active after the wake word
    # before going back to sleep waiting for the next wake word.
    active_timeout_s: float = 30.0

    # Greeting spoken when the wake word is detected
    greeting: str = "Hey! I'm Nyra, how can I help you?"


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

@dataclass
class LoggingConfig:
    """Logging settings."""

    level: str = "INFO"
    fmt: str = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
    datefmt: str = "%H:%M:%S"


# ---------------------------------------------------------------------------
# UI Configuration (PySide6 Desktop)
# ---------------------------------------------------------------------------

@dataclass
class UIConfig:
    """Design tokens and layout parameters for the PySide6 desktop UI."""

    # Obsidian Cyber-Luxe dark colour palette
    color_bg: str = "#090A0F"            # deep obsidian window background
    color_surface: str = "#141722"       # glassmorphic card / panel surfaces
    color_surface_alt: str = "#1E2235"   # elevated surface variant
    color_accent: str = "#6366F1"        # neon indigo accent (buttons, visualizer)
    color_accent_hover: str = "#818CF8"  # lighter indigo on hover
    color_cyan: str = "#06B6D4"          # neon cyan (thinking / cyber elements)
    color_purple: str = "#A855F7"        # neon purple
    color_idle: str = "#64748B"          # IDLE state — slate gray
    color_listening: str = "#10B981"     # LISTENING — emerald green
    color_processing: str = "#F59E0B"    # PROCESSING — amber
    color_speaking: str = "#3B82F6"      # SPEAKING — electric blue
    color_text: str = "#F8FAFC"          # primary bright text
    color_text_muted: str = "#94A3B8"    # secondary / muted text
    color_user_bubble: str = "#312E81"   # user chat bubble background
    color_assistant_bubble: str = "#131722"  # assistant bubble background

    # Asset paths
    avatar_path: str = field(
        default_factory=lambda: str(
            _PROJECT_ROOT / "assets" / "avatar.png"
        )
    )

    # Window geometry
    window_width: int = 480
    window_height: int = 760
    window_min_width: int = 400
    window_min_height: int = 600
    frameless: bool = False              # set True for borderless floating mode

    # Avatar Stage & Visualizer
    visualizer_fps: int = 60            # frames per second for animation
    visualizer_height: int = 220        # pixel height of the avatar waveform widget
    avatar_size: int = 110              # diameter of character avatar frame

    # Animation
    badge_anim_duration_ms: int = 300   # state badge colour crossfade duration
    drawer_anim_duration_ms: int = 280  # settings drawer slide duration


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Top-level application configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)


# Singleton — import this in all modules
CONFIG = AppConfig()
